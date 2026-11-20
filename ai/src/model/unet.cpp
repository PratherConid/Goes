#include "model/unet.h"
#include <limits>
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>

// Match x's channel count to out_ch for a parameter-free residual shortcut:
// clip the extra channels if x has more than out_ch, zero-pad if it has fewer.
static torch::Tensor match_channels(const torch::Tensor& x, int64_t out_ch) {
    int64_t in_ch = x.size(1);
    if (in_ch == out_ch) return x;
    if (in_ch > out_ch) return x.slice(1, 0, out_ch);
    auto pad_sizes = x.sizes().vec();
    pad_sizes[1] = out_ch - in_ch;
    auto pad = torch::zeros(pad_sizes, x.options());
    return torch::cat({x, pad}, 1);
}

// Normalizes across channels only, independently per spatial position (per
// "node") - matches GNN's per-node LayerNorm(hidden_dim). GroupNorm can't
// express this directly: every GroupNorm variant pools some spatial extent
// into its statistics, whereas this holds each (row,col) position's own
// C-dim feature vector fixed and normalizes only across C.
struct ChannelLayerNormImpl : torch::nn::Module {
    torch::nn::LayerNorm ln{nullptr};
    ChannelLayerNormImpl(int channels) {
        ln = register_module("ln", torch::nn::LayerNorm(torch::nn::LayerNormOptions({channels})));
    }
    torch::Tensor forward(const torch::Tensor& x) {
        // (B,C,H,W) -> (B,H,W,C): normalize over the trailing (channel) dim -> back
        auto y = x.permute({0, 2, 3, 1}).contiguous();
        y = ln->forward(y);
        return y.permute({0, 3, 1, 2});
    }
};
TORCH_MODULE(ChannelLayerNorm);

UNetImpl::UNetImpl(const BoardConfig& bc, const UNetConfig& cfg, int num_stones, int num_players)
    : cfg_(cfg), num_stones_(num_stones)
{
    // A real (always-active) check, not assert() - assert compiles out under NDEBUG (this project's
    // Release builds), which would otherwise let a non-2D-embedded board (e.g. regpoly/dodeca/icosa
    // - see board_config.h - all use emb_dim=0) fall through into the side_lens[0]/[1] reads below
    // as undefined behavior instead of failing loudly here.
    if (bc.emb.emb_dim() != 2)
        throw std::runtime_error(
            "UNet requires a 2D embedding (bc.emb.emb_dim() == 2), got emb_dim=" + std::to_string(bc.emb.emb_dim()));
    assert(bc.N > 0);

    // Pad the input descriptor's sideLens (the grid NodeSlots index into) up to a
    // square whose side is the least power of two >= both dimensions. Every
    // level's MaxPool2d(2,2) then halves exactly with no remainder, so no board
    // row/column is ever silently dropped by floor-mode pooling on an odd or
    // non-power-of-two board size.
    const auto side_lens = cfg.input_descr.at("sideLens").get<std::vector<unsigned>>();
    grid_w_ = static_cast<int>(side_lens[0]);
    grid_h_ = static_cast<int>(side_lens[1]);
    int side = 1;
    while (side < std::max(grid_w_, grid_h_)) side <<= 1;
    side_ = side;

    // Compute encoder level count, channel counts, and per-level image size
    // (side length; height == width always). Halve spatial dims until 1×1
    // (exact, since side is a power of two); C_k = hidden_dim * 2^k.
    {
        int h = side, w = side;
        for (int k = 0; ; k++) {
            level_size_.push_back(h);
            int factor = 1 << k;
            enc_channels_.push_back(cfg_.hidden_dim * factor);
            if (h == 1 && w == 1) break;
            h /= 2;
            w /= 2;
        }
        num_levels_ = static_cast<int>(enc_channels_.size());
    }

    // Helper: build a Sequential of n_convs × Conv2d (ReLU between consecutive
    // convs), followed by a single ChannelLayerNorm + ReLU at the end of the
    // block. First conv maps in_ch→out_ch; the rest keep out_ch→out_ch.
    // kernel is 3 normally, but 1 at image_size == 1: a 3×3 kernel on a 1×1
    // input would have 8 of its 9 weight positions permanently multiplying
    // zero-padding (dead weights that never receive gradient), so it
    // degenerates to a 1×1 conv anyway - just build it as one directly.
    auto build_block = [&](const std::string& name, int in_ch, int out_ch,
                            int n_convs, int image_size) {
        int kernel = (image_size == 1) ? 1 : 3;
        int pad = kernel / 2;
        torch::nn::Sequential seq;
        seq->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, out_ch, kernel).padding(pad)));
        for (int i = 1; i < n_convs; i++) {
            seq->push_back(torch::nn::ReLU());
            seq->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(out_ch, out_ch, kernel).padding(pad)));
        }
        seq->push_back(ChannelLayerNorm(out_ch));
        seq->push_back(torch::nn::ReLU());
        return register_module(name, seq);
    };

    // input_proj maps feature_dim+1 (validity channel included) -> hidden_dim
    // once, as a plain embedding step (see forward()) - so enc_blocks_[0]
    // below already starts at hidden_dim width like every other level.
    input_proj = register_module("input_proj",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(cfg_.feature_dim + 1, cfg_.hidden_dim, 1)));

    // Encoder blocks: 4 convs at k=0 (full resolution), 2 convs at deeper levels.
    int in_ch = cfg_.hidden_dim;
    for (int k = 0; k < num_levels_; k++) {
        enc_blocks_.push_back(build_block("enc_" + std::to_string(k),
            in_ch, enc_channels_[k], k == 0 ? 4 : 2, level_size_[k]));
        in_ch = enc_channels_[k];
    }

    // Decoder: for each level k (0..L-1), a 1×1 channel-reduction conv and a
    // conv block. dec_reduce_[k] maps C_{k+1}→C_k before the skip add, so
    // dec_blocks_[k] needs one fewer conv than the matching encoder level to
    // reach the same depth: 3 convs at k=0 (back to full resolution), 1 elsewhere.
    for (int k = 0; k < num_levels_ - 1; k++) {
        dec_reduce_.push_back(register_module("dec_reduce_" + std::to_string(k),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(enc_channels_[k + 1], enc_channels_[k], 1))));
        dec_blocks_.push_back(build_block("dec_" + std::to_string(k),
            enc_channels_[k], enc_channels_[k], k == 0 ? 3 : 1, level_size_[k]));
    }

    // Ownership heads: same full-resolution decoder output as the policy head
    // (see forward()), gathered per-node.
    stone_head = register_module("stone_head", torch::nn::Sequential(
        torch::nn::Linear(enc_channels_[0], enc_channels_[0]),
        torch::nn::ReLU(),
        torch::nn::Linear(enc_channels_[0], num_stones + 1)));
    territory_head = register_module("territory_head", torch::nn::Sequential(
        torch::nn::Linear(enc_channels_[0], enc_channels_[0]),
        torch::nn::ReLU(),
        torch::nn::Linear(enc_channels_[0], num_stones + 1)));

    policy_head = register_module("policy_head", UNetPolicyHead(enc_channels_[0], num_stones));
    rank_head = register_module("rank_head", RankHead(enc_channels_[0], num_rank_pairs(num_players)));
}

UNetPolicyHeadImpl::UNetPolicyHeadImpl(int hidden_dim, int num_stones)
    : num_stones_(num_stones)
{
    proj = register_module("proj", torch::nn::Linear(hidden_dim, num_stones + 1));
    attn = register_module("attn", torch::nn::Linear(hidden_dim, 1));
}

torch::Tensor UNetPolicyHeadImpl::forward(const torch::Tensor& h) {
    int64_t B = h.size(0), S = h.size(1);
    auto out          = proj->forward(h);                        // (B, S, ns+1)
    // Place logits are slot-major out of `proj` ((B,S,numStones)) but must be flattened
    // stone-major (index (stone-1)*S+slot, matching legal_mask's layout - see slot_input(),
    // features.h) - permute before reshaping to make this happen.
    auto place_logits = out.slice(-1, 0, num_stones_)
                            .permute({0, 2, 1})
                            .reshape({B, num_stones_ * S});        // (B, ns*S)
    auto pass_field   = out.select(-1, num_stones_);               // (B, S)
    auto attn_weights = torch::softmax(attn->forward(h).squeeze(-1), -1); // (B, S)
    auto pass_logit   = (attn_weights * pass_field).sum(-1, /*keepdim=*/true); // (B, 1)
    return torch::cat({place_logits, pass_logit}, -1);            // (B, ns*S+1)
}

torch::Tensor UNetImpl::features_to_grid(const torch::Tensor& x, const torch::Tensor& valid) const {
    // Slot s is grid cell (s / W, s % W) (NodeSlots' layout: dimension 0, the column, fastest);
    // the validity channel goes last.
    auto grid = torch::cat({x, valid.unsqueeze(-1)}, -1).permute({0, 2, 1})
                    .reshape({x.size(0), x.size(2) + 1, grid_h_, grid_w_});
    // Zero-pad the H×W grid's right and bottom edges out to side×side (invalid cells).
    return torch::constant_pad_nd(grid, {0, side_ - grid_w_, 0, side_ - grid_h_});
}

ModelOutput UNetImpl::forward(
    torch::Tensor x,
    torch::Tensor valid,
    torch::Tensor legal_mask)
{
    int64_t B = x.size(0);

    // Encoder: apply each block (with a parameter-free residual shortcut
    // around it, channel-matched via clip/zero-pad where channels actually
    // change), save skip output, then MaxPool to the next level. input_proj
    // brings the raw input to hidden_dim width first, as a plain embedding
    // step with no residual of its own (see class doc comment).
    std::vector<torch::Tensor> skips(num_levels_);
    auto h = torch::relu(input_proj->forward(features_to_grid(x, valid))); // (B, hidden_dim, side, side)
    for (int k = 0; k < num_levels_; k++) {
        h = enc_blocks_[k]->forward(h) + match_channels(h, enc_channels_[k]);
        skips[k] = h;
        if (k < num_levels_ - 1)
            // TODO: Instead of max-pooling, just reshape the tensor
            //       to merge all channels of the four squares
            h = torch::max_pool2d(h, {2, 2}, {2, 2});
    }
    // h = skips[L] = bottleneck: (B, C_L, 1, 1)

    // Decoder: for each level from L-1 down to 0,
    //   1. Upsample 2× via nearest-neighbor pixel replication (exact, since
    //      side is a power of two, so each level is exactly double the next)
    //   2. Reduce channels C_{k+1}→C_k via 1×1 conv
    //   3. Add encoder skip (additive residual)
    //   4. Apply decoder conv block, with a plain identity residual shortcut
    //      (dec_reduce_'s 1×1 conv plus the skip add above already bring h to
    //      dec_blocks_[k]'s constant enc_channels_[k] width, so no channel
    //      matching is ever needed here)
    for (int k = num_levels_ - 2; k >= 0; k--) {
        h = h.repeat_interleave(2, /*dim=*/2).repeat_interleave(2, /*dim=*/3);
        h = dec_reduce_[k]->forward(h);     // C_{k+1} → C_k
        h = h + skips[k];                   // additive residual from encoder
        h = dec_blocks_[k]->forward(h) + h;
    }
    // h: (B, C_0, side, side) - same full-resolution decoder output the policy head reads

    // The decoder's full-resolution output, cropped back to the H×W grid, at every slot - feeds the
    // ownership heads (stone/territory estimate), the policy head and the rank head, all of which
    // operate purely on this per-slot (B,S,C_0) tensor rather than the raw grid.
    auto h_slots = h.slice(2, 0, grid_h_).slice(3, 0, grid_w_)
                    .reshape({B, enc_channels_[0], -1})
                    .permute({0, 2, 1});                                  // (B, S, C_0)
    auto stone_est     = torch::softmax(stone_head->forward(h_slots), -1);      // (B, S, num_stones+1)
    auto territory_est = torch::softmax(territory_head->forward(h_slots), -1);  // (B, S, num_stones+1)
    auto ownership = torch::stack({stone_est, territory_est}, 1);              // (B, 2, S, num_stones+1)
    auto rank = rank_head->forward(h_slots);                            // (B, num_rank_pairs, 3)

    auto logits = policy_head->forward(h_slots); // (B, num_stones*S+1)
    const float NEG_INF = -std::numeric_limits<float>::infinity();
    logits = logits.masked_fill(legal_mask.logical_not(), NEG_INF);
    logits = logits.masked_fill(legal_mask.any(-1, true).logical_not(), 0.0f);
    auto policy = torch::softmax(logits, -1);
    return {policy, ownership, rank};
}

ModelInput UNetImpl::prepare_input(const BoardState& state, const std::shared_ptr<const GameInput>& game) const {
    return slot_input(state, cfg_.input_descr, game);
}

ModelOutput UNetImpl::evaluate_inputs(const std::vector<const ModelInput*>& inputs) {
    torch::NoGradGuard ng;
    auto dev = input_proj->weight.device();
    auto x     = stack_inputs(inputs, &ModelInput::features).to(dev);   // (B, S, F)
    auto valid = stack_valid(inputs).to(dev);                           // (B, S)
    auto mask  = stack_inputs(inputs, &ModelInput::legal_mask).to(dev); // (B, ns*S+1)
    return forward(x, valid, mask);
}
