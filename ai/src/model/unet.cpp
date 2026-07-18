#include "model/unet.h"
#include <limits>
#include <algorithm>
#include <cassert>
#include <omp.h>

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

UNetImpl::UNetImpl(const BoardConfig& bc, const UNetConfig& cfg, int num_players, int num_stones)
    : cfg_(cfg), num_players_(num_players), num_stones_(num_stones)
{
    assert(bc.emb_dim == 2 && "UNet requires a 2D embedding (bc.emb_dim == 2)");
    assert(bc.N > 0);

    // Compute the board's tight bounding box from embedding, then pad it up to
    // a square whose side is the least power of two >= both dimensions. Every
    // level's MaxPool2d(2,2) then halves exactly with no remainder, so no
    // board row/column is ever silently dropped by floor-mode pooling on an
    // odd or non-power-of-two board size.
    int gw = 0, gh = 0;
    for (int i = 0; i < bc.N; i++) {
        gw = std::max(gw, (int)bc.embed[i][0] + 1);
        gh = std::max(gh, (int)bc.embed[i][1] + 1);
    }
    int side = 1;
    while (side < std::max(gw, gh)) side <<= 1;
    side_ = side;

    // Build per-node linear index (row*side + col) and validity grid over the
    // padded side×side canvas; cells outside the original gw×gh board stay
    // invalid (valid_tensor_ = 0) and never receive a lin_idx entry.
    auto lin_cpu = torch::zeros({bc.N}, torch::kLong);
    auto val_cpu = torch::zeros({1, 1, side, side}, torch::kFloat32);
    {
        auto lin_a = lin_cpu.accessor<int64_t, 1>();
        auto val_a = val_cpu.accessor<float, 4>();
        for (int i = 0; i < bc.N; i++) {
            int col = (int)bc.embed[i][0];
            int row = (int)bc.embed[i][1];
            lin_a[i]              = row * side + col;
            val_a[0][0][row][col] = 1.0f;
        }
    }
    lin_idx_      = register_buffer("lin_idx",      lin_cpu);
    valid_tensor_ = register_buffer("valid_tensor", val_cpu);

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

    // Encoder blocks: 4 convs at k=0 (full resolution), 2 convs at deeper levels.
    int in_ch = cfg_.feature_dim + 1;  // F node features + 1 validity channel
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

    // Value heads: same full-resolution decoder output as the policy head
    // (see forward()), gathered per-node.
    stone_head = register_module("stone_head", torch::nn::Sequential(
        torch::nn::Linear(enc_channels_[0], 64),
        torch::nn::ReLU(),
        torch::nn::Linear(64, num_stones + 1)));
    territory_head = register_module("territory_head", torch::nn::Sequential(
        torch::nn::Linear(enc_channels_[0], 64),
        torch::nn::ReLU(),
        torch::nn::Linear(64, num_stones + 1)));

    policy_head = register_module("policy_head", UNetPolicyHead(enc_channels_[0], bc.N, num_stones));
}

UNetPolicyHeadImpl::UNetPolicyHeadImpl(int in_channels, int num_nodes, int num_stones)
    : num_stones_(num_stones)
{
    conv = register_module("conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, num_stones + 1, 1)));
    pass_reduce = register_module("pass_reduce", torch::nn::Linear(num_nodes, 1));
}

torch::Tensor UNetPolicyHeadImpl::forward(const torch::Tensor& h, const torch::Tensor& lin_idx) {
    int64_t N = lin_idx.size(0);
    auto out = conv->forward(h).view({h.size(0), num_stones_ + 1, -1}); // (B, ns+1, H*W)
    auto place_logits = out.slice(1, 0, num_stones_).index_select(2, lin_idx)
                            .reshape({h.size(0), num_stones_ * N});    // (B, ns*N)
    auto pass_field   = out.select(1, num_stones_).index_select(1, lin_idx); // (B, N)
    auto pass_logit   = pass_reduce->forward(pass_field);              // (B, 1)
    return torch::cat({place_logits, pass_logit}, -1);                 // (B, ns*N+1)
}

torch::Tensor UNetImpl::features_to_grid(const torch::Tensor& x) const {
    bool batched = x.dim() == 3;
    auto x3 = batched ? x : x.unsqueeze(0);
    int64_t B = x3.size(0);
    int64_t N = x3.size(1);
    int64_t F = x3.size(2);

    // flat: (B, F+1, side*side), all zeros
    auto flat = torch::zeros({B, F + 1, (int64_t)side_ * side_}, x3.options());

    // Scatter node features into channels 0..F-1
    auto idx = lin_idx_.unsqueeze(0).unsqueeze(0).expand({B, F, N});
    flat.slice(1, 0, F).scatter_(2, idx, x3.permute({0, 2, 1}));

    // Validity channel (same for all batch items)
    flat.select(1, F).copy_(valid_tensor_.view({1, -1}).expand({B, -1}));

    return flat.view({B, F + 1, side_, side_});
}

std::pair<torch::Tensor, torch::Tensor> UNetImpl::forward(
    torch::Tensor x,
    torch::Tensor legal_mask)
{
    bool batched = x.dim() == 3;
    if (!batched) {
        x          = x.unsqueeze(0);
        legal_mask = legal_mask.unsqueeze(0);
    }

    int64_t B = x.size(0);

    // Encoder: apply each block (with a parameter-free residual shortcut
    // around it, channel-matched via clip/zero-pad), save skip output, then
    // MaxPool to the next level.
    std::vector<torch::Tensor> skips(num_levels_);
    auto h = features_to_grid(x);            // (B, F+1, H, W)
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
    //   4. Apply decoder conv block, with the same residual shortcut as the
    //      encoder (always a plain identity add here, since dec_blocks_ never
    //      changes channel count)
    for (int k = num_levels_ - 2; k >= 0; k--) {
        h = h.repeat_interleave(2, /*dim=*/2).repeat_interleave(2, /*dim=*/3);
        h = dec_reduce_[k]->forward(h);     // C_{k+1} → C_k
        h = h + skips[k];                   // additive residual from encoder
        h = dec_blocks_[k]->forward(h) + match_channels(h, enc_channels_[k]);
    }
    // h: (B, C_0, H, W) - same full-resolution decoder output the policy head reads

    // Value: same input as the policy head, gathered per-node (same lin_idx_
    // mechanism the policy head uses) then passed through two independent
    // per-location softmax heads (stone estimate, territory estimate).
    auto h_nodes = h.reshape({B, enc_channels_[0], -1})
                    .index_select(2, lin_idx_)
                    .permute({0, 2, 1});                                  // (B, N, C_0)
    auto stone_est     = torch::softmax(stone_head->forward(h_nodes), -1);      // (B, N, num_stones+1)
    auto territory_est = torch::softmax(territory_head->forward(h_nodes), -1);  // (B, N, num_stones+1)
    auto ownership = torch::stack({stone_est, territory_est}, 1);              // (B, 2, N, num_stones+1)

    auto logits = policy_head->forward(h, lin_idx_); // (B, num_stones*N+1)
    const float NEG_INF = -std::numeric_limits<float>::infinity();
    logits = logits.masked_fill(legal_mask.logical_not(), NEG_INF);
    logits = logits.masked_fill(legal_mask.any(-1, true).logical_not(), 0.0f);
    auto policy = torch::softmax(logits, -1);

    if (!batched)
        return {policy.squeeze(0), ownership.squeeze(0)};
    return {policy, ownership};
}

std::pair<torch::Tensor, torch::Tensor> UNetImpl::evaluate(const BoardState& state) {
    torch::NoGradGuard ng;
    auto dev = lin_idx_.device();
    auto [ft, mask] = board_to_features(state, dev, cfg_.input_descr);
    auto [policy, ownership] = forward(ft, mask);
    return {policy, ownership};
}

static std::pair<torch::Tensor, torch::Tensor> run_batch(
    UNetImpl* self,
    const std::vector<const BoardState*>& states)
{
    torch::NoGradGuard ng;
    auto dev = self->lin_idx_.device();
    int B = static_cast<int>(states.size());
    std::vector<torch::Tensor> feats(B), masks(B);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < B; i++) {
        auto [ft, mask] = board_to_features(*states[i], dev, self->cfg_.input_descr);
        feats[i] = ft;
        masks[i] = mask;
    }
    auto x    = torch::stack(feats, 0); // (B, N, F)
    auto mask = torch::stack(masks, 0); // (B, N+1)
    auto [policy, ownership] = self->forward(x, mask);
    return {policy, ownership};
}

std::pair<torch::Tensor, torch::Tensor> UNetImpl::evaluate_batch(
    const std::vector<BoardState*>& states)
{
    std::vector<const BoardState*> cstates(states.begin(), states.end());
    return run_batch(this, cstates);
}

std::pair<torch::Tensor, torch::Tensor> UNetImpl::evaluate_batch(
    const std::vector<const BoardState*>& states)
{
    return run_batch(this, states);
}
