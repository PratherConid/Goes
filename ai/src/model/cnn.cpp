#include "model/cnn.h"
#include <limits>
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>

CNNImpl::CNNImpl(const BoardConfig& bc, const CNNConfig& cfg, int num_stones, int num_players)
    : cfg_(cfg), num_stones_(num_stones)
{
    // A real (always-active) check, not assert() - assert compiles out under NDEBUG (this project's
    // Release builds), which would otherwise let a non-2D-embedded board (e.g. regpoly/dodeca/icosa
    // - see board_config.h - all use emb_dim=0) fall through into the side_lens[0]/[1] reads below
    // as undefined behavior instead of failing loudly here.
    if (bc.emb.emb_dim() != 2)
        throw std::runtime_error(
            "CNN requires a 2D embedding (bc.emb.emb_dim() == 2), got emb_dim=" + std::to_string(bc.emb.emb_dim()));
    assert(bc.N > 0);

    // The grid is the input descriptor's sideLens (no padding), which NodeSlots index into.
    const auto side_lens = cfg.input_descr.at("sideLens").get<std::vector<unsigned>>();
    int gw = static_cast<int>(side_lens[0]), gh = static_cast<int>(side_lens[1]);
    grid_w_ = gw;
    grid_h_ = gh;

    num_blocks_ = std::max(gw, gh);

    input_proj = register_module("input_proj",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(cfg_.feature_dim + 1, cfg_.hidden_dim, 1)));

    // "Same" padding (conv_size/2, integer division - exact since conv_size is enforced odd) keeps
    // H,W unchanged across every conv, which forward()'s residual add requires.
    int pad = cfg_.conv_size / 2;
    for (int k = 0; k < num_blocks_; k++) {
        torch::nn::Sequential seq;
        seq->push_back(torch::nn::Conv2d(
            torch::nn::Conv2dOptions(cfg_.hidden_dim, cfg_.hidden_dim, cfg_.conv_size).padding(pad)
        ));
        seq->push_back(torch::nn::ReLU());
        seq->push_back(torch::nn::Conv2d(
            torch::nn::Conv2dOptions(cfg_.hidden_dim, cfg_.hidden_dim, cfg_.conv_size).padding(pad)
        ));
        blocks_.push_back(register_module("block_" + std::to_string(k), seq));
    }

    stone_head = register_module("stone_head", torch::nn::Sequential(
        torch::nn::Linear(cfg_.hidden_dim, cfg_.hidden_dim),
        torch::nn::ReLU(),
        torch::nn::Linear(cfg_.hidden_dim, num_stones + 1)));
    territory_head = register_module("territory_head", torch::nn::Sequential(
        torch::nn::Linear(cfg_.hidden_dim, cfg_.hidden_dim),
        torch::nn::ReLU(),
        torch::nn::Linear(cfg_.hidden_dim, num_stones + 1)));

    policy_head = register_module("policy_head", CNNPolicyHead(cfg_.hidden_dim, num_stones));
    rank_head = register_module("rank_head", RankHead(cfg_.hidden_dim, num_rank_pairs(num_players)));
}

CNNPolicyHeadImpl::CNNPolicyHeadImpl(int hidden_dim, int num_stones)
    : num_stones_(num_stones)
{
    proj = register_module("proj", torch::nn::Linear(hidden_dim, num_stones + 1));
    attn = register_module("attn", torch::nn::Linear(hidden_dim, 1));
}

torch::Tensor CNNPolicyHeadImpl::forward(const torch::Tensor& h) {
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

torch::Tensor CNNImpl::features_to_grid(const torch::Tensor& x, const torch::Tensor& valid) const {
    // Slot s is grid cell (s / W, s % W) (NodeSlots' layout: dimension 0, the column, fastest);
    // the validity channel goes last.
    auto grid = torch::cat({x, valid.unsqueeze(-1)}, -1).permute({0, 2, 1})
                    .reshape({x.size(0), x.size(2) + 1, grid_h_, grid_w_});
    return grid;
}

ModelOutput CNNImpl::forward(
    torch::Tensor x,
    torch::Tensor valid,
    torch::Tensor legal_mask)
{
    int64_t B = x.size(0);

    auto h = torch::relu(input_proj->forward(features_to_grid(x, valid))); // (B, hidden_dim, H, W)
    for (int k = 0; k < num_blocks_; k++) {
        auto out = blocks_[k]->forward(h);
        h = torch::relu(out + h);
    }
    // h: (B, hidden_dim, H, W)

    // The final block output at every slot - feeds the ownership heads (stone/territory estimate),
    // the policy head and the rank head, all of which operate purely on this per-slot
    // (B,S,hidden_dim) tensor rather than the raw grid.
    auto h_slots = h.reshape({B, cfg_.hidden_dim, -1}).permute({0, 2, 1});  // (B, S, hidden_dim)
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

ModelInput CNNImpl::prepare_input(const BoardState& state, const std::shared_ptr<const GameInput>& game) const {
    return slot_input(state, cfg_.input_descr, game);
}

ModelOutput CNNImpl::evaluate_inputs(const std::vector<const ModelInput*>& inputs) {
    torch::NoGradGuard ng;
    auto dev = input_proj->weight.device();
    auto x     = stack_inputs(inputs, &ModelInput::features).to(dev);   // (B, S, F)
    auto valid = stack_valid(inputs).to(dev);                           // (B, S)
    auto mask  = stack_inputs(inputs, &ModelInput::legal_mask).to(dev); // (B, ns*S+1)
    return forward(x, valid, mask);
}
