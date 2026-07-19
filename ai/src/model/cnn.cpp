#include "model/cnn.h"
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

CNNImpl::CNNImpl(const BoardConfig& bc, const CNNConfig& cfg, int num_players, int num_stones)
    : cfg_(cfg), num_players_(num_players), num_stones_(num_stones)
{
    assert(bc.emb_dim == 2 && "CNN requires a 2D embedding (bc.emb_dim == 2)");
    assert(bc.N > 0);

    // Compute the board's tight bounding box from embedding (no padding).
    int gw = 0, gh = 0;
    for (int i = 0; i < bc.N; i++) {
        gw = std::max(gw, (int)bc.embed[i][0] + 1);
        gh = std::max(gh, (int)bc.embed[i][1] + 1);
    }
    grid_w_ = gw;
    grid_h_ = gh;

    // Build per-node linear index (row*grid_w_+col) and validity grid.
    auto lin_cpu = torch::zeros({bc.N}, torch::kLong);
    auto val_cpu = torch::zeros({1, 1, gh, gw}, torch::kFloat32);
    {
        auto lin_a = lin_cpu.accessor<int64_t, 1>();
        auto val_a = val_cpu.accessor<float, 4>();
        for (int i = 0; i < bc.N; i++) {
            int col = (int)bc.embed[i][0];
            int row = (int)bc.embed[i][1];
            lin_a[i]              = row * gw + col;
            val_a[0][0][row][col] = 1.0f;
        }
    }
    lin_idx_      = register_buffer("lin_idx",      lin_cpu);
    valid_tensor_ = register_buffer("valid_tensor", val_cpu);

    num_blocks_ = std::max(gw, gh);

    // Blocks: constant hidden_dim width throughout; only block 0's input
    // channel count differs (feature_dim + 1 validity channel). "Same"
    // padding (conv_size/2, integer division - exact since conv_size is
    // enforced odd) keeps H,W unchanged across every conv, which the
    // residual add below requires (match_channels only reconciles channel
    // count, not spatial dims).
    int in_ch = cfg_.feature_dim + 1;
    int pad = cfg_.conv_size / 2;
    for (int k = 0; k < num_blocks_; k++) {
        torch::nn::Sequential seq;
        seq->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, cfg_.hidden_dim, cfg_.conv_size).padding(pad)));
        seq->push_back(torch::nn::ReLU());
        seq->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(cfg_.hidden_dim, cfg_.hidden_dim, cfg_.conv_size).padding(pad)));
        blocks_.push_back(register_module("block_" + std::to_string(k), seq));
        in_ch = cfg_.hidden_dim;
    }

    stone_head = register_module("stone_head", torch::nn::Sequential(
        torch::nn::Linear(cfg_.hidden_dim, 64),
        torch::nn::ReLU(),
        torch::nn::Linear(64, num_stones + 1)));
    territory_head = register_module("territory_head", torch::nn::Sequential(
        torch::nn::Linear(cfg_.hidden_dim, 64),
        torch::nn::ReLU(),
        torch::nn::Linear(64, num_stones + 1)));

    policy_head = register_module("policy_head", CNNPolicyHead(cfg_.hidden_dim, bc.N, num_stones));
}

CNNPolicyHeadImpl::CNNPolicyHeadImpl(int in_channels, int num_nodes, int num_stones)
    : num_stones_(num_stones)
{
    conv = register_module("conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, num_stones + 1, 1)));
    pass_reduce = register_module("pass_reduce", torch::nn::Linear(num_nodes, 1));
}

torch::Tensor CNNPolicyHeadImpl::forward(const torch::Tensor& h, const torch::Tensor& lin_idx) {
    int64_t N = lin_idx.size(0);
    auto out = conv->forward(h).view({h.size(0), num_stones_ + 1, -1}); // (B, ns+1, H*W)
    // Place channels are already stone-major/channel-first: (B, ns, N)
    // reshapes directly to (B, ns*N) with no permute needed.
    auto place_logits = out.slice(1, 0, num_stones_).index_select(2, lin_idx)
                            .reshape({h.size(0), num_stones_ * N});    // (B, ns*N)
    auto pass_field   = out.select(1, num_stones_).index_select(1, lin_idx); // (B, N)
    auto pass_logit   = pass_reduce->forward(pass_field);              // (B, 1)
    return torch::cat({place_logits, pass_logit}, -1);                 // (B, ns*N+1)
}

torch::Tensor CNNImpl::features_to_grid(const torch::Tensor& x) const {
    bool batched = x.dim() == 3;
    auto x3 = batched ? x : x.unsqueeze(0);
    int64_t B = x3.size(0);
    int64_t N = x3.size(1);
    int64_t F = x3.size(2);

    // flat: (B, F+1, H*W), all zeros
    auto flat = torch::zeros({B, F + 1, (int64_t)grid_h_ * grid_w_}, x3.options());

    // Scatter node features into channels 0..F-1
    auto idx = lin_idx_.unsqueeze(0).unsqueeze(0).expand({B, F, N});
    flat.slice(1, 0, F).scatter_(2, idx, x3.permute({0, 2, 1}));

    // Validity channel (same for all batch items)
    flat.select(1, F).copy_(valid_tensor_.view({1, -1}).expand({B, -1}));

    return flat.view({B, F + 1, grid_h_, grid_w_});
}

std::pair<torch::Tensor, torch::Tensor> CNNImpl::forward(
    torch::Tensor x,
    torch::Tensor legal_mask)
{
    bool batched = x.dim() == 3;
    if (!batched) {
        x          = x.unsqueeze(0);
        legal_mask = legal_mask.unsqueeze(0);
    }

    int64_t B = x.size(0);

    // Each block: two 3x3 convs, then a residual shortcut from the block's
    // input (channel-matched via clip/zero-pad) added before the final ReLU.
    auto h = features_to_grid(x); // (B, F+1, H, W)
    for (int k = 0; k < num_blocks_; k++) {
        auto out = blocks_[k]->forward(h);
        h = torch::relu(out + match_channels(h, cfg_.hidden_dim));
    }
    // h: (B, hidden_dim, H, W)

    // Value: gather the final block output at each node's grid position (same
    // lin_idx_ mechanism the policy head uses) to get per-node features, then
    // apply two independent per-location softmax heads (stone/territory estimate).
    auto h_nodes = h.reshape({B, cfg_.hidden_dim, -1})
                    .index_select(2, lin_idx_)
                    .permute({0, 2, 1});                                  // (B, N, hidden_dim)
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

std::pair<torch::Tensor, torch::Tensor> CNNImpl::evaluate(const BoardState& state) {
    torch::NoGradGuard ng;
    auto dev = lin_idx_.device();
    auto [ft, mask] = board_to_features(state, dev, cfg_.input_descr);
    auto [policy, ownership] = forward(ft, mask);
    return {policy, ownership};
}

static std::pair<torch::Tensor, torch::Tensor> run_batch(
    CNNImpl* self,
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

std::pair<torch::Tensor, torch::Tensor> CNNImpl::evaluate_batch(
    const std::vector<BoardState*>& states)
{
    std::vector<const BoardState*> cstates(states.begin(), states.end());
    return run_batch(this, cstates);
}

std::pair<torch::Tensor, torch::Tensor> CNNImpl::evaluate_batch(
    const std::vector<const BoardState*>& states)
{
    return run_batch(this, states);
}
