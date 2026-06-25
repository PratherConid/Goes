#include "model/cnn.h"
#include <limits>
#include <algorithm>
#include <cassert>
#include <omp.h>

ConvNNImpl::ConvNNImpl(const BoardConfig& bc, int in_dim, int hidden_dim, int num_players)
    : hidden_dim_(hidden_dim), num_players_(num_players)
{
    assert(bc.emb_dim == 2 && "ConvNN requires a 2D embedding (bc.emb_dim == 2)");
    assert(bc.N > 0);

    // Compute grid dimensions from embedding
    int gw = 0, gh = 0;
    for (int i = 0; i < bc.N; i++) {
        gw = std::max(gw, (int)bc.embed[i][0] + 1);
        gh = std::max(gh, (int)bc.embed[i][1] + 1);
    }
    grid_w_ = gw;
    grid_h_ = gh;

    // Build per-node linear index (row*W + col) and validity grid
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

    // Compute encoder level sizes and channel counts.
    // Halve spatial dims (floor) until 1×1; C_k = hidden_dim * min(2^k, 8).
    {
        int h = gh, w = gw;
        for (int k = 0; ; k++) {
            enc_sizes_.push_back({h, w});
            int factor = (k >= 3) ? 8 : (1 << k);
            enc_channels_.push_back(hidden_dim * factor);
            if (h == 1 && w == 1) break;
            h = std::max(h / 2, 1);
            w = std::max(w / 2, 1);
        }
        num_levels_ = static_cast<int>(enc_sizes_.size());
    }

    // Helper: build a Sequential of n_convs × (Conv2d + GroupNorm(1) + ReLU).
    // First conv maps in_ch→out_ch; the rest keep out_ch→out_ch.
    auto build_block = [&](const std::string& name, int in_ch, int out_ch, int n_convs) {
        torch::nn::Sequential seq;
        seq->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, out_ch, 3).padding(1)));
        seq->push_back(torch::nn::GroupNorm(torch::nn::GroupNormOptions(1, out_ch)));
        seq->push_back(torch::nn::ReLU());
        for (int i = 1; i < n_convs; i++) {
            seq->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(out_ch, out_ch, 3).padding(1)));
            seq->push_back(torch::nn::GroupNorm(torch::nn::GroupNormOptions(1, out_ch)));
            seq->push_back(torch::nn::ReLU());
        }
        return register_module(name, seq);
    };

    // Encoder blocks: 4 convs at k=0 (full resolution), 2 convs at deeper levels.
    int in_ch = in_dim + 1;  // F node features + 1 validity channel
    for (int k = 0; k < num_levels_; k++) {
        enc_blocks_.push_back(build_block("enc_" + std::to_string(k),
            in_ch, enc_channels_[k], k == 0 ? 4 : 2));
        in_ch = enc_channels_[k];
    }

    // Decoder: for each level k (0..L-1), a 1×1 channel-reduction conv and a
    // conv block. dec_reduce_[k] maps C_{k+1}→C_k before the skip add.
    // dec_blocks_[k] has 4 convs at k=0 (back to full resolution), 2 elsewhere.
    for (int k = 0; k < num_levels_ - 1; k++) {
        dec_reduce_.push_back(register_module("dec_reduce_" + std::to_string(k),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(enc_channels_[k + 1], enc_channels_[k], 1))));
        dec_blocks_.push_back(build_block("dec_" + std::to_string(k),
            enc_channels_[k], enc_channels_[k], k == 0 ? 4 : 2));
    }

    // Value head on bottleneck (flattened C_L features from the 1×1 spatial output)
    int C_L = enc_channels_[num_levels_ - 1];
    value_head = register_module("value_head", torch::nn::Sequential(
        torch::nn::Linear(C_L, 64),
        torch::nn::ReLU(),
        torch::nn::Linear(64, num_players),
        torch::nn::Tanh()));

    policy_head = register_module("policy_head",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(enc_channels_[0], 1, 1)));

    pass_logit = register_parameter("pass_logit", torch::zeros({1}));
}

torch::Tensor ConvNNImpl::features_to_grid(const torch::Tensor& x) const {
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

torch::Tensor ConvNNImpl::grid_to_node_logits(const torch::Tensor& policy_grid) const {
    // (B, H, W) → (B, H*W) → index at lin_idx_ → (B, N)
    return policy_grid.view({policy_grid.size(0), -1}).index_select(1, lin_idx_);
}

std::pair<torch::Tensor, torch::Tensor> ConvNNImpl::forward(
    torch::Tensor x,
    torch::Tensor legal_mask)
{
    bool batched = x.dim() == 3;
    if (!batched) {
        x          = x.unsqueeze(0);
        legal_mask = legal_mask.unsqueeze(0);
    }

    int64_t B = x.size(0);

    // Encoder: apply each block, save skip output, then MaxPool to the next level.
    std::vector<torch::Tensor> skips(num_levels_);
    auto h = features_to_grid(x);            // (B, F+1, H, W)
    for (int k = 0; k < num_levels_; k++) {
        h = enc_blocks_[k]->forward(h);
        skips[k] = h;
        if (k < num_levels_ - 1)
            h = torch::max_pool2d(h, {2, 2}, {2, 2});
    }
    // h = skips[L] = bottleneck: (B, C_L, 1, 1)

    // Value head: flatten bottleneck → (B, C_L) → (B, num_players)
    auto value = value_head->forward(h.flatten(1));
    // Normalise to zero-sum across players: subtract the per-row mean so the
    // per-player values sum to zero (the model cannot rate everyone as winning).
    value = value - value.mean(-1, /*keepdim=*/true);

    // Decoder: for each level from L-1 down to 0,
    //   1. Upsample to the recorded encoder spatial size (nearest-neighbor)
    //   2. Reduce channels C_{k+1}→C_k via 1×1 conv
    //   3. Add encoder skip (additive residual)
    //   4. Apply decoder conv block
    namespace F = torch::nn::functional;
    for (int k = num_levels_ - 2; k >= 0; k--) {
        auto [enc_h, enc_w] = enc_sizes_[k];
        h = F::interpolate(h, F::InterpolateFuncOptions()
                .size(std::vector<int64_t>{enc_h, enc_w})
                .mode(torch::kNearest));
        h = dec_reduce_[k]->forward(h);     // C_{k+1} → C_k
        h = h + skips[k];                   // additive residual from encoder
        h = dec_blocks_[k]->forward(h);
    }
    // h: (B, C_0, H, W)

    // Policy head: (B, 1, H, W) → squeeze → (B, H, W) → extract node logits
    auto node_logits = grid_to_node_logits(policy_head->forward(h).squeeze(1));
    auto logits = torch::cat({node_logits, pass_logit.expand({B, 1})}, -1);
    const float NEG_INF = -std::numeric_limits<float>::infinity();
    logits = logits.masked_fill(legal_mask.logical_not(), NEG_INF);
    logits = logits.masked_fill(legal_mask.any(-1, true).logical_not(), 0.0f);
    auto policy = torch::softmax(logits, -1);

    if (!batched)
        return {policy.squeeze(0), value.squeeze(0)};
    return {policy, value};
}

std::pair<torch::Tensor, torch::Tensor> ConvNNImpl::evaluate(const BoardState& state) {
    torch::NoGradGuard ng;
    auto dev = lin_idx_.device();
    auto [ft, mask] = board_to_features(state, dev);
    auto [policy, value] = forward(ft, mask);
    return {policy.cpu(), value.cpu()};
}

static std::pair<torch::Tensor, torch::Tensor> run_batch(
    ConvNNImpl* self,
    const std::vector<const BoardState*>& states)
{
    torch::NoGradGuard ng;
    auto dev = self->lin_idx_.device();
    int B = static_cast<int>(states.size());
    std::vector<torch::Tensor> feats(B), masks(B);
    // Build features on CPU inside the parallel region — calling .to(CUDA) from
    // OpenMP worker threads concurrently is unsafe. Move to device once below.
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < B; i++) {
        auto [ft, mask] = board_to_features(*states[i], torch::kCPU);
        feats[i] = ft;
        masks[i] = mask;
    }
    auto x    = torch::stack(feats, 0).to(dev); // (B, N, F)
    auto mask = torch::stack(masks, 0).to(dev); // (B, N+1)
    auto [policy, value] = self->forward(x, mask);
    return {policy.cpu(), value.cpu()};
}

std::pair<torch::Tensor, torch::Tensor> ConvNNImpl::evaluate_batch(
    const std::vector<BoardState*>& states)
{
    std::vector<const BoardState*> cstates(states.begin(), states.end());
    return run_batch(this, cstates);
}

std::pair<torch::Tensor, torch::Tensor> ConvNNImpl::evaluate_batch(
    const std::vector<const BoardState*>& states)
{
    return run_batch(this, states);
}
