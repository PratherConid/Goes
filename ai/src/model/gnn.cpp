#include "model/gnn.h"
#include <limits>
#include <cmath>
#include <omp.h>

MessagePassingGNNImpl::MessagePassingGNNImpl(const GNNConfig& cfg, int num_players,
                                              int num_stones, const AdjNorms& adj_norms)
    : cfg_(cfg), num_players_(num_players), num_stones_(num_stones)
{
    input_proj = register_module("input_proj", torch::nn::Linear(cfg_.feature_dim, cfg_.hidden_dim));

    // Untrainable random Gaussian embedding table for neighbor counts: index d
    // (0..max_degree) maps to a fixed random vector, giving degree more
    // representational capacity than a single concatenated scalar.
    int max_degree = adj_norms.max_degree;
    int deg_embed_len = static_cast<int>(std::round(
        2.0 * (std::sqrt(static_cast<double>(max_degree)) + 4.0)));
    deg_embed_ = register_buffer("deg_embed", torch::randn({max_degree + 1, deg_embed_len}));

    // Each layer aggregates [h_self, h_neighbour_avg, neighbor_count_embedding] → new h
    for (int i = 0; i < cfg_.num_layers; i++) {
        auto layer = register_module("layer_" + std::to_string(i),
            torch::nn::Linear(2 * cfg_.hidden_dim + deg_embed_len, cfg_.hidden_dim));
        auto ln = register_module("ln_" + std::to_string(i),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({cfg_.hidden_dim})));
        layers.push_back(layer);
        layer_norms.push_back(ln);
    }

    stone_head = register_module("stone_head", torch::nn::Sequential(
        torch::nn::Linear(cfg_.hidden_dim, 64),
        torch::nn::ReLU(),
        torch::nn::Linear(64, num_stones + 1)
    ));
    territory_head = register_module("territory_head", torch::nn::Sequential(
        torch::nn::Linear(cfg_.hidden_dim, 64),
        torch::nn::ReLU(),
        torch::nn::Linear(64, num_stones + 1)
    ));
    policy_head = register_module("policy_head", GNNPolicyHead(cfg_.hidden_dim, num_stones));
}

GNNPolicyHeadImpl::GNNPolicyHeadImpl(int hidden_dim, int num_stones)
    : num_stones_(num_stones)
{
    proj = register_module("proj", torch::nn::Linear(hidden_dim, num_stones + 1));
    attn = register_module("attn", torch::nn::Linear(hidden_dim, 1));
}

torch::Tensor GNNPolicyHeadImpl::forward(const torch::Tensor& h) {
    int64_t B = h.size(0), N = h.size(1);
    auto out          = proj->forward(h);                        // (B, N, ns+1)
    // Node-major (B,N,ns) -> stone-major (B,ns,N) -> flat (B,ns*N). The permute
    // is required here (unlike CNN, whose conv output is already channel-first)
    // to match legal_mask's stone-major flatten order.
    auto place_logits = out.slice(-1, 0, num_stones_)
                            .permute({0, 2, 1})
                            .reshape({B, num_stones_ * N});        // (B, ns*N)
    auto pass_field   = out.select(-1, num_stones_);               // (B, N)
    auto attn_weights = torch::softmax(attn->forward(h).squeeze(-1), -1); // (B, N)
    auto pass_logit   = (attn_weights * pass_field).sum(-1, /*keepdim=*/true); // (B, 1)
    return torch::cat({place_logits, pass_logit}, -1);            // (B, ns*N+1)
}

std::pair<torch::Tensor, torch::Tensor> MessagePassingGNNImpl::forward(
    torch::Tensor x,
    const AdjNorms& adj_norms,
    torch::Tensor legal_mask)
{
    bool batched = (x.dim() == 3);
    if (!batched) {
        x          = x.unsqueeze(0);
        legal_mask = legal_mask.unsqueeze(0);
    }

    // Input projection: (B, N, hidden)
    auto h = torch::relu(input_proj->forward(x));

    // Message-passing layers: cycle adj, adj2, adj, adj4, adj, adj2, adj, adj4, ...
    for (int i = 0; i < cfg_.num_layers; i++) {
        const torch::Tensor& a   = (i % 4 == 1) ? adj_norms.adj2
                                 : (i % 4 == 3) ? adj_norms.adj4
                                 : adj_norms.adj;
        const torch::Tensor& deg = (i % 4 == 1) ? adj_norms.deg2
                                 : (i % 4 == 3) ? adj_norms.deg4
                                 : adj_norms.deg1;
        auto agg = torch::einsum("nm,bmd->bnd", {a, h}); // (B, N, hidden)
        auto deg_idx = deg.squeeze(-1).to(torch::kLong);            // (N,)
        auto deg_vec = deg_embed_.index_select(0, deg_idx);         // (N, L)
        auto deg_b   = deg_vec.unsqueeze(0).expand({h.size(0), -1, -1}); // (B, N, L)
        auto h_cat = torch::cat({h, agg, deg_b}, -1);      // (B, N, 2*hidden+L)
        // Residual: add input h before normalising so gradients bypass the linear
        h = layer_norms[i]->forward(torch::relu(layers[i]->forward(h_cat)) + h);
    }

    // Value: per-location stone/territory ownership softmax, applied directly to
    // the per-node hidden state h (B, N, hidden) - no pooling, since every node
    // needs its own distribution over "who owns this location at game end".
    auto stone_est     = torch::softmax(stone_head->forward(h), -1);      // (B, N, num_stones+1)
    auto territory_est = torch::softmax(territory_head->forward(h), -1);  // (B, N, num_stones+1)
    auto ownership = torch::stack({stone_est, territory_est}, 1);        // (B, 2, N, num_stones+1)

    auto logits = policy_head->forward(h); // (B, num_stones*N+1)

    // Mask illegal actions with -inf
    const float NEG_INF = -std::numeric_limits<float>::infinity();
    logits = logits.masked_fill(legal_mask.logical_not(), NEG_INF);
    // Guard against all-illegal (game over): avoid NaN in softmax
    auto all_illegal = legal_mask.any(-1, true).logical_not(); // (B, 1)
    logits = logits.masked_fill(all_illegal, 0.0f);

    auto policy = torch::softmax(logits, -1); // (B, num_stones*N+1)

    if (!batched) {
        return {policy.squeeze(0), ownership.squeeze(0)};
    }
    return {policy, ownership};
}

std::pair<torch::Tensor, torch::Tensor> MessagePassingGNNImpl::evaluate(
    const BoardState& state,
    const AdjNorms& adj_norms)
{
    auto dev = adj_norms.adj.device();
    torch::NoGradGuard ng;
    auto [ft, mask] = board_to_features(state, dev, cfg_.input_descr);
    auto [policy, ownership] = forward(ft, adj_norms, mask);
    return {policy, ownership};
}

// Helper: run a batch of raw pointers
static std::pair<torch::Tensor, torch::Tensor> run_batch(
    MessagePassingGNNImpl* self,
    const std::vector<const BoardState*>& states,
    const AdjNorms& adj_norms)
{
    torch::NoGradGuard ng;
    auto dev = adj_norms.adj.device();
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
    auto [policy, ownership] = self->forward(x, adj_norms, mask);
    return {policy, ownership};
}

std::pair<torch::Tensor, torch::Tensor> MessagePassingGNNImpl::evaluate_batch(
    const std::vector<BoardState*>& states,
    const AdjNorms& adj_norms)
{
    std::vector<const BoardState*> cstates(states.begin(), states.end());
    return run_batch(this, cstates, adj_norms);
}

std::pair<torch::Tensor, torch::Tensor> MessagePassingGNNImpl::evaluate_batch(
    const std::vector<const BoardState*>& states,
    const AdjNorms& adj_norms)
{
    return run_batch(this, states, adj_norms);
}
