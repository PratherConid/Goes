#include "model/gnn.h"
#include <limits>
#include <omp.h>

MessagePassingGNNImpl::MessagePassingGNNImpl(int in_dim, int hidden_dim, int num_layers, int num_players)
    : hidden_dim_(hidden_dim), num_layers_(num_layers), num_players_(num_players)
{
    input_proj = register_module("input_proj", torch::nn::Linear(in_dim, hidden_dim));

    // Each layer aggregates [h_self, h_neighbour_avg] → new h
    for (int i = 0; i < num_layers; i++) {
        auto layer = register_module("layer_" + std::to_string(i),
            torch::nn::Linear(2 * hidden_dim, hidden_dim));
        auto ln = register_module("ln_" + std::to_string(i),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));
        layers.push_back(layer);
        layer_norms.push_back(ln);
    }

    value_head = register_module("value_head", torch::nn::Sequential(
        torch::nn::Linear(hidden_dim, 64),
        torch::nn::ReLU(),
        torch::nn::Linear(64, num_players),
        torch::nn::Tanh()
    ));
    policy_head = register_module("policy_head", torch::nn::Linear(hidden_dim, 1));
    pass_logit  = register_parameter("pass_logit", torch::zeros({1}));
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

    int64_t B = x.size(0);

    // Input projection: (B, N, hidden)
    auto h = torch::relu(input_proj->forward(x));

    // Message-passing layers: cycle adj, adj2, adj, adj4, adj, adj2, adj, adj4, ...
    for (int i = 0; i < num_layers_; i++) {
        const torch::Tensor& a = (i % 4 == 1) ? adj_norms.adj2
                               : (i % 4 == 3) ? adj_norms.adj4
                               : adj_norms.adj;
        auto agg = torch::einsum("nm,bmd->bnd", {a, h}); // (B, N, hidden)
        auto h_cat = torch::cat({h, agg}, -1);             // (B, N, 2*hidden)
        // Residual: add input h before normalising so gradients bypass the linear
        h = layer_norms[i]->forward(torch::relu(layers[i]->forward(h_cat)) + h);
    }

    // Value head: global average pool → (B, num_players)
    auto global_feat = h.mean(1);                           // (B, hidden)
    auto value = value_head->forward(global_feat);          // (B, num_players)
    // Normalise to zero-sum across players: subtract the per-row mean so the
    // per-player values sum to zero (the model cannot rate everyone as winning).
    value = value - value.mean(-1, /*keepdim=*/true);

    // Policy head: per-node logit + learnable pass logit
    auto node_logits = policy_head->forward(h).squeeze(-1);     // (B, N)
    auto pass_expand = pass_logit.expand({B, 1});                // (B, 1)
    auto logits = torch::cat({node_logits, pass_expand}, -1);   // (B, N+1)

    // Mask illegal actions with -inf
    const float NEG_INF = -std::numeric_limits<float>::infinity();
    logits = logits.masked_fill(legal_mask.logical_not(), NEG_INF);
    // Guard against all-illegal (game over): avoid NaN in softmax
    auto all_illegal = legal_mask.any(-1, true).logical_not(); // (B, 1)
    logits = logits.masked_fill(all_illegal, 0.0f);

    auto policy = torch::softmax(logits, -1); // (B, N+1)

    if (!batched) {
        return {policy.squeeze(0), value.squeeze(0)};
    }
    return {policy, value};
}

std::pair<torch::Tensor, torch::Tensor> MessagePassingGNNImpl::evaluate(
    const BoardState& state,
    const AdjNorms& adj_norms)
{
    auto dev = adj_norms.adj.device();
    torch::NoGradGuard ng;
    auto [ft, mask] = board_to_features(state, dev);
    auto [policy, value] = forward(ft, adj_norms, mask);
    return {policy.cpu(), value.cpu()};
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
        auto [ft, mask] = board_to_features(*states[i], dev);
        feats[i] = ft;
        masks[i] = mask;
    }
    auto x    = torch::stack(feats, 0); // (B, N, F)
    auto mask = torch::stack(masks, 0); // (B, N+1)
    auto [policy, value] = self->forward(x, adj_norms, mask);
    return {policy.cpu(), value.cpu()};
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
