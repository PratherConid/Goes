#include "model/gnn.h"
#include <algorithm>
#include <limits>
#include <cmath>

MessagePassingGNNImpl::MessagePassingGNNImpl(const GNNConfig& cfg, int num_stones, int num_players)
    : cfg_(cfg), num_stones_(num_stones)
{
    input_proj = register_module("input_proj", torch::nn::Linear(cfg_.feature_dim, cfg_.hidden_dim));

    // Untrainable random Gaussian embedding table for neighbor counts: index d (0..nBound - no
    // board the model runs on has more nodes, so no node has more neighbors) maps to a fixed random
    // vector, giving degree more representational capacity than a single concatenated scalar.
    int n_bound = cfg_.input_descr.at("nBound").get<int>();
    int deg_embed_len = static_cast<int>(std::round(
        2.0 * (std::sqrt(static_cast<double>(n_bound)) + 4.0)));
    deg_embed_ = register_buffer("deg_embed", torch::randn({n_bound + 1, deg_embed_len}));

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
        torch::nn::Linear(cfg_.hidden_dim, cfg_.hidden_dim),
        torch::nn::ReLU(),
        torch::nn::Linear(cfg_.hidden_dim, num_stones + 1)
    ));
    territory_head = register_module("territory_head", torch::nn::Sequential(
        torch::nn::Linear(cfg_.hidden_dim, cfg_.hidden_dim),
        torch::nn::ReLU(),
        torch::nn::Linear(cfg_.hidden_dim, num_stones + 1)
    ));
    policy_head = register_module("policy_head", GNNPolicyHead(cfg_.hidden_dim, num_stones));
    rank_head = register_module("rank_head", RankHead(cfg_.hidden_dim, num_rank_pairs(num_players)));
}

GNNPolicyHeadImpl::GNNPolicyHeadImpl(int hidden_dim, int num_stones)
    : num_stones_(num_stones)
{
    proj = register_module("proj", torch::nn::Linear(hidden_dim, num_stones + 1));
    attn = register_module("attn", torch::nn::Linear(hidden_dim, 1));
}

torch::Tensor GNNPolicyHeadImpl::forward(const torch::Tensor& h) {
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

AdjNorms MessagePassingGNNImpl::adj_norms_of(const std::shared_ptr<const BoardConfig>& bc) const {
    torch::Device dev = input_proj->weight.device();
    std::lock_guard<std::mutex> lock(adj_cache_mutex_);
    for (auto it = adj_cache_.begin(); it != adj_cache_.end(); ++it) {
        if (it->first != bc || it->second.adj.device() != dev) continue;
        adj_cache_.splice(adj_cache_.begin(), adj_cache_, it);
        return adj_cache_.front().second;
    }
    adj_cache_.emplace_front(bc, compute_adj_norms(*bc, compute_node_slots(*bc, cfg_.input_descr), dev));
    if (adj_cache_.size() > kAdjCacheBoards) adj_cache_.pop_back();
    return adj_cache_.front().second;
}

ModelOutput MessagePassingGNNImpl::forward(
    torch::Tensor x,
    const std::vector<std::shared_ptr<const BoardConfig>>& boards,
    torch::Tensor legal_mask)
{
    int64_t B = x.size(0), S = x.size(1);

    // Every row's padded adjacency set: one (S,S) set for the whole batch when every row is on the
    // same board, else one per row, stacked to (B,S,S).
    const bool shared = std::all_of(boards.begin(), boards.end(), [&](const auto& b) { return b == boards[0]; });
    AdjNorms a;
    if (shared) {
        a = adj_norms_of(boards[0]);
    } else {
        std::vector<AdjNorms> rows;
        rows.reserve(boards.size());
        for (const auto& b : boards) rows.push_back(adj_norms_of(b));
        auto stack = [&](torch::Tensor AdjNorms::*field) {
            std::vector<torch::Tensor> t;
            t.reserve(rows.size());
            for (const AdjNorms& r : rows) t.push_back(r.*field);
            return torch::stack(t);
        };
        a = {stack(&AdjNorms::adj), stack(&AdjNorms::adj2), stack(&AdjNorms::adj4),
             stack(&AdjNorms::deg1), stack(&AdjNorms::deg2), stack(&AdjNorms::deg4)};
    }

    // Input projection: (B, S, hidden)
    auto h = torch::relu(input_proj->forward(x));

    // Message-passing layers: cycle adj, adj2, adj, adj4, adj, adj2, adj, adj4, ...
    for (int i = 0; i < cfg_.num_layers; i++) {
        const torch::Tensor& m   = (i % 4 == 1) ? a.adj2 : (i % 4 == 3) ? a.adj4 : a.adj;
        const torch::Tensor& deg = (i % 4 == 1) ? a.deg2 : (i % 4 == 3) ? a.deg4 : a.deg1;
        auto agg = torch::einsum(shared ? "nm,bmd->bnd" : "bnm,bmd->bnd", {m, h});  // (B, S, hidden)
        auto deg_idx = deg.squeeze(-1).to(torch::kLong).flatten();                    // (S,) or (B*S,)
        auto deg_b   = deg_embed_.index_select(0, deg_idx)
                           .view({shared ? 1 : B, S, -1}).expand({B, S, -1});        // (B, S, L)
        auto h_cat = torch::cat({h, agg, deg_b}, -1);      // (B, S, 2*hidden+L)
        // Residual: add input h before normalising so gradients bypass the linear
        h = layer_norms[i]->forward(torch::relu(layers[i]->forward(h_cat)) + h);
    }

    // Ownership: per-location stone/territory softmax, applied directly to the per-slot hidden
    // state h (B, S, hidden) - no pooling, since every node needs its own distribution over "who
    // owns this location at game end".
    auto stone_est     = torch::softmax(stone_head->forward(h), -1);      // (B, S, num_stones+1)
    auto territory_est = torch::softmax(territory_head->forward(h), -1);  // (B, S, num_stones+1)
    auto ownership = torch::stack({stone_est, territory_est}, 1);              // (B, 2, S, num_stones+1)
    auto rank = rank_head->forward(h);                            // (B, num_rank_pairs, 3)

    auto logits = policy_head->forward(h); // (B, num_stones*S+1)

    // Mask illegal actions with -inf
    const float NEG_INF = -std::numeric_limits<float>::infinity();
    logits = logits.masked_fill(legal_mask.logical_not(), NEG_INF);
    // Guard against all-illegal (game over): avoid NaN in softmax
    auto all_illegal = legal_mask.any(-1, true).logical_not(); // (B, 1)
    logits = logits.masked_fill(all_illegal, 0.0f);

    auto policy = torch::softmax(logits, -1); // (B, num_stones*S+1)
    return {policy, ownership, rank};
}

ModelInput MessagePassingGNNImpl::prepare_input(const BoardState& state) const {
    return slot_input(state, cfg_.input_descr);
}

ModelOutput MessagePassingGNNImpl::evaluate_inputs(const std::vector<const ModelInput*>& inputs) {
    torch::NoGradGuard ng;
    auto dev = input_proj->weight.device();
    auto x     = stack_inputs(inputs, &ModelInput::features).to(dev);   // (B, S, F)
    auto mask  = stack_inputs(inputs, &ModelInput::legal_mask).to(dev); // (B, ns*S+1)
    std::vector<std::shared_ptr<const BoardConfig>> boards;
    boards.reserve(inputs.size());
    for (const ModelInput* in : inputs) boards.push_back(in->board);
    return forward(x, boards, mask);
}
