#include "model/gnn.h"
#include <cmath>

MessagePassingGNNImpl::MessagePassingGNNImpl(const GNNConfig& cfg, int num_stones, int num_players)
    : cfg_(cfg)
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

    heads.register_on(*this, cfg_.hidden_dim, num_stones, num_players);
}

ModelOutput MessagePassingGNNImpl::forward(
    torch::Tensor x,
    const std::vector<std::shared_ptr<const GameInput>>& games,
    torch::Tensor legal_mask)
{
    int64_t B = x.size(0), S = x.size(1);

    // Every row's neighbor tiers (NeighborTiers, features.h) over the batch's B*S slots: tier t's
    // edges take h at src[t] into the mean at dst[t] (dst follows from the degrees, since each row's
    // src lists its slots' neighbors in slot order).
    std::vector<torch::Tensor> degs;
    degs.reserve(games.size());
    for (const auto& g : games) degs.push_back(g->tiers.deg);
    auto deg = stack_cpu(degs).to(x.device());  // (B, 3, S)
    std::array<torch::Tensor, 3> src, dst;
    for (int t = 0; t < 3; t++) {
        std::vector<torch::Tensor> rows;
        rows.reserve(games.size());
        for (const auto& g : games) rows.push_back(g->tiers.src[t]);
        auto deg_t = deg.select(1, t).reshape({-1});
        dst[t] = torch::repeat_interleave(torch::arange(B * S, deg_t.options()), deg_t);
        src[t] = torch::cat(rows).to(x.device()).to(torch::kLong) + dst[t].div(S, "floor") * S;
    }

    // Input projection: (B, S, hidden)
    auto h = torch::relu(input_proj->forward(x));

    // Message-passing layers: cycle tiers 0, 1, 0, 2, 0, 1, 0, 2, ...
    for (int i = 0; i < cfg_.num_layers; i++) {
        int t = (i % 4 == 1) ? 1 : (i % 4 == 3) ? 2 : 0;
        auto deg_t = deg.select(1, t).reshape({-1});                                     // (B*S,)
        auto h_flat = h.reshape({B * S, -1});
        auto agg = torch::zeros_like(h_flat).index_add(0, dst[t], h_flat.index_select(0, src[t]))
                   / deg_t.clamp_min(1).unsqueeze(1).to(h.scalar_type());
        auto deg_b = deg_embed_.index_select(0, deg_t).view({B, S, -1});                // (B, S, L)
        auto h_cat = torch::cat({h, agg.view({B, S, -1}), deg_b}, -1);  // (B, S, 2*hidden+L)
        // Residual: add input h before normalising so gradients bypass the linear
        h = layer_norms[i]->forward(torch::relu(layers[i]->forward(h_cat)) + h);
    }

    return heads.forward(h, legal_mask);
}

ModelInput MessagePassingGNNImpl::prepare_input(const BoardState& state, const std::shared_ptr<const GameInput>& game) const {
    return slot_input(state, cfg_.input_descr, game);
}

ModelOutput MessagePassingGNNImpl::evaluate_inputs(const std::vector<const ModelInput*>& inputs) {
    torch::NoGradGuard ng;
    auto dev = input_proj->weight.device();
    auto x     = stack_inputs(inputs, &ModelInput::features).to(dev);   // (B, S, F)
    auto mask  = stack_inputs(inputs, &ModelInput::legal_mask).to(dev); // (B, ns*S+1)
    std::vector<std::shared_ptr<const GameInput>> games;
    games.reserve(inputs.size());
    for (const ModelInput* in : inputs) games.push_back(in->game);
    return forward(x, games, mask);
}
