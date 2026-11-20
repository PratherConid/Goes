#pragma once
#include <torch/torch.h>
#include "game/board_state.h"
#include "model/features.h"
#include "model/heads.h"
#include "model/model_config.h"
#include "model/model_output.h"
#include <memory>
#include <vector>

// Graph neural network for policy + per-location ownership estimation.
//
// Input features: F = cfg.feature_dim per slot (see model_config.h; computed via
// compute_input_descr(), model/input_descr.h, at model-build time). Every input and output is in
// the model's slot space (NodeSlots, features.h), and each row aggregates over its own game's
// neighbor tiers (GameInput, features.h), so one batch can mix boards.
// Heads: OutputHeads (heads.h) over the final per-slot hidden states.
struct MessagePassingGNNImpl : torch::nn::Module {
    torch::nn::Linear input_proj{nullptr};
    std::vector<torch::nn::Linear> layers;
    std::vector<torch::nn::LayerNorm> layer_norms;
    OutputHeads heads;

    GNNConfig cfg_;
    // Untrainable random Gaussian embedding table for neighbor counts:
    // (nBound+1, L) - row d is the fixed vector used whenever a node has
    // d neighbors in the relevant adjacency tier. Sized from the input descriptor's
    // nBound, which bounds every node count, and so every degree, the model runs on
    // (deg_embed_ is saved/loaded with the model like any other buffer).
    torch::Tensor deg_embed_;

    MessagePassingGNNImpl(const GNNConfig& cfg, int num_stones, int num_players);

    // x (B,S,F), games (each row's GameInput), legal_mask (B,numStones*S+1). Returns a ModelOutput
    // (model_output.h). Layer i averages over each row's tier 0 (i%4==0,2), 1 (i%4==1) or 2
    // (i%4==3) neighbors (NeighborTiers, features.h); each layer also concatenates that tier's
    // per-slot neighbor-count embedding (deg_embed_) so aggregation isn't degree-agnostic.
    ModelOutput forward(
        torch::Tensor x,
        const std::vector<std::shared_ptr<const GameInput>>& games,
        torch::Tensor legal_mask);

    // One state's network input (model_output.h's ModelInput), built on the CPU from it and its
    // game's GameInput (make_game_input(), features.h). Thread-safe.
    ModelInput prepare_input(const BoardState& state, const std::shared_ptr<const GameInput>& game) const;

    // Evaluates prepared inputs (from any boards that fit) in one forward pass, copying each stacked
    // input tensor to the model's device once. Every ModelOutput tensor is left on the model's
    // device - callers needing CPU access (e.g. via .accessor<T,N>()) must call .cpu() themselves
    // at their point of use.
    ModelOutput evaluate_inputs(const std::vector<const ModelInput*>& inputs);
};

TORCH_MODULE(MessagePassingGNN);
