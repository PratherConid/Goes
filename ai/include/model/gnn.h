#pragma once
#include <torch/torch.h>
#include "game/board_state.h"
#include "model/features.h"
#include "model/model_config.h"
#include "model/model_output.h"
#include "model/rank_head.h"
#include <memory>
#include <vector>

// Policy head: per-node linear produces num_stones place-logit channels and 1
// pass-field channel; the pass field is reduced to a single pass logit via a
// learned attention-weighted sum over nodes, so it stays valid for any N
// (attention weights come from node features, not fixed node positions).
// Place logits are node-major out of `proj` ((B,N,numStones)) but must be
// flattened stone-major (index (stone-1)*N+pos, matching legal_mask's layout
// in features.cpp) - forward() permutes before reshaping to make this happen.
struct GNNPolicyHeadImpl : torch::nn::Module {
    torch::nn::Linear proj{nullptr};  // hidden_dim -> num_stones+1 (place logits, pass field)
    torch::nn::Linear attn{nullptr};  // hidden_dim -> 1 (unnormalised attention score)
    int num_stones_;

    GNNPolicyHeadImpl(int hidden_dim, int num_stones);

    // h: (B, S, hidden_dim) over the model's slots -> (B, num_stones*S+1) logits
    torch::Tensor forward(const torch::Tensor& h);
};
TORCH_MODULE(GNNPolicyHead);

// Graph neural network for policy + per-location ownership estimation.
//
// Input features per node: F = cfg.feature_dim (see model_config.h; computed
// via compute_input_descr(), training/self_play.h, at model-build time)
// Outputs:
// Every input and output is in the model's slot space (NodeSlots, features.h), and each row
// aggregates over its own game's neighbor tiers (GameInput, features.h), so one batch can mix
// boards.
//   policy    : (B, numStones*S+1) softmax over legal actions (numStones*S placements + pass)
//   ownership : (B, 2, S, num_stones+1) per-location stone/territory ownership estimates -
//               index 0 = stone estimate, index 1 = territory estimate; each (N, num_stones+1)
//               slice is a softmax over "which stone type occupies/holds territory at this
//               location at game end" (channel 0 = none) - stone-type indexed to match
//               ScoreData/BoardState::board; player-level aggregation (via stone_to_player_map)
//               happens downstream, e.g. in estimate_player_totals() (evaluator.h)
//   rank      : (B, num_rank_pairs, 3) pairwise outcome probabilities - RankHead (rank_head.h) over
//               the final per-node states
struct MessagePassingGNNImpl : torch::nn::Module {
    torch::nn::Linear input_proj{nullptr};
    std::vector<torch::nn::Linear> layers;
    std::vector<torch::nn::LayerNorm> layer_norms;
    torch::nn::Sequential stone_head{nullptr};
    torch::nn::Sequential territory_head{nullptr};
    GNNPolicyHead policy_head{nullptr};
    RankHead rank_head{nullptr};

    GNNConfig cfg_;
    int num_stones_;
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
    // per-node neighbor-count embedding (deg_embed_) so aggregation isn't degree-agnostic.
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
