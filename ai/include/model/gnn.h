#pragma once
#include <torch/torch.h>
#include "game/board_state.h"
#include "model/features.h"
#include <vector>

// Graph neural network for policy + value estimation.
//
// Input features per node: F = 2*num_stones + 4
// Outputs:
//   policy : (N+1,) softmax over legal actions (N placements + pass)
//   value  : (num_players,) estimated reward for each player (in player-ID order 1..P)
struct MessagePassingGNNImpl : torch::nn::Module {
    torch::nn::Linear input_proj{nullptr};
    std::vector<torch::nn::Linear> layers;
    std::vector<torch::nn::LayerNorm> layer_norms;
    torch::nn::Sequential value_head{nullptr};
    torch::nn::Linear policy_head{nullptr};
    torch::Tensor pass_logit;

    int hidden_dim_;
    int num_layers_;
    int num_players_;

    MessagePassingGNNImpl(int in_dim, int hidden_dim, int num_layers, int num_players);

    // Returns (policy, value).
    //   Unbatched: x (N,F), legal_mask (N+1,)   → policy (N+1,), value (num_players,)
    //   Batched:   x (B,N,F), legal_mask (B,N+1) → policy (B,N+1), value (B,num_players)
    // Layer i uses: adj (i%4==0,2), adj2 (i%4==1), adj4 (i%4==3).
    std::pair<torch::Tensor, torch::Tensor> forward(
        torch::Tensor x,
        const AdjNorms& adj_norms,
        torch::Tensor legal_mask);

    // Evaluate a single BoardState. Returns (policy (N+1,), value (num_players,)).
    std::pair<torch::Tensor, torch::Tensor> evaluate(
        const BoardState& state,
        const AdjNorms& adj_norms);

    // Evaluate a list of states in one forward pass (all must share the same board).
    // Returns policy_batch (B, N+1) and value_batch (B,).
    std::pair<torch::Tensor, torch::Tensor> evaluate_batch(
        const std::vector<BoardState*>& states,
        const AdjNorms& adj_norms);

    // Convenience overload for const states
    std::pair<torch::Tensor, torch::Tensor> evaluate_batch(
        const std::vector<const BoardState*>& states,
        const AdjNorms& adj_norms);
};

TORCH_MODULE(MessagePassingGNN);
