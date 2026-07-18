#pragma once
#include <torch/torch.h>
#include "game/board_state.h"
#include "model/features.h"
#include "model/model_config.h"
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

    // h: (B, N, hidden_dim) -> (B, num_stones*N+1) logits
    torch::Tensor forward(const torch::Tensor& h);
};
TORCH_MODULE(GNNPolicyHead);

// Graph neural network for policy + per-location ownership estimation.
//
// Input features per node: F = cfg.feature_dim (see model_config.h; computed
// via compute_input_descr(), training/self_play.h, at model-build time)
// Outputs:
//   policy    : (numStones*N+1,) softmax over legal actions (numStones*N placements + pass)
//   ownership : (2, N, num_stones+1) per-location stone/territory ownership estimates -
//               index 0 = stone estimate, index 1 = territory estimate; each (N, num_stones+1)
//               slice is a softmax over "which stone type occupies/holds territory at this
//               location at game end" (channel 0 = none) - stone-type indexed to match
//               ScoreData/BoardState::board; player-level aggregation (via stone_to_player_map)
//               happens downstream, e.g. in estimate_player_rewards() (evaluator.h)
struct MessagePassingGNNImpl : torch::nn::Module {
    torch::nn::Linear input_proj{nullptr};
    std::vector<torch::nn::Linear> layers;
    std::vector<torch::nn::LayerNorm> layer_norms;
    torch::nn::Sequential stone_head{nullptr};
    torch::nn::Sequential territory_head{nullptr};
    GNNPolicyHead policy_head{nullptr};

    GNNConfig cfg_;
    int num_players_;
    int num_stones_;
    // Untrainable random Gaussian embedding table for neighbor counts:
    // (max_degree+1, L) - row d is the fixed vector used whenever a node has
    // d neighbors in the relevant adjacency tier. Sized from AdjNorms::max_degree
    // at construction time, so the model must be built with the AdjNorms it will
    // run against (deg_embed_ is saved/loaded with the model like any other buffer).
    torch::Tensor deg_embed_;

    MessagePassingGNNImpl(const GNNConfig& cfg, int num_players, int num_stones,
                          const AdjNorms& adj_norms);

    // Returns (policy, ownership).
    //   Unbatched: x (N,F), legal_mask (numStones*N+1,)   → policy (numStones*N+1,), ownership (2,N,num_stones+1)
    //   Batched:   x (B,N,F), legal_mask (B,numStones*N+1) → policy (B,numStones*N+1), ownership (B,2,N,num_stones+1)
    // Layer i uses: adj (i%4==0,2), adj2 (i%4==1), adj4 (i%4==3); each layer
    // also concatenates that adjacency tier's per-node neighbor-count embedding
    // (deg_embed_ looked up via adj_norms.deg1/deg2/deg4) so aggregation isn't
    // degree-agnostic.
    std::pair<torch::Tensor, torch::Tensor> forward(
        torch::Tensor x,
        const AdjNorms& adj_norms,
        torch::Tensor legal_mask);

    // Evaluate a single BoardState. Returns (policy (numStones*N+1,), ownership (2,N,num_stones+1)),
    // both left on the model's device - callers needing CPU access (e.g. via
    // .accessor<T,N>()) must call .cpu() themselves at their point of use.
    std::pair<torch::Tensor, torch::Tensor> evaluate(
        const BoardState& state,
        const AdjNorms& adj_norms);

    // Evaluate a list of states in one forward pass (all must share the same board).
    // Returns policy_batch (B, numStones*N+1) and ownership_batch (B, 2, N, num_stones+1), both
    // left on the model's device (see evaluate()'s comment).
    std::pair<torch::Tensor, torch::Tensor> evaluate_batch(
        const std::vector<BoardState*>& states,
        const AdjNorms& adj_norms);

    // Convenience overload for const states
    std::pair<torch::Tensor, torch::Tensor> evaluate_batch(
        const std::vector<const BoardState*>& states,
        const AdjNorms& adj_norms);
};

TORCH_MODULE(MessagePassingGNN);
