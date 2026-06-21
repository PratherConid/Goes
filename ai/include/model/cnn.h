#pragma once
#include <torch/torch.h>
#include "game/board_state.h"
#include "game/board_config.h"
#include "model/features.h"
#include <vector>
#include <utility>

// U-Net CNN for policy + value estimation on boards with a 2D integer embedding.
//
// Encoder: progressively pool (H,W) → 1×1 via MaxPool2d, doubling channels at
//          each level (C_k = hidden_dim * min(2^k, 8)).
// Decoder: upsample back to each encoder spatial size via nearest-neighbor
//          interpolation, reduce channels with a 1×1 conv, then add the
//          matching encoder skip connection (additive residual).
// Value:   extracted from the 1×1 bottleneck.
// Policy:  extracted from the full-resolution decoder output via the node
//          embedding map (same as the GNN).
//
// Conv counts: 4 at encoder level 0, 2 at deeper encoder levels; 4 at decoder
//              level 0, 2 at deeper decoder levels.
// num_layers is not a parameter - depth is determined by grid dimensions.
// Requires bc.emb_dim == 2.
struct ConvNNImpl : torch::nn::Module {
    std::vector<torch::nn::Sequential> enc_blocks_;  // [k=0..L]: encoder conv groups
    std::vector<torch::nn::Conv2d>     dec_reduce_;  // [k=0..L-1]: C_{k+1}→C_k, 1×1
    std::vector<torch::nn::Sequential> dec_blocks_;  // [k=0..L-1]: decoder conv groups
    torch::nn::Sequential value_head{nullptr};
    torch::nn::Conv2d     policy_head{nullptr};
    torch::Tensor pass_logit;

    int hidden_dim_;
    int num_players_;
    int num_levels_;                            // L+1 (total encoder blocks)
    int grid_h_, grid_w_;
    std::vector<std::pair<int,int>> enc_sizes_;   // (H_k, W_k) at encoder level k
    std::vector<int>                enc_channels_; // C_k at encoder level k

    torch::Tensor lin_idx_;      // (N,) long - row*W+col for each board node
    torch::Tensor valid_tensor_; // (1,1,H,W) float - 1 at valid grid cells

    ConvNNImpl(const BoardConfig& bc, int in_dim, int hidden_dim, int num_players);

    // x: (N,F) or (B,N,F) → (1,F+1,H,W) or (B,F+1,H,W)
    torch::Tensor features_to_grid(const torch::Tensor& x) const;

    // policy_grid: (B,H,W) → (B,N) via embedding map
    torch::Tensor grid_to_node_logits(const torch::Tensor& policy_grid) const;

    // Returns (policy, value).
    //   Unbatched: x (N,F), legal_mask (N+1,)   → policy (N+1,), value (num_players,)
    //   Batched:   x (B,N,F), legal_mask (B,N+1) → policy (B,N+1), value (B,num_players)
    std::pair<torch::Tensor, torch::Tensor> forward(
        torch::Tensor x,
        torch::Tensor legal_mask);

    // Evaluate a single BoardState. Returns (policy (N+1,), value (num_players,)).
    std::pair<torch::Tensor, torch::Tensor> evaluate(const BoardState& state);

    // Evaluate a batch of states in one forward pass (all must share the same board).
    std::pair<torch::Tensor, torch::Tensor> evaluate_batch(
        const std::vector<BoardState*>& states);
    std::pair<torch::Tensor, torch::Tensor> evaluate_batch(
        const std::vector<const BoardState*>& states);
};

TORCH_MODULE(ConvNN);
