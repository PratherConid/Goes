#pragma once
#include <torch/torch.h>
#include "game/board_state.h"
#include "game/board_config.h"
#include "model/features.h"
#include "model/model_config.h"
#include <vector>
#include <utility>

// Policy head: 1×1 conv over the final block's feature map produces, per node,
// num_stones place-logit channels and 1 pass-field channel; the pass field is
// reduced to a single pass logit via a learned affine combination over the
// board's valid nodes. Place logits are flattened stone-major (index
// (stone-1)*N+pos) and concatenated with the pass logit into one (B,
// num_stones*N+1) vector - a single flattened softmax over the whole vector
// (computed by the caller) then chooses among every (stone,pos) placement or
// pass, matching legal_mask's layout (see features.cpp).
struct CNNPolicyHeadImpl : torch::nn::Module {
    torch::nn::Conv2d conv{nullptr};
    torch::nn::Linear pass_reduce{nullptr};
    int num_stones_;

    CNNPolicyHeadImpl(int in_channels, int num_nodes, int num_stones);

    // h: (B, C, H, W), lin_idx: (N,) long node positions in the flattened grid.
    // Returns (B, num_stones*N+1) logits.
    torch::Tensor forward(const torch::Tensor& h, const torch::Tensor& lin_idx);
};
TORCH_MODULE(CNNPolicyHead);

// Plain residual CNN for policy + per-location ownership estimation on boards
// with a 2D integer embedding. Unlike UNet, there is no pooling/downsampling:
// every block operates at the board's tight bounding box resolution (no
// power-of-two padding either).
//
// Blocks: max(grid_w_, grid_h_) blocks, each two 3×3 convs (no
// normalization) at a constant hidden_dim width, with a residual shortcut
// from the block's input to its output (channel-matched via clip/zero-pad,
// not a learned projection - see match_channels() in cnn.cpp). Only the
// first block's shortcut actually pads (in_dim+1 -> hidden_dim); every
// later block already matches (hidden_dim -> hidden_dim).
// Ownership: per-node gather (via lin_idx_, same mechanism as the policy head) of the
//            final block output, then two independent Linear/ReLU/Linear softmax heads
//            (stone estimate, territory estimate) - see MessagePassingGNN's ownership doc.
// Policy: extracted from the final block output via CNNPolicyHead.
// Requires bc.emb_dim == 2.
struct CNNImpl : torch::nn::Module {
    std::vector<torch::nn::Sequential> blocks_;  // [k=0..num_blocks_-1]: Conv3x3->ReLU->Conv3x3
    torch::nn::Sequential stone_head{nullptr};
    torch::nn::Sequential territory_head{nullptr};
    CNNPolicyHead policy_head{nullptr};

    CNNConfig cfg_;
    int num_players_;
    int num_stones_;
    int num_blocks_;
    int grid_h_, grid_w_;  // tight bounding box (no padding)

    torch::Tensor lin_idx_;      // (N,) long - row*grid_w_+col for each board node
    torch::Tensor valid_tensor_; // (1,1,grid_h_,grid_w_) float - 1 at valid (true board) cells

    // cfg.feature_dim: per-node feature dimension F (as produced by board_to_features), NOT
    // the number of input channels to the first convolution. features_to_grid appends
    // a validity channel, so the first conv actually receives feature_dim + 1 channels.
    CNNImpl(const BoardConfig& bc, const CNNConfig& cfg, int num_players, int num_stones);

    // x: (N,F) or (B,N,F) → (1,F+1,H,W) or (B,F+1,H,W)
    torch::Tensor features_to_grid(const torch::Tensor& x) const;

    // Returns (policy, ownership).
    //   Unbatched: x (N,F), legal_mask (numStones*N+1,)   → policy (numStones*N+1,), ownership (2,N,num_stones+1)
    //   Batched:   x (B,N,F), legal_mask (B,numStones*N+1) → policy (B,numStones*N+1), ownership (B,2,N,num_stones+1)
    std::pair<torch::Tensor, torch::Tensor> forward(
        torch::Tensor x,
        torch::Tensor legal_mask);

    // Evaluate a single BoardState. Returns (policy (numStones*N+1,), ownership (2,N,num_stones+1)),
    // both left on the model's device - callers needing CPU access (e.g. via
    // .accessor<T,N>()) must call .cpu() themselves at their point of use.
    std::pair<torch::Tensor, torch::Tensor> evaluate(const BoardState& state);

    // Evaluate a batch of states in one forward pass (all must share the same board).
    // Returns tensors on the model's device (see evaluate()'s comment).
    std::pair<torch::Tensor, torch::Tensor> evaluate_batch(
        const std::vector<BoardState*>& states);
    std::pair<torch::Tensor, torch::Tensor> evaluate_batch(
        const std::vector<const BoardState*>& states);
};

TORCH_MODULE(CNN);
