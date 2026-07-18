#pragma once
#include <torch/torch.h>
#include "game/board_state.h"
#include "game/board_config.h"
#include "model/features.h"
#include "model/model_config.h"
#include <vector>
#include <utility>

// Policy head: 1×1 conv over the decoder's full-resolution feature map
// produces, per node, num_stones place-logit channels and 1 pass-field
// channel; the pass field is reduced to a single pass logit via a learned
// affine combination over the board's valid nodes. Place logits are
// flattened stone-major (index (stone-1)*N+pos), matching legal_mask's
// layout in features.cpp.
struct UNetPolicyHeadImpl : torch::nn::Module {
    torch::nn::Conv2d conv{nullptr};
    torch::nn::Linear pass_reduce{nullptr};
    int num_stones_;

    UNetPolicyHeadImpl(int in_channels, int num_nodes, int num_stones);

    // h: (B, C, H, W), lin_idx: (N,) long node positions in the flattened grid.
    // Returns (B, num_stones*N+1) logits.
    torch::Tensor forward(const torch::Tensor& h, const torch::Tensor& lin_idx);
};
TORCH_MODULE(UNetPolicyHead);

// U-Net for policy + per-location ownership estimation on boards with a 2D integer embedding.
//
// The board's tight bounding box is padded up to a square whose side (side_)
// is the least power of two >= both dimensions; cells outside the true board
// are marked invalid in valid_tensor_. This guarantees every encoder level's
// MaxPool2d(2,2) halves exactly, so no board row/column is ever silently
// dropped by floor-mode pooling.
//
// Encoder: progressively pool (H,W) → 1×1 via MaxPool2d, doubling channels at
//          each level (C_k = hidden_dim * 2^k). Each level's conv block also
//          has its own residual shortcut (block(h) + h, channel-matched via
//          clip/zero-pad rather than a learned 1×1 projection).
// Decoder: upsample 2× via nearest-neighbor pixel replication (exact, since
//          each level is exactly double the next), reduce channels with a
//          1×1 conv, then add the matching encoder skip connection
//          (additive residual). Each decoder block has the same per-block
//          residual shortcut as the encoder.
// Ownership: same full-resolution decoder output as the policy head, gathered per-node
//            (via lin_idx_) and passed through two independent softmax heads (stone
//            estimate, territory estimate) - see MessagePassingGNN's ownership doc.
// Policy:    extracted from the full-resolution decoder output via UNetPolicyHead.
//
// Conv counts: 4 at encoder level 0, 2 at deeper encoder levels; 3 at decoder
//              level 0, 1 at deeper decoder levels (one fewer than the
//              matching encoder level, since dec_reduce_'s 1×1 conv already
//              contributes one conv's worth of depth at each level).
// num_layers is not a parameter - depth is determined by grid dimensions.
// Requires bc.emb_dim == 2.
struct UNetImpl : torch::nn::Module {
    std::vector<torch::nn::Sequential> enc_blocks_;  // [k=0..L]: encoder conv groups
    std::vector<torch::nn::Conv2d>     dec_reduce_;  // [k=0..L-1]: C_{k+1}→C_k, 1×1
    std::vector<torch::nn::Sequential> dec_blocks_;  // [k=0..L-1]: decoder conv groups
    torch::nn::Sequential stone_head{nullptr};
    torch::nn::Sequential territory_head{nullptr};
    UNetPolicyHead policy_head{nullptr};

    UNetConfig cfg_;
    int num_players_;
    int num_stones_;
    int num_levels_;                            // L+1 (total encoder blocks)
    int side_;                                  // padded power-of-two square side
    std::vector<int>                enc_channels_; // C_k at encoder level k
    std::vector<int>                level_size_;   // image side length at encoder level k

    torch::Tensor lin_idx_;      // (N,) long - row*side+col for each board node
    torch::Tensor valid_tensor_; // (1,1,side,side) float - 1 at valid (true board) cells

    // cfg.feature_dim: per-node feature dimension F (as produced by board_to_features), NOT
    // the number of input channels to the first convolution. features_to_grid appends
    // a validity channel, so the first conv actually receives feature_dim + 1 channels.
    UNetImpl(const BoardConfig& bc, const UNetConfig& cfg, int num_players, int num_stones);

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

TORCH_MODULE(UNet);
