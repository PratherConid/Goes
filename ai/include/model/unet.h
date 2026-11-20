#pragma once
#include <torch/torch.h>
#include "game/board_state.h"
#include "model/features.h"
#include "model/heads.h"
#include "model/model_config.h"
#include "model/model_output.h"
#include <vector>
#include <utility>

// U-Net for policy + per-location ownership estimation on boards with a 2D integer embedding.
//
// The input descriptor's sideLens are padded up to a square whose side (side_)
// is the least power of two >= both dimensions; cells outside the true board
// are marked invalid in the validity channel. This guarantees every encoder level's
// MaxPool2d(2,2) halves exactly, so no board row/column is ever silently
// dropped by floor-mode pooling.
//
// input_proj (1×1 conv, same role as CNN's/MessagePassingGNN's own
// input_proj) maps the feature_dim+1 input channels (validity channel
// included) to hidden_dim once, as a plain embedding step with no residual
// around it - unlike the repeated encoder/decoder blocks below, it isn't
// part of the architecture's doubling pattern, so there's nothing meaningful
// to shortcut around it.
// Encoder: progressively pool (H,W) → 1×1 via MaxPool2d, doubling channels at
//          each level (C_k = hidden_dim * 2^k). Each level's conv block also
//          has its own residual shortcut (block(h) + h, channel-matched via
//          clip/zero-pad rather than a learned 1×1 projection where channels
//          actually change - level 0, right after input_proj, already has
//          matching channels, so match_channels is a plain identity there).
// Decoder: upsample 2× via nearest-neighbor pixel replication (exact, since
//          each level is exactly double the next), reduce channels with a
//          1×1 conv, then add the matching encoder skip connection
//          (additive residual). Each decoder block's own shortcut is a plain
//          identity add (dec_reduce_'s 1×1 conv plus the skip add above
//          already bring h to dec_blocks_[k]'s constant enc_channels_[k]
//          width, so no channel matching is ever needed here).
// Heads:   OutputHeads (heads.h) over the full-resolution decoder output at every slot.
//
// Conv counts: 4 at encoder level 0, 2 at deeper encoder levels; 3 at decoder
//              level 0, 1 at deeper decoder levels (one fewer than the
//              matching encoder level, since dec_reduce_'s 1×1 conv already
//              contributes one conv's worth of depth at each level).
// num_layers is not a parameter - depth is determined by grid dimensions.
// Requires the input descriptor's embDim == 2.
struct UNetImpl : torch::nn::Module {
    torch::nn::Conv2d input_proj{nullptr};           // 1x1 conv: feature_dim+1 -> hidden_dim
    std::vector<torch::nn::Sequential> enc_blocks_;  // [k=0..L]: encoder conv groups
    std::vector<torch::nn::Conv2d>     dec_reduce_;  // [k=0..L-1]: C_{k+1}→C_k, 1×1
    std::vector<torch::nn::Sequential> dec_blocks_;  // [k=0..L-1]: decoder conv groups
    OutputHeads heads;

    UNetConfig cfg_;
    int num_levels_;                            // L+1 (total encoder blocks)
    int grid_h_, grid_w_;                       // the input descriptor's sideLens (before padding)
    int side_;                                  // padded power-of-two square side
    std::vector<int>                enc_channels_; // C_k at encoder level k
    std::vector<int>                level_size_;   // image side length at encoder level k

    // cfg.feature_dim: per-node feature dimension F (as produced by board_to_features), NOT
    // the number of input channels to input_proj. features_to_grid appends
    // a validity channel, so input_proj actually receives feature_dim + 1 channels.
    UNetImpl(const UNetConfig& cfg, int num_stones, int num_players);

    // x: (B,S,F), valid: (B,S) → (B,F+1,side,side), the validity channel last
    torch::Tensor features_to_grid(const torch::Tensor& x, const torch::Tensor& valid) const;

    // x (B,S,F), valid (B,S), legal_mask (B,numStones*S+1) - the model's slot space (NodeSlots,
    // model/features.h). Returns a ModelOutput (model_output.h).
    ModelOutput forward(
        torch::Tensor x,
        torch::Tensor valid,
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

TORCH_MODULE(UNet);
