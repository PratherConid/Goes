#pragma once
#include <torch/torch.h>
#include "game/board_state.h"
#include "game/board_config.h"
#include "model/features.h"
#include "model/model_config.h"
#include "model/model_output.h"
#include "model/rank_head.h"
#include <vector>
#include <utility>

// Policy head: per-node Linear produces num_stones place-logit channels and 1
// pass-field channel; the pass field is reduced to a single pass logit via a
// learned attention-weighted sum over nodes, so it stays valid for any N
// (attention weights come from node features, not fixed node positions) -
// identical in structure to MessagePassingGNN's own policy head
// (GNNPolicyHeadImpl, gnn.h/gnn.cpp). Place logits are flattened stone-major
// (index (stone-1)*N+pos) and concatenated with the pass logit into one (B,
// num_stones*N+1) vector - a single flattened softmax over the whole vector
// (computed by the caller) then chooses among every (stone,pos) placement or
// pass, matching legal_mask's layout (see features.cpp).
struct CNNPolicyHeadImpl : torch::nn::Module {
    torch::nn::Linear proj{nullptr};  // hidden_dim -> num_stones+1 (place logits, pass field)
    torch::nn::Linear attn{nullptr};  // hidden_dim -> 1 (unnormalised attention score)
    int num_stones_;

    CNNPolicyHeadImpl(int hidden_dim, int num_stones);

    // h: (B, S, hidden_dim) over the model's slots. Returns (B, num_stones*S+1) logits.
    torch::Tensor forward(const torch::Tensor& h);
};
TORCH_MODULE(CNNPolicyHead);

// Plain residual CNN for policy + per-location ownership estimation on boards
// with a 2D integer embedding. Unlike UNet, there is no pooling/downsampling:
// every block operates at the input descriptor's sideLens (no power-of-two
// padding either).
//
// A 1×1 conv (input_proj, same role as MessagePassingGNN's own input_proj)
// maps the feature_dim+1 input channels (validity channel included - see
// features_to_grid) to hidden_dim once, up front - so every block afterward
// operates at a constant hidden_dim width on both sides of its residual
// shortcut, which is then a plain identity add (no channel-matching needed
// anywhere in the block loop, unlike UNet's per-level clip/zero-pad).
//
// Blocks: max(grid_w_, grid_h_) blocks, each two cfg.conv_size×cfg.conv_size
// convs (no normalization, "same" padding = conv_size/2 so spatial dims
// never change) at a constant hidden_dim width, with a residual shortcut
// from the block's input to its output.
// Ownership: the final block output at every slot (same as the policy head reads), then two
//            independent Linear/ReLU/Linear softmax heads (stone estimate, territory estimate) -
//            see MessagePassingGNN's ownership doc.
// Policy: extracted from the final block output via CNNPolicyHead.
// Rank: RankHead (rank_head.h) over the same per-slot output.
// Requires bc.emb.emb_dim() == 2.
struct CNNImpl : torch::nn::Module {
    torch::nn::Conv2d input_proj{nullptr};       // 1x1 conv: feature_dim+1 -> hidden_dim
    std::vector<torch::nn::Sequential> blocks_;  // [k=0..num_blocks_-1]: Conv->ReLU->Conv (cfg.conv_size)
    torch::nn::Sequential stone_head{nullptr};
    torch::nn::Sequential territory_head{nullptr};
    CNNPolicyHead policy_head{nullptr};
    RankHead rank_head{nullptr};

    CNNConfig cfg_;
    int num_stones_;
    int num_blocks_;
    int grid_h_, grid_w_;  // the input descriptor's sideLens (no padding)

    // cfg.feature_dim: per-node feature dimension F (as produced by board_to_features), NOT
    // the number of input channels to input_proj. features_to_grid appends
    // a validity channel, so input_proj actually receives feature_dim + 1 channels.
    CNNImpl(const BoardConfig& bc, const CNNConfig& cfg, int num_stones, int num_players);

    // x: (B,S,F), valid: (B,S) → (B,F+1,H,W), the validity channel last
    torch::Tensor features_to_grid(const torch::Tensor& x, const torch::Tensor& valid) const;

    // x (B,S,F), valid (B,S), legal_mask (B,numStones*S+1) - the model's slot space (NodeSlots,
    // model/features.h). Returns a ModelOutput (model_output.h).
    ModelOutput forward(
        torch::Tensor x,
        torch::Tensor valid,
        torch::Tensor legal_mask);

    // One state's network input (model_output.h's ModelInput), built on the CPU. Thread-safe.
    ModelInput prepare_input(const BoardState& state) const;

    // Evaluates prepared inputs (from any boards that fit) in one forward pass, copying each stacked
    // input tensor to the model's device once. Every ModelOutput tensor is left on the model's
    // device - callers needing CPU access (e.g. via .accessor<T,N>()) must call .cpu() themselves
    // at their point of use.
    ModelOutput evaluate_inputs(const std::vector<const ModelInput*>& inputs);
};

TORCH_MODULE(CNN);
