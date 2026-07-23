#pragma once
#include <torch/torch.h>
#include "game/board_state.h"
#include "game/board_config.h"
#include "model/features.h"
#include "model/model_config.h"
#include <vector>
#include <utility>

// Policy head: identical structure/formula to CNNPolicyHeadImpl/GNNPolicyHeadImpl/
// UNetPolicyHeadImpl (duplicated here rather than shared, matching this codebase's existing
// per-architecture convention - no common base class exists for these heads today). Per-node
// linear produces num_stones place-logit channels and 1 pass-field channel; the pass field is
// reduced to a single pass logit via a learned attention-weighted sum over nodes.
struct TransformerPolicyHeadImpl : torch::nn::Module {
    torch::nn::Linear proj{nullptr};  // hidden_dim -> num_stones+1 (place logits, pass field)
    torch::nn::Linear attn{nullptr};  // hidden_dim -> 1 (unnormalised attention score)
    int num_stones_;

    TransformerPolicyHeadImpl(int hidden_dim, int num_stones);

    // h: (B, N, hidden_dim) -> (B, num_stones*N+1) logits
    torch::Tensor forward(const torch::Tensor& h);
};
TORCH_MODULE(TransformerPolicyHead);

// One pre-LN attention + feed-forward block, shared shape for both the history self-attention
// stack and the cross-attention stack (only the query/key-value wiring differs at the call site -
// see TransformerImpl::forward()). Sequence-first (L,B,D) layout throughout - this libtorch
// build's torch::nn::MultiheadAttention has no batch_first option.
struct TransformerBlockImpl : torch::nn::Module {
    torch::nn::MultiheadAttention mha{nullptr};
    torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr};
    torch::nn::Sequential ffn{nullptr};  // Linear(D,4D) -> ReLU -> Linear(4D,D)

    TransformerBlockImpl(int hidden_dim, int num_heads);

    // q: (Lq,B,D) query sequence; kv: (Lk,B,D) key/value sequence (pass the same tensor as q for
    // self-attention); key_padding_mask: (B,Lk) bool, True = ignore that position.
    torch::Tensor forward(const torch::Tensor& q, const torch::Tensor& kv, const torch::Tensor& key_padding_mask);
};
TORCH_MODULE(TransformerBlock);

// History-aware architecture: the only one of the four that can support forcedPassOnly=true,
// since it's the only one that looks at more than the current board state.
//
// Encoders: TWO separate flatten-MLP encoders, since the current ply and past plies use
//          different-width feature descriptors (see TransformerConfig::input_descr vs
//          history_descr) - the current ply uses the SAME full descriptor CNN/UNet/GNN do
//          (legalPlace/liberty/groupSize/etc. all included, since only one current state is ever
//          evaluated at a time - no risk of the same physical position getting different
//          legalPlace bits across time), while every PAST ply uses a much narrower
//          plyMod+stoneOccupancy-only descriptor (legalPlace in particular depends on transient
//          per-turn context, not physical board content, so it can't be reused across time - see
//          history_features_at_ply()'s doc comment, model/features.h). encoder_in/encoder_out
//          (N*feature_dim -> D -> D) embeds the current ply only; hist_encoder_in/hist_encoder_out
//          (N*history_feature_dim -> D -> D) embeds every past ply, with the SAME shared weights
//          applied to each one. Both land in the same D-dim space. Topology-agnostic (no
//          adjacency/shape assumption - works for any board type identically), though N-specific
//          like every other architecture already is per checkpoint.
// History set: every PAST state's (D,) embedding (from hist_encoder_in/hist_encoder_out) forms an
//          unordered set, refined by self_attn_layers_ (num_attn_layers deep) - permutation-
//          EQUIVARIANT (no positional/recency signal ever added), so the set stays order-symmetric
//          by construction. A learned, never-masked history_sentinel_ token is always prepended so
//          attention never faces an all-masked row (avoids softmax(-inf,...) -> NaN) even for a
//          genesis state with zero past plies.
// Cross-attention: the CURRENT state's single (D,) embedding (from encoder_in/encoder_out) is the
//          query - it never joins the history set, playing a structurally different role instead
//          (cross_attn_layers_, same depth as the history stack), producing an enriched h*.
// Decoder: h* -> one monolithic MLP (decoder_in -> ReLU -> decoder_out) -> reshape (N, hidden_dim)
//          - the literal inverse of the encoder's flatten, no per-node operation anywhere in this
//          path (unlike the other three architectures' topology-aware bodies).
// Heads: the resulting (N, hidden_dim) tensor feeds the same stone_head/territory_head/policy
//          head shapes CNN/UNet/GNN already share, unmodified.
//
// Known cost tradeoff: unlike the other three architectures' O(1)-per-leaf evaluation, this is
// O(ply_count) per leaf (loops every historical ply via history_features_at_ply()) - materially
// more expensive per MCTS simulation for long games, though each past-ply reconstruction now skips
// the group_liberty() traversal entirely (the minimal history descriptor never needs
// liberty/groupSize - see board_to_features_at_ply()'s perf guard, features.cpp); only the
// current-ply's full-descriptor call still pays for it.
struct TransformerImpl : torch::nn::Module {
    torch::nn::Linear encoder_in{nullptr};   // N*feature_dim -> hidden_dim (current ply only)
    torch::nn::Linear encoder_out{nullptr};  // hidden_dim -> hidden_dim (current ply only)

    torch::nn::Linear hist_encoder_in{nullptr};   // N*history_feature_dim -> hidden_dim (past plies only)
    torch::nn::Linear hist_encoder_out{nullptr};  // hidden_dim -> hidden_dim (past plies only)

    std::vector<TransformerBlock> self_attn_layers_;   // cfg.num_attn_layers, history set only
    std::vector<TransformerBlock> cross_attn_layers_;  // cfg.num_attn_layers, query = current-state token

    torch::Tensor history_sentinel_;  // (1,1,hidden_dim) learned parameter, never masked

    torch::nn::Linear decoder_in{nullptr};   // hidden_dim -> hidden_dim
    torch::nn::Linear decoder_out{nullptr};  // hidden_dim -> N*hidden_dim

    torch::nn::Sequential stone_head{nullptr};
    torch::nn::Sequential territory_head{nullptr};
    TransformerPolicyHead policy_head{nullptr};

    TransformerConfig cfg_;
    int num_players_;
    int num_stones_;
    int N_;
    int history_feature_dim_;  // cfg_.history_descr's totalDims, cached at construction time
    static constexpr int kNumHeads = 4;  // hardcoded, not CLI-configurable - see .cpp ctor

    TransformerImpl(const BoardConfig& bc, const TransformerConfig& cfg, int num_players, int num_stones);

    // hist_x: (B,T,N,F) or (T,N,F) float32 (T past plies, zero-padded to the batch's own max);
    // hist_mask: (B,T) or (T,) bool, True = padded/invalid slot; cur_x: (B,N,F) or (N,F) - the
    // current ply; legal_mask: (B,numStones*N+1) or (numStones*N+1,).
    // Returns (policy, ownership) - same shapes as CNN/UNet/GNN's forward().
    std::pair<torch::Tensor, torch::Tensor> forward(
        torch::Tensor hist_x, torch::Tensor hist_mask,
        torch::Tensor cur_x, torch::Tensor legal_mask);

    // Evaluate a single BoardState. Returns (policy (numStones*N+1,), ownership (2,N,num_stones+1)),
    // both left on the model's device - matching CNN/GNN/UNet's evaluate() contract exactly.
    std::pair<torch::Tensor, torch::Tensor> evaluate(const BoardState& state);

    // Evaluate a batch of states (arbitrary, independently-varying history lengths) in one forward
    // pass. Returns tensors on the model's device (see evaluate()'s comment).
    std::pair<torch::Tensor, torch::Tensor> evaluate_batch(const std::vector<BoardState*>& states);
    std::pair<torch::Tensor, torch::Tensor> evaluate_batch(const std::vector<const BoardState*>& states);
};

TORCH_MODULE(Transformer);
