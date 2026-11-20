#pragma once
#include <torch/torch.h>
#include "model/model_output.h"
#include "model/rank_pairs.h"

// Policy head, shared by every architecture: a per-slot Linear produces num_stones place-logit
// channels and 1 pass-field channel; the pass field is reduced to a single pass logit via a learned
// attention-weighted sum over slots, so it stays valid for any slot count (attention weights come
// from slot features, not fixed slot positions).
struct PolicyHeadImpl : torch::nn::Module {
    torch::nn::Linear proj{nullptr};  // hidden_dim -> num_stones+1 (place logits, pass field)
    torch::nn::Linear attn{nullptr};  // hidden_dim -> 1 (unnormalised attention score)
    int num_stones_;

    PolicyHeadImpl(int hidden_dim, int num_stones);

    // h: (B, S, hidden_dim) over the model's slots -> (B, num_stones*S+1) logits, placements
    // stone-major (index (stone-1)*S+slot) and pass last.
    torch::Tensor forward(const torch::Tensor& h);
};
TORCH_MODULE(PolicyHead);

// Pairwise-outcome ("rank") head, shared by every architecture: mean- and max-pools the trunk's
// per-slot hidden states, then an MLP gives, for every player pair (p, q), the
// probabilities that p finishes ahead of q, that they tie, and that q finishes ahead - see
// ai/Readme.md's "Rank Head" section for why MCTS needs this rather than a value derived from the
// ownership head.
struct RankHeadImpl : torch::nn::Module {
    torch::nn::Sequential mlp{nullptr};
    int num_pairs_;

    RankHeadImpl(int hidden_dim, int num_pairs);

    // h: (B, S, hidden_dim) -> (B, num_pairs, 3) probabilities, each row summing to 1.
    torch::Tensor forward(const torch::Tensor& h);
};
TORCH_MODULE(RankHead);

// The heads every architecture puts on its trunk's final per-slot hidden states. register_on()
// registers them on the architecture's own module, as "stone_head", "territory_head",
// "policy_head" and "rank_head", not under a submodule of their own.
//   stone_head/territory_head: Linear -> ReLU -> Linear, softmaxed at each slot over "which stone
//       type occupies/holds territory at this location at game end" (channel 0 = none) -
//       stone-type indexed to match ScoreData/BoardState::board; player-level aggregation (via
//       stone_to_player_map) happens downstream, e.g. in estimate_player_totals() (evaluator.h)
//   policy_head: PolicyHead, softmaxed over legal_mask's legal actions
//   rank_head: RankHead
struct OutputHeads {
    torch::nn::Sequential stone_head{nullptr};
    torch::nn::Sequential territory_head{nullptr};
    PolicyHead policy_head{nullptr};
    RankHead rank_head{nullptr};

    void register_on(torch::nn::Module& owner, int hidden_dim, int num_stones, int num_players);

    // h: (B, S, hidden_dim), legal_mask: (B, num_stones*S+1) -> ModelOutput (model_output.h). A row
    // with no legal action (a finished game) gets a uniform policy.
    ModelOutput forward(const torch::Tensor& h, const torch::Tensor& legal_mask);
};
