#pragma once
#include <torch/torch.h>
#include <utility>
#include <vector>

// Unordered player pairs (p, q), p < q, 1-indexed, in the order the rank head's output rows and
// GameRecord::rank_target use: (1,2), (1,3), ..., (1,P), (2,3), ..., (P-1,P).
inline std::vector<std::pair<int, int>> rank_pairs(int num_players) {
    std::vector<std::pair<int, int>> pairs;
    for (int p = 1; p <= num_players; p++)
        for (int q = p + 1; q <= num_players; q++) pairs.push_back({p, q});
    return pairs;
}

inline int num_rank_pairs(int num_players) { return num_players * (num_players - 1) / 2; }

// Class indices of one pair's 3-way outcome, as predicted by RankHead and stored in
// GameRecord::rank_target.
enum RankOutcome : int64_t { FIRST_AHEAD = 0, TIED = 1, SECOND_AHEAD = 2 };

// Pairwise-outcome ("rank") head, shared by every architecture: mean- and max-pools the trunk's
// per-node hidden states over the board, then an MLP gives, for every player pair (p, q), the
// probabilities that p finishes ahead of q, that they tie, and that q finishes ahead - see
// ai/Readme.md's "Rank Head" section for why MCTS needs this rather than a value derived from the
// ownership head.
struct RankHeadImpl : torch::nn::Module {
    torch::nn::Sequential mlp{nullptr};
    int num_pairs_;

    RankHeadImpl(int hidden_dim, int num_pairs);

    // h: (B, N, hidden_dim) -> (B, num_pairs, 3) probabilities, each row summing to 1.
    torch::Tensor forward(const torch::Tensor& h);
};
TORCH_MODULE(RankHead);
