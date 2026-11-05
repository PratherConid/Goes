#pragma once
#include <functional>
#include <stdexcept>
#include <vector>
#include <utility>
#include <string>
#include <unordered_map>
#include <torch/torch.h>
#include "game/board_state.h"
#include "model/model_output.h"
#include "model/rank_head.h"

// Type-erased neural-network evaluator used by MCTS and self-play. Wraps any of the four
// architectures (see make_evaluator(), any_model.h) - for a MessagePassingGNN, its adj_norms are
// captured inside fn, so every wrapped model presents the same states-in/ModelOutput-out
// interface regardless of what it needs internally.
struct Evaluator {
    std::function<ModelOutput(const std::vector<const BoardState*>&)> fn;

    ModelOutput evaluate_batch(const std::vector<const BoardState*>& states) const {
        return fn(states);
    }
};

// Combines the model's stone/territory ownership estimates into expected
// per-stone-type point totals under `rule` ("stone" | "territoryonly" |
// "area" | "territory"), mirroring BoardState::compute_points() - summed over
// all N locations. "territory" matches "territoryonly" here (captures are
// player- not stone-indexed - see estimate_player_rewards(), which folds
// them in separately). `ownership` may be the model's own (B,2,N,num_stones+1)
// softmax output, or a one-hot-encoded ground-truth tensor of the same shape
// (see train.cpp's point_loss, which uses this for both the predicted and
// actual point totals).
inline torch::Tensor estimate_stone_points(const torch::Tensor& ownership, const std::string& rule) {
    auto stone_est     = ownership.select(1, 0);  // (B, 2, N, num_stones+1) -> (B, N, num_stones+1)
    auto territory_est = ownership.select(1, 1);  // (B, N, num_stones+1)
    torch::Tensor points_est = (rule == "stone")                                  ? stone_est
                              : (rule == "territoryonly" || rule == "territory")  ? territory_est
                              :                                                     stone_est + territory_est;  // "area"

    int64_t S = points_est.size(-1) - 1;
    // Expected per-stone-type points: sum over locations of that stone type's channel.
    return points_est.slice(-1, 1, S + 1).sum(1); // (B, num_stones)
}

// The model's estimate of compute_player_rewards()'s (board_state.cpp) rank+fraction reward for
// a non-terminal state, per player:
//   - rank term: the expected rank_diff, from `rank` (ModelOutput::rank, a RankHead's pairwise
//     outcome probabilities) - P(p ahead of q) - P(q ahead of p), summed over p's opponents q -
//     rather than anything derived from the ownership head (see ai/Readme.md's "Rank Head"
//     section for why).
//   - point-fraction term: from estimate_stone_points() - expected per-stone-type points
//     aggregated to players - which mirrors BoardState::compute_points(), so it stays consistent
//     with the terminal ground truth whatever the game's scoring rule.
// MCTS batches always share one game's config (see Evaluator::evaluate_batch's "all must share
// the same board" contract), so a single rule/map per call is correct. Capture counts differ
// between positions, though, so capture_counts holds one row per batch entry (that state's
// BoardState::capture_count()).
inline torch::Tensor estimate_player_rewards(
    const torch::Tensor& ownership, const torch::Tensor& rank, const std::vector<std::vector<int>>& capture_counts,
    const std::string& rule, const std::unordered_map<int, std::vector<int>>& stone_to_player_map,
    int num_players, const std::vector<float>& komi)
{
    auto stone_points = estimate_stone_points(ownership, rule); // (B, num_stones)
    int64_t num_stones = stone_points.size(-1);

    // Aggregate per-stone-type totals to per-player totals via stone_to_player_map,
    // mirroring compute_player_rewards()'s (board_state.cpp) own stone->player
    // aggregation - built as a small (num_stones,num_players) membership matrix
    // (num_stones/num_players are always tiny, this is cheap CPU-side setup).
    // A stone mapping to several players credits each of them the same full
    // total (not split), matching compute_player_rewards().
    auto agg = torch::zeros({num_stones, (int64_t)num_players}, torch::kFloat32);
    auto agg_a = agg.accessor<float, 2>();
    for (auto& [stone, players] : stone_to_player_map) {
        if (stone < 1 || stone > num_stones) continue;
        for (int player : players)
            if (player >= 1 && player <= num_players)
                agg_a[stone - 1][player - 1] = 1.0f;
    }
    auto total = torch::matmul(stone_points, agg.to(stone_points.device())); // (B, num_players)

    // Fold in komi (and, under "territory", capture_counts - captures so far
    // are a fully-known quantity at any ply, not a prediction, so this is a
    // direct extension of the same treatment) before computing the
    // point-fraction denominator (below), matching compute_player_rewards()'s
    // (board_state.cpp) treatment - keeps the point-fraction term zero-sum
    // across players.
    if ((int64_t)komi.size() == num_players) {
        auto komi_t = torch::from_blob((void*)komi.data(), {(int64_t)komi.size()}, torch::kFloat32)
                          .to(total.device());
        total = total + komi_t;
    }
    if (rule == "territory") {
        auto capture_t = torch::zeros({total.size(0), (int64_t)num_players}, torch::kFloat32);
        auto capture_a = capture_t.accessor<float, 2>();
        for (int64_t b = 0; b < total.size(0); b++)
            for (int p = 0; p < num_players; p++)
                capture_a[b][p] = static_cast<float>(capture_counts.at(b).at(p));
        total = total + capture_t.to(total.device());
    }

    int64_t P = total.size(-1);
    if (P <= 1) return torch::zeros_like(total);

    // clamp_min (not an abs/sign-safe clamp) is safe here: komi is required to
    // be >= 0 (see the BoardState constructor assert) and the estimated
    // stone/territory/capture points are never negative either, so sum_total
    // can never be negative - clamp_min only guards against an exact/near-zero sum.
    auto sum_total = total.sum(-1, /*keepdim=*/true);  // (B, 1)
    auto sf = total / sum_total.clamp_min(1e-8f);       // (B, P) - ~0 when sum_total ~ 0

    auto pairs = rank_pairs(static_cast<int>(P));
    if (rank.size(1) != (int64_t)pairs.size())
        throw std::runtime_error("estimate_player_rewards: rank head scores " + std::to_string(rank.size(1)) +
                                 " player pairs, but " + std::to_string(P) + " players have " +
                                 std::to_string(pairs.size()));
    // pair_sign[k][p-1] = +1 / -1 for pair k's first / second player - the margin
    // P(first ahead) - P(second ahead) counts for the first and against the second.
    auto pair_sign = torch::zeros({(int64_t)pairs.size(), P}, torch::kFloat32);
    auto sign_a = pair_sign.accessor<float, 2>();
    for (size_t k = 0; k < pairs.size(); k++) {
        sign_a[k][pairs[k].first - 1] = 1.0f;
        sign_a[k][pairs[k].second - 1] = -1.0f;
    }
    auto margin = rank.select(-1, FIRST_AHEAD) - rank.select(-1, SECOND_AHEAD);   // (B, pairs)
    auto rank_diff = torch::matmul(margin, pair_sign.to(margin.device()));          // (B, P)

    float denom = static_cast<float>(2 * P - 1);
    auto rank_reward  = rank_diff * (2.0f / denom);
    auto point_reward = (sf - 1.0f / static_cast<float>(P)) / denom;
    return rank_reward + point_reward;  // (B, P)
}
