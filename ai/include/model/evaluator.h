#pragma once
#include <functional>
#include <vector>
#include <utility>
#include <string>
#include <unordered_map>
#include <torch/torch.h>
#include "game/board_state.h"

// Type-erased neural-network evaluator used by MCTS and self-play.
// Wraps either a MessagePassingGNN (with adj_norms captured in fn), a UNet, or a CNN.
struct Evaluator {
    std::function<std::pair<torch::Tensor, torch::Tensor>(
        const std::vector<const BoardState*>&)> fn;

    std::pair<torch::Tensor, torch::Tensor>
    evaluate_batch(const std::vector<const BoardState*>& states) const {
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

// Derives a per-player backup scalar from estimate_stone_points(), using the
// same rank+fraction reward formula as compute_player_rewards() (board_state.cpp),
// but vectorized over continuous expected points instead of ground-truth integer
// points. Since estimate_stone_points() already mirrors BoardState::compute_points(),
// MCTS's non-terminal leaf evaluation stays consistent with the terminal ground
// truth, whatever the game's scoring rule. MCTS batches always share one game's
// config (see Evaluator::evaluate_batch's "all must share the same board"
// contract), so a single rule/map per call is correct.
inline torch::Tensor estimate_player_rewards(
    const torch::Tensor& ownership, const std::string& rule,
    const std::unordered_map<int, std::vector<int>>& stone_to_player_map, int num_players,
    const std::vector<float>& komi, const std::vector<int>& capture_count)
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

    // Fold in komi (and, under "territory", capture_count - captures so far
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
    if (rule == "territory" && (int64_t)capture_count.size() == num_players) {
        std::vector<float> capture_count_f(capture_count.begin(), capture_count.end());
        auto capture_t = torch::from_blob((void*)capture_count_f.data(), {(int64_t)capture_count_f.size()}, torch::kFloat32)
                              .to(total.device());
        total = total + capture_t;
    }

    int64_t P = total.size(-1);
    if (P <= 1) return torch::zeros_like(total);

    // clamp_min (not an abs/sign-safe clamp) is safe here: komi is required to
    // be >= 0 (see the BoardState constructor assert) and the estimated
    // stone/territory/capture points are never negative either, so sum_total
    // can never be negative - clamp_min only guards against an exact/near-zero sum.
    auto sum_total = total.sum(-1, /*keepdim=*/true);  // (B, 1)
    auto sf = total / sum_total.clamp_min(1e-8f);       // (B, P) - ~0 when sum_total ~ 0

    auto diff = total.unsqueeze(2) - total.unsqueeze(1);   // (B, P, P): diff[b,p,q] = total[p]-total[q]
    auto rank_diff = torch::sign(diff).sum(-1);             // (B, P)

    float denom = static_cast<float>(2 * P - 1);
    auto rank_reward  = rank_diff * (2.0f / denom);
    auto point_reward = (sf - 1.0f / static_cast<float>(P)) / denom;
    return rank_reward + point_reward;  // (B, P)
}
