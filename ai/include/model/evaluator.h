#pragma once
#include <functional>
#include <stdexcept>
#include <vector>
#include <utility>
#include <string>
#include <unordered_map>
#include <torch/torch.h>
#include "game/board_state.h"
#include "model/features.h"
#include "model/model_output.h"
#include "model/rank_head.h"

// Type-erased neural-network evaluator used by MCTS and self-play, split as KataGo's NNEvaluator
// splits an evaluation: prepare() builds one position's input on the CPU in whichever thread wants
// it evaluated (thread-safe), and run() evaluates a batch of prepared inputs in one forward pass.
// Wraps any of the four architectures (see make_evaluator(), any_model.h) - for a
// MessagePassingGNN, its adj_norms are captured inside run, so every wrapped model presents the same
// interface regardless of what it needs internally.
struct Evaluator {
    std::function<ModelInput(const BoardState&)> prepare;
    std::function<ModelOutput(const std::vector<const ModelInput*>&)> run;

    // prepare() for each state, then one run().
    ModelOutput evaluate_batch(const std::vector<const BoardState*>& states) const {
        std::vector<ModelInput> inputs;
        inputs.reserve(states.size());
        for (const BoardState* s : states) inputs.push_back(prepare(*s));
        std::vector<const ModelInput*> ptrs;
        ptrs.reserve(inputs.size());
        for (const ModelInput& in : inputs) ptrs.push_back(&in);
        return run(ptrs);
    }
};

// Combines the model's stone/territory ownership estimates into expected
// per-stone-type point totals under `rule` ("stone" | "territoryonly" |
// "area" | "territory"), mirroring BoardState::compute_points() - summed over
// all N locations. "territory" matches "territoryonly" here (captures are
// player- not stone-indexed - see estimate_player_totals(), which folds
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

// The model's expected final per-player totals for a batch of non-terminal states: the ownership
// heads' expected per-stone-type points (estimate_stone_points()), credited to players through
// stone_to_player_map, plus komi and, under "territory", each state's captures so far - the same
// quantities compute_player_totals() (board_state.h) sums exactly at a terminal state. A stone
// mapping to several players credits each of them its full total. A player no stone credits gets
// komi alone. MCTS batches always share one game's config, so a single rule/map per call is
// correct; capture counts differ between positions, so capture_counts has one row per batch entry
// (BoardState::capture_count()). Returns (B, num_players).
inline torch::Tensor estimate_player_totals(
    const torch::Tensor& ownership, const std::vector<std::vector<int>>& capture_counts,
    const std::string& rule, const std::unordered_map<int, std::vector<int>>& stone_to_player_map,
    int num_players, const std::vector<float>& komi)
{
    auto stone_points = estimate_stone_points(ownership, rule); // (B, num_stones)
    int64_t num_stones = stone_points.size(-1);

    auto agg = torch::zeros({num_stones, (int64_t)num_players}, torch::kFloat32);
    auto agg_a = agg.accessor<float, 2>();
    for (auto& [stone, players] : stone_to_player_map) {
        if (stone < 1 || stone > num_stones) continue;
        for (int player : players)
            if (player >= 1 && player <= num_players)
                agg_a[stone - 1][player - 1] = 1.0f;
    }
    auto total = torch::matmul(stone_points, agg.to(stone_points.device())); // (B, num_players)

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
    return total;
}

// Stand-in for a network before any has been trained - the analog of KataGo's "random" net, which
// self-play uses until enough data exists to train the first model. Costs no forward pass: the
// policy is uniform over each state's legal actions, every node's ownership is uniform over
// empty/each stone type, and every player pair's outcome is uniform. prepare() builds only the legal
// mask. All states must share one game config (N nodes, num_stones stone types). Outputs are on the
// CPU.
inline Evaluator make_uniform_evaluator(int N, int num_stones, int num_players) {
    return Evaluator{
        [](const BoardState& s) { return ModelInput{{}, legal_action_mask(s), {}}; },
        [N, num_stones, num_players](const std::vector<const ModelInput*>& inputs) {
            auto legal = stack_inputs(inputs, &ModelInput::legal_mask).to(torch::kFloat32);  // (B, ns*N+1)
            auto policy = legal / legal.sum(-1, true).clamp_min(1.0f);
            int64_t B = legal.size(0);
            auto ownership = torch::full({B, 2, (int64_t)N, (int64_t)num_stones + 1},
                                         1.0f / static_cast<float>(num_stones + 1));
            auto rank = torch::full({B, (int64_t)num_rank_pairs(num_players), 3}, 1.0f / 3.0f);
            return ModelOutput{policy, ownership, rank};
        }};
}
