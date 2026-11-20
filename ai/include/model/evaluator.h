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
#include "model/rank_pairs.h"

// Type-erased neural-network evaluator used by MCTS and self-play, split as KataGo's NNEvaluator
// splits an evaluation: prepare() builds one position's input on the CPU in whichever thread wants
// it evaluated (thread-safe), and run() evaluates a batch of prepared inputs in one forward pass.
// Wraps any of the four architectures (see make_evaluator(), any_model.h) or the untrained
// stand-in (make_uniform_evaluator(), below) behind one interface. prepare() builds a position's
// input from its state and its game's GameInput (make_game_input(), features.h, built once when the
// game starts), which the input then carries, so one run() can mix boards.
struct Evaluator {
    std::function<ModelInput(const BoardState&, const std::shared_ptr<const GameInput>&)> prepare;
    std::function<ModelOutput(const std::vector<const ModelInput*>&)> run;
};

// Combines the model's stone/territory ownership estimates into expected
// per-stone-type point totals under `rule` ("stone" | "territoryonly" |
// "area" | "territory"), mirroring BoardState::compute_points() - summed over
// every location (a model's slot-space estimate must be zeroed at slots no node occupies first). "territory" matches "territoryonly" here (captures are
// player- not stone-indexed - see estimate_player_totals(), which folds
// them in separately). `ownership` may be the model's own (B,2,S,num_stones+1)
// softmax output, or a one-hot-encoded ground-truth tensor of the same shape
// (see train.cpp's compute_losses(), whose point loss uses this for both the
// predicted and actual point totals).
inline torch::Tensor estimate_stone_points(const torch::Tensor& ownership, const std::string& rule) {
    auto stone_est     = ownership.select(1, 0);  // (B, 2, S, num_stones+1) -> (B, S, num_stones+1)
    auto territory_est = ownership.select(1, 1);  // (B, S, num_stones+1)
    torch::Tensor points_est = (rule == "stone")                                  ? stone_est
                              : (rule == "territoryonly" || rule == "territory")  ? territory_est
                              :                                                     stone_est + territory_est;  // "area"

    // Expected per-stone-type points: sum over locations of that stone type's channel.
    return points_est.slice(-1, 1).sum(1); // (B, num_stones)
}

// The model's expected final per-player totals for a batch of non-terminal states: the ownership
// heads' expected per-stone-type points (estimate_stone_points()), credited to players through
// stone_to_player_map, plus komi and, under "territory", each state's captures so far - the same
// quantities compute_player_totals() (board_state.h) sums exactly at a terminal state. A stone
// mapping to several players credits each of them its full total. A player no stone credits gets
// only komi (and, under "territory", its captures). Every position in a batch shares one game
// config - an NNServer (mcts/nn_server.h) batches one SelfPlayRunner's games, and server.cpp
// evaluates one position at a time - so a single rule/map per call is correct; capture counts
// differ between positions, so capture_counts has one row per batch entry
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
// mask, in its game's slot space as a model's does, so states from different boards batch together.
// Outputs are on the CPU.
inline Evaluator make_uniform_evaluator(int num_stones, int num_players) {
    return Evaluator{
        [](const BoardState& s, const std::shared_ptr<const GameInput>& game) {
            return ModelInput{{}, actions_to_slots(legal_action_mask(s), *game), {}, game};
        },
        [num_stones, num_players](const std::vector<const ModelInput*>& inputs) {
            auto legal = stack_inputs(inputs, &ModelInput::legal_mask).to(torch::kFloat32);  // (B, ns*S+1)
            auto policy = legal / legal.sum(-1, true).clamp_min(1.0f);
            int64_t B = legal.size(0), S = inputs[0]->game->slots.valid.size(0);
            auto ownership = torch::full({B, 2, S, (int64_t)num_stones + 1}, 1.0f / static_cast<float>(num_stones + 1));
            auto rank = torch::full({B, (int64_t)num_rank_pairs(num_players), 3}, 1.0f / 3.0f);
            return ModelOutput{policy, ownership, rank};
        }};
}
