#include "training/self_play.h"
#include "model/features.h"
#include <chrono>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <random>

static std::mt19937 rng(std::random_device{}());

BoardState new_state(const GameConfig& cfg, const BoardConfig& bc) {
    return BoardState(
        cfg.num_stones, cfg.num_players,
        cfg.turn_stone_list, cfg.stone_to_player_map,
        cfg.forced_pass_only,
        std::vector<int>(bc.N, 0),
        bc);
}

std::vector<GameRecord> trajectory_to_records(
    const std::vector<PlyResult>& trajectory,
    const std::unordered_map<int,int>& stone_count,
    const std::unordered_map<int,int>& stone_to_player_map,
    int num_players)
{
    auto player_reward = compute_player_rewards(stone_count, stone_to_player_map);

    auto v_tensor = torch::zeros({num_players}, torch::kFloat32);
    auto va = v_tensor.accessor<float, 1>();
    for (int p = 1; p <= num_players; p++) {
        auto it = player_reward.find(p);
        if (it != player_reward.end()) va[p - 1] = it->second;
    }

    std::vector<GameRecord> records;
    records.reserve(trajectory.size());
    for (auto& pr : trajectory) {
        auto p_tensor = torch::zeros({(int)pr.policy.size()}, torch::kFloat32);
        auto pa = p_tensor.accessor<float, 1>();
        for (int i = 0; i < (int)pr.policy.size(); i++) pa[i] = pr.policy[i];
        records.push_back({pr.features, pr.legal_mask, p_tensor, v_tensor});
    }
    return records;
}

// Not thread-safe: uses the file-scope std::mt19937 rng to seed each MCTS instance.
std::pair<std::vector<PlyResult>, MCTSTiming> generate_one_ply_per_game(
    Evaluator& evaluator,
    const std::vector<BoardState*>& states,
    torch::Device device,
    int num_simulations,
    int temperature_threshold,
    float c_puct,
    int verbosity,
    std::optional<int> max_plies)
{
    MCTS mcts(evaluator, c_puct, rng());

    std::vector<float> temps;
    temps.reserve(states.size());
    for (auto* s : states)
        temps.push_back(s->ply_count() < temperature_threshold ? 1.0f : 0.0f);

    auto t_search0 = std::chrono::high_resolution_clock::now();
    auto [results, timing] = mcts.search_batch(
        states, num_simulations,
        NoiseConfig{/*add_noise=*/true, 0.3f, 0.25f}, temps, max_plies);
    double search_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_search0).count();
    timing.search = search_ms / 1000.0;

    std::vector<PlyResult> ply_results;
    ply_results.reserve(states.size());
    for (int j = 0; j < (int)states.size(); j++) {
        auto* s = states[j];
        auto& [policy, move] = results[j];
        auto [ft, mask] = board_to_features(*s, device);
        int stone = s->next_player;

        if (verbosity >= 2) {
            std::string move_str = (move == s->N) ? "pass" : std::to_string(move);
            std::cout << "  slot=" << j << " ply=" << s->ply_count()
                      << " stone=" << stone << " move=" << move_str
                      << " p=" << policy[move] << std::endl;
        }

        std::optional<int> k = (move == s->N) ? std::nullopt : std::optional<int>(move);
        bool ok = s->make_move(k);
        assert(ok && "MCTS returned illegal move");

        ply_results.push_back({ft, mask, std::move(policy), stone, move});
    }

    return {std::move(ply_results), timing};
}
