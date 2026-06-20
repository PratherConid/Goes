#pragma once
#include "game/board_config.h"
#include "game/board_state.h"
#include "model/gnn.h"
#include "mcts/mcts.h"
#include "training/replay_buffer.h"
#include <vector>
#include <unordered_map>
#include <optional>

struct GameConfig {
    int num_stones;
    int num_players;
    std::vector<int> turn_stone_list;
    std::unordered_map<int,int> stone_to_player_map;
    bool forced_pass_only = true;
};

// Per-ply result: features captured before the move, plus the chosen move.
struct PlyResult {
    torch::Tensor features;
    torch::Tensor legal_mask;
    std::vector<float> policy;
    int stone;
    int move;
};

// Create a fresh starting board state for a game.
BoardState new_state(const GameConfig& cfg, const BoardConfig& bc);

// Convert a completed game's trajectory to GameRecords with outcome value targets.
// Value for each ply = (stones belonging to that ply's player) / (total stones on board),
// using the final stone_count from the terminal BoardState. Returns 0 if the board is empty.
std::vector<GameRecord> trajectory_to_records(
    const std::vector<PlyResult>& trajectory,
    const std::unordered_map<int,int>& stone_count,
    const std::unordered_map<int,int>& stone_to_player_map,
    int num_players);

// Produce one training record per active game: run batched MCTS on all states,
// capture board features before the move, then apply the chosen move in place.
//
// Call only on states that are not yet game-over. After returning, check each
// state for game_over() (and max_plies, if used) to detect finished games and
// collect their trajectories.
//
// Dirichlet noise is always added to root priors for self-play exploration.
// temperature_threshold maps to the MCTS temperature parameter: plies before the
// threshold use temperature=1 (sampling proportional to visit counts); at or
// after it, temperature=0 (argmax).
//
// verbosity >=2: one line per ply plus a timing summary per batch.
std::pair<std::vector<PlyResult>, MCTSTiming> generate_one_ply_per_game(
    MessagePassingGNN& model,
    const std::vector<BoardState*>& states,
    const AdjNorms& adj_norms,
    int num_simulations,
    int temperature_threshold,
    float c_puct,
    int verbosity = 0,
    std::optional<int> max_plies = std::nullopt);
