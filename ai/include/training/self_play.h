#pragma once
#include "game/board_config.h"
#include "game/board_state.h"
#include "model/features.h"
#include "model/evaluator.h"
#include "model/model_config.h"
#include "mcts/mcts.h"
#include "training/replay_buffer.h"
#include "nlohmann/json.hpp"
#include <vector>
#include <unordered_map>
#include <optional>
#include <string>

struct GameConfig {
    // Mirrors shared/types.ts's GameConfig.boardType/boardArgs.
    std::string board_type;
    std::vector<int> board_args;
    int num_stones;
    int num_players;
    std::vector<TurnInfo> turn_list;
    // [stone-1][player-1]; nullopt = unlimited. If left empty, new_state()
    // defaults to all-unlimited, sized to num_stones x num_players.
    std::vector<std::vector<std::optional<int>>> player_stone_place_limit;
    // [stone-1]; nullopt = unlimited. If left empty, new_state() defaults to
    // all-unlimited, sized to num_stones.
    std::vector<std::optional<int>> global_stone_place_limit;
    std::unordered_map<int, std::vector<int>> stone_to_player_map;
    bool forced_pass_only = true;
    // Scoring rule ("stone" | "territoryonly" | "area") - see BoardState::compute_points().
    std::string score_rule = "area";
    // Per-player scoring handicap (see BoardState::komi). If left empty,
    // new_state() defaults to all-zero, sized to num_players.
    std::vector<float> komi;
    // Superko variant (see BoardState::ko_rule). Situational matches the
    // engine's pre-existing (and only) behavior before koRule was ported.
    KoRule ko_rule = KoRule::Situational;
    // Whether a move that leaves the mover's own group with zero liberties is
    // legal (captures that own group immediately, rather than being rejected).
    bool allow_suicide = false;
    // Fixed cap on BoardState::max_plies (nullopt = no fixed cap) - primarily
    // for server.cpp's /move endpoint, which receives this directly from the
    // client rather than sampling it. Combines with linear_move_bound (below)
    // rather than being overridden by it: new_state() takes the smaller of
    // this and any linear_move_bound sample, so a game-generation run that
    // sets both is still bounded by whichever is tighter.
    std::optional<int> max_plies;
    // If set, new_state() samples a bound uniformly between {k1,k2}.first*N
    // and {k1,k2}.second*N and applies it to the fresh game's
    // BoardState::max_plies (see max_plies, above, for how the two combine).
    std::optional<std::pair<float,float>> linear_move_bound;

    // Serialises to the same wire shape as shared/types.ts's GameConfig.toJSON()
    // (the mirror of parse_game_cfg, below) - used by train.cpp's checkpoint
    // config.json. GameConfig has no `players` field (the engine doesn't track
    // PlayerInfo), so that TS-side key is omitted; every other key matches by name.
    nlohmann::json to_json() const;
};

// True iff a and b agree on every field that used to determine
// GameConfig::model_tag()'s (now-removed) human-readable checkpoint
// directory name: board_type/board_args, num_stones, num_players, turn_list,
// stone_to_player_map, forced_pass_only, allow_suicide, score_rule, komi,
// ko_rule. stone_to_player_map compares as a sorted (stone,player) pair set
// (order within one stone's player list doesn't matter), matching the old
// tag's own flattening. playerStonePlaceLimit/globalStonePlaceLimit/maxPlies
// are intentionally excluded, same as the old tag - use strong_equal (below)
// where those need to match too. Used by server.cpp to find which checkpoint
// directory (now an opaque hash name) satisfies a /move request's config.
bool weak_equal(const GameConfig& a, const GameConfig& b);

// True iff a and b are identical in every field (== GameConfig::to_json()
// equality). Used by train.cpp's --resume validation, where the game rules
// must match the resumed checkpoint's exactly, not just on the weak_equal
// subset.
bool strong_equal(const GameConfig& a, const GameConfig& b);

// Parses a GameConfig-shaped JSON object into a GameConfig, matching
// shared/types.ts's GameConfig.toJSON() wire shape - the same parser used by
// server.cpp's /move handler (its request `config` object) and train.cpp's
// --game-config file, so there's one JSON->GameConfig implementation instead
// of two independently-maintained copies. linear_move_bound has no
// shared/types.ts analog (a self-play-only max_plies sampling knob), so it's
// never set here - callers that want it set it explicitly afterward.
GameConfig parse_game_cfg(const nlohmann::json& cfg);

// Resolves an empty player_stone_place_limit/global_stone_place_limit
// ("not set" - see GameConfig's own field doc comments, above) to an
// explicit all-unlimited array of the right shape. Shared by new_state()
// and any caller (e.g. train.cpp's model input-width computation) that
// needs the resolved shape before a BoardState exists.
std::vector<std::vector<std::optional<int>>> resolve_player_stone_place_limit(
    const std::vector<std::vector<std::optional<int>>>& limit, int num_stones, int num_players);
std::vector<std::optional<int>> resolve_global_stone_place_limit(
    const std::vector<std::optional<int>>& limit, int num_stones);

// Per-ply result: features captured before the move, plus the chosen move.
struct PlyResult {
    torch::Tensor features;
    torch::Tensor legal_mask;
    std::vector<float> policy;
    int move;    // flat MCTS action index (stone-major: (stone-1)*N+pos for a
                 // placement, numStones*N for pass) - see Policy Head and
                 // Action Space in ai/Readme.md.
    int stone;   // stone actually placed (1-indexed); 0 = pass. Decoded from
                 // `move` after MCTS picks it - a turn can now offer several
                 // stones, so this isn't knowable beforehand.

    // move/stone/policy only - features/legal_mask are never persisted (large
    // per-node tensors; not needed for the trajectory dump - see train.cpp -
    // which exists for human inspection via analysis.py, not replay).
    nlohmann::json to_json() const;
};

// Inverse of PlyResult::to_json() - move/stone/policy only; features/legal_mask
// are left default-constructed (undefined) torch::Tensors, since those are
// never part of the dumped JSON to begin with.
PlyResult parse_ply_result(const nlohmann::json& j);

// Builds the full self-describing feature-block descriptor for a fresh
// training run - the JSON stored as ModelConfig::input_descr (model/model_config.h),
// consumed by board_to_features() (model/features.h). Call exactly once, at
// the start of a fresh run (train.cpp), for callers that have a full
// GameConfig on hand (board size N is separate - see BoardConfig::N - since
// GameConfig has no board-geometry fields of its own) - never recompute this
// on --resume or in server.cpp's load_model(); both only ever read whatever
// descriptor is already stored in a checkpoint's _config.json, so a checkpoint
// stays self-describing even if this function's own block formulas change later.
//
// Shape: {"blocks": [[name, ...args], ...], "totalDims": N}. Each block's
// args already fully determine its width without needing a GameConfig to
// reinterpret - see board_to_features()'s doc comment for the exact per-block
// arg meaning and the "0 bits = absent, dense grid" convention used by
// playerStoneBudget/globalStoneBudget. Passing cfg's
// player_stone_place_limit/global_stone_place_limit unresolved (rather than
// through resolve_player_stone_place_limit/resolve_global_stone_place_limit)
// is fine here: an empty/unset field contributes the same all-zero-bits grid
// either way - resolution only matters where BoardState needs a
// concretely-shaped array to index into, not for this size computation.
nlohmann::json compute_input_descr(const GameConfig& cfg, int N);

// Create a fresh starting board state for a game.
BoardState new_state(const GameConfig& cfg, const BoardConfig& bc);

// Convert a completed game's trajectory to a single GameRecord with per-ply
// tensors stacked along a new leading dimension, and stone_owner/territory_owner
// ground truth (shared by the whole game) taken directly from the final board and
// ScoreData::territory_owner - both already stone-type indexed, matching the
// network's ownership head (see evaluator.h). Use this when the trajectory's
// PlyResults already carry features/legal_mask (i.e. straight from self-play,
// not round-tripped through PlyResult::to_json()/parse_ply_result()) and the
// final board/territory ownership are already on hand.
GameRecord trajectory_and_result_to_record(
    const std::vector<PlyResult>& trajectory,
    const std::vector<int>& board,
    const std::vector<int>& territory_owner_stone);

// Reconstructs a GameRecord from a trajectory whose PlyResults only carry
// move/stone/policy - e.g. parsed back from a trajectory dump via
// parse_ply_result(), which never has features/legal_mask or a final board
// state (see PlyResult::to_json()). Replays the moves through a fresh
// BoardState (built from cfg/bc) to regenerate each ply's features/legal_mask
// via board_to_features() (descr must match whatever descriptor the original
// game was actually recorded under - a mismatch produces a differently-shaped
// or differently-populated tensor, not necessarily a crash) and to obtain the
// game's final board/territory ownership, then delegates to
// trajectory_and_result_to_record(). Used by train.cpp's resume() to rebuild
// ReplayBuffer state from historical trajectory dumps.
//
// Always builds features/legal_mask on CPU - ReplayBuffer entries live on
// CPU regardless of the training device, and only get moved to GPU (if any)
// for the duration of one sampled training batch (see train.cpp's training
// loop) - self-play/MCTS itself never reads these particular tensors (they're
// built purely for storage; MCTS evaluation uses its own, separately-placed
// features - see generate_one_ply_per_game()).
GameRecord trajectory_to_record(
    const std::vector<PlyResult>& trajectory,
    const GameConfig& cfg,
    const BoardConfig& bc,
    const nlohmann::json& descr);

// Produce one training record per active game: run batched MCTS on all states,
// capture board features before the move, then apply the chosen move in place.
//
// Call only on states that are not yet game-over. After returning, check each
// state for game_over() (which now also covers its own max_plies, if set) to
// detect finished games and collect their trajectories.
//
// Dirichlet noise is always added to root priors for self-play exploration.
// temperature_threshold maps to the MCTS temperature parameter: plies before the
// threshold use temperature=1 (sampling proportional to visit counts); at or
// after it, temperature=0 (argmax).
//
// verbosity >=2: one line per ply plus a timing summary per batch.
//
// The returned PlyResults' features/legal_mask are always built on CPU (see
// trajectory_to_record()'s doc comment) - `evaluator` (not a device param
// here) already carries its own model's device internally, and is all MCTS
// itself needs for evaluation; these particular tensors exist only to be
// stored into the replay buffer via the caller's trajectory_and_result_to_record().
std::pair<std::vector<PlyResult>, MCTSTiming> generate_one_ply_per_game(
    Evaluator& evaluator,
    const std::vector<BoardState*>& states,
    const nlohmann::json& descr,
    int num_simulations,
    int temperature_threshold,
    float c_puct,
    int verbosity = 0);
