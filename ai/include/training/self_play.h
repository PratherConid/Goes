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
#include <map>
#include <optional>
#include <cstdint>
#include <string>

struct GameConfig {
    // Mirrors shared/gameConfig.ts's GameConfig.boardDescr, but held as cleg source text (its
    // toJSON() form), not a parsed ClegProgram - re-parsed on demand (game/cleg_parser.h's
    // parse_cleg) wherever a board is needed, since nothing here keeps the AST around.
    std::string board_descr;
    // Mirrors shared/gameConfig.ts's GameConfig.boardSeed; nullopt = null. server.cpp requires it on
    // a /move request's config. train.cpp ignores it and draws its own seed per run.
    std::optional<uint32_t> board_seed;
    int num_stones;
    int num_players;
    std::vector<TurnInfo> turn_list;
    // Mirrors shared/gameConfig.ts's GameConfig.playerStonePlaceLimit; nullopt = null. If left
    // empty, new_state() defaults to all-unlimited, sized to num_stones x num_players.
    std::vector<std::vector<std::optional<int>>> player_stone_place_limit;
    // Mirrors shared/gameConfig.ts's GameConfig.globalStonePlaceLimit; nullopt = null. If left
    // empty, new_state() defaults to all-unlimited, sized to num_stones.
    std::vector<std::optional<int>> global_stone_place_limit;
    std::unordered_map<int, std::vector<int>> stone_to_player_map;
    bool forced_pass_only = true;
    // Mirrors shared/gameConfig.ts's GameConfig.scoreRule, as a plain string.
    std::string score_rule = "area";
    // Mirrors shared/gameConfig.ts's GameConfig.komi. If left empty, new_state() defaults to
    // all-zero, sized to num_players.
    std::vector<float> komi;
    // Mirrors shared/gameConfig.ts's GameConfig.koRule.
    KoRule ko_rule = KoRule::Situational;
    // Mirrors shared/gameConfig.ts's GameConfig.allowSuicide.
    bool allow_suicide = false;
    // Mirrors shared/gameConfig.ts's GameConfig.maxPlies; nullopt = null. new_state() takes the
    // smaller of this and any linear_move_bound sample (below).
    std::optional<int> max_plies;
    // If set, new_state() samples a bound uniformly between {k1,k2}.first*N
    // and {k1,k2}.second*N and applies it to the fresh game's
    // BoardState::max_plies (see max_plies, above, for how the two combine).
    std::optional<std::pair<float,float>> linear_move_bound;

    // Mirrors shared/gameConfig.ts's GameConfig.toJSON(), minus `players` (the engine doesn't
    // track PlayerInfo). Used for train.cpp's checkpoint config.json.
    nlohmann::json to_json() const;
};

// True iff a and b agree on every field that used to determine
// GameConfig::model_tag()'s (now-removed) human-readable checkpoint
// directory name: board_descr, num_stones, num_players, turn_list,
// stone_to_player_map, forced_pass_only, allow_suicide, score_rule, komi,
// ko_rule. board_descr is compared as plain text (exact string equality) - together with
// board_seed it fully determines the actual board topology, so a checkpoint trained on one board
// description must never be treated as resumable with a differently-worded (even if it would
// evaluate to an isomorphic board) one. board_seed itself is excluded: a training run builds its
// board from a seed of its own (see train.cpp), so a checkpoint is matched by description alone.
// stone_to_player_map compares as a sorted (stone,player) pair set
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

// Mirrors shared/gameConfig.ts's GameConfig.fromJSON(), minus `players`. Shared by server.cpp's
// /move handler (its request `config` object) and train.cpp's --game-config file.
// linear_move_bound has no TS analog, so it's never set here - callers that want it set it
// explicitly afterward.
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
//
// Used identically by all four architectures for their CURRENT-ply-only input - including the
// Transformer, whose separate, much smaller per-ply HISTORY descriptor (plyMod + stoneOccupancy
// only) is built directly in train.cpp and stored as TransformerConfig::history_descr, entirely
// independent of this function.
nlohmann::json compute_input_descr(const GameConfig& cfg, int N);

// Create a fresh starting board state for a game.
BoardState new_state(const GameConfig& cfg, const BoardConfig& bc);

// KataGo's self-play settings (its PlaySettings) that Goes ports - defaults are KataGo's
// cpp/configs/training/selfplay1.cfg values. See ai/Readme.md's "Self-Play".
struct SelfPlayParams {
    int full_visits = 600;                  // maxVisits
    int cheap_visits = 100;                 // cheapSearchVisits
    float cheap_prob = 0.75f;               // cheapSearchProb - cheap searches are not trained on
    bool reduce_visits = true;              // fewer visits and less weight once a game is decided
    float reduce_visits_threshold = 0.9f;   // search win term some player must stay above...
    int reduce_visits_lookback = 3;         // ...for this many consecutive searched plies
    int reduced_visits_min = 100;
    float reduced_visits_weight = 0.1f;
    float policy_surprise_data_weight = 0.5f;
    float value_surprise_data_weight = 0.1f;
    float policy_init_area_prop = 0.04f;    // mean opening moves sampled from the raw policy, per node
    float policy_init_temperature = 1.0f;
    float move_temperature_early = 0.75f;   // chosenMoveTemperatureEarly
    float move_temperature = 0.15f;         // chosenMoveTemperature
    float temperature_halflife = 19.0f;     // chosenMoveTemperatureHalflife
    float root_policy_temperature_early = 1.25f;
    float root_policy_temperature = 1.1f;
    float forced_playouts_coeff = 2.0f;     // rootDesiredPerChildVisitsCoeff
};

// One ply of a game in progress.
struct PlyRecord {
    int move = 0;                            // flat action (stone-major, num_stones*N = pass)
    bool searched = false;                   // false: an opening move sampled from the raw policy
    std::vector<float> policy_target;        // searched plies: the search's policy target
    std::vector<float> search_pair_probs;    // searched plies: the root's search-averaged pair outcomes
    std::vector<float> nn_pair_probs;        // searched plies: the root's network pair outcomes
    float target_weight = 0.0f;              // before surprise reweighting
    float policy_surprise = 0.0f;            // KL(policy_target || root prior)
    int row_count = 0;                       // resolved training multiplicity
    torch::Tensor features, legal_mask;      // searched plies, CPU, captured before the move
    torch::Tensor history_features;          // every ply, Transformer runs only
};

// Builds a finished game's GameRecord: rows are the searched plies with row_count > 0, each with
// the blended pair-outcome target (see ai/Readme.md's "Value Targets"); stone/territory ownership
// comes from final_state.
GameRecord build_game_record(const std::vector<PlyRecord>& plies, const BoardState& final_state);

struct FinishedGame {
    GameRecord record;
    nlohmann::json trajectory;  // {"plies": [...]} - see SelfPlayRunner's trajectory format
    BoardState final_state;
};

// A pool of concurrent self-play games. Each step() plays one ply in every game - an opening move
// sampled from the raw policy, or a batched search (cheap or full, see SelfPlayParams) - and
// returns the games that just finished, each slot immediately restarting with a fresh game.
//
// Trajectory format, one object per game: {"plies": [ply, ...]} (train.cpp adds "model": the
// iteration of the model that played it, -1 = uniform evaluator), where an opening ply is
// {"move": a} and a searched ply is {"move": a, "search": [num_rank_pairs*3 floats], "weight": w}
// plus "policy": [num_stones*N+1 floats] when w > 0 - everything replay_trajectory() needs to
// rebuild the game's GameRecord exactly.
class SelfPlayRunner {
public:
    SelfPlayRunner(const GameConfig& cfg, const BoardConfig& bc, nlohmann::json input_descr,
                   const nlohmann::json* history_descr, SelfPlayParams params, SearchParams search_params,
                   int num_games, uint64_t seed);

    std::vector<FinishedGame> step(const Evaluator& evaluator, MCTSTiming& timing);

private:
    struct Slot {
        BoardState state;
        std::vector<PlyRecord> plies;
        int opening_moves_left = 0;
        std::vector<float> recent_win;  // per searched ply, the best player's search win term
    };

    const GameConfig& cfg_;
    const BoardConfig& bc_;
    nlohmann::json input_descr_;
    const nlohmann::json* history_descr_;
    SelfPlayParams params_;
    SearchParams search_params_;
    std::mt19937_64 rng_;
    std::vector<Slot> slots_;

    Slot new_slot();
    FinishedGame finish(Slot& slot);
};

// Rebuilds a finished game's GameRecord from its trajectory (SelfPlayRunner's format) by replaying
// its moves through a fresh BoardState - used by --resume/--retrain to refill the training window.
GameRecord replay_trajectory(const nlohmann::json& trajectory, const GameConfig& cfg, const BoardConfig& bc,
                             const nlohmann::json& descr, const nlohmann::json* history_descr);
