#pragma once
#include "game/board_config.h"
#include "game/board_state.h"
#include "model/features.h"
#include "model/evaluator.h"
#include "model/model_config.h"
#include "mcts/mcts.h"
#include "mcts/nn_server.h"
#include "training/replay_buffer.h"
#include "nlohmann/json.hpp"
#include <vector>
#include <unordered_map>
#include <map>
#include <optional>
#include <cstdint>
#include <string>
#include <condition_variable>
#include <mutex>
#include <thread>

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
    // Mirrors shared/gameConfig.ts's GameConfig.maxPlies; nullopt = null.
    std::optional<int> max_plies;

    // Mirrors shared/gameConfig.ts's GameConfig.toJSON(), minus `players` (the engine doesn't
    // track PlayerInfo). Used for train.cpp's checkpoint config.json.
    nlohmann::json to_json() const;
};

// True iff a and b agree on the fields that determine which trained model fits a ruleset:
// board_descr (exact text), num_stones, num_players, turn_list, stone_to_player_map (as a sorted
// (stone,player) set), which place limits are set (not their values), forced_pass_only,
// allow_suicide, score_rule, komi and ko_rule. board_seed is excluded because training draws its
// own board seed; max_plies is excluded too. Used by server.cpp to find the checkpoint directory
// for a /move request's config.
bool weak_equal(const GameConfig& a, const GameConfig& b);

// True iff a and b are identical in every field (== GameConfig::to_json()
// equality). Used by train.cpp's --resume/--retrain validation, where the game
// rules must match the checkpoint's exactly, not just on the weak_equal subset.
bool strong_equal(const GameConfig& a, const GameConfig& b);

// Mirrors shared/gameConfig.ts's GameConfig.fromJSON(), minus `players`. Shared by server.cpp's
// /move handler (its request `config` object) and train.cpp's --game-config file.
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

// Builds the full self-describing feature-block descriptor for a training run - the JSON stored
// as ModelConfig::input_descr (model/model_config.h), consumed by board_to_features()
// (model/features.h). Board size is separate, since GameConfig has no board-geometry fields of its
// own: `side_lens` (goes_train's --side-lens, else the board's own) is stored as "sideLens", the
// CNN/UNet grid, which every board they run on must fit in (fits_in(), game/board_config.h), and
// `n_bound`, an upper bound on the board's node count (goes_train's --n-bound, else
// num_cells(side_lens)), as "nBound", which sizes the liberty and groupSize blocks. Only train.cpp calls this (on --resume, the
// result must match the checkpoint's stored model config); server.cpp's load_model() reads the descriptor stored in a
// checkpoint's _config.json, so a checkpoint stays self-describing even if this function's own
// block formulas change later.
//
// Shape: {"blocks": [[name, ...args], ...], "totalDims": F, "sideLens": side_lens, "nBound": n_bound}.
// Each block's
// args already fully determine its width without needing a GameConfig to
// reinterpret - see board_to_features()'s doc comment for the exact per-block
// arg meaning and the "0 bits = absent, dense grid" convention used by
// playerStoneBudget/globalStoneBudget. cfg's place limits may be left empty (unset); this
// resolves them itself.
//
// Used identically by all four architectures for their CURRENT-ply-only input - including the
// Transformer, whose separate, much smaller per-ply HISTORY descriptor (plyMod + stoneOccupancy
// only) is built directly in train.cpp and stored as TransformerConfig::history_descr, entirely
// independent of this function.
nlohmann::json compute_input_descr(const GameConfig& cfg, const std::vector<unsigned>& side_lens, int n_bound);

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
    // If set, each game ends after Uniform(K1,K2)*N plies, sampled per game - combined with the
    // config's max_plies by taking the tighter bound.
    std::optional<std::pair<float,float>> linear_move_bound;
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

// Concurrent self-play games, one thread per game - KataGo's self-play structure. Each thread plays
// its own game at its own pace - opening moves sampled from the raw policy, then a cheap or full
// search per move (see SelfPlayParams) - with every search's network requests batched by one shared
// NNServer, and starts a new game as soon as one finishes.
//
// play() runs every thread until `num_games` games have finished, then pauses each thread at its
// next ply boundary and returns every game finished by then (possibly more than `num_games`, since
// a thread's last ply can finish its game too). Unfinished games carry over to the next play() call,
// continuing with whatever evaluators that call passes. Network server thread i uses
// evaluators[i % evaluators.size()] (NNServer::set_evaluators()).
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
                   int num_games, int nn_server_threads, int nn_max_batch, uint64_t seed);
    ~SelfPlayRunner();

    std::vector<FinishedGame> play(const std::vector<Evaluator>& evaluators, int num_games);

    // Network rows, batches and evaluation time since the previous call.
    NNServer::Stats take_nn_stats() { return nn_.take_stats(); }

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
    NNServer nn_;

    std::mutex mutex_;
    std::condition_variable workers_cv_, main_cv_;
    bool running_ = false, stopping_ = false;
    int paused_ = 0;
    int target_games_ = 0;
    std::vector<FinishedGame> finished_;
    std::vector<std::thread> threads_;

    void worker(uint64_t seed);
    Slot new_slot(std::mt19937_64& rng) const;
    void play_ply(Slot& slot, MCTS& mcts, std::mt19937_64& rng);
    FinishedGame finish(Slot& slot, std::mt19937_64& rng) const;
};

// Rebuilds a finished game's GameRecord from its trajectory (SelfPlayRunner's format) by replaying
// its moves through a fresh BoardState - used by --resume (to refill the training window) and --retrain.
GameRecord replay_trajectory(const nlohmann::json& trajectory, const GameConfig& cfg, const BoardConfig& bc,
                             const nlohmann::json& descr, const nlohmann::json* history_descr);
