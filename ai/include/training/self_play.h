#pragma once
#include "game/board_config.h"
#include "game/board_state.h"
#include "game/game_config.h"
#include "model/features.h"
#include "model/evaluator.h"
#include "mcts/mcts.h"
#include "mcts/nn_server.h"
#include "training/replay_buffer.h"
#include "nlohmann/json.hpp"
#include <vector>
#include <optional>
#include <cstdint>
#include <memory>
#include <random>
#include <condition_variable>
#include <mutex>
#include <thread>

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
// comes from final_state; `game` is the game's GameInput.
GameRecord build_game_record(const std::vector<PlyRecord>& plies, const BoardState& final_state,
                             std::shared_ptr<const GameInput> game);

struct FinishedGame {
    GameRecord record;
    nlohmann::json trajectory;  // {"boardSeed": s, "plies": [...]} - see SelfPlayRunner's trajectory format
    BoardState final_state;
};

// Concurrent self-play games, one thread per game - KataGo's self-play structure. Each thread plays
// its own game at its own pace - opening moves sampled from the raw policy, then a cheap or full
// search per move (see SelfPlayParams) - with every search's network requests batched by one shared
// NNServer, and starts a new game as soon as one finishes. Every thread plays all its games on its
// own board: the game config's board program run with a seed no other thread has. The constructor
// builds every thread's board before starting any, and throws std::runtime_error, naming the seed,
// if one doesn't fit the input descriptor (compute_node_slots(), model/features.h). `cfg` must be a
// config new_state() accepts: a game thread has no way to report its error.
//
// play() runs every thread until `num_games` games have finished, then pauses each thread at its
// next ply boundary and returns every game finished by then (possibly more than `num_games`, since
// a thread's last ply can finish its game too). Unfinished games carry over to the next play() call,
// continuing with whatever evaluators that call passes. Network server thread i uses
// evaluators[i % evaluators.size()] (NNServer::set_evaluators()).
//
// Trajectory format, one object per game: {"boardSeed": s, "plies": [ply, ...]}, s the seed of the
// game's board (train.cpp adds "model": the
// iteration of the model that played it, -1 = uniform evaluator), where an opening ply is
// {"move": a} and a searched ply is {"move": a, "search": [num_rank_pairs*3 floats], "weight": w}
// plus "policy": [num_stones*N+1 floats] when w > 0 - everything replay_trajectory() needs to
// rebuild the game's GameRecord exactly.
class SelfPlayRunner {
public:
    SelfPlayRunner(const GameConfig& cfg, nlohmann::json input_descr,
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
        uint32_t board_seed = 0;        // the seed of state's board
        std::shared_ptr<const GameInput> game_input;  // built once, in new_slot()
    };

    const GameConfig& cfg_;
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

    void worker(uint64_t seed, uint32_t board_seed, std::shared_ptr<const BoardConfig> board);
    Slot new_slot(std::shared_ptr<const BoardConfig> board, uint32_t board_seed, std::mt19937_64& rng) const;
    void play_ply(Slot& slot, MCTS& mcts, std::mt19937_64& rng);
    FinishedGame finish(Slot& slot, std::mt19937_64& rng) const;
};

// Rebuilds a finished game's GameRecord from its trajectory (SelfPlayRunner's format) by replaying
// its moves through a fresh BoardState on `bc`, the board its "boardSeed" builds - used by --resume
// (to refill the training window) and --retrain.
GameRecord replay_trajectory(const nlohmann::json& trajectory, const GameConfig& cfg,
                             std::shared_ptr<const BoardConfig> bc, const nlohmann::json& descr,
                             const nlohmann::json* history_descr);
