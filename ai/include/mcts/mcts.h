#pragma once
#include "game/board_state.h"
#include "model/evaluator.h"
#include "mcts/utility.h"
#include <memory>
#include <random>
#include <vector>

struct MCTSTiming {
    double eval = 0.0;      // network evaluation (roots and leaves)
    double select = 0.0;    // tree descent
    double simulate = 0.0;  // the whole simulation loop
    double teardown = 0.0;  // destroying the search trees
    double search = 0.0;    // the whole search() call
};

// Settings shared by every root of a search - the subset of KataGo's SearchParams that Goes ports
// (see ai/Readme.md's "Search"). self_play() holds KataGo's self-play values
// (cpp/configs/training/selfplay1.cfg); server() holds the values KataGo plays with outside
// self-play (cpp/program/setup.cpp's defaults) - only the parameter values, nothing of its protocol.
struct SearchParams {
    float cpuct = 1.1f;              // cpuctExploration
    float cpuct_log = 0.0f;          // cpuctExplorationLog - cpuct grows as log((N + base) / base)
    float cpuct_base = 500.0f;       // cpuctExplorationBase
    float fpu_reduction_max = 0.2f;  // non-root first-play urgency reduction
    float fpu_parent_weight_pow = 2.0f;  // fpuParentWeightByVisitedPolicyPow
    UtilityParams utility;
    float noise_total_concentration = 10.83f;  // rootDirichletNoiseTotalConcentration
    float noise_weight = 0.25f;                // rootDirichletNoiseWeight
    float lcb_stdevs = 5.0f;
    float min_visit_prop_for_lcb = 0.15f;
    bool lcb_for_move = false;  // false: LCB shapes only the policy target, as KataGo's self-play does
    float chosen_move_prune = 1.0f;
    // Visits searched before the dynamic score centre is re-read from the root's own search
    // average (KataGo's short pre-search after clearing the tree).
    int center_presearch_visits = 10;

    static SearchParams self_play();
    static SearchParams server();
};

// Per-root settings, so one batched search can mix KataGo's full and cheap searches.
struct RootSettings {
    int visits = 600;                     // root visits to reach, including the root's own evaluation
    bool noise = false;                   // Dirichlet root noise
    float policy_temperature = 1.0f;      // root policy raised to 1/T before noise
    float fpu_reduction_max = 0.0f;       // rootFpuReductionMax
    float forced_playouts_coeff = 0.0f;   // rootDesiredPerChildVisitsCoeff
    float move_temperature = 0.0f;        // temperature on the play-selection values; ~0 = argmax
};

// One root's search outcome. Action vectors are dense over num_stones*N+1 actions (stone-major,
// num_stones*N = pass - see ai/Readme.md's "Policy Head and Action Space").
struct SearchResult {
    std::vector<float> policy_target;  // normalized play-selection values (pruned, LCB-adjusted)
    int move = 0;
    std::vector<float> root_prior;     // policy the root was searched with (tempered, noised)
    std::vector<float> utility;        // per player, search average
    std::vector<float> win;            // per player, search average of the win term
    std::vector<float> pair_probs;     // num_rank_pairs*3, search average
    std::vector<float> nn_pair_probs;  // num_rank_pairs*3, the root's own network evaluation
    int visits = 0;
};

// KataGo's interpolateEarly(): `late` + (`early` - `late`) * 0.5^(ply/halflife * 19/sqrt(N)) - a
// schedule that decays with the move number, faster on larger boards.
float interpolate_early(float halflife, float early, float late, int ply, int N);

struct MCTSNode;

// Batched MCTS over independent roots. Every simulation step descends one path per unfinished root,
// evaluates all new leaves in one network call, and backs each leaf's value up its own path.
class MCTS {
public:
    MCTS(Evaluator evaluator, SearchParams params, uint64_t seed);

    // All states must share one game config and board, and none may be game-over.
    std::pair<std::vector<SearchResult>, MCTSTiming> search(
        const std::vector<BoardState*>& states, const std::vector<RootSettings>& settings);

private:
    Evaluator evaluator_;
    SearchParams params_;
    std::mt19937 rng_;

    double explore_scaling(double total_child_visits) const;
    int choose_child(const MCTSNode& node, bool is_root, const RootSettings& root_settings) const;
    std::vector<double> play_selection_values(const MCTSNode& root, bool use_lcb) const;
    void add_root_noise(std::vector<float>& prior);
    int choose_with_temperature(const std::vector<double>& values, float temperature);
};
