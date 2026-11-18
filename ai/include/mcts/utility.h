#pragma once
#include "game/board_state.h"
#include <vector>

// The quantity MCTS maximizes for each player - KataGo's search utility, generalized from two
// players to any number by summing over the rank_pairs() pairs (model/rank_head.h) a player is part
// of. For player p with opponents q (P players):
//
//   win_p    = 1/(P-1) * sum_q [P(p ahead of q) - P(q ahead of p)]                     in [-1, 1]
//   static_p = 1/(P-1) * sum_q (2/pi) atan(d_pq / (2 sqrt(N)))
//   dyn_p    = 1/(P-1) * sum_q (2/pi) atan((d_pq - c_pq) / (dynamic_center_scale sqrt(N)))
//   utility_p = win_loss_factor * win_p + static_score_factor * static_p + dynamic_score_factor * dyn_p
//
// d_pq is the (expected) final total of p minus that of q, N is the board's node count, and c_pq
// is the search's dynamic score centre for that pair (see dynamic_score_centers()). For P = 2 this
// is exactly KataGo's (W - L) + static + dynamic score utility, from each player's own perspective;
// every term is antisymmetric in (p, q), so utilities always sum to 0 across players. KataGo takes
// the atan's expectation under the score head's predicted spread; Goes has no score-spread head,
// so a non-terminal position uses its expected difference alone.
struct UtilityParams {
    float win_loss_factor = 1.0f;
    float static_score_factor = 0.0f;
    float dynamic_score_factor = 0.4f;
    float dynamic_center_zero_weight = 0.25f;
    float dynamic_center_scale = 0.5f;
    static constexpr float kStaticScoreScale = 2.0f;

    // Largest possible |utility| - KataGo's utilityRangeRadius, used by LCB.
    float radius() const { return win_loss_factor + static_score_factor + dynamic_score_factor; }
};

// One position's evaluation per rank_pairs() pair: the outcome distribution (FIRST_AHEAD, TIED,
// SECOND_AHEAD - model/rank_head.h's RankOutcome order) and the expected final total difference
// (first player's minus second's).
struct PairEval {
    std::vector<float> probs;  // num_rank_pairs * 3
    std::vector<float> diff;   // num_rank_pairs
};

// A finished game's exact PairEval: one-hot outcomes and exact differences of
// compute_player_totals() (a player no stone credits is ranked on komi alone).
PairEval exact_pair_eval(const BoardState& final_state);

// KataGo's recentScoreCenter, per pair: the expected difference pulled toward 0 by
// dynamic_center_zero_weight, but never more than dynamic_center_scale * sqrt(N) away from it.
std::vector<float> dynamic_score_centers(const std::vector<float>& expected_diff, int N, const UtilityParams& params);

// Per-player utility and win term (both length num_players) of `eval` under `centers`.
void pair_eval_to_utility(const PairEval& eval, const std::vector<float>& centers, int num_players, int N,
                          const UtilityParams& params, std::vector<float>& utility, std::vector<float>& win);
