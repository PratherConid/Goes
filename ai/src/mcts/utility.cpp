#include "mcts/utility.h"
#include "model/rank_head.h"
#include <algorithm>
#include <cmath>

static constexpr float kTwoOverPi = 0.63661977236758134f;

PairEval exact_pair_eval(const BoardState& s) {
    auto totals = compute_player_totals(BoardState::compute_points(s.score_rule, s.score()),
                                        s.stone_to_player_map, s.komi, s.score_rule, s.capture_count());
    auto total_of = [&](int p) {
        auto it = totals.find(p);
        if (it != totals.end()) return it->second;
        return p <= (int)s.komi.size() ? s.komi[p - 1] : 0.0f;
    };
    auto pairs = rank_pairs(s.num_players);
    PairEval e{std::vector<float>(pairs.size() * 3, 0.0f), std::vector<float>(pairs.size(), 0.0f)};
    for (size_t k = 0; k < pairs.size(); k++) {
        float d = total_of(pairs[k].first) - total_of(pairs[k].second);
        e.diff[k] = d;
        e.probs[k * 3 + (d > 0 ? FIRST_AHEAD : d < 0 ? SECOND_AHEAD : TIED)] = 1.0f;
    }
    return e;
}

std::vector<float> dynamic_score_centers(const std::vector<float>& expected_diff, int N, const UtilityParams& params) {
    float cap = std::sqrt(static_cast<float>(N)) * params.dynamic_center_scale;
    std::vector<float> c(expected_diff.size());
    for (size_t k = 0; k < c.size(); k++) {
        float d = expected_diff[k];
        c[k] = std::clamp(d * (1.0f - params.dynamic_center_zero_weight), d - cap, d + cap);
    }
    return c;
}

void pair_eval_to_utility(const PairEval& eval, const std::vector<float>& centers, int num_players, int N,
                          const UtilityParams& params, std::vector<float>& utility, std::vector<float>& win) {
    utility.assign(num_players, 0.0f);
    win.assign(num_players, 0.0f);
    if (num_players <= 1) return;
    float sqrt_n = std::sqrt(static_cast<float>(N));
    float static_scale = UtilityParams::kStaticScoreScale * sqrt_n;
    float dynamic_scale = params.dynamic_center_scale * sqrt_n;
    auto pairs = rank_pairs(num_players);
    for (size_t k = 0; k < pairs.size(); k++) {
        float margin = eval.probs[k * 3 + FIRST_AHEAD] - eval.probs[k * 3 + SECOND_AHEAD];
        float score = params.static_score_factor * kTwoOverPi * std::atan(eval.diff[k] / static_scale)
                    + params.dynamic_score_factor * kTwoOverPi * std::atan((eval.diff[k] - centers[k]) / dynamic_scale);
        float u = params.win_loss_factor * margin + score;
        int p = pairs[k].first - 1, q = pairs[k].second - 1;
        win[p] += margin;  win[q] -= margin;
        utility[p] += u;   utility[q] -= u;
    }
    float inv = 1.0f / static_cast<float>(num_players - 1);
    for (int p = 0; p < num_players; p++) { utility[p] *= inv; win[p] *= inv; }
}
