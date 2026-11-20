#include "mcts/mcts.h"
#include "model/features.h"
#include "model/rank_head.h"
#include <algorithm>
#include <cmath>
#include <limits>

SearchParams SearchParams::self_play() { return SearchParams{}; }

SearchParams SearchParams::server() {
    SearchParams p;
    p.cpuct = 1.0f;
    p.cpuct_log = 0.45f;
    p.utility.static_score_factor = 0.1f;
    p.utility.dynamic_score_factor = 0.3f;
    p.utility.dynamic_center_zero_weight = 0.2f;
    p.utility.dynamic_center_scale = 0.75f;
    p.lcb_for_move = true;
    return p;
}

float interpolate_early(float halflife, float early, float late, int ply, int N) {
    double halflives = static_cast<double>(ply) / halflife * 19.0 / std::sqrt(static_cast<double>(N));
    return static_cast<float>(late + (early - late) * std::pow(0.5, halflives));
}

// ── Nodes ─────────────────────────────────────────────────────────────────────

// A position's own evaluation: the network's (or, for a terminal position, the exact) pair
// evaluation and the per-player utility/win term it implies under its root's score centres.
struct LeafEval {
    PairEval pair;
    std::vector<float> utility, win;
};

struct MCTSNode {
    BoardState state;
    bool evaluated = false;
    bool terminal = false;
    std::vector<int> actions;                         // legal actions
    std::vector<float> prior;                         // aligned with actions
    std::vector<std::unique_ptr<MCTSNode>> children;  // aligned with actions; null = never visited
    LeafEval own;
    // Sums over every visit of this node's subtree, its own evaluation counted as the first.
    int visits = 0;
    std::vector<double> utility_sum, utility_sq_sum, win_sum, pair_sum, diff_sum;

    explicit MCTSNode(BoardState s) : state(std::move(s)) {}

    void add(const LeafEval& v) {
        if (utility_sum.empty()) {
            utility_sum.assign(v.utility.size(), 0.0);
            utility_sq_sum.assign(v.utility.size(), 0.0);
            win_sum.assign(v.win.size(), 0.0);
            pair_sum.assign(v.pair.probs.size(), 0.0);
            diff_sum.assign(v.pair.diff.size(), 0.0);
        }
        visits++;
        for (size_t p = 0; p < v.utility.size(); p++) {
            utility_sum[p] += v.utility[p];
            utility_sq_sum[p] += static_cast<double>(v.utility[p]) * v.utility[p];
            win_sum[p] += v.win[p];
        }
        for (size_t k = 0; k < v.pair.probs.size(); k++) pair_sum[k] += v.pair.probs[k];
        for (size_t k = 0; k < v.pair.diff.size(); k++) diff_sum[k] += v.pair.diff[k];
    }

    double mean_utility(int player_idx) const { return utility_sum[player_idx] / visits; }
};

// ── MCTS ─────────────────────────────────────────────────────────────────────

MCTS::MCTS(SearchParams params, uint64_t seed)
    : params_(params), rng_(static_cast<std::mt19937::result_type>(seed)) {}

double MCTS::explore_scaling(double total_child_visits) const {
    double cpuct = params_.cpuct
        + params_.cpuct_log * std::log((total_child_visits + params_.cpuct_base) / params_.cpuct_base);
    return cpuct * std::sqrt(total_child_visits + 0.01);
}

// PUCT over a node's legal actions, from the mover's perspective. An unvisited child is valued at
// the first-play urgency: the parent's value - its search average blended toward its own
// evaluation the less of its prior mass has been visited - minus fpu_reduction_max *
// sqrt(visited prior mass). At the root, a child below its forced-playout floor
// sqrt(coeff * prior * total visits) is always taken first.
int MCTS::choose_child(const MCTSNode& node, bool is_root, const RootSettings& root_settings) const {
    int m = node.state.next_turn.player - 1;
    int L = static_cast<int>(node.actions.size());
    double total = 0.0, visited_mass = 0.0;
    for (int k = 0; k < L; k++)
        if (node.children[k] && node.children[k]->visits > 0) {
            total += node.children[k]->visits;
            visited_mass += node.prior[k];
        }
    double avg_weight = std::min(1.0, std::pow(visited_mass, static_cast<double>(params_.fpu_parent_weight_pow)));
    double parent_for_fpu = avg_weight * node.mean_utility(m) + (1.0 - avg_weight) * node.own.utility[m];
    double reduction_max = is_root ? root_settings.fpu_reduction_max : params_.fpu_reduction_max;
    double fpu = parent_for_fpu - reduction_max * std::sqrt(visited_mass);
    double scale = explore_scaling(total);
    double forced_coeff = is_root ? root_settings.forced_playouts_coeff : 0.0f;

    int best = 0;
    double best_value = -std::numeric_limits<double>::infinity();
    for (int k = 0; k < L; k++) {
        const MCTSNode* child = node.children[k].get();
        double n = child ? child->visits : 0.0;
        double value;
        if (forced_coeff > 0.0 && node.prior[k] > 0.0f && n < std::sqrt(node.prior[k] * total * forced_coeff))
            value = 1e20;
        else
            value = (n > 0 ? child->mean_utility(m) : fpu) + scale * node.prior[k] / (1.0 + n);
        if (value > best_value) { best_value = value; best = k; }
    }
    return best;
}

// Root play-selection values, aligned with root.actions: each visited child's visit count, then
// (1) every child but the most stably explored one cut back to the visits PUCT would have given it
// had it known the final values (KataGo's policy target pruning - this removes forced playouts),
// (2) with use_lcb, the child with the best lower confidence bound raised enough to outrank every
// child it beats on LCB, and (3) values below chosen_move_prune (capped at max/64) zeroed.
std::vector<double> MCTS::play_selection_values(const MCTSNode& r, bool use_lcb) const {
    int L = static_cast<int>(r.actions.size());
    int m = r.state.next_turn.player - 1;
    std::vector<double> v(L, 0.0);
    double total = 0.0, best_goodness = -1e30;
    int best = -1;
    for (int k = 0; k < L; k++) {
        const MCTSNode* c = r.children[k].get();
        if (!c || c->visits <= 0) continue;
        double n = c->visits;
        v[k] = n;
        total += n;
        double g = n * std::max(0.0, n - 1.0) / std::max(1.0, n) + 2.0 * r.prior[k];
        if (g > best_goodness) { best_goodness = g; best = k; }
    }

    if (best >= 0) {
        double scale = explore_scaling(total);
        double best_weight = v[best];
        double best_value = r.children[best]->mean_utility(m) + scale * r.prior[best] / (1.0 + best_weight);
        for (int k = 0; k < L; k++) {
            if (k == best || v[k] <= 0) continue;
            double explore = best_value - r.children[k]->mean_utility(m);
            double wanted = explore <= 0 ? 1e100 : std::max(0.0, scale * r.prior[k] / explore - 1.0);
            v[k] = std::ceil(std::min(v[k], wanted));
        }

        if (use_lcb) {
            double radius_range = params_.utility.radius();
            std::vector<double> lcb(L, -1e10), radius(L, 0.0);
            double best_lcb = -1e10;
            int best_lcb_idx = -1;
            for (int k = 0; k < L; k++) {
                const MCTSNode* c = r.children[k].get();
                if (!c || c->visits <= 0) continue;
                // Utility mean/variance with KataGo's small max-variance prior, so few-visit
                // children get wide bounds.
                double n = c->visits;
                double avg = c->mean_utility(m);
                double sq = std::max(c->utility_sq_sum[m] / n, avg * avg + 1e-8);
                double prior_w = n / (n * n * n);
                sq = (sq * n + (sq + radius_range * radius_range) * prior_w) / (n + prior_w);
                double w = n + prior_w, w_sq = n + prior_w * prior_w;
                double ess = w * w / w_sq;
                radius[k] = std::sqrt(std::max(0.0, sq - avg * avg) / ess) * params_.lcb_stdevs;
                lcb[k] = avg - radius[k];
                if (v[k] > 0 && v[k] >= params_.min_visit_prop_for_lcb * best_weight && lcb[k] > best_lcb) {
                    best_lcb = lcb[k];
                    best_lcb_idx = k;
                }
            }
            if (best_lcb_idx >= 0) {
                double adjusted = v[best_lcb_idx];
                for (int k = 0; k < L; k++) {
                    if (k == best_lcb_idx || v[k] <= 0) continue;
                    double excess = best_lcb - lcb[k];
                    if (excess < 0) continue;
                    double factor = (radius[k] + excess) / (radius[k] + 0.2 * excess);
                    adjusted = std::max(adjusted, factor * factor * v[k]);
                }
                v[best_lcb_idx] = adjusted;
            }
        }
    }

    double max_v = *std::max_element(v.begin(), v.end());
    if (max_v <= 1e-50) {
        for (int k = 0; k < L; k++) v[k] = r.prior[k];
        max_v = *std::max_element(v.begin(), v.end());
        if (max_v <= 1e-50) { std::fill(v.begin(), v.end(), 1.0); max_v = 1.0; }
    }
    double prune = std::min(static_cast<double>(params_.chosen_move_prune), max_v / 64.0);
    for (double& x : v) if (x < prune) x = 0.0;
    return v;
}

// KataGo's root noise: a Dirichlet draw over legal moves whose concentrations sum to
// noise_total_concentration - half spread uniformly, half by how far each move's log-policy sits
// above the mean - mixed into the prior with weight noise_weight.
void MCTS::add_root_noise(std::vector<float>& prior) {
    int L = static_cast<int>(prior.size());
    if (L == 0) return;
    std::vector<double> alpha(L);
    double log_sum = 0.0;
    for (int k = 0; k < L; k++) {
        alpha[k] = std::log(std::min(0.01, static_cast<double>(prior[k])) + 1e-20);
        log_sum += alpha[k];
    }
    double log_mean = log_sum / L, prop_sum = 0.0;
    for (int k = 0; k < L; k++) {
        alpha[k] = std::max(0.0, alpha[k] - log_mean);
        prop_sum += alpha[k];
    }
    double uniform = 1.0 / L;
    for (int k = 0; k < L; k++)
        alpha[k] = prop_sum <= 0.0 ? uniform : 0.5 * (alpha[k] / prop_sum + uniform);

    std::vector<double> draw(L);
    double draw_sum = 0.0;
    for (int k = 0; k < L; k++) {
        std::gamma_distribution<double> gamma(alpha[k] * params_.noise_total_concentration, 1.0);
        draw[k] = gamma(rng_);
        draw_sum += draw[k];
    }
    if (draw_sum <= 0.0) return;
    for (int k = 0; k < L; k++)
        prior[k] = static_cast<float>(draw[k] / draw_sum * params_.noise_weight + prior[k] * (1.0 - params_.noise_weight));
}

// KataGo's chooseIndexWithTemperature(): argmax at temperature ~0, otherwise sample in proportion
// to (value / max)^(1/temperature).
int MCTS::choose_with_temperature(const std::vector<double>& values, float temperature) {
    int best = static_cast<int>(std::max_element(values.begin(), values.end()) - values.begin());
    if (temperature <= 1e-4f) return best;
    double log_max = std::log(values[best]);
    std::vector<double> w(values.size(), 0.0);
    for (size_t k = 0; k < values.size(); k++)
        if (values[k] > 0) w[k] = std::exp((std::log(values[k]) - log_max) / temperature);
    std::discrete_distribution<int> dist(w.begin(), w.end());
    return dist(rng_);
}

std::vector<NNRow> output_rows(const ModelOutput& out, const std::vector<const ModelInput*>& inputs,
                               const std::vector<const BoardState*>& states) {
    std::vector<std::vector<int>> captures;
    captures.reserve(states.size());
    for (const BoardState* s : states) captures.push_back(s->capture_count());
    const BoardState& s0 = *states[0];
    int P = s0.num_players;
    int64_t B = static_cast<int64_t>(states.size());

    // Slots no node occupies own nothing.
    auto ownership = out.ownership * stack_valid(inputs).to(out.ownership.device()).view({B, 1, -1, 1});
    auto totals = estimate_player_totals(ownership, captures, s0.score_rule, s0.stone_to_player_map, P, s0.komi)
                      .to(torch::kCPU).contiguous();
    auto rank = out.rank.to(torch::kCPU).contiguous().view({B, -1});
    auto policy = out.policy.to(torch::kCPU).contiguous();
    auto to_vec = [](const torch::Tensor& t) {
        const float* p = t.data_ptr<float>();
        return std::vector<float>(p, p + t.numel());
    };
    std::vector<NNRow> rows(states.size());
    for (int64_t j = 0; j < B; j++) {
        auto row_policy = policy[j].index_select(0, inputs[j]->game->action_index);
        rows[j] = NNRow{to_vec(row_policy.contiguous()), to_vec(totals[j]), to_vec(rank[j])};
    }
    return rows;
}

std::vector<NNRow> evaluate_rows(const Evaluator& evaluator, const std::vector<const BoardState*>& states,
                                 const std::shared_ptr<const GameInput>& game) {
    std::vector<ModelInput> inputs;
    inputs.reserve(states.size());
    for (const BoardState* s : states) inputs.push_back(evaluator.prepare(*s, game));
    std::vector<const ModelInput*> ptrs;
    ptrs.reserve(inputs.size());
    for (const ModelInput& in : inputs) ptrs.push_back(&in);
    return output_rows(evaluator.run(ptrs), ptrs, states);
}

// Fills a non-terminal node's legal actions, prior and own evaluation from its network row. With
// set_centers, `centers` (the root's dynamic score centres) are first set from the node's own
// expected differences.
static void evaluate_node(MCTSNode& n, const NNRow& row, std::vector<float>& centers, bool set_centers,
                          const UtilityParams& utility_params) {
    int N = n.state.N, P = n.state.num_players, num_pairs = num_rank_pairs(P);
    n.actions = legal_actions(n.state);
    n.prior.resize(n.actions.size());
    n.children.resize(n.actions.size());
    double sum = 0.0;
    for (size_t k = 0; k < n.actions.size(); k++) { n.prior[k] = row.policy[n.actions[k]]; sum += n.prior[k]; }
    for (float& p : n.prior) p = sum > 0 ? static_cast<float>(p / sum) : 1.0f / n.prior.size();

    auto pairs = rank_pairs(P);
    n.own.pair.probs = row.rank;
    n.own.pair.diff.resize(num_pairs);
    for (int k = 0; k < num_pairs; k++)
        n.own.pair.diff[k] = row.totals[pairs[k].first - 1] - row.totals[pairs[k].second - 1];
    if (set_centers) centers = dynamic_score_centers(n.own.pair.diff, N, utility_params);
    pair_eval_to_utility(n.own.pair, centers, P, N, utility_params, n.own.utility, n.own.win);
    n.evaluated = true;
}

SearchResult MCTS::search(const BoardState& state, const RootSettings& settings, const NNQuery& nn) {
    int N = state.N, A = state.num_stones * N + 1;
    // The tree's own HistoryManager, declared before the root so it outlives every node.
    HistoryManager hm;
    auto root = std::make_unique<MCTSNode>(state.copy_with_hm(&hm));
    std::vector<float> centers;
    evaluate_node(*root, nn(root->state), centers, /*set_centers=*/true, params_.utility);

    if (settings.policy_temperature != 1.0f && !root->prior.empty()) {
        float max_p = *std::max_element(root->prior.begin(), root->prior.end());
        double sum = 0.0;
        for (float& p : root->prior) {
            if (p > 0) p = static_cast<float>(std::exp((std::log(p) - std::log(max_p)) / settings.policy_temperature));
            sum += p;
        }
        for (float& p : root->prior) p = static_cast<float>(p / sum);
    }
    if (settings.noise) add_root_noise(root->prior);
    root->add(root->own);

    bool recentered = false;
    std::vector<MCTSNode*> path;
    while (root->visits < settings.visits && !root->actions.empty()) {
        // Descend to a leaf: an unvisited child (evaluated exactly if terminal, else by the
        // network) or a terminal node already in the tree.
        path.clear();
        MCTSNode* node = root.get();
        path.push_back(node);
        while (node->evaluated && !node->terminal && !node->actions.empty()) {
            int k = choose_child(*node, node == root.get(), settings);
            std::unique_ptr<MCTSNode>& child = node->children[k];
            if (!child) {
                BoardState s = node->state.copy();
                apply_action(s, node->actions[k]);
                // Resigned players' turns are passed automatically, as the TS game flow does after
                // every move (BoardState.advanceResigned()).
                s.advance_resigned();
                child = std::make_unique<MCTSNode>(std::move(s));
                if (child->state.game_over()) {
                    child->terminal = true;
                    child->evaluated = true;
                    child->own.pair = exact_pair_eval(child->state);
                    pair_eval_to_utility(child->own.pair, centers, child->state.num_players, N, params_.utility,
                                         child->own.utility, child->own.win);
                } else {
                    evaluate_node(*child, nn(child->state), centers, /*set_centers=*/false, params_.utility);
                }
                node = child.get();
                path.push_back(node);
                break;
            }
            node = child.get();
            path.push_back(node);
        }
        for (MCTSNode* n : path) n->add(node->own);

        if (!recentered && root->visits >= std::min(params_.center_presearch_visits, settings.visits)) {
            std::vector<float> avg_diff(root->diff_sum.size());
            for (size_t k = 0; k < avg_diff.size(); k++)
                avg_diff[k] = static_cast<float>(root->diff_sum[k] / root->visits);
            centers = dynamic_score_centers(avg_diff, N, params_.utility);
            recentered = true;
        }
    }

    SearchResult res;
    res.visits = root->visits;
    res.policy_target.assign(A, 0.0f);
    res.root_prior.assign(A, 0.0f);
    for (size_t k = 0; k < root->actions.size(); k++) res.root_prior[root->actions[k]] = root->prior[k];
    if (!root->actions.empty()) {
        auto target_values = play_selection_values(*root, /*use_lcb=*/true);
        auto move_values = params_.lcb_for_move ? target_values : play_selection_values(*root, false);
        double sum = 0.0;
        for (double x : target_values) sum += x;
        for (size_t k = 0; k < root->actions.size(); k++)
            res.policy_target[root->actions[k]] = static_cast<float>(target_values[k] / sum);
        res.move = root->actions[choose_with_temperature(move_values, settings.move_temperature)];
    } else {
        res.move = A - 1;
    }
    auto mean = [&](const std::vector<double>& s) {
        std::vector<float> m(s.size());
        for (size_t k = 0; k < s.size(); k++) m[k] = static_cast<float>(s[k] / root->visits);
        return m;
    };
    res.utility = mean(root->utility_sum);
    res.win = mean(root->win_sum);
    res.pair_probs = mean(root->pair_sum);
    res.nn_pair_probs = root->own.pair.probs;
    return res;
}
