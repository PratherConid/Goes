#include "mcts/mcts.h"
#include "model/rank_head.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <omp.h>

using Clock = std::chrono::high_resolution_clock;
static double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

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

static std::vector<int> legal_actions(const BoardState& s) {
    int N = s.N, ns = s.num_stones;
    std::vector<int> a;
    for (int st = 1; st <= ns; st++)
        for (int i = 0; i < N; i++)
            if (s.legal_moves_data().captures[st][i].has_value()) a.push_back((st - 1) * N + i);
    if (!s.game_over() && (!s.forced_pass_only || s.no_trad_legal())) a.push_back(ns * N);
    return a;
}

static void apply_action(BoardState& s, int action) {
    int N = s.N, ns = s.num_stones;
    if (action == ns * N) s.make_move(std::nullopt, std::nullopt);
    else s.make_move(action % N, action / N + 1);
}

struct RootCtx {
    MCTSNode* root = nullptr;
    RootSettings settings;
    std::vector<float> centers;  // dynamic score centre per pair
    bool recentered = false;
};

// ── MCTS ─────────────────────────────────────────────────────────────────────

MCTS::MCTS(Evaluator evaluator, SearchParams params, uint64_t seed)
    : evaluator_(std::move(evaluator)), params_(params), rng_(static_cast<std::mt19937::result_type>(seed)) {}

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

// Evaluates `nodes` (all non-terminal, sharing one game config) in one network call, filling each
// one's actions/prior and own evaluation. With set_centers, each node is its root and its root's
// dynamic score centres are first set from its own expected differences.
static void evaluate_nodes(const Evaluator& evaluator, const UtilityParams& utility_params,
                           const std::vector<MCTSNode*>& nodes, const std::vector<std::vector<float>*>& centers,
                           bool set_centers) {
    std::vector<const BoardState*> states;
    std::vector<std::vector<int>> captures;
    states.reserve(nodes.size());
    for (MCTSNode* n : nodes) { states.push_back(&n->state); captures.push_back(n->state.capture_count()); }
    const BoardState& s0 = *states[0];
    int N = s0.N, P = s0.num_players, num_pairs = num_rank_pairs(P);

    ModelOutput out = evaluator.evaluate_batch(states);
    auto totals = estimate_player_totals(out.ownership, captures, s0.score_rule, s0.stone_to_player_map, P, s0.komi)
                      .to(torch::kCPU).contiguous();
    auto rank = out.rank.to(torch::kCPU).contiguous();
    auto policy = out.policy.to(torch::kCPU).contiguous();
    auto totals_a = totals.accessor<float, 2>();
    auto rank_a = rank.accessor<float, 3>();
    auto policy_a = policy.accessor<float, 2>();
    auto pairs = rank_pairs(P);

    for (size_t j = 0; j < nodes.size(); j++) {
        MCTSNode& n = *nodes[j];
        n.actions = legal_actions(n.state);
        n.prior.resize(n.actions.size());
        n.children.resize(n.actions.size());
        double sum = 0.0;
        for (size_t k = 0; k < n.actions.size(); k++) { n.prior[k] = policy_a[j][n.actions[k]]; sum += n.prior[k]; }
        for (float& p : n.prior) p = sum > 0 ? static_cast<float>(p / sum) : 1.0f / n.prior.size();

        n.own.pair.probs.resize(num_pairs * 3);
        n.own.pair.diff.resize(num_pairs);
        for (int k = 0; k < num_pairs; k++) {
            for (int c = 0; c < 3; c++) n.own.pair.probs[k * 3 + c] = rank_a[j][k][c];
            n.own.pair.diff[k] = totals_a[j][pairs[k].first - 1] - totals_a[j][pairs[k].second - 1];
        }
        if (set_centers) *centers[j] = dynamic_score_centers(n.own.pair.diff, N, utility_params);
        pair_eval_to_utility(n.own.pair, *centers[j], P, N, utility_params, n.own.utility, n.own.win);
        n.evaluated = true;
    }
}

// Descends from ctx.root to a leaf, recording the path (root first, leaf last). A newly created
// terminal child is given its exact evaluation here; any other new child is returned unevaluated
// for the caller's batched network call.
static MCTSNode* descend(RootCtx& ctx, std::vector<MCTSNode*>& path, const UtilityParams& utility_params,
                         const std::function<int(const MCTSNode&, bool)>& choose) {
    path.clear();
    MCTSNode* node = ctx.root;
    path.push_back(node);
    while (node->evaluated && !node->terminal && !node->actions.empty()) {
        int k = choose(*node, node == ctx.root);
        std::unique_ptr<MCTSNode>& child = node->children[k];
        if (!child) {
            BoardState s = node->state.copy();
            apply_action(s, node->actions[k]);
            child = std::make_unique<MCTSNode>(std::move(s));
            if (child->state.game_over()) {
                child->terminal = true;
                child->evaluated = true;
                child->own.pair = exact_pair_eval(child->state);
                pair_eval_to_utility(child->own.pair, ctx.centers, child->state.num_players, child->state.N,
                                     utility_params, child->own.utility, child->own.win);
            }
            path.push_back(child.get());
            return child.get();
        }
        node = child.get();
        path.push_back(node);
    }
    return node;
}

std::pair<std::vector<SearchResult>, MCTSTiming> MCTS::search(
    const std::vector<BoardState*>& states, const std::vector<RootSettings>& settings)
{
    auto t_start = Clock::now();
    MCTSTiming timing;
    int n = static_cast<int>(states.size());
    const BoardState& s0 = *states[0];
    int N = s0.N, A = s0.num_stones * N + 1;

    // One HistoryManager per root keeps every tree independent, so descents and backups can run in
    // parallel across roots. Declared before the roots so it outlives their nodes.
    std::vector<HistoryManager> hms(n);
    std::vector<std::unique_ptr<MCTSNode>> roots(n);
    std::vector<RootCtx> ctx(n);
    for (int i = 0; i < n; i++) {
        roots[i] = std::make_unique<MCTSNode>(states[i]->copy_with_hm(&hms[i]));
        ctx[i].root = roots[i].get();
        ctx[i].settings = settings[i];
    }

    {
        std::vector<MCTSNode*> nodes;
        std::vector<std::vector<float>*> centers;
        for (int i = 0; i < n; i++) { nodes.push_back(roots[i].get()); centers.push_back(&ctx[i].centers); }
        auto t0 = Clock::now();
        evaluate_nodes(evaluator_, params_.utility, nodes, centers, /*set_centers=*/true);
        timing.eval += seconds_since(t0);
    }
    for (int i = 0; i < n; i++) {
        MCTSNode& r = *roots[i];
        float T = settings[i].policy_temperature;
        if (T != 1.0f && !r.prior.empty()) {
            float max_p = *std::max_element(r.prior.begin(), r.prior.end());
            double sum = 0.0;
            for (float& p : r.prior) {
                if (p > 0) p = static_cast<float>(std::exp((std::log(p) - std::log(max_p)) / T));
                sum += p;
            }
            for (float& p : r.prior) p = static_cast<float>(p / sum);
        }
        if (settings[i].noise) add_root_noise(r.prior);
        r.add(r.own);
    }

    std::vector<std::function<int(const MCTSNode&, bool)>> choosers;
    for (int i = 0; i < n; i++)
        choosers.push_back([this, &settings, i](const MCTSNode& node, bool is_root) {
            return choose_child(node, is_root, settings[i]);
        });

    auto t_sim0 = Clock::now();
    std::vector<std::vector<MCTSNode*>> paths(n);
    std::vector<MCTSNode*> leaves(n, nullptr);
    while (true) {
        std::vector<int> active;
        for (int i = 0; i < n; i++)
            if (roots[i]->visits < settings[i].visits && !roots[i]->actions.empty()) active.push_back(i);
        if (active.empty()) break;

        auto t_sel0 = Clock::now();
        #pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < (int)active.size(); j++) {
            int i = active[j];
            leaves[i] = descend(ctx[i], paths[i], params_.utility, choosers[i]);
        }
        timing.select += seconds_since(t_sel0);

        std::vector<MCTSNode*> to_eval;
        std::vector<std::vector<float>*> to_eval_centers;
        for (int i : active)
            if (!leaves[i]->evaluated) { to_eval.push_back(leaves[i]); to_eval_centers.push_back(&ctx[i].centers); }
        if (!to_eval.empty()) {
            auto t0 = Clock::now();
            evaluate_nodes(evaluator_, params_.utility, to_eval, to_eval_centers, /*set_centers=*/false);
            timing.eval += seconds_since(t0);
        }

        #pragma omp parallel for schedule(static)
        for (int j = 0; j < (int)active.size(); j++) {
            int i = active[j];
            for (MCTSNode* node : paths[i]) node->add(leaves[i]->own);
        }

        for (int i : active) {
            if (ctx[i].recentered || roots[i]->visits < std::min(params_.center_presearch_visits, settings[i].visits))
                continue;
            std::vector<float> avg_diff(roots[i]->diff_sum.size());
            for (size_t k = 0; k < avg_diff.size(); k++)
                avg_diff[k] = static_cast<float>(roots[i]->diff_sum[k] / roots[i]->visits);
            ctx[i].centers = dynamic_score_centers(avg_diff, N, params_.utility);
            ctx[i].recentered = true;
        }
    }
    timing.simulate = seconds_since(t_sim0);

    std::vector<SearchResult> results(n);
    for (int i = 0; i < n; i++) {
        const MCTSNode& r = *roots[i];
        SearchResult& res = results[i];
        res.visits = r.visits;
        res.policy_target.assign(A, 0.0f);
        res.root_prior.assign(A, 0.0f);
        for (size_t k = 0; k < r.actions.size(); k++) res.root_prior[r.actions[k]] = r.prior[k];
        if (!r.actions.empty()) {
            auto target_values = play_selection_values(r, /*use_lcb=*/true);
            auto move_values = params_.lcb_for_move ? target_values : play_selection_values(r, false);
            double sum = 0.0;
            for (double x : target_values) sum += x;
            for (size_t k = 0; k < r.actions.size(); k++)
                res.policy_target[r.actions[k]] = static_cast<float>(target_values[k] / sum);
            res.move = r.actions[choose_with_temperature(move_values, settings[i].move_temperature)];
        } else {
            res.move = A - 1;
        }
        auto mean = [&](const std::vector<double>& s) {
            std::vector<float> m(s.size());
            for (size_t k = 0; k < s.size(); k++) m[k] = static_cast<float>(s[k] / r.visits);
            return m;
        };
        res.utility = mean(r.utility_sum);
        res.win = mean(r.win_sum);
        res.pair_probs = mean(r.pair_sum);
        res.nn_pair_probs = r.own.pair.probs;
    }

    // Destroy the trees explicitly so their (recursive) teardown is timed. Trees are disjoint and
    // only share the read-only adj/adlist shared_ptrs, whose refcounts are atomic.
    auto t_tear0 = Clock::now();
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) roots[i].reset();
    roots.clear();
    hms.clear();
    timing.teardown = seconds_since(t_tear0);
    timing.search = seconds_since(t_start);
    return {std::move(results), timing};
}
