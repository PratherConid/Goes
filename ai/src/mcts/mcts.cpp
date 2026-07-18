#include "mcts/mcts.h"
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <omp.h>
#include "nlohmann/json.hpp"

// Debug toggle: when true, search_batch prints each root's MCTS subtree as JSON.
#define DBG_PRINT_TREE false

using json = nlohmann::json;

// Recursively serialise an MCTS node and its subtree to JSON.
//   "move"           action taken from the parent state (omitted for the root)
//   "visit_count"    N(parent, move): visits to the edge into this node (non-root)
//   "total_value"    W(parent, move): summed backup value of that edge (non-root)
//   "prior"          P(parent, move): model policy prior for that edge (non-root)
//   "board"          current board stones (root only)
//   "last_move"      description of the move that produced this state (root only)
//   "ply"            ply number of this node's state
//   "value_estimate" per-player reward estimate (empty if not yet evaluated)
//   "proven"         true if value_estimate is an exact terminal outcome
//                    rather than a GNN estimate (see MCTSNode::proven)
//   "subtree"        list of child nodes
static json mcts_node_to_json(const MCTSNode* node, std::optional<int> move,
                              int visit_count, float total_value, float prior) {
    json j;
    if (move.has_value()) {
        j["move"] = move.value();
        j["visit_count"] = visit_count;
        j["total_value"] = total_value;
        j["prior"] = prior;
    } else {
        // Only the root is serialised with no move from a parent; record the
        // board and the move that produced this state instead.
        j["board"] = node->state.board;
        j["last_move"] = move_to_string(node->state.last_move());
    }
    j["ply"] = node->state.ply_count();
    json ve = json::object();
    if (node->reward_estimate.has_value())
        for (const auto& [player, val] : *node->reward_estimate)
            ve[std::to_string(player)] = val;
    j["value_estimate"] = std::move(ve);
    j["proven"] = node->proven;
    json sub = json::array();
    for (const auto& [action, child] : node->children)
        sub.push_back(mcts_node_to_json(child.get(), action,
                          node->visit_count[action], node->total_value[action],
                          node->prior[action]));
    j["subtree"] = std::move(sub);
    return j;
}

// ── MCTSNode ──────────────────────────────────────────────────────────────────

MCTSNode::MCTSNode(BoardState s, std::vector<float> p)
    : state(std::move(s)),
      prior(std::move(p)),
      visit_count(state.N * state.num_stones + 1, 0),
      total_value(state.N * state.num_stones + 1, 0.0f)
{}

std::vector<float> MCTSNode::q_values() const {
    std::vector<float> q(visit_count.size());
    // Turn ownership (who is moving) is independent of stone_to_player_map
    // (which is scoring-only and can now map a stone to several players) -
    // next_turn.player is the sole source of truth, mirrors shared/boardState.ts.
    int mover = state.next_turn.player;
    for (size_t i = 0; i < q.size(); i++) {
        auto cit = children.find(static_cast<int>(i));
        if (cit != children.end() && cit->second->proven) {
            // Exact, ground-truth value - use it directly instead of the
            // (possibly still-averaging-over-few-samples) running average.
            auto rit = cit->second->reward_estimate->find(mover);
            q[i] = (rit != cit->second->reward_estimate->end()) ? rit->second : 0.0f;
        } else {
            q[i] = visit_count[i] > 0 ? total_value[i] / visit_count[i] : 0.0f;
        }
    }
    return q;
}

std::vector<float> MCTSNode::ucb_scores(float c_puct) const {
    int total = std::accumulate(visit_count.begin(), visit_count.end(), 0);
    float sqrt_total = std::sqrt(static_cast<float>(total + 1));
    auto q = q_values();
    std::vector<float> scores(q.size());
    for (size_t i = 0; i < scores.size(); i++)
        scores[i] = q[i] + c_puct * prior[i] * sqrt_total / (1.0f + visit_count[i]);
    return scores;
}

// ── MCTS ─────────────────────────────────────────────────────────────────────

MCTS::MCTS(Evaluator evaluator, float c_puct, uint64_t seed)
    : model_(std::move(evaluator)), c_puct_(c_puct),
      rng_(static_cast<unsigned>(seed))
{}

std::vector<float> MCTS::dirichlet_sample(int n, float alpha) {
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    std::vector<float> v(n);
    float sum = 0.0f;
    for (float& x : v) { x = gamma(rng_); sum += x; }
    for (float& x : v) x /= sum;
    return v;
}

std::vector<bool> MCTS::legal_mask(const BoardState& state) {
    int N = state.N, ns = state.num_stones;
    std::vector<bool> mask(ns * N + 1, false);
    for (int s = 1; s <= ns; s++)
        for (int i = 0; i < N; i++)
            if (state.legal_moves_data().captures[s][i].has_value()) mask[(s - 1) * N + i] = true;
    bool can_pass = (!state.forced_pass_only) || state.no_trad_legal();
    if (can_pass && !state.game_over()) mask[ns * N] = true;
    return mask;
}

// Walk tree via UCB to a leaf. Returns (path, leaf, per-player-rewards-or-nullopt).
//
// cached_rewards is non-nullopt only for terminal leaves (no GNN call needed):
// either game-over states or states that have reached their own max_plies.
// New non-terminal leaves have is_expanded=false; caller must evaluate them.
std::tuple<std::vector<std::pair<MCTSNode*, int>>,
           MCTSNode*,
           std::optional<std::unordered_map<int,float>>>
MCTS::select(MCTSNode* root) {
    std::vector<std::pair<MCTSNode*, int>> path;
    MCTSNode* node = root;
    std::optional<std::unordered_map<int,float>> cached_rewards;

    // game_over() checks max_plies live (against ply_count()), independently of
    // MoveInfo.all_passed (which only covers the consecutive-passes cause) -
    // mirroring shared/boardState.ts. Once true, make_move() itself will always
    // refuse further moves, so a max_plies-truncated leaf's score is exact/final
    // in the same sense as any other game-over leaf (see the "proven" comment
    // on simulate_batch()).
    auto is_terminal = [&](const MCTSNode* n) { return n->state.game_over(); };

    while (node->is_expanded && !is_terminal(node)) {
        auto scores = node->ucb_scores(c_puct_);
        auto mask = legal_mask(node->state);
        // Apply legal mask
        bool any_legal = false;
        for (size_t i = 0; i < scores.size(); i++) {
            if (!mask[i]) scores[i] = -std::numeric_limits<float>::infinity();
            else any_legal = true;
        }
        if (!any_legal) break;
        int action = static_cast<int>(
            std::max_element(scores.begin(), scores.end()) - scores.begin());
        path.push_back({node, action});

        if (!node->children.count(action)) {
            // Create new child
            auto child_state = node->state.copy();
            int N = node->state.N, ns = node->state.num_stones;
            std::optional<int> k, stone;
            if (action != ns * N) {
                stone = action / N + 1; // 1-indexed stone color
                k     = action % N;     // 0-indexed position
            }
            child_state.make_move(k, stone);
            auto new_child = std::make_unique<MCTSNode>(
                std::move(child_state),
                std::vector<float>(ns * N + 1, 0.0f));
            MCTSNode* child_ptr = new_child.get();
            if (is_terminal(child_ptr)) {
                child_ptr->is_expanded = true;
                cached_rewards = compute_player_rewards(
                    BoardState::compute_points(child_ptr->state.score_rule, child_ptr->state.score()),
                    child_ptr->state.stone_to_player_map, child_ptr->state.komi,
                    child_ptr->state.score_rule, child_ptr->state.capture_count());
            }
            node->children[action] = std::move(new_child);
            node = child_ptr;
            break;
        }
        node = node->children.at(action).get();
    }

    if (!cached_rewards.has_value() && is_terminal(node))
        cached_rewards = compute_player_rewards(
            BoardState::compute_points(node->state.score_rule, node->state.score()),
            node->state.stone_to_player_map, node->state.komi,
            node->state.score_rule, node->state.capture_count());

    return {std::move(path), node, std::move(cached_rewards)};
}

// Hybrid backup: plain running average for every edge (as before), plus a
// "proven" override. If an expanded child is proven (its reward_estimate is
// an exact terminal outcome, not a GNN estimate - see MCTSNode::proven) and
// its value is currently at least as good as every other action available at
// this node, this node adopts that child's exact reward vector as its own
// and is itself marked proven, since its own mover would simply always take
// that guaranteed-best option. This only ever propagates *exact* values
// upward - an unproven, few-visit GNN estimate never overrides anything, so
// this stays safe for large-branching-factor games: averaging alone still
// governs every uncertain estimate, exactly as in plain AlphaZero-style MCTS.
void MCTS::backup(const std::vector<std::pair<MCTSNode*, int>>& path,
                  const std::unordered_map<int,float>& rewards) {
    for (int i = static_cast<int>(path.size()) - 1; i >= 0; i--) {
        auto [node, action] = path[i];
        int mover = node->state.next_turn.player;
        float value = 0.0f;
        {
            auto rit = rewards.find(mover);
            if (rit != rewards.end()) value = rit->second;
        }
        node->visit_count[action] += 1;
        node->total_value[action] += value;

        float best = -std::numeric_limits<float>::infinity();
        bool best_proven = false;
        const std::unordered_map<int,float>* best_vec = nullptr;
        for (auto& [a, child] : node->children) {
            float v;
            if (child->proven) {
                auto rit = child->reward_estimate->find(mover);
                v = (rit != child->reward_estimate->end()) ? rit->second : 0.0f;
            } else if (node->visit_count[a] > 0) {
                v = node->total_value[a] / node->visit_count[a];
            } else {
                continue;
            }
            if (v > best) {
                best = v;
                best_proven = child->proven;
                best_vec = child->proven ? &child->reward_estimate.value() : nullptr;
            }
        }
        node->proven = best_proven;
        if (best_proven) node->reward_estimate = *best_vec;
    }
}

// Run one simulation step across all roots with a single batched GNN call.
//
// Each simulation selects a leaf per root: follows UCB scores (exploitation Q +
// exploration prior) down existing edges until reaching an unvisited child or a
// terminal node. For non-terminal leaves, estimate_player_rewards() derives a
// per-player reward from the model's ownership output, standing in for
// rolling the game out to the end. When a new child is created, the GNN is
// called once to get its prior and value; the child is immediately marked
// expanded so a second evaluation is not triggered on the next visit.
//
// All non-terminal leaves across roots are evaluated together in one batched GNN
// call. Backup looks up per-player rewards so each node's Q-value reflects its
// own player's outcome rather than relying on a zero-sum negation.
//
// Returns timing: seconds spent in MCTS::select (all roots) and model.evaluate_batch
//                 (eval=0.0 if all leaves were terminal and no model call was needed)
MCTSTiming MCTS::simulate_batch(const std::vector<MCTSNode*>& roots) {
    int n = static_cast<int>(roots.size());
    std::vector<std::vector<std::pair<MCTSNode*, int>>> paths(n);
    std::vector<MCTSNode*> nodes(n);
    std::vector<std::optional<std::unordered_map<int,float>>> leaf_values(n);
    std::vector<int> eval_indices;

    auto t_sel0 = std::chrono::high_resolution_clock::now();
    // Each root's tree is independent (separate HistoryManager), so select and
    // backup are safe to run in parallel across roots.
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        auto [path, node, cached] = select(roots[i]);
        paths[i] = std::move(path);
        nodes[i] = node;
        if (cached.has_value())
            leaf_values[i] = std::move(cached.value());
        else if (node->state.game_over())
            leaf_values[i] = compute_player_rewards(
                BoardState::compute_points(node->state.score_rule, node->state.score()),
                node->state.stone_to_player_map, node->state.komi,
                node->state.score_rule, node->state.capture_count());
        // leaf_values[i] stays nullopt when GNN evaluation is needed
    }
    auto t_sel1 = std::chrono::high_resolution_clock::now();
    double select_time = std::chrono::duration<double>(t_sel1 - t_sel0).count();

    // Collect eval indices serially after the parallel region
    for (int i = 0; i < n; i++) {
        if (!leaf_values[i].has_value())
            eval_indices.push_back(i);
    }

    double eval_time = 0.0;
    if (!eval_indices.empty()) {
        std::vector<const BoardState*> batch_states;
        batch_states.reserve(eval_indices.size());
        for (int i : eval_indices) batch_states.push_back(&nodes[i]->state);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto [policy_t, ownership_t] = model_.evaluate_batch(batch_states);
        auto t1 = std::chrono::high_resolution_clock::now();
        eval_time = std::chrono::duration<double>(t1 - t0).count();

        auto reward_t = estimate_player_rewards(ownership_t, batch_states[0]->score_rule,
                                                batch_states[0]->stone_to_player_map, batch_states[0]->num_players,
                                                batch_states[0]->komi, batch_states[0]->capture_count());
        // Both tensors arrive on the model's device (see evaluate_batch()'s comment) -
        // transfer only here, right where CPU-side accessor reads actually happen.
        auto policy_cpu = policy_t.cpu();
        auto reward_cpu = reward_t.cpu();
        auto pol_a = policy_cpu.accessor<float, 2>();
        auto reward_a = reward_cpu.accessor<float, 2>();
        for (int j = 0; j < (int)eval_indices.size(); j++) {
            int i = eval_indices[j];
            int Np1 = nodes[i]->state.N * nodes[i]->state.num_stones + 1;
            for (int k = 0; k < Np1; k++)
                nodes[i]->prior[k] = pol_a[j][k];
            nodes[i]->is_expanded = true;
            int P = nodes[i]->state.num_players;
            std::unordered_map<int,float> rewards;
            for (int p = 0; p < P; p++)
                rewards[p + 1] = reward_a[j][p];
            leaf_values[i] = std::move(rewards);
        }
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        if (leaf_values[i].has_value()) {
            nodes[i]->reward_estimate = leaf_values[i];
            // Only a genuine game-over state has an exact, ground-truth
            // reward; a max_plies-truncated leaf's "reward" is a heuristic
            // approximation like a GNN estimate, so it must not be treated
            // as proven either.
            nodes[i]->proven = nodes[i]->state.game_over();
            backup(paths[i], leaf_values[i].value());
        }
    }
    return {eval_time, select_time};
}

std::pair<std::vector<float>, int>
MCTS::visit_counts_to_policy(const std::vector<int>& vc, float temperature,
                              std::mt19937& rng) {
    std::vector<int> vc2 = vc;
    int total = std::accumulate(vc2.begin(), vc2.end(), 0);
    if (total == 0) std::fill(vc2.begin(), vc2.end(), 1);

    int n = static_cast<int>(vc2.size());
    std::vector<float> dist(n);
    int move;

    if (temperature == 0.0f) {
        move = static_cast<int>(
            std::max_element(vc2.begin(), vc2.end()) - vc2.begin());
        std::fill(dist.begin(), dist.end(), 0.0f);
        dist[move] = 1.0f;
    } else {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            dist[i] = std::pow(static_cast<float>(vc2[i]), 1.0f / temperature);
            sum += dist[i];
        }
        for (float& d : dist) d /= sum;
        // Sample from dist
        std::discrete_distribution<int> distr(dist.begin(), dist.end());
        move = distr(rng);
    }
    return {std::move(dist), move};
}

// Batched MCTS search: build a search tree per state and return a move for each.
//
// Runs `num_simulations` rounds of simulation. In each round one simulation step
// is taken across all trees simultaneously, with all GNN evaluations batched into
// a single forward pass. For a single state pass a one-element vector.
//
// Each simulation traverses a tree to a leaf via UCB scores, evaluates non-terminal
// leaves with the GNN, and backs the value up to the root. The root's visit-count
// distribution is the policy target (stronger than the raw GNN prior because it
// reflects the result of lookahead). The move is sampled from that distribution
// raised to the power 1/temperature; temperature=0 picks the most-visited action.
//
// temperatures: per-state; empty defaults to 1.0 for all.
// All states must share the same adjacency matrix.
// noise_cfg: Dirichlet root-noise settings (added to root priors for self-play exploration).
// Move-count truncation is read from each state's own BoardState::max_plies
// (a state at or beyond its own bound is treated as terminal - value from
// stone counts - rather than evaluated by the GNN); it propagates to every
// node of that state's search tree via copy()/copy_with_hm().
//
// Each root gets its own HistoryManager so the per-root search trees are fully
// independent. This is required for parallel select: HistoryManager::store_board/lookup
// are not thread-safe.
//
// Returns (per-state (visit-count dist, move index (stone-major flat index,
//          num_stones*N=pass)), timing including initial batched root evaluation)
std::pair<std::vector<std::pair<std::vector<float>, int>>, MCTSTiming>
MCTS::search_batch(
    std::vector<BoardState*> states,
    int num_simulations,
    NoiseConfig noise_cfg,
    std::vector<float> temperatures)
{
    int n = static_cast<int>(states.size());
    if (temperatures.empty()) temperatures.assign(n, 1.0f);

    // Initial root-prior batch evaluation
    std::vector<const BoardState*> cstates(states.begin(), states.end());
    auto t0 = std::chrono::high_resolution_clock::now();
    auto [policy_t, ownership_t] = model_.evaluate_batch(cstates);
    auto t1 = std::chrono::high_resolution_clock::now();
    MCTSTiming total;
    total.eval = std::chrono::duration<double>(t1 - t0).count();

    auto reward_t = estimate_player_rewards(ownership_t, states[0]->score_rule,
                                            states[0]->stone_to_player_map, states[0]->num_players,
                                            states[0]->komi, states[0]->capture_count());
    // Both tensors arrive on the model's device (see evaluate_batch()'s comment) -
    // transfer only here, right where CPU-side accessor reads actually happen.
    auto policy_cpu = policy_t.cpu();
    auto reward_cpu = reward_t.cpu();
    auto pol_a = policy_cpu.accessor<float, 2>();
    auto reward_a = reward_cpu.accessor<float, 2>();

    // hms must be declared before roots so it outlives the MCTSNodes (LIFO destruction).
    // One HistoryManager per root so each search tree is fully independent.
    std::vector<HistoryManager> hms(n);
    std::vector<std::unique_ptr<MCTSNode>> roots;
    roots.reserve(n);
    for (int i = 0; i < n; i++) {
        int Np1 = states[i]->N * states[i]->num_stones + 1;
        std::vector<float> prior(Np1);
        for (int k = 0; k < Np1; k++) prior[k] = pol_a[i][k];

        if (noise_cfg.add_noise) {
            auto noise = dirichlet_sample(Np1, noise_cfg.dirichlet_alpha);
            for (int k = 0; k < Np1; k++)
                prior[k] = (1.0f - noise_cfg.noise_weight) * prior[k] + noise_cfg.noise_weight * noise[k];
        }
        auto root = std::make_unique<MCTSNode>(states[i]->copy_with_hm(&hms[i]), std::move(prior));
        root->is_expanded = true;
        // Debug: store the root's per-player reward estimate from the initial eval.
        std::unordered_map<int,float> root_rewards;
        for (int p = 0; p < states[i]->num_players; p++)
            root_rewards[p + 1] = reward_a[i][p];
        root->reward_estimate = std::move(root_rewards);
        roots.push_back(std::move(root));
    }

    std::vector<MCTSNode*> root_ptrs;
    root_ptrs.reserve(n);
    for (auto& r : roots) root_ptrs.push_back(r.get());

    auto t_sim0 = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < num_simulations; s++)
        total.add(simulate_batch(root_ptrs));

    std::vector<std::pair<std::vector<float>, int>> results;
    results.reserve(n);
    for (int i = 0; i < n; i++) {
        auto [dist, move] = visit_counts_to_policy(
            roots[i]->visit_count, temperatures[i], rng_);
        results.push_back({std::move(dist), move});
    }
    total.simulate = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_sim0).count();

    if (DBG_PRINT_TREE) {
        json roots_json = json::array();
        for (int i = 0; i < n; i++)
            roots_json.push_back(mcts_node_to_json(root_ptrs[i], std::nullopt, 0, 0.0f, 0.0f));
        std::filesystem::create_directories(".logs");
        std::ofstream(".logs/tree.json") << roots_json.dump(2) << std::endl;
    }

    // Explicitly tear down the search trees here so the recursive MCTSNode
    // destruction is timed rather than hiding in the function's return unwind.
    // Each root tree is independent (its own HistoryManager hms[i]); the only
    // shared object is the read-only adj shared_ptr, whose refcount is atomic.
    // So destroying disjoint trees in parallel is race-free.
    root_ptrs.clear();
    auto t_tear0 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++)
        roots[i].reset();  // recursively destroy tree i
    roots.clear();         // now just drops null unique_ptrs (cheap)
    total.teardown = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_tear0).count();
    hms.clear();

    return {std::move(results), total};
}
