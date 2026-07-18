#pragma once
#include "game/board_state.h"
#include "model/evaluator.h"
#include <vector>
#include <unordered_map>
#include <memory>
#include <optional>
#include <random>
#include <tuple>

struct MCTSTiming {
    double eval = 0.0;
    double select = 0.0;
    double simulate = 0.0;  // wall time for the simulation loop
    double teardown = 0.0;  // wall time to destroy the search trees + history managers
    double search = 0.0;    // wall time for the full search_batch call
    MCTSTiming& add(const MCTSTiming& o) { eval += o.eval; select += o.select; return *this; }
};

struct MCTSNode {
    BoardState state;
    std::vector<float> prior;        // (num_stones*N+1,) from model policy, stone-major + pass
    std::vector<int>   visit_count;  // (num_stones*N+1,)
    std::vector<float> total_value;  // (num_stones*N+1,)
    std::unordered_map<int, std::unique_ptr<MCTSNode>> children;
    bool is_expanded = false;
    // True only for genuine game-over terminal states (not max_plies
    // truncation): reward_estimate is then an exact, ground-truth value
    // rather than a GNN estimate, and backup() lets it override averaging -
    // see the comment on MCTS::backup().
    bool proven = false;
    // Debug: this node's per-player reward estimate when first evaluated
    // (terminal value via compute_player_rewards, or derived from the model's
    // ownership output via estimate_player_rewards()), or - once proven -
    // the exact terminal value adopted from its best proven child. nullopt
    // until the node is evaluated as a leaf / root.
    std::optional<std::unordered_map<int,float>> reward_estimate;

    MCTSNode(BoardState s, std::vector<float> p);

    std::vector<float> q_values() const;
    std::vector<float> ucb_scores(float c_puct) const;
};

// Dirichlet root-noise settings for search_batch (self-play exploration).
struct NoiseConfig {
    bool  add_noise       = false;  // add Dirichlet noise to root priors
    float dirichlet_alpha = 0.3f;   // Dirichlet concentration
    float noise_weight    = 0.25f;  // mixing weight of noise into the prior
};

// AlphaZero-style MCTS.
//
// Value convention: each node stores Q-values from the perspective of the player
// whose turn it is at that node. Backup does not negate - instead it looks up the
// per-player reward map and reads the value for each node's own player. Terminal
// rewards come from compute_player_rewards(); model leaves contribute only the value
// for the leaf's current player (others get 0 for that simulation).
class MCTS {
public:
    MCTS(Evaluator evaluator, float c_puct, uint64_t seed = 42);

    // Move-count truncation is read per-state from BoardState::max_plies
    // (propagated to every node of a game's own search tree via copy() /
    // copy_with_hm()), not passed in here.
    std::pair<std::vector<std::pair<std::vector<float>, int>>, MCTSTiming> search_batch(
        std::vector<BoardState*> states,
        int num_simulations = 200,
        NoiseConfig noise_cfg = {},
        std::vector<float> temperatures = {});

private:
    Evaluator model_;
    float c_puct_;
    std::mt19937 rng_;

    // Returns (path, leaf, per-player-rewards-or-nullopt)
    std::tuple<std::vector<std::pair<MCTSNode*, int>>,
               MCTSNode*,
               std::optional<std::unordered_map<int,float>>>
    select(MCTSNode* root);

    MCTSTiming simulate_batch(const std::vector<MCTSNode*>& roots);
    void backup(const std::vector<std::pair<MCTSNode*, int>>& path,
                const std::unordered_map<int,float>& rewards);

    static std::pair<std::vector<float>, int>
    visit_counts_to_policy(const std::vector<int>& vc, float temperature,
                           std::mt19937& rng);

    static std::vector<bool> legal_mask(const BoardState& state);
    std::vector<float> dirichlet_sample(int n, float alpha);
};
