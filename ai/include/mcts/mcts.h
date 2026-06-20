#pragma once
#include "game/board_state.h"
#include "model/gnn.h"
#include <vector>
#include <unordered_map>
#include <memory>
#include <optional>
#include <random>
#include <tuple>

struct MCTSTiming {
    double eval = 0.0;
    double select = 0.0;
    MCTSTiming& add(const MCTSTiming& o) { eval += o.eval; select += o.select; return *this; }
};

struct MCTSNode {
    BoardState state;
    std::vector<float> prior;        // (N+1,) from GNN policy
    std::vector<int>   visit_count;  // (N+1,)
    std::vector<float> total_value;  // (N+1,)
    std::unordered_map<int, std::unique_ptr<MCTSNode>> children;
    bool is_expanded = false;

    MCTSNode(BoardState s, std::vector<float> p);

    std::vector<float> q_values() const;
    std::vector<float> ucb_scores(float c_puct) const;
};

// AlphaZero-style MCTS.
//
// Value convention: each node stores Q-values from the perspective of the player
// whose turn it is at that node. Backup does not negate — instead it looks up the
// per-player reward map and reads the value for each node's own player. Terminal
// rewards come from compute_player_rewards(); GNN leaves contribute only the value
// for the leaf's current player (others get 0 for that simulation).
class MCTS {
public:
    MCTS(MessagePassingGNN model,
         float c_puct,           // exploration constant: higher = trust GNN prior more vs. Q values
         AdjNorms adj_norms,     // pre-computed multi-scale adjacency, device already set
         uint64_t seed = 42);

    std::pair<std::vector<std::pair<std::vector<float>, int>>, MCTSTiming> search_batch(
        std::vector<BoardState*> states,
        int num_simulations = 200,
        bool add_noise = false,
        float dirichlet_alpha = 0.3f,
        float noise_weight = 0.25f,
        std::vector<float> temperatures = {},
        std::optional<int> max_plies = std::nullopt);

private:
    MessagePassingGNN model_;
    float c_puct_;
    AdjNorms adj_norms_;
    std::mt19937 rng_;

    // Returns (path, leaf, per-player-rewards-or-nullopt)
    std::tuple<std::vector<std::pair<MCTSNode*, int>>,
               MCTSNode*,
               std::optional<std::unordered_map<int,float>>>
    select(MCTSNode* root, std::optional<int> max_plies);

    MCTSTiming simulate_batch(const std::vector<MCTSNode*>& roots,
                              std::optional<int> max_plies);
    void backup(const std::vector<std::pair<MCTSNode*, int>>& path,
                const std::unordered_map<int,float>& rewards);

    static std::pair<std::vector<float>, int>
    visit_counts_to_policy(const std::vector<int>& vc, float temperature,
                           std::mt19937& rng);

    static std::vector<bool> legal_mask(const BoardState& state);
    std::vector<float> dirichlet_sample(int n, float alpha);
};
