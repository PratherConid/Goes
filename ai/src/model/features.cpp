#include "model/features.h"
#include <algorithm>

AdjNorms compute_adj_norms(const BoardConfig& bc, torch::Device device) {
    int N = bc.N;

    // Binary adjacency without self-loops
    auto A_raw = torch::zeros({N, N}, torch::kFloat32);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (bc.adj[i][j]) A_raw[i][j] = 1.0f;

    // A_self = A + I; (A_self)^k entry > 0 iff node reachable in ≤k hops
    auto A_self = A_raw + torch::eye(N, torch::kFloat32);
    auto A2 = torch::mm(A_self, A_self);
    auto A3 = torch::mm(A2,    A_self);
    auto A4 = torch::mm(A3,    A_self);

    auto normalise = [](torch::Tensor mask_float) {
        return mask_float / mask_float.sum(1, true).clamp_min(1.0f);
    };

    // adj:  1-hop + self
    // adj2: reachable in ≤2 hops but not ≤1  (exclusive 2-hop ring)
    // adj4: reachable in ≤4 hops but not ≤3  (exclusive 4-hop ring)
    auto f1 = A_self.gt(0).to(torch::kFloat32);
    auto f2 = (A2.gt(0).to(torch::kFloat32) - f1).clamp_min(0.0f);
    auto f4 = (A4.gt(0).to(torch::kFloat32) - A3.gt(0).to(torch::kFloat32)).clamp_min(0.0f);

    return {normalise(f1).to(device), normalise(f2).to(device), normalise(f4).to(device)};
}

// Convert a BoardState to tensors for the GNN.
//
// Returns
//   features  : float32 (N, F)   per-node feature matrix
//   legal_mask: bool    (N+1,)   True = legal action (last entry = pass)
//
// Feature layout (F = num_stones + 4 + turn_stone_list.size()):
//   [0 .. num_stones-1]                      is_stone[k]  (one-hot stone occupancy)
//   [num_stones]                              is_empty
//   [num_stones+1]                            is_legal (legal PLACE for current player)
//   [num_stones+2]                            liberty_count / N
//   [num_stones+3]                            group_size / N
//   [num_stones+4 .. num_stones+3+tsl.size()] ply-mod one-hot: one-hot at index
//                                             (ply_count % turn_stone_list.size()),
//                                             broadcast to all nodes. Encodes position
//                                             in the turn cycle rather than stone type,
//                                             so players with multiple consecutive turns
//                                             (e.g. turn_stone_list=[1,1,2,2]) can be
//                                             distinguished.
FeatureTensors board_to_features(const BoardState& state, torch::Device device) {
    int N = state.N;
    int ns = state.num_stones;
    // F = num_stones + 4 + turn_stone_list.size()
    int F = ns + 4 + static_cast<int>(state.turn_stone_list.size());

    auto features = torch::zeros({N, F}, torch::kFloat32);
    auto feat_a = features.accessor<float, 2>();

    // Stone occupancy / empty flag
    for (int i = 0; i < N; i++) {
        int stone = state.board[i];
        if (stone > 0)
            feat_a[i][stone - 1] = 1.0f;
        else
            feat_a[i][ns] = 1.0f;
    }

    // Legal PLACE moves
    for (int i = 0; i < N; i++)
        if (state.legal_moves_with_take[i].has_value())
            feat_a[i][ns + 1] = 1.0f;

    // Liberty count and group size (skip if game already over)
    if (!state.game_over()) {
        auto gdict = group_liberty(state.board, *state.adj, N);
        std::vector<int> node_liberty(N, 0), node_group_size(N, 0);
        for (auto& [color, entries] : gdict) {
            for (auto& [group, libs] : entries) {
                int lib_count  = static_cast<int>(libs.size());
                int group_size = static_cast<int>(group.size());
                for (int node : group) {
                    node_liberty[node]    = lib_count;
                    node_group_size[node] = group_size;
                }
            }
        }
        float inv_N = 1.0f / std::max(N, 1);
        for (int i = 0; i < N; i++) {
            feat_a[i][ns + 2] = node_liberty[i]    * inv_N;
            feat_a[i][ns + 3] = node_group_size[i] * inv_N;
        }
    }

    // Ply-mod one-hot (broadcast to all nodes)
    int tsl_len = static_cast<int>(state.turn_stone_list.size());
    if (tsl_len > 0) {
        int ply_mod = state.ply_count() % tsl_len;
        for (int i = 0; i < N; i++)
            feat_a[i][ns + 4 + ply_mod] = 1.0f;
    }

    // Legal mask: N place actions + 1 pass
    auto legal_mask = torch::zeros({N + 1}, torch::kBool);
    auto mask_a = legal_mask.accessor<bool, 1>();
    for (int i = 0; i < N; i++)
        if (state.legal_moves_with_take[i].has_value())
            mask_a[i] = true;
    if (!state.game_over()) {
        bool can_pass = (!state.forced_pass_only) || state.no_trad_legal();
        if (can_pass) mask_a[N] = true;
    }

    return {features.to(device), legal_mask.to(device)};
}
