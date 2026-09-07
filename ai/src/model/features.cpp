#include "model/features.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdexcept>

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

    // Neighbor count per node in each hop-tier's mask, before row-normalisation.
    auto deg1 = f1.sum(1, /*keepdim=*/true); // (N,1)
    auto deg2 = f2.sum(1, /*keepdim=*/true);
    auto deg4 = f4.sum(1, /*keepdim=*/true);
    int max_degree = static_cast<int>(std::max({deg1.max().item<float>(),
                                                  deg2.max().item<float>(),
                                                  deg4.max().item<float>()}));

    return {normalise(f1).to(device), normalise(f2).to(device), normalise(f4).to(device),
            deg1.to(device), deg2.to(device), deg4.to(device), max_degree};
}

// Soft, monotonic stand-in for "bit i of value": 1 at value=0, ramping down
// linearly to 0 at value=2^i and clamped at 0 beyond - i.e. max(0, 1 -
// value/2^i). Used (in place of raw binary bit-extraction) for every
// channel of the "liberty"/"playerStoneBudget"/"globalStoneBudget" blocks
// below, one channel per bit_index in [0, bits), so a small change in value
// always moves every channel by a small, continuous amount rather than
// flipping a bit discontinuously.
static float clamp_scale(int value, int bit_index) {
    return std::max(0.0f, 1.0f - static_cast<float>(value) / static_cast<float>(1 << bit_index));
}

// Computes the (N,F) feature tensor for `state` as of a given ply - shared by board_to_features()
// (ply = state.ply_count(), i.e. "now") and history_features_at_ply() (any past ply). Every block
// reads ply-indexed accessors (board_at(), turn_list[ply % len], legal_moves_data_at(),
// player_stone_place_cnt_at(), consecutive_passes_at()) rather than state's own "live" fields
// (state.board, state.next_turn, state.legal_moves_data(), ...) - those live fields are themselves
// exactly the ply_count()-indexed case of each accessor (e.g. next_turn ==
// turn_list[ply_count() % len], confirmed via make_move()'s own use of that formula), so this is
// behaviorally identical to the old board_to_features() when ply == state.ply_count(). All of
// these accessors are O(1), backed by HistoryManager's per-ply interning - no BoardState mutation
// (withdraw_move()) is ever needed to reconstruct a past ply's features.
static torch::Tensor board_to_features_at_ply(const BoardState& state, int ply, const nlohmann::json& descr) {
    if (ply < 0 || ply > state.ply_count()) {
        std::cerr << "board_to_features_at_ply: ply " << ply << " out of bounds [0, "
                  << state.ply_count() << "]\n";
        throw std::out_of_range("board_to_features_at_ply: ply out of bounds");
    }

    int N = state.N;
    int ns = state.num_stones;
    int F = descr.at("totalDims").get<int>();
    int tl_len = (int)state.turn_list.size();

    auto features = torch::zeros({N, F}, torch::kFloat32);
    auto feat_a = features.accessor<float, 2>();

    const std::vector<int>& board = state.board_at(ply);
    const TurnInfo& turn_at_ply = state.turn_list[ply % tl_len];

    // group_liberty() is only needed to fill node_liberty/node_group_size, which only the
    // "liberty"/"groupSize" blocks read - skip the traversal entirely when descr requests neither
    // (e.g. the Transformer's minimal history descriptor never does), rather than always paying
    // for it and discarding the result.
    bool needs_liberty = false;
    for (auto& entry : descr.at("blocks")) {
        std::string name = entry.at(0).get<std::string>();
        if (name == "liberty" || name == "groupSize") { needs_liberty = true; break; }
    }

    std::vector<int> node_liberty(N, 0), node_group_size(N, 0);
    if (needs_liberty) {
        auto gdict = group_liberty(board, *state.adj, N, turn_at_ply.friendly);
        for (auto& [color, entries] : gdict) {
            for (auto& e : entries) {
                int lib_count  = static_cast<int>(e.liberties.size());
                int group_size = static_cast<int>(e.group.size());
                for (int node : e.group) {
                    node_liberty[node]    = lib_count;
                    node_group_size[node] = group_size;
                }
            }
        }
    }

    auto& legal     = state.legal_moves_data_at(ply);
    auto& place_cnt = state.player_stone_place_cnt_at(ply);

    int off = 0;
    for (auto& entry : descr.at("blocks")) {
        std::string name = entry.at(0).get<std::string>();

        if (name == "stoneOccupancy") {
            for (int i = 0; i < N; i++) {
                int stone = board[i];
                if (stone > 0) feat_a[i][off + stone - 1] = 1.0f;
                else            feat_a[i][off + ns] = 1.0f;
            }
            off += ns + 1;
        } else if (name == "legalPlace") {
            for (int s = 1; s <= ns; s++)
                for (int i = 0; i < N; i++)
                    if (legal.captures[s][i].has_value())
                        feat_a[i][off + s - 1] = 1.0f;
            off += ns;
        } else if (name == "liberty") {
            int bits = entry.at(1).get<int>();
            for (int i = 0; i < N; i++) {
                int lib = node_liberty[i];
                for (int b = 0; b < bits; b++)
                    feat_a[i][off + b] = clamp_scale(lib, b);
            }
            off += bits;
        } else if (name == "groupSize") {
            float inv_N = 1.0f / std::max(N, 1);
            for (int i = 0; i < N; i++)
                feat_a[i][off] = node_group_size[i] * inv_N;
            off += 1;
        } else if (name == "plyMod") {
            int block_tl_len = entry.at(1).get<int>();
            if (block_tl_len > 0) {
                int ply_mod = ply % block_tl_len;
                for (int i = 0; i < N; i++)
                    feat_a[i][off + ply_mod] = 1.0f;
            }
            off += block_tl_len;
        } else if (name == "consectivePassOneHot") {
            int bits = entry.at(1).get<int>();
            int cp = state.consecutive_passes_at(ply);
            for (int i = 0; i < N; i++)
                feat_a[i][off + cp] = 1.0f;
            off += bits;
        } else if (name == "playerStoneBudget") {
            auto& rows = entry.at(1);
            for (size_t s = 0; s < rows.size(); s++) {
                auto& row = rows[s];
                for (size_t p = 0; p < row.size(); p++) {
                    int bits = row[p].get<int>();
                    if (bits == 0) continue;
                    int lim = *state.player_stone_place_limit[s][p];
                    int remaining = lim - place_cnt[s][p];
                    for (int b = 0; b < bits; b++) {
                        float v = clamp_scale(remaining, b);
                        for (int i = 0; i < N; i++) feat_a[i][off + b] = v;
                    }
                    off += bits;
                }
            }
        } else if (name == "globalStoneBudget") {
            auto& row = entry.at(1);
            for (size_t s = 0; s < row.size(); s++) {
                int bits = row[s].get<int>();
                if (bits == 0) continue;
                int lim = *state.global_stone_place_limit[s];
                int placed = 0;
                for (int p = 0; p < state.num_players; p++) placed += place_cnt[s][p];
                int remaining = lim - placed;
                for (int b = 0; b < bits; b++) {
                    float v = clamp_scale(remaining, b);
                    for (int i = 0; i < N; i++) feat_a[i][off + b] = v;
                }
                off += bits;
            }
        } else {
            throw std::runtime_error("board_to_features_at_ply: unknown feature block: " + name);
        }
    }
    assert(off == F && "board_to_features_at_ply offset accounting must match descr's totalDims");

    return features;
}

// Convert a BoardState to tensors for the GNN.
//
// Returns
//   features  : float32 (N, F)               per-node feature matrix
//   legal_mask: bool    (num_stones*N + 1,)   True = legal action (stone-major
//                                             flattened: index (stone-1)*N+pos;
//                                             last entry = pass) - see the
//                                             matching policy-head layout in
//                                             cnn.cpp/gnn.cpp/unet.cpp.
//
// descr is the self-describing feature-block descriptor built once by
// compute_input_descr() (training/self_play.h) at the start of a fresh
// training run and persisted verbatim thereafter (ModelConfig::input_descr,
// model_config.h) - shape {"blocks": [[name, ...args], ...], "totalDims": F}.
// See board_to_features_at_ply() above for the recognized block names/args
// and per-block fill logic (shared with this function).
//
// Features come from board_to_features_at_ply() at the current ply; the legal mask stays here
// rather than moving into that shared function, since it depends on game_over()/no_trad_legal(),
// concepts that are only meaningful for the current position, not an arbitrary past ply.
FeatureTensors board_to_features(const BoardState& state, torch::Device device, const nlohmann::json& descr) {
    int N = state.N;
    int ns = state.num_stones;
    auto features = board_to_features_at_ply(state, state.ply_count(), descr);

    // Legal mask: num_stones*N place actions (stone-major) + 1 pass -
    // independent of descr.
    auto legal_mask = torch::zeros({ns * N + 1}, torch::kBool);
    auto mask_a = legal_mask.accessor<bool, 1>();
    for (int s = 1; s <= ns; s++)
        for (int i = 0; i < N; i++)
            if (state.legal_moves_data().captures[s][i].has_value())
                mask_a[(s - 1) * N + i] = true;
    if (!state.game_over()) {
        bool can_pass = (!state.forced_pass_only) || state.no_trad_legal();
        if (can_pass) mask_a[ns * N] = true;
    }

    return {features.to(device), legal_mask.to(device)};
}

torch::Tensor history_features_at_ply(const BoardState& state, int ply, const nlohmann::json& descr) {
    return board_to_features_at_ply(state, ply, descr);
}
