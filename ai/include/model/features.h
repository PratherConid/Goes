#pragma once
#include <torch/torch.h>
#include "game/board_state.h"
#include "nlohmann/json.hpp"
#include <vector>

struct FeatureTensors {
    torch::Tensor features;    // (N, F) float32
    torch::Tensor legal_mask;  // (num_stones*N + 1,) bool
};

// Multi-scale adjacency matrices, computed once per BoardConfig.
// Layers cycle through these as: adj, adj2, adj, adj4, adj, adj2, adj, adj4, ...
struct AdjNorms {
    torch::Tensor adj;   // row-normalised 1-hop + self
    torch::Tensor adj2;  // row-normalised exactly-2-hop neighbors
    torch::Tensor adj4;  // row-normalised exactly-4-hop neighbors
    torch::Tensor deg1;  // (N,1) float - neighbor count per node in adj's mask (1-hop+self)
    torch::Tensor deg2;  // (N,1) float - neighbor count per node in adj2's mask
    torch::Tensor deg4;  // (N,1) float - neighbor count per node in adj4's mask
    int max_degree;      // max over deg1, deg2, deg4 (all nodes, all three tiers)
};

// Compute per-node feature matrix and legal mask for one state, as CPU tensors. descr is the
// self-describing feature-block descriptor built by
// compute_input_descr() (training/self_play.h) - see
// that function's doc comment for the JSON shape and this function's own doc
// comment (features.cpp) for the recognized block names/args. The feature
// width F is read directly from descr["totalDims"], not recomputed here.
// server.cpp never calls compute_input_descr: train.cpp persists the
// descriptor into the checkpoint's config JSON and server.cpp reads it back.
FeatureTensors board_to_features(const BoardState& state, const nlohmann::json& descr);

// The current position's legal actions in the flat stone-major action encoding - placement of
// stone s (1-based) at node i is (s-1)*N + i, pass is num_stones*N: every legal placement in that
// order, then pass unless forced_pass_only forbids it. A resigned player to move may only pass;
// a finished game has none.
std::vector<int> legal_actions(const BoardState& state);

// Plays a flat action (see legal_actions()) on `state`. Throws std::runtime_error if
// make_move() rejects it.
void apply_action(BoardState& state, int action);

// legal_actions() as a (num_stones*N + 1,) bool CPU tensor - board_to_features()'s legal_mask.
torch::Tensor legal_action_mask(const BoardState& state);

// Computes per-node features for one historical ply of `state`, under whatever descr is passed -
// ply may be anything in [0, state.ply_count()], including the current ply. Fully descriptor-
// agnostic - a thin forward to features.cpp's board_to_features_at_ply(), the same per-ply fill
// board_to_features() runs at ply_count(). Used exclusively by the Transformer architecture
// (transformer.cpp), for every PAST ply (0..ply_count()-1); the current ply's features and legal
// mask come from board_to_features(). The Transformer passes TransformerConfig::history_descr
// here - a small 2-block (plyMod, stoneOccupancy) descriptor built directly in train.cpp,
// independent of compute_input_descr() (training/self_play.h), not board_to_features()'s full
// descr. Returns (N, F) float32 on CPU (caller stacks across plies/states and moves to device
// once). No legal mask - only the current ply's legality matters for the policy output, so
// callers that need a mask use board_to_features() for that ply instead.
torch::Tensor history_features_at_ply(const BoardState& state, int ply, const nlohmann::json& descr);

// Compute the multi-scale adjacency set. Call once per BoardConfig.
AdjNorms compute_adj_norms(const BoardConfig& bc, torch::Device device);
