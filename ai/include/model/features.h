#pragma once
#include <torch/torch.h>
#include "game/board_state.h"
#include "nlohmann/json.hpp"

struct FeatureTensors {
    torch::Tensor features;    // (N, F) float32
    torch::Tensor legal_mask;  // (num_stones*N + 1,) bool
};

// Multi-scale adjacency matrices, computed once per BoardConfig.
// Layers cycle through these as: adj, adj2, adj, adj4, adj, adj2, adj, adj4, ...
struct AdjNorms {
    torch::Tensor adj;   // row-normalised 1-hop + self  (original adj_norm)
    torch::Tensor adj2;  // row-normalised exactly-2-hop neighbors
    torch::Tensor adj4;  // row-normalised exactly-4-hop neighbors
    torch::Tensor deg1;  // (N,1) float - neighbor count per node in adj's mask (1-hop+self)
    torch::Tensor deg2;  // (N,1) float - neighbor count per node in adj2's mask
    torch::Tensor deg4;  // (N,1) float - neighbor count per node in adj4's mask
    int max_degree;      // max over deg1, deg2, deg4 (all nodes, all three tiers)
};

// Compute per-node feature matrix and legal mask for one state. descr is the
// self-describing feature-block descriptor built by
// compute_input_descr(const GameConfig&, int, const std::string&) (training/self_play.h) - see
// that function's doc comment for the JSON shape and this function's own doc
// comment (features.cpp) for the recognized block names/args. The feature
// width F is read directly from descr["totalDims"], not recomputed here.
// server.cpp never calls compute_input_descr: it persists a checkpoint's
// descriptor into the config JSON once (at train time) and reads it back
// directly rather than recomputing it.
FeatureTensors board_to_features(const BoardState& state, torch::Device device, const nlohmann::json& descr);

// Computes per-node features for one historical ply of `state`, under whatever descr is passed -
// ply may be anything in [0, state.ply_count()], including the current ply. Fully descriptor-
// agnostic (just a thin forward to board_to_features_at_ply(), below); used exclusively by the
// Transformer architecture (transformer.cpp), which applies this identically to every PAST ply
// (0..ply_count()-1) via BoardState::board_at()/consecutive_passes_at() rather than the live
// state.board/state.next_turn/state.last_move() board_to_features() reads - the current ply's own
// features (and its legal mask) still come from the unmodified board_to_features() above, since
// that already works unchanged for any descr, current-ply-only case. The Transformer passes
// TransformerConfig::history_descr here - a small 2-block (plyMod, stoneOccupancy) descriptor
// built directly in train.cpp, independent of compute_input_descr() (training/self_play.h), not
// board_to_features()'s full descr. Returns (N, F) float32 on CPU (caller stacks across
// plies/states and moves to device once). No legal mask - only the current ply's legality matters
// for the policy output, so callers that need a mask use board_to_features() for that ply instead.
torch::Tensor history_features_at_ply(const BoardState& state, int ply, const nlohmann::json& descr);

// Compute the multi-scale adjacency set. Call once per BoardConfig.
AdjNorms compute_adj_norms(const BoardConfig& bc, torch::Device device);
