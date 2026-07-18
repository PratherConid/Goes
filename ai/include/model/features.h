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
// compute_input_descr(const GameConfig&, int) (training/self_play.h) - see
// that function's doc comment for the JSON shape and this function's own doc
// comment (features.cpp) for the recognized block names/args. The feature
// width F is read directly from descr["totalDims"], not recomputed here.
// server.cpp never calls compute_input_descr: it persists a checkpoint's
// descriptor into the config JSON once (at train time) and reads it back
// directly rather than recomputing it.
FeatureTensors board_to_features(const BoardState& state, torch::Device device, const nlohmann::json& descr);

// Compute the multi-scale adjacency set. Call once per BoardConfig.
AdjNorms compute_adj_norms(const BoardConfig& bc, torch::Device device);
