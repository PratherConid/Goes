#pragma once
#include <torch/torch.h>
#include "game/board_state.h"

struct FeatureTensors {
    torch::Tensor features;    // (N, F) float32
    torch::Tensor legal_mask;  // (N+1,) bool
};

// Multi-scale adjacency matrices, computed once per BoardConfig.
// Layers cycle through these as: adj, adj2, adj, adj4, adj, adj2, adj, adj4, ...
struct AdjNorms {
    torch::Tensor adj;   // row-normalised 1-hop + self  (original adj_norm)
    torch::Tensor adj2;  // row-normalised exactly-2-hop neighbors
    torch::Tensor adj4;  // row-normalised exactly-4-hop neighbors
};

// Compute per-node feature matrix and legal mask for one state.
FeatureTensors board_to_features(const BoardState& state, torch::Device device);

// Compute the multi-scale adjacency set. Call once per BoardConfig.
AdjNorms compute_adj_norms(const BoardConfig& bc, torch::Device device);
