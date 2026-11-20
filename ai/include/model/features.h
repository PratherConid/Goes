#pragma once
#include <torch/torch.h>
#include "game/board_state.h"
#include "model/model_output.h"
#include "nlohmann/json.hpp"
#include <vector>

struct FeatureTensors {
    torch::Tensor features;    // (N, F) float32
    torch::Tensor legal_mask;  // (num_stones*N + 1,) bool
};

// One board's multi-scale adjacency matrices, padded to the GNN's slots (NodeSlots): a slot no
// node occupies has no edges and degree 0. The GNN computes them once per board
// (MessagePassingGNNImpl::adj_norms_of()). Layers cycle through these as: adj, adj2, adj, adj4,
// adj, adj2, adj, adj4, ...
struct AdjNorms {
    torch::Tensor adj;   // (S,S) row-normalised 1-hop + self
    torch::Tensor adj2;  // (S,S) row-normalised exactly-2-hop neighbors
    torch::Tensor adj4;  // (S,S) row-normalised exactly-4-hop neighbors
    torch::Tensor deg1;  // (S,1) float - neighbor count per node in adj's mask (1-hop+self)
    torch::Tensor deg2;  // (S,1) float - neighbor count per node in adj2's mask
    torch::Tensor deg4;  // (S,1) float - neighbor count per node in adj4's mask
};

// A board's nodes' slots in the zero-extended space every network's input and output live in,
// whose layout the input descriptor alone fixes, so one model serves - and one batch mixes -
// every board that fits: with an embedding, node i's slot is its position's index in the
// descriptor's sideLens box (dimension 0 fastest); without one, it is i, out of the descriptor's
// nBound. CPU tensors, cheap (O(N * emb_dim)) to compute.
struct NodeSlots {
    torch::Tensor slot;   // (N,) long - node i's slot
    torch::Tensor valid;  // (S,) float - 1 at the board's slots (S = num_node_slots())
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

// bc's multi-scale adjacency set, padded to `slots`.
AdjNorms compute_adj_norms(const BoardConfig& bc, const NodeSlots& slots, torch::Device device);

// The networks' slot count for boards like `bc` (see NodeSlots): num_cells(descr's sideLens) with
// an embedding, else descr's nBound.
int num_node_slots(const BoardConfig& bc, const nlohmann::json& descr);

// bc's NodeSlots under `descr`. Throws std::runtime_error if bc doesn't fit: its N in nBound and,
// with an embedding, its side lengths in sideLens (fits_in(), game/board_config.h).
NodeSlots compute_node_slots(const BoardConfig& bc, const nlohmann::json& descr);

// Every flat stone-major action's slot-space index: placement (s-1)*N + i -> (s-1)*S + slot[i], pass
// num_stones*N -> num_stones*S.
torch::Tensor slot_action_index(const NodeSlots& slots, int num_stones);

// `per_node`, whose dimension `dim` is indexed by node, zero-extended along it to the S slots.
torch::Tensor nodes_to_slots(const torch::Tensor& per_node, const NodeSlots& slots, int64_t dim = 0);

// A (num_stones*N+1,) per-action tensor (a legal mask, a policy target) zero-extended to the
// (num_stones*S+1,) slot-space actions.
torch::Tensor actions_to_slots(const torch::Tensor& per_action, const NodeSlots& slots, int num_stones);

// One state's network input in the model's slot space: board_to_features() under `descr`,
// zero-extended to the NodeSlots of the state's board, plus those slots and the board itself.
// Every architecture's prepare_input() builds on it (the Transformer adds history).
ModelInput slot_input(const BoardState& state, const nlohmann::json& descr);
