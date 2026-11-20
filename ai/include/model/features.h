#pragma once
#include <torch/torch.h>
#include "game/board_state.h"
#include "model/model_output.h"
#include "nlohmann/json.hpp"
#include <array>
#include <memory>
#include <vector>

struct FeatureTensors {
    torch::Tensor features;    // (N, F) float32
    torch::Tensor legal_mask;  // (num_stones*N + 1,) bool
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

// A board's GNN message-passing neighborhoods in slot indexing. Tier 0 is a node's neighbors and
// itself, tier 1 the nodes exactly 2 hops away, tier 2 those exactly 4 hops away; a node's tier-t
// aggregate is the mean over its tier-t neighbors. src[t] lists every slot's tier-t neighbors,
// grouped by slot in increasing slot order, and deg[t] counts each slot's (0 at slots no node
// occupies). The GNN's layers cycle through the tiers 0, 1, 0, 2, 0, 1, 0, 2, ...
struct NeighborTiers {
    std::array<torch::Tensor, 3> src;  // (E_t,) int32
    torch::Tensor deg;                 // (3, S) int64
};

// The board-derived part of every network input of one game, built once when the game starts
// (make_game_input()) and shared by each of its positions' ModelInput.
struct GameInput {
    NodeSlots slots;
    torch::Tensor action_index;  // (num_stones*N+1,) - slot_action_index(slots, num_stones)
    NeighborTiers tiers;
    torch::Tensor conn_img;      // (S, conn_px^embDim) float - the connImg block's channels
                                 // (render_conn_img()); undefined without that block
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

// The networks' slot count under `descr` (see NodeSlots): num_cells(sideLens) with an embedding
// (embDim > 0), else nBound.
int num_node_slots(const nlohmann::json& descr);

// bc's NodeSlots under `descr`. Throws std::runtime_error if bc doesn't fit: its embedding dimension
// must be embDim, its N at most nBound and, with an embedding, its side lengths within sideLens
// (fits_in(), game/board_config.h).
NodeSlots compute_node_slots(const BoardConfig& bc, const nlohmann::json& descr);

// Every flat stone-major action's slot-space index: placement (s-1)*N + i -> (s-1)*S + slot[i], pass
// num_stones*N -> num_stones*S.
torch::Tensor slot_action_index(const NodeSlots& slots, int num_stones);

// `per_node`, whose dimension `dim` is indexed by node, zero-extended along it to the S slots.
torch::Tensor nodes_to_slots(const torch::Tensor& per_node, const NodeSlots& slots, int64_t dim = 0);

// A (num_stones*N+1,) per-action tensor (a legal mask, a policy target) of one of `game`'s
// positions, zero-extended to the (num_stones*S+1,) slot-space actions.
torch::Tensor actions_to_slots(const torch::Tensor& per_action, const GameInput& game);

// bc's neighbor tiers in `slots`' indexing, by a breadth-first search to depth 4 from every node.
NeighborTiers neighbor_tiers(const BoardConfig& bc, const NodeSlots& slots);

// The connImg block's channels for bc under `descr`, conn_px the block's arg: every point of the
// sideLens box owns a cube of conn_px pixels per side, centered on the point, and every edge is drawn
// between its two nodes' cube centers as a line one pixel wide, anti-aliased as in Xiaolin Wu's
// algorithm generalized to any dimension (at each pixel center along the line's longest axis, weight
// 1 is split among the pixels around the line across the other axes by linear interpolation along
// each; overlapping lines keep the larger value). Row s lists slot s's cube, dimension 0 fastest:
// (S, conn_px^embDim).
torch::Tensor render_conn_img(const BoardConfig& bc, const nlohmann::json& descr, int conn_px);

// A position's (N, F) features zero-extended to its game's slots, with the game's connImg channels,
// when the descriptor has that block, filled in at every slot.
torch::Tensor slot_features(const torch::Tensor& node_features, const GameInput& game);

// The GameInput of the game `state` is a position of, under the run's input descriptor `descr`.
// Its neighbor tiers are built whatever the architecture, so a game's input doesn't depend on which
// model evaluates it. Throws std::runtime_error if its board doesn't fit (compute_node_slots()).
std::shared_ptr<const GameInput> make_game_input(const BoardState& state, const nlohmann::json& descr);

// One position's network input in the model's slot space: board_to_features() under `descr`,
// zero-extended to its game's slots. Every architecture's prepare_input() builds on it (the
// Transformer adds history).
ModelInput slot_input(const BoardState& state, const nlohmann::json& descr, std::shared_ptr<const GameInput> game);

// Every input's game's slot validity, stacked: (B, S) float32.
torch::Tensor stack_valid(const std::vector<const ModelInput*>& inputs);
