#pragma once
#include "game/game_config.h"
#include "nlohmann/json.hpp"
#include <vector>

// Builds the full self-describing feature-block descriptor for a training run - the JSON stored
// as ModelConfig::input_descr (model/model_config.h), consumed by board_to_features()
// (model/features.h). Board geometry is separate, since GameConfig has no board-geometry fields of
// its own: `emb_dim`, the board's embedding dimension (0 without one), is stored as "embDim";
// `side_lens` (goes_train's --side-lens, else the board's own) is stored as "sideLens", the
// box of the slot space every network's input and output live in (NodeSlots, model/features.h), and
// `n_bound`, an upper bound on the board's node count (goes_train's --n-bound, else
// num_cells(side_lens)), as "nBound", which sizes the liberty and groupSize blocks. A positive
// `conn_px` (goes_train's --conn-px) appends the connImg block, always last (see board_to_features()).
// Only train.cpp calls this (on --resume, the
// result must match the checkpoint's stored model config); server.cpp's load_model() reads the descriptor stored in a
// checkpoint's _config.json, so a checkpoint stays self-describing even if this function's own
// block formulas change later.
//
// Shape: {"blocks": [[name, ...args], ...], "totalDims": F, "embDim": emb_dim, "sideLens": side_lens,
// "nBound": n_bound}. Each block's
// args, with the top-level fields, already fully determine its width without needing a GameConfig to
// reinterpret - see board_to_features()'s doc comment for the exact per-block
// arg meaning and the "0 bits = absent, dense grid" convention used by
// playerStoneBudget/globalStoneBudget. cfg's place limits may be left empty (unset); this
// resolves them itself.
//
// Used identically by all four architectures for their CURRENT-ply-only input - including the
// Transformer, whose separate, much smaller per-ply HISTORY descriptor is compute_history_descr()'s.
nlohmann::json compute_input_descr(const GameConfig& cfg, unsigned emb_dim, const std::vector<unsigned>& side_lens,
                                   int n_bound, int conn_px);

// The Transformer's per-PAST-ply descriptor, stored as TransformerConfig::history_descr
// (model/model_config.h): just the turn position and stones, {"blocks": [["plyMod",
// turn_list_len], ["stoneOccupancy", num_stones]], "totalDims": F}. Like compute_input_descr(),
// only train.cpp calls this; server.cpp reads the checkpoint's stored copy.
nlohmann::json compute_history_descr(const GameConfig& cfg);
