#pragma once
#include "model/features.h"
#include "nlohmann/json.hpp"
#include <memory>
#include <string>

// Model-building metadata, disjoint from GameConfig (game rules, shared with
// the TS client, see training/self_play.h) - written alongside a GameConfig
// to each checkpoint's <arch>_config.json by train.cpp (the two to_json()
// objects are joined into one file), read back separately by server.cpp's
// load_model(). Has no shared/types.ts analog - it's purely an
// ai/-internal artifact. Lives in model/ (rather than training/, alongside
// GameConfig) so cnn.h/unet.h/gnn.h can depend on it without a model/ ->
// training/ layering inversion.
//
// Base holds only the fields every architecture needs (modelType/featureDim/
// hiddenDim); architecture-specific fields (e.g. GNNConfig::num_layers) live
// on the matching subclass below, so e.g. CNNImpl only ever sees the fields
// it actually uses.
struct ModelConfig {
    std::string model_type;  // "cnn" | "unet" | "gnn" - matches the checkpoint filename prefix
    int feature_dim;
    // Hidden dim of whichever architecture model_type names. train.cpp's Args
    // keeps three separate CLI-configurable defaults (--cnn-hidden-dim etc.,
    // since a training run may target any one architecture) but only the one
    // actually trained is persisted here.
    int hidden_dim;
    // Self-describing feature-block descriptor this model was trained with -
    // see compute_input_descr() (training/self_play.h) for how it's built and
    // board_to_features() (features.cpp) for how it's consumed. Common to all
    // architectures (unlike num_layers below), so it lives on the base rather
    // than a subclass. No meaningful default exists without a GameConfig to
    // build one from - callers must always pass one explicitly.
    nlohmann::json input_descr;

    ModelConfig(std::string model_type_, int feature_dim_, int hidden_dim_,
                nlohmann::json input_descr_)
        : model_type(std::move(model_type_)), feature_dim(feature_dim_), hidden_dim(hidden_dim_),
          input_descr(std::move(input_descr_)) {}
    virtual ~ModelConfig() = default;

    // modelType/featureDim/hiddenDim/inputDescr only - subclasses with
    // additional fields (e.g. GNNConfig::num_layers) override this and extend
    // the result. Join with a GameConfig::to_json() (training/self_play.h) at
    // the call site for the full checkpoint JSON.
    virtual nlohmann::json to_json() const;
};

// CNN-specific model config. CNN's block count (num_blocks_) is derived from
// board geometry at construction time, not user-configurable (see CNNImpl's
// ctor) - kept as its own type for symmetry with UNetConfig/GNNConfig and so
// CNNImpl only depends on CNN's own fields.
struct CNNConfig : ModelConfig {
    // Convolution kernel size for every conv in every block (see CNNImpl's
    // ctor) - must be odd (so "same" padding via conv_size/2 keeps spatial
    // dims exactly unchanged, required for the block's residual add) and > 1
    // (a 1x1 kernel can't be validated the same way and defeats the point of
    // a spatial conv here) - enforced by train.cpp's --cnn-conv-size parsing.
    int conv_size;

    CNNConfig(int feature_dim, int hidden_dim, nlohmann::json input_descr, int conv_size_ = 5)
        : ModelConfig("cnn", feature_dim, hidden_dim, std::move(input_descr)), conv_size(conv_size_) {}

    nlohmann::json to_json() const override;
};

// UNet-specific model config. No additional fields today - UNet's depth is
// derived from board grid dimensions, not user-configurable (see UNetImpl's
// ctor doc comment) - kept as its own type for the same reason as CNNConfig.
struct UNetConfig : ModelConfig {
    UNetConfig(int feature_dim, int hidden_dim, nlohmann::json input_descr)
        : ModelConfig("unet", feature_dim, hidden_dim, std::move(input_descr)) {}
};

// GNN-specific model config.
struct GNNConfig : ModelConfig {
    int num_layers;  // message-passing layer count - see MessagePassingGNNImpl's ctor.

    GNNConfig(int feature_dim, int hidden_dim, int num_layers_, nlohmann::json input_descr)
        : ModelConfig("gnn", feature_dim, hidden_dim, std::move(input_descr)), num_layers(num_layers_) {}

    nlohmann::json to_json() const override;
};

// Transformer-specific model config.
struct TransformerConfig : ModelConfig {
    // Cross-attention stack depth AND history self-attention stack depth (same count for both -
    // see TransformerImpl's ctor doc comment for why one CLI-configurable count serves both
    // stacks). Deliberately NOT GNNConfig::num_layers/--num-layers (GNN's own message-passing
    // depth flag, default 9) - kept as its own flag/field, per an explicit decision not to
    // complicate the shared --num-layers default or its meaning.
    int num_attn_layers;
    // Minimal per-ply feature descriptor for PAST plies only (plyMod + stoneOccupancy) - built
    // directly in train.cpp from the GameConfig, independent of compute_input_descr(). The
    // inherited ModelConfig::input_descr now uniformly means "the full descriptor," used for the
    // CURRENT ply only, same meaning as CNN/UNet/GNN's input_descr. See TransformerImpl's ctor for
    // how this sizes the separate history-encoder MLP.
    nlohmann::json history_descr;

    TransformerConfig(int feature_dim, int hidden_dim, int num_attn_layers_,
                       nlohmann::json input_descr, nlohmann::json history_descr_)
        : ModelConfig("transformer", feature_dim, hidden_dim, std::move(input_descr)),
          num_attn_layers(num_attn_layers_), history_descr(std::move(history_descr_)) {}

    nlohmann::json to_json() const override;
};

// Parses a checkpoint's <arch>_config.json modelType/featureDim/hiddenDim
// (/numLayers for gnn) keys into the matching concrete subclass, selected by
// the modelType key - call parse_game_cfg (training/self_play.h) on the same
// JSON separately for the GameConfig (numStones/numPlayers/etc.) fields
// joined into the same file. Used by server.cpp's load_model() instead of
// reading individual keys inline.
std::unique_ptr<ModelConfig> parse_model_config(const nlohmann::json& cfg);

// True iff a and b are identical in every field (== ModelConfig::to_json()
// equality, which each subclass already overrides to include its own extra
// fields, e.g. GNNConfig::numLayers). Used by train.cpp's --resume validation.
bool strong_equal(const ModelConfig& a, const ModelConfig& b);
