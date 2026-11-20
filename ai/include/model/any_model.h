#pragma once
#include "model/gnn.h"
#include "model/unet.h"
#include "model/cnn.h"
#include "model/transformer.h"
#include "model/evaluator.h"
#include "model/features.h"
#include "model/model_config.h"
#include "game/board_config.h"
#include <stdexcept>
#include <string>
#include <variant>
#include <type_traits>

// Variant holding either model type; used in both train and server.
using AnyModel = std::variant<MessagePassingGNN, UNet, CNN, Transformer>;

// A freshly initialized model of the architecture cfg.model_type names; cfg's dynamic type must be
// that architecture's ModelConfig subclass (as parse_model_config() returns).
inline AnyModel build_model(const BoardConfig& bc, const ModelConfig& cfg, int num_stones, int num_players) {
    if (cfg.model_type == "cnn")
        return CNN(bc, static_cast<const CNNConfig&>(cfg), num_stones, num_players);
    if (cfg.model_type == "unet")
        return UNet(bc, static_cast<const UNetConfig&>(cfg), num_stones, num_players);
    if (cfg.model_type == "transformer")
        return Transformer(bc, static_cast<const TransformerConfig&>(cfg), num_stones, num_players);
    // adj_norms only sizes the GNN's neighbor-count embedding table.
    if (cfg.model_type == "gnn")
        return MessagePassingGNN(static_cast<const GNNConfig&>(cfg), num_stones, num_players,
                                 compute_adj_norms(bc, torch::kCPU));
    throw std::runtime_error("build_model: unknown model type '" + cfg.model_type + "'");
}

// The input descriptor `model` was built with (ModelConfig::input_descr).
inline const nlohmann::json& input_descr_of(const AnyModel& model) {
    return std::visit([](const auto& m) -> const nlohmann::json& { return m->cfg_.input_descr; }, model);
}

// Build a type-erased Evaluator from a model variant, capturing by value the board tensors it runs
// against, so the returned Evaluator is self-contained: adj_norms for GNN, node_slots for
// UNet/CNN/Transformer (the other is unused).
inline Evaluator make_evaluator(AnyModel& model, const AdjNorms& adj_norms, const NodeSlots& node_slots) {
    return std::visit([&](auto& m) -> Evaluator {
        using M = std::decay_t<decltype(m)>;
        auto prepare = [m](const BoardState& s) { return m->prepare_input(s); };
        if constexpr (std::is_same_v<M, MessagePassingGNN>) {
            auto adj = adj_norms;
            return Evaluator{prepare, [m, adj](const std::vector<const ModelInput*>& in) mutable {
                return m->evaluate_inputs(in, adj);
            }};
        } else {
            auto slots = node_slots;
            return Evaluator{prepare, [m, slots](const std::vector<const ModelInput*>& in) mutable {
                return m->evaluate_inputs(in, slots);
            }};
        }
    }, model);
}
