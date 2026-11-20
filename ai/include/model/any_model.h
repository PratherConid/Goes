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
        return Transformer(static_cast<const TransformerConfig&>(cfg), num_stones, num_players);
    if (cfg.model_type == "gnn")
        return MessagePassingGNN(static_cast<const GNNConfig&>(cfg), num_stones, num_players);
    throw std::runtime_error("build_model: unknown model type '" + cfg.model_type + "'");
}

// The input descriptor `model` was built with (ModelConfig::input_descr).
inline const nlohmann::json& input_descr_of(const AnyModel& model) {
    return std::visit([](const auto& m) -> const nlohmann::json& { return m->cfg_.input_descr; }, model);
}

// Build a type-erased Evaluator from a model variant.
inline Evaluator make_evaluator(AnyModel& model) {
    return std::visit([](auto& m) -> Evaluator {
        return Evaluator{[m](const BoardState& s, const std::shared_ptr<const GameInput>& game) {
                             return m->prepare_input(s, game);
                         },
                         [m](const std::vector<const ModelInput*>& in) mutable { return m->evaluate_inputs(in); }};
    }, model);
}
