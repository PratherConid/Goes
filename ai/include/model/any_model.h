#pragma once
#include "model/gnn.h"
#include "model/unet.h"
#include "model/cnn.h"
#include "model/transformer.h"
#include "model/evaluator.h"
#include "model/features.h"
#include "model/model_config.h"
#include <stdexcept>
#include <string>
#include <variant>

// Variant holding either model type; used in both train and server.
using AnyModel = std::variant<MessagePassingGNN, UNet, CNN, Transformer>;

// A freshly initialized model of the architecture cfg.model_type names; cfg's dynamic type must be
// that architecture's ModelConfig subclass (as parse_model_config() returns). Throws
// std::runtime_error if the architecture can't take cfg (e.g. a CNN/UNet descriptor whose embDim
// isn't 2).
inline AnyModel build_model(const ModelConfig& cfg, int num_stones, int num_players) {
    if (cfg.model_type == "cnn")
        return CNN(static_cast<const CNNConfig&>(cfg), num_stones, num_players);
    if (cfg.model_type == "unet")
        return UNet(static_cast<const UNetConfig&>(cfg), num_stones, num_players);
    if (cfg.model_type == "transformer")
        return Transformer(static_cast<const TransformerConfig&>(cfg), num_stones, num_players);
    if (cfg.model_type == "gnn")
        return MessagePassingGNN(static_cast<const GNNConfig&>(cfg), num_stones, num_players);
    throw std::runtime_error("build_model: unknown model type '" + cfg.model_type + "'");
}

// Loads the weights torch::save() wrote to `source` (a file path or a std::istream) into `model`,
// then moves it to `device`.
template <class Source>
void load_weights(AnyModel& model, Source&& source, torch::Device device) {
    std::visit([&](auto& m) { torch::load(m, source); m->to(device); }, model);
}

// An eval-mode model of cfg's architecture (build_model()) holding `source`'s weights, on `device`.
template <class Source>
AnyModel load_eval_model(const ModelConfig& cfg, int num_stones, int num_players, Source&& source,
                         torch::Device device) {
    AnyModel model = build_model(cfg, num_stones, num_players);
    load_weights(model, source, device);
    std::visit([](auto& m) { m->eval(); }, model);
    return model;
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
