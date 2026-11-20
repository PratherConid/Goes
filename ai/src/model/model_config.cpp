#include "model/model_config.h"
#include <stdexcept>

using json = nlohmann::json;

nlohmann::json ModelConfig::to_json() const {
    json j;
    j["modelType"]  = model_type;
    j["featureDim"] = feature_dim;
    j["hiddenDim"]  = hidden_dim;
    j["inputDescr"] = input_descr;
    return j;
}

nlohmann::json GNNConfig::to_json() const {
    json j = ModelConfig::to_json();
    j["numLayers"] = num_layers;
    return j;
}

nlohmann::json CNNConfig::to_json() const {
    json j = ModelConfig::to_json();
    j["convSize"] = conv_size;
    return j;
}

nlohmann::json TransformerConfig::to_json() const {
    json j = ModelConfig::to_json();
    j["numAttnLayers"] = num_attn_layers;
    j["historyDescr"] = history_descr;
    return j;
}

std::unique_ptr<ModelConfig> parse_model_config(const json& cfg) {
    // .at(), not operator[] - see parse_game_cfg() (game/game_config.cpp) for why.
    std::string model_type = cfg.at("modelType").get<std::string>();
    int feature_dim         = cfg.at("featureDim").get<int>();
    int hidden_dim           = cfg.at("hiddenDim").get<int>();
    json descr               = cfg.at("inputDescr");
    if (model_type == "cnn")
        return std::make_unique<CNNConfig>(feature_dim, hidden_dim, descr, cfg.at("convSize").get<int>());
    if (model_type == "unet") return std::make_unique<UNetConfig>(feature_dim, hidden_dim, descr);
    if (model_type == "transformer")
        return std::make_unique<TransformerConfig>(feature_dim, hidden_dim, cfg.at("numAttnLayers").get<int>(),
                                                     descr, cfg.at("historyDescr"));
    if (model_type == "gnn")
        return std::make_unique<GNNConfig>(feature_dim, hidden_dim, cfg.at("numLayers").get<int>(), descr);
    throw std::runtime_error("parse_model_config: unknown modelType '" + model_type + "'");
}

bool strong_equal(const ModelConfig& a, const ModelConfig& b) {
    return a.to_json() == b.to_json();
}
