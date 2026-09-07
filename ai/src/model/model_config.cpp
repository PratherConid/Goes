#include "model/model_config.h"

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
    std::string model_type = cfg["modelType"].get<std::string>();
    int feature_dim         = cfg["featureDim"].get<int>();
    int hidden_dim           = cfg["hiddenDim"].get<int>();
    json descr               = cfg.value("inputDescr", json::object());
    // convSize defaults to 3 (the fixed kernel size before --cnn-conv-size
    // existed) so checkpoints saved before this option was added still load.
    if (model_type == "cnn")  return std::make_unique<CNNConfig>(feature_dim, hidden_dim, descr, cfg.value("convSize", 3));
    if (model_type == "unet") return std::make_unique<UNetConfig>(feature_dim, hidden_dim, descr);
    if (model_type == "transformer")
        return std::make_unique<TransformerConfig>(feature_dim, hidden_dim, cfg["numAttnLayers"].get<int>(),
                                                     descr, cfg["historyDescr"]);
    return std::make_unique<GNNConfig>(feature_dim, hidden_dim, cfg["numLayers"].get<int>(), descr);
}

bool strong_equal(const ModelConfig& a, const ModelConfig& b) {
    return a.to_json() == b.to_json();
}
