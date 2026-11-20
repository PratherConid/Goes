#include "model/cnn.h"
#include <algorithm>
#include <stdexcept>
#include <string>

CNNImpl::CNNImpl(const CNNConfig& cfg, int num_stones, int num_players)
    : cfg_(cfg)
{
    // Thrown rather than asserted (Release builds compile assert() out): any other embDim would
    // send the side_lens[0]/[1] reads below out of bounds.
    const unsigned emb_dim = cfg.input_descr.at("embDim").get<unsigned>();
    if (emb_dim != 2)
        throw std::runtime_error("CNN requires a 2D embedding (embDim == 2), got embDim=" + std::to_string(emb_dim));

    // The grid is the input descriptor's sideLens (no padding), which NodeSlots index into.
    const auto side_lens = cfg.input_descr.at("sideLens").get<std::vector<unsigned>>();
    int gw = static_cast<int>(side_lens[0]), gh = static_cast<int>(side_lens[1]);
    grid_w_ = gw;
    grid_h_ = gh;

    num_blocks_ = std::max(gw, gh);

    input_proj = register_module("input_proj",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(cfg_.feature_dim + 1, cfg_.hidden_dim, 1)));

    // "Same" padding (conv_size/2, integer division - exact since conv_size is enforced odd) keeps
    // H,W unchanged across every conv, which forward()'s residual add requires.
    int pad = cfg_.conv_size / 2;
    for (int k = 0; k < num_blocks_; k++) {
        torch::nn::Sequential seq;
        seq->push_back(torch::nn::Conv2d(
            torch::nn::Conv2dOptions(cfg_.hidden_dim, cfg_.hidden_dim, cfg_.conv_size).padding(pad)
        ));
        seq->push_back(torch::nn::ReLU());
        seq->push_back(torch::nn::Conv2d(
            torch::nn::Conv2dOptions(cfg_.hidden_dim, cfg_.hidden_dim, cfg_.conv_size).padding(pad)
        ));
        blocks_.push_back(register_module("block_" + std::to_string(k), seq));
    }

    heads.register_on(*this, cfg_.hidden_dim, num_stones, num_players);
}

torch::Tensor CNNImpl::features_to_grid(const torch::Tensor& x, const torch::Tensor& valid) const {
    // Slot s is grid cell (s / W, s % W) (NodeSlots' layout: dimension 0, the column, fastest);
    // the validity channel goes last.
    auto grid = torch::cat({x, valid.unsqueeze(-1)}, -1).permute({0, 2, 1})
                    .reshape({x.size(0), x.size(2) + 1, grid_h_, grid_w_});
    return grid;
}

ModelOutput CNNImpl::forward(
    torch::Tensor x,
    torch::Tensor valid,
    torch::Tensor legal_mask)
{
    int64_t B = x.size(0);

    auto h = torch::relu(input_proj->forward(features_to_grid(x, valid))); // (B, hidden_dim, H, W)
    for (int k = 0; k < num_blocks_; k++) {
        auto out = blocks_[k]->forward(h);
        h = torch::relu(out + h);
    }
    // h: (B, hidden_dim, H, W)

    auto h_slots = h.reshape({B, cfg_.hidden_dim, -1}).permute({0, 2, 1});  // (B, S, hidden_dim)
    return heads.forward(h_slots, legal_mask);
}

ModelInput CNNImpl::prepare_input(const BoardState& state, const std::shared_ptr<const GameInput>& game) const {
    return slot_input(state, cfg_.input_descr, game);
}

ModelOutput CNNImpl::evaluate_inputs(const std::vector<const ModelInput*>& inputs) {
    torch::NoGradGuard ng;
    auto dev = input_proj->weight.device();
    auto x     = stack_inputs(inputs, &ModelInput::features).to(dev);   // (B, S, F)
    auto valid = stack_valid(inputs).to(dev);                           // (B, S)
    auto mask  = stack_inputs(inputs, &ModelInput::legal_mask).to(dev); // (B, ns*S+1)
    return forward(x, valid, mask);
}
