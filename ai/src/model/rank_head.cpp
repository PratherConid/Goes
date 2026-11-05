#include "model/rank_head.h"

RankHeadImpl::RankHeadImpl(int hidden_dim, int num_pairs) : num_pairs_(num_pairs) {
    mlp = register_module("mlp", torch::nn::Sequential(
        torch::nn::Linear(2 * hidden_dim, hidden_dim),
        torch::nn::ReLU(),
        torch::nn::Linear(hidden_dim, num_pairs * 3)));
}

torch::Tensor RankHeadImpl::forward(const torch::Tensor& h) {
    auto pooled = torch::cat({h.mean(1), std::get<0>(h.max(1))}, -1);   // (B, 2*hidden_dim)
    auto logits = mlp->forward(pooled).reshape({h.size(0), num_pairs_, 3});
    return torch::softmax(logits, -1);
}
