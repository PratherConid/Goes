#include "training/replay_buffer.h"
#include <algorithm>
#include <numeric>

ReplayBuffer::ReplayBuffer(int capacity) : capacity_(capacity) {}

void ReplayBuffer::add(std::vector<GameRecord> records) {
    for (auto& r : records) {
        if ((int)buf_.size() >= capacity_) buf_.pop_front();
        buf_.push_back(std::move(r));
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ReplayBuffer::sample(int batch_size, std::mt19937& rng) {
    int n = std::min(batch_size, (int)buf_.size());

    // Draw n unique indices
    std::vector<int> indices(buf_.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    indices.resize(n);

    std::vector<torch::Tensor> feats, masks, policies, values;
    feats.reserve(n);
    masks.reserve(n);
    policies.reserve(n);
    values.reserve(n);

    for (int idx : indices) {
        const auto& r = buf_[idx];
        feats.push_back(r.features);
        masks.push_back(r.legal_mask);
        policies.push_back(r.policy_target);
        values.push_back(r.value_target);
    }

    auto x     = torch::stack(feats, 0);    // (B, N, F)
    auto mask  = torch::stack(masks, 0);    // (B, N+1)
    auto p_tgt = torch::stack(policies, 0); // (B, N+1)
    auto v_tgt = torch::stack(values, 0);   // (B, num_players)

    return {x, mask, p_tgt, v_tgt};
}
