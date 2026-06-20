#pragma once
#include <torch/torch.h>
#include <deque>
#include <tuple>
#include <random>

struct GameRecord {
    torch::Tensor features;       // (N, F) float32
    torch::Tensor legal_mask;     // (N+1,) bool
    torch::Tensor policy_target;  // (N+1,) float32
    torch::Tensor value_target;   // (num_players,) float32 — reward for each player in ID order 1..P
};

// Circular buffer of GameRecord objects.
class ReplayBuffer {
public:
    explicit ReplayBuffer(int capacity = 100000);

    int size() const { return static_cast<int>(buf_.size()); }
    void add(std::vector<GameRecord> records);

    // Returns (features, legal_mask, policy_target, value_target) batched tensors:
    //   features      : (B, N, F)        float32
    //   legal_mask    : (B, N+1)         bool
    //   policy_target : (B, N+1)         float32
    //   value_target  : (B, num_players) float32
    // adj_norm is assumed shared / caller-supplied externally
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    sample(int batch_size, std::mt19937& rng);

private:
    int capacity_;
    std::deque<GameRecord> buf_;
};
