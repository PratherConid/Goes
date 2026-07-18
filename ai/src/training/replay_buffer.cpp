#include "training/replay_buffer.h"
#include <algorithm>
#include <unordered_set>

ReplayBuffer::ReplayBuffer(int capacity) : capacity_(capacity) {}

void ReplayBuffer::add(GameRecord record) {
    int64_t num_plies = record.features.size(0);
    uint64_t start_idx = removed_ply_count_ + static_cast<uint64_t>(live_ply_count_);

    size_t slot;
    if ((int)buf_.size() < capacity_) {
        slot = buf_.size();
        buf_.push_back(Slot{});
    } else {
        slot = removed_game_count_ % capacity_;
        Slot& old = buf_[slot];
        for (uint64_t i = old.start_idx; i < old.start_idx + (uint64_t)old.num_plies; i++)
            idx_map_.erase(i);
        removed_ply_count_ += old.num_plies;
        removed_game_count_ += 1;
        live_ply_count_ -= old.num_plies;
    }

    buf_[slot] = Slot{std::move(record), start_idx, num_plies};
    for (uint64_t i = start_idx; i < start_idx + (uint64_t)num_plies; i++)
        idx_map_[i] = slot;
    live_ply_count_ += num_plies;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ReplayBuffer::sample(int batch_size, std::mt19937& rng) {
    int n = std::min((int64_t)batch_size, live_ply_count_);

    std::uniform_int_distribution<uint64_t> dist(0, (uint64_t)live_ply_count_ - 1);
    std::unordered_set<uint64_t> chosen_offsets;
    std::vector<uint64_t> offsets;
    offsets.reserve(n);
    while ((int)offsets.size() < n) {
        uint64_t offset = dist(rng);
        if (chosen_offsets.insert(offset).second) offsets.push_back(offset);
    }

    std::vector<torch::Tensor> feats, masks, policies, stone_owners, territory_owners;
    feats.reserve(n); masks.reserve(n); policies.reserve(n);
    stone_owners.reserve(n); territory_owners.reserve(n);

    for (uint64_t offset : offsets) {
        uint64_t global_idx = removed_ply_count_ + offset;
        size_t slot = idx_map_.at(global_idx);
        const Slot& s = buf_[slot];
        int64_t ply_offset = static_cast<int64_t>(global_idx - s.start_idx);
        feats.push_back(s.record.features[ply_offset]);
        masks.push_back(s.record.legal_mask[ply_offset]);
        policies.push_back(s.record.policy_target[ply_offset]);
        stone_owners.push_back(s.record.stone_owner);
        territory_owners.push_back(s.record.territory_owner);
    }

    auto x       = torch::stack(feats, 0);            // (B, N, F)
    auto mask    = torch::stack(masks, 0);             // (B, N+1)
    auto p_tgt   = torch::stack(policies, 0);           // (B, N+1)
    auto so_tgt  = torch::stack(stone_owners, 0);       // (B, N)
    auto to_tgt  = torch::stack(territory_owners, 0);   // (B, N)

    return {x, mask, p_tgt, so_tgt, to_tgt};
}
