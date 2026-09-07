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

std::vector<std::pair<size_t, int64_t>> ReplayBuffer::pick_samples(int batch_size, std::mt19937& rng) {
    int n = std::min((int64_t)batch_size, live_ply_count_);

    std::uniform_int_distribution<uint64_t> dist(0, (uint64_t)live_ply_count_ - 1);
    std::unordered_set<uint64_t> chosen_offsets;
    std::vector<uint64_t> offsets;
    offsets.reserve(n);
    while ((int)offsets.size() < n) {
        uint64_t offset = dist(rng);
        if (chosen_offsets.insert(offset).second) offsets.push_back(offset);
    }

    std::vector<std::pair<size_t, int64_t>> result;
    result.reserve(n);
    for (uint64_t offset : offsets) {
        uint64_t global_idx = removed_ply_count_ + offset;
        size_t slot = idx_map_.at(global_idx);
        int64_t ply_offset = static_cast<int64_t>(global_idx - buf_[slot].start_idx);
        result.emplace_back(slot, ply_offset);
    }
    return result;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ReplayBuffer::sample(int batch_size, std::mt19937& rng) {
    auto picks = pick_samples(batch_size, rng);

    std::vector<torch::Tensor> feats, masks, policies, stone_owners, territory_owners;
    feats.reserve(picks.size()); masks.reserve(picks.size()); policies.reserve(picks.size());
    stone_owners.reserve(picks.size()); territory_owners.reserve(picks.size());

    for (auto& [slot, ply_offset] : picks) {
        const Slot& s = buf_[slot];
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

HistoryBatch ReplayBuffer::sample_with_history(int batch_size, std::mt19937& rng) {
    auto picks = pick_samples(batch_size, rng);
    int B = static_cast<int>(picks.size());

    int64_t Tmax = 0;
    for (auto& [slot, ply_offset] : picks) Tmax = std::max(Tmax, ply_offset);

    int64_t N = buf_[picks[0].first].record.history_features.size(1);
    int64_t F = buf_[picks[0].first].record.history_features.size(2);

    auto hist_features = torch::zeros({B, Tmax, N, F}, torch::kFloat32);
    auto hist_mask      = torch::ones({B, Tmax}, torch::kBool);  // True = pad
    std::vector<torch::Tensor> cur_feats, masks, policies, stone_owners, territory_owners;
    cur_feats.reserve(B); masks.reserve(B); policies.reserve(B);
    stone_owners.reserve(B); territory_owners.reserve(B);

    for (int i = 0; i < B; i++) {
        auto [slot, ply_offset] = picks[i];
        const Slot& s = buf_[slot];
        if (ply_offset > 0) {
            hist_features[i].slice(0, 0, ply_offset).copy_(s.record.history_features.slice(0, 0, ply_offset));
            hist_mask[i].slice(0, 0, ply_offset).fill_(false);
        }
        cur_feats.push_back(s.record.features[ply_offset]);
        masks.push_back(s.record.legal_mask[ply_offset]);
        policies.push_back(s.record.policy_target[ply_offset]);
        stone_owners.push_back(s.record.stone_owner);
        territory_owners.push_back(s.record.territory_owner);
    }

    return HistoryBatch{
        hist_features,
        hist_mask,
        torch::stack(cur_feats, 0),
        torch::stack(masks, 0),
        torch::stack(policies, 0),
        torch::stack(stone_owners, 0),
        torch::stack(territory_owners, 0),
    };
}
