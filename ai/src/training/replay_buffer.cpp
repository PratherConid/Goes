#include "training/replay_buffer.h"
#include <algorithm>
#include <stdexcept>
#include <unordered_set>

int64_t GameRecord::num_rows() const {
    int64_t n = 0;
    for (int c : row_count) n += c;
    return n;
}

void ReplayBuffer::add(GameRecord record) {
    if (record.row_count.empty()) return;
    Slot slot;
    slot.start = total_rows_;
    slot.cum.reserve(record.row_count.size());
    int64_t acc = 0;
    for (int c : record.row_count) { slot.cum.push_back(acc); acc += c; }
    slot.rows = acc;
    slot.record = std::move(record);
    total_rows_ += acc;
    slots_.push_back(std::move(slot));
}

int64_t ReplayBuffer::stored_rows() const {
    return slots_.empty() ? 0 : total_rows_ - slots_.front().start;
}

void ReplayBuffer::trim(int64_t window_rows) {
    int64_t window_start = total_rows_ - window_rows;
    while (!slots_.empty() && slots_.front().start + slots_.front().rows <= window_start) slots_.pop_front();
}

std::vector<std::pair<size_t, int>> ReplayBuffer::pick(int batch_size, int64_t window_rows, std::mt19937& rng) const {
    if (slots_.empty()) throw std::runtime_error("ReplayBuffer::pick: buffer is empty");
    int64_t lo = std::max(total_rows_ - window_rows, slots_.front().start);
    int64_t span = total_rows_ - lo;
    int n = static_cast<int>(std::min<int64_t>(batch_size, span));
    std::uniform_int_distribution<int64_t> dist(lo, total_rows_ - 1);
    std::unordered_set<int64_t> chosen;
    std::vector<std::pair<size_t, int>> picks;
    picks.reserve(n);
    while ((int)picks.size() < n) {
        int64_t g = dist(rng);
        if (!chosen.insert(g).second) continue;
        auto it = std::upper_bound(slots_.begin(), slots_.end(), g,
                                   [](int64_t x, const Slot& s) { return x < s.start; });
        size_t si = static_cast<size_t>(std::distance(slots_.begin(), it)) - 1;
        const Slot& s = slots_[si];
        int64_t local = g - s.start;
        int row = static_cast<int>(std::upper_bound(s.cum.begin(), s.cum.end(), local) - s.cum.begin()) - 1;
        picks.emplace_back(si, row);
    }
    return picks;
}

Batch ReplayBuffer::gather(const std::vector<std::pair<size_t, int>>& picks) const {
    std::vector<torch::Tensor> feats, masks, policies, stones, territories, ranks;
    for (auto& [si, row] : picks) {
        const GameRecord& r = slots_[si].record;
        feats.push_back(r.features[row]);
        masks.push_back(r.legal_mask[row]);
        policies.push_back(r.policy_target[row]);
        stones.push_back(r.stone_owner);
        territories.push_back(r.territory_owner);
        ranks.push_back(r.rank_target[row]);
    }
    return Batch{
        torch::stack(feats, 0).to(torch::kFloat32),
        torch::stack(masks, 0),
        torch::stack(policies, 0),
        torch::stack(stones, 0),
        torch::stack(territories, 0),
        torch::stack(ranks, 0),
    };
}

Batch ReplayBuffer::sample(int batch_size, int64_t window_rows, std::mt19937& rng) const {
    return gather(pick(batch_size, window_rows, rng));
}

HistoryBatch ReplayBuffer::sample_with_history(int batch_size, int64_t window_rows, std::mt19937& rng) const {
    auto picks = pick(batch_size, window_rows, rng);
    int B = static_cast<int>(picks.size());
    int64_t t_max = 0;
    for (auto& [si, row] : picks) t_max = std::max<int64_t>(t_max, slots_[si].record.row_ply[row]);
    const torch::Tensor& h0 = slots_[picks[0].first].record.history_features;
    auto hist = torch::zeros({B, t_max, h0.size(1), h0.size(2)}, torch::kFloat32);
    auto hist_mask = torch::ones({B, t_max}, torch::kBool);
    for (int i = 0; i < B; i++) {
        const GameRecord& r = slots_[picks[i].first].record;
        int64_t ply = r.row_ply[picks[i].second];
        if (ply > 0) {
            hist[i].slice(0, 0, ply).copy_(r.history_features.slice(0, 0, ply));
            hist_mask[i].slice(0, 0, ply).fill_(false);
        }
    }
    return HistoryBatch{hist, hist_mask, gather(picks)};
}
