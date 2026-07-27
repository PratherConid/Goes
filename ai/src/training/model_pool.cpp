#include "training/model_pool.h"
#include <cassert>

ModelPool::ModelPool(int capacity) : capacity_(capacity) {
    assert(capacity_ >= 1 && "ModelPool capacity must be >= 1");
}

std::optional<int> ModelPool::add(int id, AnyModel model) {
    entries_.push_back({id, std::move(model)});
    if ((int)entries_.size() <= capacity_) return std::nullopt;
    int evicted_id = entries_.front().first;
    entries_.erase(entries_.begin());
    return evicted_id;
}

int ModelPool::random_id(std::mt19937& rng) const {
    std::uniform_int_distribution<size_t> dist(0, entries_.size() - 1);
    return entries_[dist(rng)].first;
}
