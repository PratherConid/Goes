#pragma once
#include "model/any_model.h"
#include <vector>
#include <utility>
#include <optional>
#include <random>

// Bounded, FIFO pool of recent model snapshots used by multi-model self-play (see
// BoardState::player_model_id, MCTS::models_, train.cpp) - the id a given entry is tagged with is
// the training iteration it was captured at. Small by design (capacity is --num-selfplay-models,
// expected to be a handful at most), so a plain vector + erase(begin()) on overflow is simpler
// than ReplayBuffer's circular-index scheme and just as fast at this size.
class ModelPool {
public:
    explicit ModelPool(int capacity);

    // Adds a fresh snapshot tagged with the given id, evicting the oldest entry if now over
    // capacity. Returns the evicted entry's id, or nullopt if no eviction happened (the pool is
    // still growing towards capacity).
    std::optional<int> add(int id, AnyModel model);

    // Uniformly random id from the current window - used both to assign a brand-new game's
    // players and (in train.cpp's refresh_player()) to reassign a player whose model was just
    // evicted.
    int random_id(std::mt19937& rng) const;

    // Current window, oldest-first - used to (re)build the evaluators map after add(). Non-const:
    // callers pass entries straight into make_evaluator(AnyModel&, const AdjNorms&)
    // (model/any_model.h), which itself takes a non-const reference.
    std::vector<std::pair<int, AnyModel>>& entries() { return entries_; }

private:
    int capacity_;
    std::vector<std::pair<int, AnyModel>> entries_;  // oldest-first
};
