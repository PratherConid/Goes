#pragma once
#include <functional>
#include <vector>
#include <utility>
#include <torch/torch.h>
#include "game/board_state.h"

// Type-erased neural-network evaluator used by MCTS and self-play.
// Wraps either a MessagePassingGNN (with adj_norms captured in fn) or a ConvNN.
struct Evaluator {
    std::function<std::pair<torch::Tensor, torch::Tensor>(
        const std::vector<const BoardState*>&)> fn;

    std::pair<torch::Tensor, torch::Tensor>
    evaluate_batch(const std::vector<const BoardState*>& states) const {
        return fn(states);
    }
};
