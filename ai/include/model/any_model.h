#pragma once
#include "model/gnn.h"
#include "model/cnn.h"
#include "model/evaluator.h"
#include <variant>
#include <type_traits>

// Variant holding either model type; used in both train and server.
using AnyModel = std::variant<MessagePassingGNN, ConvNN>;

// Build a type-erased Evaluator from a model variant.
// For GNN, adj_norms is captured by value so the returned Evaluator is self-contained.
// For CNN, adj_norms is unused.
inline Evaluator make_evaluator(AnyModel& model, const AdjNorms& adj_norms) {
    return std::visit([&](auto& m) -> Evaluator {
        using M = std::decay_t<decltype(m)>;
        if constexpr (std::is_same_v<M, MessagePassingGNN>) {
            auto adj = adj_norms;
            return Evaluator{[m, adj](const std::vector<const BoardState*>& s) mutable {
                return m->evaluate_batch(s, adj);
            }};
        } else {
            return Evaluator{[m](const std::vector<const BoardState*>& s) mutable {
                return m->evaluate_batch(s);
            }};
        }
    }, model);
}
