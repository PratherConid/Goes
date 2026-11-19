#pragma once
#include <torch/torch.h>
#include <cstring>
#include <vector>

// What every architecture's forward()/evaluate_inputs() returns, all left on the model's device.
struct ModelOutput {
    torch::Tensor policy;     // (B, numStones*N+1) softmax over legal actions
    torch::Tensor ownership;  // (B, 2, N, numStones+1) stone/territory ownership estimates
    torch::Tensor rank;       // (B, num_rank_pairs, 3) pairwise outcome probabilities (RankHead,
                              // rank_head.h)
};

// One position's network input, on the CPU. Built by the thread that wants the position evaluated -
// as KataGo's search threads fill their own input rows (NNInputs::fillRow*) before queueing them -
// so the thread running the network only stacks a batch and copies it to the device.
struct ModelInput {
    torch::Tensor features;    // (N, F) float32 - the current ply's features
    torch::Tensor legal_mask;  // (numStones*N+1,) bool
    torch::Tensor history;     // Transformer only: (T, N, F_history) float32, every past ply
};

// Stacks equally-shaped, contiguous CPU tensors into one (rows.size(), ...) CPU tensor with plain
// memcpys - KataGo's server thread copies each queued input row into its batch buffer the same way.
// Unlike torch::stack, it never starts a libtorch parallel region (whose idle workers spin), however
// large the result.
inline torch::Tensor stack_cpu(const std::vector<torch::Tensor>& rows) {
    const torch::Tensor& first = rows.at(0);
    std::vector<int64_t> shape = {static_cast<int64_t>(rows.size())};
    for (int64_t d : first.sizes()) shape.push_back(d);
    auto out = torch::empty(shape, first.options());
    size_t bytes = static_cast<size_t>(first.numel()) * first.element_size();
    auto* dst = static_cast<char*>(out.data_ptr());
    for (size_t i = 0; i < rows.size(); i++) {
        TORCH_CHECK(rows[i].is_contiguous() && rows[i].sizes() == first.sizes() &&
                    rows[i].scalar_type() == first.scalar_type(), "stack_cpu: rows must match and be contiguous");
        std::memcpy(dst + i * bytes, rows[i].data_ptr(), bytes);
    }
    return out;
}

// stack_cpu() over one field of every input.
inline torch::Tensor stack_inputs(const std::vector<const ModelInput*>& inputs, torch::Tensor ModelInput::*field) {
    std::vector<torch::Tensor> rows;
    rows.reserve(inputs.size());
    for (const ModelInput* in : inputs) rows.push_back(in->*field);
    return stack_cpu(rows);
}
