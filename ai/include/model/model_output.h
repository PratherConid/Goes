#pragma once
#include <torch/torch.h>

// What every architecture's forward()/evaluate_batch() returns, all left on the model's device.
// Batched shapes shown; an unbatched forward() drops the leading B.
struct ModelOutput {
    torch::Tensor policy;     // (B, numStones*N+1) softmax over legal actions
    torch::Tensor ownership;  // (B, 2, N, numStones+1) stone/territory ownership estimates
    torch::Tensor rank;       // (B, num_rank_pairs, 3) pairwise outcome probabilities (RankHead,
                              // rank_head.h)
};
