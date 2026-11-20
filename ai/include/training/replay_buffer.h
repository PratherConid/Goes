#pragma once
#include <torch/torch.h>
#include "model/features.h"
#include <cstdint>
#include <deque>
#include <memory>
#include <random>
#include <vector>

// One finished self-play game's training data. Rows are the positions self-play chose to train on
// (see ai/Readme.md's "Self-Play"), each stored once with a multiplicity - KataGo writes a row once
// per unit of its resolved target weight; sampling treats a row of multiplicity k as k rows.
struct GameRecord {
    torch::Tensor features;         // (R, N, F) float16
    torch::Tensor legal_mask;       // (R, numStones*N+1) bool
    torch::Tensor policy_target;    // (R, numStones*N+1) float32 - pruned search visit distribution
    torch::Tensor rank_target;      // (R, num_rank_pairs, 3) float32 - blended pair-outcome target
    std::vector<int> row_count;     // (R) multiplicity of each row, >= 1
    std::vector<int> row_ply;       // (R) the row's position among all of the game's plies (0-based)
    torch::Tensor stone_owner;      // (N,) int64 - stone type occupying each node at game end, 0 = empty
    torch::Tensor territory_owner;  // (N,) int64 - stone type whose territory each node is at game end
    // Transformer runs only: every ply's minimal history features (TransformerConfig::history_descr),
    // so any row's full history prefix is available. Undefined otherwise.
    torch::Tensor history_features;  // (T, N, F_hist) float16
    std::shared_ptr<const GameInput> game;  // the game's board-derived network input

    int64_t num_rows() const;
};

// Sampled rows, each zero-extended to its game's node slots - the networks' slot space, so rows from
// different boards batch together.
struct Batch {
    torch::Tensor features;         // (B, S, F) float32
    torch::Tensor valid;            // (B, S) float32 - 1 at each row's board's slots
    torch::Tensor legal_mask;       // (B, numStones*S+1) bool
    torch::Tensor policy_target;    // (B, numStones*S+1) float32
    torch::Tensor stone_owner;      // (B, S) int64 - 0 at slots no node occupies
    torch::Tensor territory_owner;  // (B, S) int64 - 0 at slots no node occupies
    torch::Tensor rank_target;      // (B, num_rank_pairs, 3) float32
    std::vector<std::shared_ptr<const GameInput>> games;  // each row's game's GameInput
};

// Batch for the Transformer, whose forward() also needs each sampled row's preceding plies.
struct HistoryBatch {
    torch::Tensor hist_features;  // (B, T_max, S, F_hist) float32, zero-padded
    torch::Tensor hist_mask;      // (B, T_max) bool - true = padded slot
    Batch current;                // the sampled rows themselves
};

// The training window: every row ever added gets the next global index, and sampling draws
// uniformly from the newest `window_rows` of them (KataGo's shuffle window - see ai/Readme.md's
// "Training Loop"). Games whose rows have all fallen out of the window can be dropped with trim().
class ReplayBuffer {
public:
    void add(GameRecord record);
    int64_t total_rows() const { return total_rows_; }
    int64_t stored_rows() const;
    // Drops every game whose rows all lie before the newest `window_rows` rows.
    void trim(int64_t window_rows);
    Batch sample(int batch_size, int64_t window_rows, std::mt19937& rng) const;
    HistoryBatch sample_with_history(int batch_size, int64_t window_rows, std::mt19937& rng) const;

private:
    struct Slot {
        GameRecord record;
        int64_t start;             // global index of this game's first row
        std::vector<int64_t> cum;  // cum[r] = rows before row r within this game (prefix sums of row_count)
        int64_t rows;
    };
    std::deque<Slot> slots_;
    int64_t total_rows_ = 0;

    // (slot, row) of `batch_size` distinct uniformly random global indices in the window.
    std::vector<std::pair<size_t, int>> pick(int batch_size, int64_t window_rows, std::mt19937& rng) const;
    Batch gather(const std::vector<std::pair<size_t, int>>& picks) const;
};
