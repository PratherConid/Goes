#pragma once
#include <torch/torch.h>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <random>
#include <cstdint>

struct GameRecord {
    torch::Tensor features;         // (P, N, F) float32 - P = plies in this game
    torch::Tensor legal_mask;       // (P, N+1) bool
    torch::Tensor policy_target;    // (P, N+1) float32
    torch::Tensor stone_owner;      // (N,) int64 - stone type occupying each node at game end
                                     // (BoardState::board), 0 = empty
    torch::Tensor territory_owner;  // (N,) int64 - stone type whose territory each node is at game
                                     // end (ScoreData::territory_owner), 0 = none/dame/occupied
    // stone_owner/territory_owner are shared by every ply of this game (only meaningful once the
    // game ends). Stone-type indexed, matching the network's ownership head (see evaluator.h).

    // Minimal-descriptor per-ply features (TransformerConfig::history_descr's plyMod+stoneOccupancy
    // descriptor) - only populated for a Transformer run (see PlyResult::history_features,
    // training/self_play.h); left default-constructed (undefined) otherwise. Used only via
    // ReplayBuffer::sample_with_history()'s history prefix - `features` above (the full
    // descriptor) is still what's used whenever a ply is sampled as the "current" one.
    torch::Tensor history_features;  // (P, N, F_hist) float32, Transformer runs only
};

// Batch returned by ReplayBuffer::sample_with_history() - used only by the Transformer
// architecture, whose forward() needs every ply's embedding up to the sampled ply, not just the
// sampled ply's own row (unlike sample(), below).
struct HistoryBatch {
    torch::Tensor hist_features;    // (B, T_max, N, F) float32, zero-padded
    torch::Tensor hist_mask;        // (B, T_max) bool - True = padded/invalid slot
    torch::Tensor cur_features;     // (B, N, F) float32 - the sampled ply's own features
    torch::Tensor legal_mask;       // (B, N+1) bool
    torch::Tensor policy_target;    // (B, N+1) float32
    torch::Tensor stone_owner;      // (B, N) int64
    torch::Tensor territory_owner;  // (B, N) int64
};

// Circular buffer of GameRecord objects (one per game).
//
// Storage is per-game, but sampling must be uniform per-ply. To reconcile
// the two, imagine every ply ever produced by self-play (across the whole
// run, not just what's currently buffered) appended in generation order to
// one persistent, conceptual array ALL_REC - it never shrinks or reorders.
// Each game occupies a contiguous range of ALL_REC indices
// (see Slot::start_idx). `ReplayBuffer` tracks:
//   - removed_ply_count_  = plies belonging to already-evicted games, i.e.
//                           how far the front of ALL_REC has been forgotten.
//   - live_ply_count_     = plies still resident in buf_.
// Together these define the currently-live window of ALL_REC as the
// half-open range [removed_ply_count_, removed_ply_count_ + live_ply_count_).
// idx_map_ maps each live ALL_REC index in that window to the buf_ slot
// holding its game (offset `by removed_game_count_`), so sample() can
// pick a uniformly random index in the window and land on any ply with
// equal probability, regardless of which game (or how long that game was) it came from.
class ReplayBuffer {
public:
    explicit ReplayBuffer(int capacity = 2048);

    // Number of ply records currently stored (not number of games).
    int size() const { return static_cast<int>(live_ply_count_); }
    void add(GameRecord record);

    // Returns (features, legal_mask, policy_target, stone_owner, territory_owner) batched tensors:
    //   features        : (B, N, F) float32
    //   legal_mask      : (B, N+1)  bool
    //   policy_target   : (B, N+1)  float32
    //   stone_owner     : (B, N)    int64
    //   territory_owner : (B, N)    int64
    // adj_norm is assumed shared / caller-supplied externally
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    sample(int batch_size, std::mt19937& rng);

    // Same uniform-over-every-live-ply sampling distribution as sample(), but for each sampled
    // ply also returns its full preceding-ply prefix as history, plus that ply's own row as
    // "current" - used only by the Transformer architecture, whose two encoders need different
    // widths for the two: history comes from GameRecord.history_features (the minimal
    // plyMod+stoneOccupancy descriptor), current comes from GameRecord.features (the full
    // descriptor, same one CNN/UNet/GNN use). Requires GameRecord.history_features to have been
    // populated - true automatically for any run using --net-arch transformer, since
    // train.cpp/self_play.cpp thread TransformerConfig::history_descr through unconditionally for
    // that architecture.
    HistoryBatch sample_with_history(int batch_size, std::mt19937& rng);

private:
    struct Slot {
        GameRecord record;
        uint64_t start_idx;  // this game's first ply's index in ALL_REC; its
                              // plies occupy [start_idx, start_idx + num_plies)
        int64_t num_plies;
    };

    // Picks `n` distinct uniformly-random offsets into the live ALL_REC window and resolves each
    // to a (slot, ply_offset) pair - the sampling-distribution core shared by sample() and
    // sample_with_history(), factored out so both draw from the exact same distribution with no
    // duplicated logic.
    std::vector<std::pair<size_t, int64_t>> pick_samples(int batch_size, std::mt19937& rng);

    int capacity_;  // max games
    std::vector<Slot> buf_;
    uint64_t removed_ply_count_ = 0;   // # plies belonging to evicted games (front of the live ALL_REC window)
    uint64_t removed_game_count_ = 0;  // # games evicted so far (also: next eviction targets slot removed_game_count_ % capacity_)
    int64_t live_ply_count_ = 0;       // # plies currently in buf_ (width of the live ALL_REC window)
    std::unordered_map<uint64_t, size_t> idx_map_;  // live ALL_REC index -> slot in buf_
};
