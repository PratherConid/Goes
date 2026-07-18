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

private:
    struct Slot {
        GameRecord record;
        uint64_t start_idx;  // this game's first ply's index in ALL_REC; its
                              // plies occupy [start_idx, start_idx + num_plies)
        int64_t num_plies;
    };

    int capacity_;  // max games
    std::vector<Slot> buf_;
    uint64_t removed_ply_count_ = 0;   // # plies belonging to evicted games (front of the live ALL_REC window)
    uint64_t removed_game_count_ = 0;  // # games evicted so far (also: next eviction targets slot removed_game_count_ % capacity_)
    int64_t live_ply_count_ = 0;       // # plies currently in buf_ (width of the live ALL_REC window)
    std::unordered_map<uint64_t, size_t> idx_map_;  // live ALL_REC index -> slot in buf_
};
