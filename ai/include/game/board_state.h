#pragma once
#include "board_config.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <string>
#include <memory>
#include <cstdint>

enum class MoveType { NOMOVE = 0, ILLEGAL = 1, PLACE = 2, PASS = 3, GAMEOVER = 4 };

// Compute per-player reward in (-1, 1) from a terminal board's stone distribution.
// rank_reward = (# opp with fewer stones - # opp with more) * 2 / (2P-1)  [0 if P==1]
// stone_reward = tanh(stone_fraction - 1/P) / (2P-1)   [empty board: fraction = 0]
// reward[p] = rank_reward + stone_reward
std::unordered_map<int,float> compute_player_rewards(
    const std::unordered_map<int,int>& stone_count,
    const std::unordered_map<int,int>& stone_to_player_map);

struct MoveInfo {
    MoveType move_type;
    std::optional<int> pos;
    std::vector<int> captures;
    std::unordered_set<int> passed_players;
};

// nullopt = illegal; value = set of captured node indices
using LegalMove = std::optional<std::vector<int>>;
using LegalMoves = std::vector<LegalMove>;

// Returns stone_color -> list of (group_nodes, liberty_nodes)
using GroupLib = std::vector<std::pair<std::vector<int>, std::vector<int>>>;
using GroupDict = std::unordered_map<int, GroupLib>;

GroupDict group_liberty(const std::vector<int>& board,
                        const std::vector<std::vector<int>>& adj,
                        int N);

// Interns (ply_mod, board) pairs and returns stable uint64_t IDs.
// Multiple BoardState copies sharing a HistoryManager avoid storing redundant
// board data: copy() copies only the ID list, not the actual board vectors.
class HistoryManager {
public:
    // Insert (ply_mod, board) if not present; return its ID.
    uint64_t intern(int ply_mod, const std::vector<int>& board);

    // Return the ID if (ply_mod, board) is already interned, without inserting.
    std::optional<uint64_t> lookup(int ply_mod, const std::vector<int>& board) const;

    // Retrieve board by ID (used by retract_move to restore the previous board).
    const std::vector<int>& board_of(uint64_t id) const;

private:
    static std::string make_key(int ply_mod, const std::vector<int>& board);
    std::unordered_map<std::string, uint64_t> key_to_id_;
    std::vector<std::vector<int>> boards_;
    uint64_t next_id_ = 0;
};

class BoardState {
public:
    BoardState(int num_stones,
               int num_players,
               std::vector<int> turn_stone_list,
               std::unordered_map<int,int> stone_to_player_map,
               bool forced_pass_only,
               std::vector<int> initial_board,
               const BoardConfig& bc);

    // Disable accidental copy; use copy() or copy_with_hm() instead
    BoardState(const BoardState&) = delete;
    BoardState& operator=(const BoardState&) = delete;
    BoardState(BoardState&&) = default;
    BoardState& operator=(BoardState&&) = default;

    const MoveInfo& last_move() const { return last_moves_.back(); }
    bool no_trad_legal() const;
    std::vector<int> legal_move_list() const;
    bool make_move(std::optional<int> k);
    void retract_move();

    // Copy sharing the same HistoryManager (copies IDs only, not board data).
    BoardState copy() const;

    // Copy re-interning all history into a different HistoryManager.
    // Use this at MCTS search entry points to scope board storage to the search.
    BoardState copy_with_hm(HistoryManager* hm) const;

    bool game_over() const { return last_move().move_type == MoveType::GAMEOVER; }
    int ply_count() const { return static_cast<int>(history_ids_.size()) - 1; }

    // Adjusts next_player to reflect a given absolute ply (used by inference server).
    void set_ply(int ply);

    // Public fields accessed by features, MCTS, self-play
    int num_stones;
    int num_players;
    std::vector<int> turn_stone_list;
    std::unordered_map<int,int> stone_to_player_map;
    bool forced_pass_only;
    int next_player;
    std::vector<int> board;
    std::shared_ptr<const std::vector<std::vector<int>>> adj; // shared, read-only
    int N;
    std::unordered_map<int,int> stone_count;
    std::vector<int> winners;
    LegalMoves legal_moves_with_take;

private:
    std::unique_ptr<HistoryManager> owned_hm_; // non-null only when self-owned
    HistoryManager* hm_ = nullptr;             // always valid during lifetime
    std::vector<uint64_t> history_ids_;        // ID per history entry
    // Reference-counted set of IDs currently in history_ids_, used for fast ko-rule
    // lookup in calculate_legal_moves. A plain unordered_set is incorrect: pass moves
    // are not required to avoid board state collision, so two states occurring in the
    // same game could have the same unique ID. If the same ID appears at
    // multiple positions in history_ids_, erasing it once on retract_move must not
    // remove it from the set entirely - the earlier occurrence is still in history.
    // Reference counting (ID → count) ensures the ID stays in the set until all
    // occurrences have been retracted.
    std::unordered_map<uint64_t, int> history_id_set_;
    std::vector<MoveInfo> last_moves_;
    std::vector<LegalMoves> legal_move_history_;

    void add_to_history_and_after_move();
    void after_move();
    void count_stones();
    static BoardState make_copy_skeleton(const BoardState& src);

    // Used by copy() / copy_with_hm() / make_copy_skeleton() without constructor logic
    BoardState() = default;
};
