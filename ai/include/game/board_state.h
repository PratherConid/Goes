#pragma once
#include "board_config.h"
#include <vector>
#include <unordered_map>
#include <map>
#include <optional>
#include <string>
#include <memory>
#include <cstdint>

enum class MoveType { NOMOVE = 0, PLACE = 1, PASS = 2 };

// Mirrors shared/types.ts's KoRule.
enum class KoRule { Positional, Situational };

// Mirrors shared/types.ts's TurnInfo. (`protected` is a C++ keyword, hence `is_protected`.)
struct TurnInfo {
    int player;
    std::vector<int> stones;
    std::vector<int> is_protected;
    std::vector<int> friendly;
};

// Compute per-player reward in (-1, 1) from a terminal board's point distribution
// (see BoardState::compute_points() - depending on the caller's scoring rule,
// `points` may represent stones only, territory only, or stones + territory).
// A stone mapping to multiple players credits each of them the stone's FULL
// point value (not split) - stone_to_player_map is stone -> set of players.
// komi is always added; capture_count (see ScoreData) is added only when
// score_rule == "territory" (real-world Japanese-style scoring is territory +
// prisoners) - both are player-indexed, so they're folded in here rather
// than inside compute_points(), which stays purely stone-indexed.
// rank_reward  = (# opp with fewer points - # opp with more) * 2 / (2P-1)  [0 if P==1]
// point_reward = (point_fraction - 1/P) / (2P-1)   [empty board: fraction = 0]
// reward[p] = rank_reward + point_reward
std::unordered_map<int,float> compute_player_rewards(
    const std::unordered_map<int,int>& points,
    const std::unordered_map<int, std::vector<int>>& stone_to_player_map,
    const std::vector<float>& komi,
    const std::string& score_rule,
    const std::vector<int>& capture_count);

// Mirrors shared/types.ts's MoveInfo.
struct MoveInfo {
    MoveType move_type;
    std::optional<int> pos;
    std::optional<int> stone;   // stone actually placed; nullopt for PASS/NOMOVE
    std::vector<int> captures;
    int consecutive_passes = 0;
    bool all_passed = false;
};

// Human-readable description of a move: "PLACE <cell>", "PASS", etc.
std::string move_to_string(const MoveInfo& m);

// Mirrors shared/types.ts's ScoreData.
struct ScoreData {
    std::unordered_map<int,int> stone_count;  // stones on the board, per stone type (1..num_stones)
    std::unordered_map<int,int> territory;    // territory points, per stone type (1..num_stones)
    std::vector<int> territory_owner;         // length N; stone type whose territory this node
                                               // belongs to, or 0 if occupied/neutral (dame) -
                                               // same 0-sentinel convention as `board`
    std::vector<int> capture_count;
};

// nullopt = illegal; value = set of captured node indices for that (stone,pos) placement
using LegalMove = std::optional<std::vector<int>>;
using LegalMoves = std::vector<LegalMove>;   // length N, one stone color's legality

// Mirrors shared/types.ts's LegalMovesData.captures[stone][loc]. Its fixed num_stones+1 width
// (unoffered stones all-nullopt) is also what keeps the NN's input/output tensor width constant
// turn-to-turn.
using LegalMovesByStone = std::vector<LegalMoves>;

// Mirrors shared/types.ts's LegalMovesData - snake_case, std::vector in place of Set, nullopt in
// place of null.
struct LegalMovesData {
    std::vector<int> pass_capture;
    LegalMovesByStone captures;
    std::vector<std::vector<int>> legals_for_stone;
    std::vector<std::vector<int>> legals_for_location;
    int place_legals = 0;
};

// Mirrors shared/types.ts's HistoryEntry. TS's per-ply Situation has no struct counterpart: board is
// content-interned via HistoryManager (history_ids_), and ply_count/len_turn_list are implicit
// (vector index / turn_list.size()). Entries are stored in HistoryManager
// (store_history_entry()/history_entry_of()) rather than a per-BoardState array - see there.
struct HistoryEntry {
    MoveInfo move_info;
    LegalMovesData legal_moves;
    ScoreData score;
    std::vector<std::vector<int>> player_stone_place_cnt;
};

// Mirrors shared/boardState.ts's GroupEntry; std::vector in place of Set.
struct GroupEntry {
    std::vector<int> group;
    std::vector<int> liberties;
    std::vector<int> non_liberties;
};
using GroupLib = std::vector<GroupEntry>;
using GroupDict = std::unordered_map<int, GroupLib>;

// Mirrors shared/boardState.ts's groupLiberty().
GroupDict group_liberty(const std::vector<int>& board,
                        const std::vector<std::vector<int>>& adj,
                        int N,
                        const std::vector<int>& friendly_stones);

// Interns (ply_mod, board) pairs and returns stable uint64_t IDs.
// Multiple BoardState copies sharing a HistoryManager avoid storing redundant
// board data: copy() copies only the ID list, not the actual board vectors.
class HistoryManager {
public:
    // Insert (ply_mod, board) if not present; return its ID.
    uint64_t store_board(int ply_mod, const std::vector<int>& board);

    // Return the ID if (ply_mod, board) is already interned, without inserting.
    std::optional<uint64_t> lookup(int ply_mod, const std::vector<int>& board) const;

    // Retrieve board by ID (used by withdraw_move to restore the previous board).
    const std::vector<int>& board_of(uint64_t id) const;

    // Store a per-ply HistoryEntry (move_info + legal_moves + score + player_stone_place_cnt) and
    // return its ID. Append-only and NOT content-interned (unlike boards): history entries rarely
    // recur and are expensive to hash, and sharing across BoardState copies (the MCTS hot path -
    // see BoardState::copy()/copy_with_hm()) is the only goal. Storing the WHOLE entry here means
    // every per-ply field lives in exactly one place, so BoardState::copy() (called once per MCTS
    // node) is O(1) for all of it rather than O(ply).
    uint64_t store_history_entry(HistoryEntry entry);

    // Retrieve a history entry by the ID returned from store_history_entry.
    const HistoryEntry& history_entry_of(uint64_t id) const;

private:
    static std::string make_key(int ply_mod, const std::vector<int>& board);
    std::unordered_map<std::string, uint64_t> key_to_id_;
    std::vector<std::vector<int>> boards_;
    std::vector<HistoryEntry> history_entries_;
    uint64_t next_id_ = 0;
};

class BoardState {
public:
    BoardState(int num_stones,
               int num_players,
               std::vector<TurnInfo> turn_list,
               std::vector<std::vector<std::optional<int>>> player_stone_place_limit,
               std::vector<std::optional<int>> global_stone_place_limit,
               std::unordered_map<int, std::vector<int>> stone_to_player_map,
               bool forced_pass_only,
               std::string score_rule,
               std::vector<float> komi,
               std::vector<int> player_model_id,
               KoRule ko_rule,
               bool allow_suicide,
               std::optional<int> max_plies,
               std::vector<int> initial_board,
               const BoardConfig& bc);

    // Disable accidental copy; use copy() or copy_with_hm() instead
    BoardState(const BoardState&) = delete;
    BoardState& operator=(const BoardState&) = delete;
    BoardState(BoardState&&) = default;
    BoardState& operator=(BoardState&&) = default;

    const MoveInfo& last_move() const { return hm_->history_entry_of(history_entry_ids_.back()).move_info; }
    bool no_trad_legal() const;
    // Mirrors shared/boardState.ts's legalPlaceList(), as (pos, stone) pairs.
    std::vector<std::pair<int,int>> legal_move_list() const;
    // Mirrors shared/boardState.ts's makeMove(); nullopt `k` is a pass.
    bool make_move(std::optional<int> k, std::optional<int> stone = std::nullopt);
    void withdraw_move();

    // Copy sharing the same HistoryManager (copies IDs only, not board data).
    BoardState copy() const;

    // Copy re-interning all history into a different HistoryManager.
    // Use this at MCTS search entry points to scope board storage to the search.
    BoardState copy_with_hm(HistoryManager* hm) const;

    // True iff the last move (if any) completed a full round of consecutive
    // passes - see MoveInfo::all_passed.
    bool all_passed() const {
        return !history_entry_ids_.empty() &&
               hm_->history_entry_of(history_entry_ids_.back()).move_info.all_passed;
    }

    // Mirrors shared/boardState.ts's gameOver().
    bool game_over() const {
        if (num_players - (int)resigned_players().size() <= 1) return true;
        if (max_plies.has_value() && ply_count() >= *max_plies) return true;
        return all_passed();
    }
    int ply_count() const { return static_cast<int>(history_ids_.size()) - 1; }

    // Board content at any reached ply (0 = genesis, ..., ply_count() = current) - O(1), backed by
    // HistoryManager's existing board interning (no extra storage). Used by the Transformer
    // architecture's per-ply feature reconstruction (features.h's history_features_at_ply()) - no
    // other architecture needs per-ply history, only the current board.
    const std::vector<int>& board_at(int ply) const { return hm_->board_of(history_ids_[ply]); }

    // MoveInfo::consecutive_passes as it stood at the end of the given ply - O(1), backed by the
    // existing per-ply HistoryEntry interning. Same genesis-is-ply-0 convention as last_move().
    // Used alongside board_at() by history_features_at_ply().
    int consecutive_passes_at(int ply) const {
        return hm_->history_entry_of(history_entry_ids_[ply]).move_info.consecutive_passes;
    }

    // Legal-move table as it stood at the given ply - O(1), same per-ply HistoryEntry interning as
    // consecutive_passes_at(). legal_moves_data_at(ply_count()) == legal_moves_data() (both read
    // history_entry_ids_.back()). Used by features.cpp's shared board_to_features_at_ply() for the
    // legalPlace block at any ply, not just the current one.
    const LegalMovesData& legal_moves_data_at(int ply) const {
        return hm_->history_entry_of(history_entry_ids_[ply]).legal_moves;
    }

    // Stone placement counts as they stood at the given ply, indexed [stone-1][player-1] - O(1),
    // same per-ply HistoryEntry interning as consecutive_passes_at(). Used by features.cpp's
    // shared board_to_features_at_ply() for the playerStoneBudget/globalStoneBudget blocks.
    const std::vector<std::vector<int>>& player_stone_place_cnt_at(int ply) const {
        return hm_->history_entry_of(history_entry_ids_[ply]).player_stone_place_cnt;
    }

    // Mirrors shared/boardState.ts's score().
    const ScoreData& score() const { return hm_->history_entry_of(history_entry_ids_.back()).score; }

    // Mirrors shared/boardState.ts's legalMovesData(), backed by HistoryManager interning rather
    // than a per-ply array (see HistoryEntry).
    const LegalMovesData& legal_moves_data() const { return hm_->history_entry_of(history_entry_ids_.back()).legal_moves; }

    // Mirrors shared/boardState.ts's computePoints().
    static std::unordered_map<int,int> compute_points(const std::string& rule, const ScoreData& score);

    // Mirrors shared/boardState.ts's playerStonePlaceCnt().
    const std::vector<std::vector<int>>& player_stone_place_cnt() const {
        return hm_->history_entry_of(history_entry_ids_.back()).player_stone_place_cnt;
    }

    // Mirrors shared/boardState.ts's captureCount().
    const std::vector<int>& capture_count() const {
        return hm_->history_entry_of(history_entry_ids_.back()).score.capture_count;
    }

    // Mirrors shared/boardState.ts's resignedPlayers getter.
    std::vector<int> resigned_players() const;
    // Mirrors shared/boardState.ts's resign().
    void resign(int player);
    // Mirrors shared/boardState.ts's advanceResigned().
    void advance_resigned();

    // Public fields accessed by features, MCTS, self-play. Unless noted, each mirrors
    // shared/boardState.ts's same-named BoardState field (nullopt in place of null).
    int num_stones;
    int num_players;
    std::vector<TurnInfo> turn_list;
    std::vector<std::vector<std::optional<int>>> player_stone_place_limit; // [stone-1][player-1]; nullopt = unlimited
    std::vector<std::optional<int>> global_stone_place_limit;
    // std::vector in place of Set.
    std::unordered_map<int, std::vector<int>> stone_to_player_map;
    bool forced_pass_only;
    std::string score_rule;
    // Per-player scoring handicap added to each non-resigned player's point
    // count before winner determination (compute_winners()) and before the
    // MCTS reward formulas (compute_player_rewards() below,
    // estimate_player_rewards() in evaluator.h) - 1-indexed via komi[p-1].
    std::vector<float> komi;
    // Training/self-play concept only - server.cpp's per-request /move inference path never reads
    // this (a session's BoardState always gets the default all-1s from new_state(), and the server
    // always evaluates with a single model regardless of this field's contents). Which model
    // searches each player's turns during self-play - 1-indexed via player_model_id[p-1], an id
    // key into MCTS::models_ (mcts.h). Lets different players in the same self-play game be
    // played by different models (e.g. a candidate checkpoint vs. a previous one); a player's
    // model evaluates every node of the search it runs on its turn (see MCTS's class comment). new_state()
    // starts every player at id 1; train.cpp then reassigns them per game from the currently-active
    // snapshot set (assign_random_models()/refresh_player()/assign_by_sequence()).
    std::vector<int> player_model_id;
    KoRule ko_rule;
    bool allow_suicide;
    std::optional<int> max_plies;
    TurnInfo next_turn;
    std::vector<int> board;
    std::shared_ptr<const std::vector<std::vector<int>>> adj; // shared, read-only
    int N;
    std::optional<std::vector<int>> winners;

private:
    std::unique_ptr<HistoryManager> owned_hm_; // non-null only when self-owned
    HistoryManager* hm_ = nullptr;             // always valid during lifetime
    std::vector<uint64_t> history_ids_;        // ID per history entry (board interning)
    // Reference-counted set of IDs currently in history_ids_, used for fast ko-rule
    // lookup in calculate_legal_moves. A plain unordered_set is incorrect: pass moves
    // are not required to avoid board state collision, so two states occurring in the
    // same game could have the same unique ID. If the same ID appears at
    // multiple positions in history_ids_, erasing it once on withdraw_move must not
    // remove it from the set entirely - the earlier occurrence is still in history.
    // Reference counting (ID → count) ensures the ID stays in the set until all
    // occurrences have been withdrawn.
    std::unordered_map<uint64_t, int> history_id_set_;
    // Per-ply HistoryEntry (move_info + legal_moves + score + player_stone_place_cnt), interned
    // in the HistoryManager exactly like history_ids_: each entry is an ID into
    // hm_->history_entry_of(...), so copies share the data instead of deep-copying it. The sole
    // backing store for last_move()/all_passed()/score()/legal_moves_data()/
    // player_stone_place_cnt()/capture_count() (all read the last entry).
    std::vector<uint64_t> history_entry_ids_;
    // Mirrors shared/boardState.ts's `resigns`; nothing here reads the ply key itself, only the
    // flattened resigned_players().
    std::map<int, std::vector<int>> resigns_;

    void after_move(MoveInfo move_info, std::vector<std::vector<int>> new_player_stone_place_cnt,
                     std::vector<int> new_capture_count);
    // Mirrors shared/boardState.ts's _computeLegalMoves().
    LegalMovesData compute_legal_moves(bool all_passed, const std::vector<std::vector<int>>& player_stone_place_cnt) const;
    // Mirrors shared/boardState.ts's _countScore().
    ScoreData count_score() const;
    // Mirrors shared/boardState.ts's _computeWinners(), but takes the already-computed `points`
    // (see compute_points()) rather than `history`.
    std::vector<int> compute_winners(const std::unordered_map<int,int>& points) const;
    // Mirrors shared/boardState.ts's _refreshWinners().
    void refresh_winners();
    static BoardState make_copy_skeleton(const BoardState& src);

    // Used by copy() / copy_with_hm() / make_copy_skeleton() without constructor logic
    BoardState() = default;
};
