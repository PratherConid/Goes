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

// Superko variant enforced by calculate_legal_moves()'s repeat-position check
// (mirrors shared/types.ts's KoRule). Situational: a board is only a repeat
// if the player-to-move also matches (ply % turn_list.size()). Positional:
// any repeated board is a violation, regardless of whose turn it is.
enum class KoRule { Positional, Situational };

// One slot in the turn order (mirrors shared/types.ts's TurnInfo). `player` is
// the sole source of truth for turn ownership - independent of
// stone_to_player_map, which is used only for scoring. `stones`/`is_protected`/
// `friendly` each have length num_stones (0/1 bit arrays): stone colors
// offered this turn (at least one must be offered), stone colors that can
// never be removed this turn even at zero liberties, and stone colors that
// don't block anyone else's liberties this turn, respectively.
// (`protected` is a C++ keyword, hence `is_protected`.)
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

struct MoveInfo {
    MoveType move_type;
    std::optional<int> pos;
    std::optional<int> stone;   // stone actually placed; nullopt for PASS/NOMOVE
    std::vector<int> captures;
    // Number of consecutive pass moves ending with (and including, if this move
    // itself is a pass) this move. Resets to 0 on a PLACE move. The game ends
    // once this reaches turn_list.size(), since a stone appearing multiple
    // times in turn_list (one player controlling several stones) must pass
    // on each of its turns, not just once, before the round can be considered over.
    int consecutive_passes = 0;
    // True iff this move was the pass that completed a full round of
    // consecutive passes (consecutive_passes reached turn_list.size()) - a
    // per-ply-intrinsic fact, set once when the move is created and never
    // retroactively mutated afterward. This is only ONE of
    // BoardState::game_over()'s three conditions (see there, which also
    // checks max_plies and resigned_players live) - e.g. a max_plies-
    // triggered PLACE move has all_passed=false even though it ends the game.
    bool all_passed = false;
};

// Human-readable description of a move: "PLACE <cell>", "PASS", etc.
std::string move_to_string(const MoveInfo& m);

struct ScoreData {
    std::unordered_map<int,int> stone_count;  // stones on the board, per stone type (1..num_stones)
    std::unordered_map<int,int> territory;    // territory points, per stone type (1..num_stones)
    std::vector<int> territory_owner;         // length N; stone type whose territory this node
                                               // belongs to, or 0 if occupied/neutral (dame) -
                                               // same 0-sentinel convention as `board`
    // Cumulative stones captured so far, indexed [player-1] - unlike
    // stone_count/territory (stone-indexed, board-derived every ply), this is
    // player-indexed and a running total across the whole game (captured
    // stones are simply gone, so it can't be recomputed from the current
    // board) - see BoardState::make_move()/capture_count(). Used for the
    // "territory" score_rule (real-world Japanese-style scoring: territory +
    // prisoners); folded in at the player-aggregation layer
    // (compute_winners/compute_player_rewards), the same way komi is, rather
    // than inside compute_points (which stays stone-indexed).
    std::vector<int> capture_count;
};

// nullopt = illegal; value = set of captured node indices for that (stone,pos) placement
using LegalMove = std::optional<std::vector<int>>;
using LegalMoves = std::vector<LegalMove>;   // length N, one stone color's legality

// Per-stone legal-move table: length num_stones+1 (index 0 unused, 1-indexed
// stone color - mirrors shared/types.ts's LegalMovesData.captures[stone][loc]).
// Always this fixed width regardless of how many stones are offered this
// turn - stones not offered are simply all-nullopt for every position, so
// the NN's input/output tensor width never changes turn-to-turn.
using LegalMovesByStone = std::vector<LegalMoves>;

// Mirrors shared/types.ts's LegalMovesData class field-for-field (see there
// for the full doc comment) - fields are snake_case and nullopt replaces
// null, otherwise identical.
struct LegalMovesData {
    std::vector<int> pass_capture;
    LegalMovesByStone captures;
    std::vector<std::vector<int>> legals_for_stone;
    std::vector<std::vector<int>> legals_for_location;
    int place_legals = 0;
};

// Mirrors shared/types.ts's HistoryEntry field-for-field. board/ply_count/len_turn_list aren't
// duplicated per-entry here the way TS's Situation is - board is content-interned via
// HistoryManager (history_ids_), and ply_count/len_turn_list are implicit (vector index /
// turn_list.size()). Unlike TS, this whole struct is interned via HistoryManager
// (store_history_entry()/history_entry_of()) rather than stored as a plain per-BoardState array -
// see HistoryManager's doc comment for why (BoardState::copy() is the MCTS hot path).
struct HistoryEntry {
    MoveInfo move_info;
    LegalMovesData legal_moves;
    ScoreData score;
    std::vector<std::vector<int>> player_stone_place_cnt;
};

// group        - list of node indices belonging to the group
// liberties    - empty node indices adjacent to the group, plus any occupied
//                neighbor whose color is friendly this turn (see friendly_stones
//                in group_liberty() - such a neighbor doesn't block a liberty)
// non_liberties - occupied, non-friendly node indices (belonging to another
//                group) adjacent to the group - used to detect when capturing a
//                neighboring group frees a liberty for this one (see calculate_legal_moves)
struct GroupEntry {
    std::vector<int> group;
    std::vector<int> liberties;
    std::vector<int> non_liberties;
};
using GroupLib = std::vector<GroupEntry>;
using GroupDict = std::unordered_map<int, GroupLib>;

// Returns stone_color -> list of groups of that color, each with its
// liberties/non_liberties (see GroupEntry). A group may legitimately have
// zero liberties - a protected color (see calculate_legal_moves) can be left
// on the board at zero liberties instead of being captured.
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
    // see BoardState::copy()/copy_with_hm()) is the only goal. Interning the WHOLE entry (not just
    // legal_moves) means BoardState never needs a separate, plainly-deep-copied per-ply array for
    // score/player_stone_place_cnt/move_info either - every per-ply field lives in exactly one
    // place, and BoardState::copy() (called once per MCTS node - the dominant term in an earlier
    // memory investigation that traced O(ply)-per-node deep copies to multi-tens-of-GB usage)
    // becomes O(1) per call for all of it, not just legal_moves.
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
    // Legal (stone, position) PLACE move pairs.
    std::vector<std::pair<int,int>> legal_move_list() const;
    // Make a move. Pass nullopt for `k` for a pass move (no stone needed). For
    // a PLACE move, `stone` selects which offered color to play; it may be
    // omitted only when the current turn offers exactly one stone (the
    // unambiguous case) - otherwise the caller must choose. Returns true if
    // the move was legal.
    bool make_move(std::optional<int> k, std::optional<int> stone = std::nullopt);
    void withdraw_move();

    // Copy sharing the same HistoryManager (copies IDs only, not board data).
    BoardState copy() const;

    // Copy re-interning all history into a different HistoryManager.
    // Use this at MCTS search entry points to scope board storage to the search.
    BoardState copy_with_hm(HistoryManager* hm) const;

    // True iff the last move (if any) completed a full round of consecutive
    // passes - see MoveInfo::all_passed. Just that flag's snapshot, same
    // "no separate live field to keep in sync" pattern as score()/etc. below.
    bool all_passed() const { return !history_entry_ids_.empty() && hm_->history_entry_of(history_entry_ids_.back()).move_info.all_passed; }

    // True iff the game has ended, via any of three independent conditions:
    // too few non-resigned players remain, max_plies has been reached, or the
    // last move completed a full round of consecutive passes. The first two
    // are checked live against always-current state (resigned_players()/
    // ply_count()) rather than stamped onto a move, so nothing ever needs
    // retroactive fixing up after resign() or withdraw_move() (see
    // compute_legal_moves()).
    bool game_over() const {
        if (num_players - (int)resigned_players().size() <= 1) return true;
        if (max_plies.has_value() && ply_count() >= *max_plies) return true;
        return all_passed();
    }
    int ply_count() const { return static_cast<int>(history_ids_.size()) - 1; }

    // Current score (stone count + territory, both per stone type), i.e. the
    // last history entry's.
    const ScoreData& score() const { return hm_->history_entry_of(history_entry_ids_.back()).score; }

    // Legal-move table for the position just reached, i.e. history_entry_ids_'s
    // last entry - mirrors shared/boardState.ts's legalMovesData(). Backed by
    // HistoryManager interning rather than a dense per-ply array (unlike TS) -
    // BoardState::copy()/copy_with_hm() are the MCTS hot path, so this keeps
    // each copy O(1) (an ID vector) instead of O(ply) HistoryEntry structs.
    const LegalMovesData& legal_moves_data() const { return hm_->history_entry_of(history_entry_ids_.back()).legal_moves; }

    // Pure: converts ScoreData into a per-stone-type point map under the given
    // scoring rule ("stone" | "territoryonly" | "area" | "territory") -
    // "stone" counts stones on the board only, "territoryonly" and
    // "territory" both count territory only here (real-world Japanese-style
    // scoring also adds captures, but those are player- not stone-indexed -
    // see ScoreData::capture_count and compute_winners, which folds them in
    // separately), "area" counts stones + territory (Chinese-style; today's
    // default).
    static std::unordered_map<int,int> compute_points(const std::string& rule, const ScoreData& score);

    // Running count of stones placed so far, indexed [stone-1][player-1] (same
    // as player_stone_place_limit) - just the last history entry's snapshot,
    // so there's no separate live field to keep in sync: make_move() computes
    // each new entry's count directly, and withdraw_move() "rewinds" it for
    // free simply by popping. Compared against player_stone_place_limit in
    // calculate_legal_moves.
    const std::vector<std::vector<int>>& player_stone_place_cnt() const {
        return hm_->history_entry_of(history_entry_ids_.back()).player_stone_place_cnt;
    }

    // Cumulative stones captured so far, indexed [player-1] - same "just the
    // last history entry's snapshot" pattern as player_stone_place_cnt()
    // above: make_move() computes each new entry's count directly, and
    // withdraw_move() rewinds it for free simply by popping.
    const std::vector<int>& capture_count() const {
        return hm_->history_entry_of(history_entry_ids_.back()).score.capture_count;
    }

    // Flattens every resignation recorded so far, regardless of ply - resigning
    // is permanent and never undone by withdraw_move().
    std::vector<int> resigned_players() const;
    // Mark a player (1-indexed) as resigned: thereafter they may only pass and
    // are excluded from scoring. Recomputes winners immediately.
    void resign(int player);
    // Auto-pass on behalf of any resigned player whose turn it is, until a
    // non-resigned player is to move or the game ends.
    void advance_resigned();

    // Public fields accessed by features, MCTS, self-play
    int num_stones;
    int num_players;
    std::vector<TurnInfo> turn_list;
    std::vector<std::vector<std::optional<int>>> player_stone_place_limit; // [stone-1][player-1]; nullopt = unlimited
    // Total placements of each stone color allowed across ALL players combined
    // (unlike player_stone_place_limit); length num_stones, indexed [stone-1];
    // nullopt = unlimited. No separate count field - derived in
    // calculate_legal_moves by summing player_stone_place_cnt[stone-1] across
    // every player.
    std::vector<std::optional<int>> global_stone_place_limit;
    // Stone color -> set of players it scores for (used only for scoring, see
    // compute_player_rewards/compute_winners) - independent of turn_list, so a
    // stone's scoring owner(s) need not be whoever's turn places it. A stone
    // maps to multiple players when each should get its full point value (not
    // split); a stone absent from the map, or mapped to an empty vector,
    // scores for no one. Players are 1-indexed.
    std::unordered_map<int, std::vector<int>> stone_to_player_map;
    bool forced_pass_only;
    std::string score_rule;
    // Per-player scoring handicap added to each non-resigned player's point
    // count before winner determination (compute_winners()) and before the
    // MCTS reward formulas (compute_player_rewards() below,
    // estimate_player_rewards() in evaluator.h) - 1-indexed via komi[p-1].
    std::vector<float> komi;
    // Superko variant enforced by calculate_legal_moves()'s repeat-position
    // check ('positional' | 'situational').
    KoRule ko_rule;
    // Whether a move that leaves the mover's own group with zero liberties is
    // legal (captures that own group immediately, rather than being rejected).
    bool allow_suicide;
    // Max plies before the game auto-ends (see make_move); nullopt = unlimited.
    std::optional<int> max_plies;
    // The turn_list entry for the upcoming ply.
    TurnInfo next_turn;
    std::vector<int> board;
    std::shared_ptr<const std::vector<std::vector<int>>> adj; // shared, read-only
    int N;
    // Only set once game_over() is true (see refresh_winners).
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
    // Ply -> players (1-indexed) who resigned at that ply, in resignation
    // order. Ply-keyed for structural parity with shared/boardState.ts's
    // `resigns` Map; nothing currently reads the ply key itself, only the
    // flattened resigned_players().
    std::map<int, std::vector<int>> resigns_;

    void after_move(MoveInfo move_info, std::vector<std::vector<int>> new_player_stone_place_cnt,
                     std::vector<int> new_capture_count);
    // Pure: computes the legal-move table for the current position (board/history_id_set_/etc.).
    // Gated on all_passed (a parameter, not a call back to this->all_passed()/
    // this->player_stone_place_cnt(): the new HistoryEntry isn't interned yet when after_move()
    // calls this, so those methods would still read the previous ply's entry) rather than
    // game_over(): all_passed only reflects the last move's own intrinsic,
    // never-retroactively-mutated flag, so this method's result for a given ply is fixed forever
    // from the moment it's computed - no caller ever needs to recompute a cached
    // history_entry_ids_ entry, not even the last one. Trade-off: a game that ends via
    // resign()/max_plies (game_over() true, all_passed false, since neither retroactively
    // touches history_entry_ids_) keeps reporting the position's actual board-legal moves here
    // rather than an empty/terminal table - harmless, since make_move() independently blocks
    // further moves via its own game_over() check regardless of what this returns.
    LegalMovesData compute_legal_moves(bool all_passed, const std::vector<std::vector<int>>& player_stone_place_cnt) const;
    // Pure: computes the score (stone count + territory, both per stone type)
    // for the current board.
    ScoreData count_score() const;
    // Pure: computes winners from the given per-stone-type point map (see
    // compute_points()). Only meaningful once game_over() is true (see refresh_winners).
    std::vector<int> compute_winners(const std::unordered_map<int,int>& points) const;
    // Refreshes `winners`: only set once game_over() is true, nullopt otherwise
    // (e.g. after withdraw_move() un-ends a finished game).
    void refresh_winners();
    static BoardState make_copy_skeleton(const BoardState& src);

    // Used by copy() / copy_with_hm() / make_copy_skeleton() without constructor logic
    BoardState() = default;
};
