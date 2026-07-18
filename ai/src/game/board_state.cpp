#include "game/board_state.h"
#include <cassert>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_set>

// ── Reward ────────────────────────────────────────────────────────────────────

std::unordered_map<int,float> compute_player_rewards(
    const std::unordered_map<int,int>& points,
    const std::unordered_map<int, std::vector<int>>& stone_to_player_map,
    const std::vector<float>& komi,
    const std::string& score_rule,
    const std::vector<int>& capture_count)
{
    std::unordered_map<int,int> player_points;
    for (auto& [stone, cnt] : points) {
        auto it = stone_to_player_map.find(stone);
        if (it == stone_to_player_map.end()) continue;
        for (int player : it->second) player_points[player] += cnt;
    }
    for (auto& [stone, players] : stone_to_player_map)
        for (int player : players) player_points.emplace(player, 0);

    // Komi-adjusted (and, under "territory", capture-adjusted) per-player
    // totals - both are folded in before computing `total` (rather than just
    // added to each numerator) so the point-fraction term stays exactly
    // zero-sum across players (see Readme's "Reward" section), matching how
    // BoardState::compute_winners() folds them into its point counts before
    // comparing them.
    std::unordered_map<int,float> adj_points;
    float total = 0.0f;
    bool use_captures = score_rule == "territory";
    for (auto& [p, cnt] : player_points) {
        float v = static_cast<float>(cnt);
        if (p >= 1 && p <= (int)komi.size()) v += komi[p - 1];
        if (use_captures && p >= 1 && p <= (int)capture_count.size()) v += capture_count[p - 1];
        adj_points[p] = v;
        total += v;
    }

    int P = static_cast<int>(player_points.size());

    std::unordered_map<int,float> rewards;
    for (auto& [p, my_cnt] : adj_points) {
        if (P <= 1) { rewards[p] = 0.0f; continue; }

        float sf = (total != 0.0f) ? (my_cnt / total) : 0.0f;

        float rank_diff = 0.0f;
        for (auto& [q, their_cnt] : adj_points) {
            if (q == p) continue;
            if (their_cnt < my_cnt)      rank_diff += 1.0f;
            else if (their_cnt > my_cnt) rank_diff -= 1.0f;
        }
        float rank_reward  = rank_diff * 2.0f / (2 * P - 1);
        float point_reward = (sf - 1.0f / P) / (2 * P - 1);
        rewards[p] = rank_reward + point_reward;
    }
    return rewards;
}

std::string move_to_string(const MoveInfo& m) {
    switch (m.move_type) {
        case MoveType::PLACE: return "PLACE " + std::to_string(m.pos.value_or(-1));
        // Only reflects the all-passed cause specifically (not max_plies/resignation) -
        // debug-only string (see mcts.cpp's DBG_PRINT_TREE), not worth threading full
        // game-over state through for.
        case MoveType::PASS:  return m.all_passed ? "PASS [ALL PASSED]" : "PASS";
        default:              return "NONE";   // NOMOVE
    }
}

// ── HistoryManager ────────────────────────────────────────────────────────────

std::string HistoryManager::make_key(int ply_mod, const std::vector<int>& board) {
    std::string key(board.size() + 1, '\0');
    key[0] = static_cast<char>(ply_mod + 1);
    for (int i = 0; i < (int)board.size(); i++)
        key[i + 1] = static_cast<char>(board[i] + 1);
    return key;
}

uint64_t HistoryManager::store_board(int ply_mod, const std::vector<int>& board) {
    std::string key = make_key(ply_mod, board);
    auto it = key_to_id_.find(key);
    if (it != key_to_id_.end()) return it->second;
    uint64_t id = next_id_++;
    key_to_id_.emplace(std::move(key), id);
    boards_.push_back(board);
    return id;
}

std::optional<uint64_t> HistoryManager::lookup(int ply_mod, const std::vector<int>& board) const {
    std::string key = make_key(ply_mod, board);
    auto it = key_to_id_.find(key);
    if (it == key_to_id_.end()) return std::nullopt;
    return it->second;
}

const std::vector<int>& HistoryManager::board_of(uint64_t id) const {
    assert(id < boards_.size());
    return boards_[id];
}

uint64_t HistoryManager::store_history_entry(HistoryEntry entry) {
    uint64_t id = history_entries_.size();
    history_entries_.push_back(std::move(entry));
    return id;
}

const HistoryEntry& HistoryManager::history_entry_of(uint64_t id) const {
    assert(id < history_entries_.size());
    return history_entries_[id];
}

// ── Group / liberty computation ───────────────────────────────────────────────

GroupDict group_liberty(const std::vector<int>& board,
                        const std::vector<std::vector<int>>& adj,
                        int N,
                        const std::vector<int>& friendly_stones) {
    auto is_friendly = [&](int stone) { return friendly_stones[stone - 1] == 1; };

    std::vector<int> aff_color(N, 0);
    std::vector<int> aff_gid(N, 0);
    std::unordered_map<int, std::vector<std::vector<int>>> groups;
    std::unordered_map<int, int> color_gid;

    for (int i = 0; i < N; i++) {
        if (board[i] == 0 || aff_color[i] != 0) continue;
        int color = board[i];
        if (!groups.count(color)) { groups[color] = {}; color_gid[color] = 0; }
        int gid = color_gid[color];
        std::vector<int> allel = {i};
        aff_color[i] = color;
        aff_gid[i]   = gid;
        std::vector<int> stack = {i};
        while (!stack.empty()) {
            std::vector<int> nxt;
            for (int node : stack) {
                const auto& row = adj[node];
                for (int j = 0; j < N; j++) {
                    if (row[j] && aff_color[j] == 0 && board[j] == color) {
                        aff_color[j] = color;
                        aff_gid[j]   = gid;
                        allel.push_back(j);
                        nxt.push_back(j);
                    }
                }
            }
            stack = std::move(nxt);
        }
        groups[color].push_back(std::move(allel));
        color_gid[color] = gid + 1;
    }

    // liberties: empty neighbors, plus occupied neighbors of a different but
    // friendly-this-turn color. non_liberties: occupied neighbors of a
    // different, non-friendly color.
    std::unordered_map<int, std::vector<std::unordered_set<int>>> lib_sets;
    std::unordered_map<int, std::vector<std::unordered_set<int>>> non_lib_sets;
    for (auto& [c, grps] : groups) {
        lib_sets[c].resize(grps.size());
        non_lib_sets[c].resize(grps.size());
    }
    for (int i = 0; i < N; i++) {
        if (aff_color[i] == 0) continue;
        const auto& row = adj[i];
        for (int j = 0; j < N; j++) {
            if (!row[j]) continue;
            if (aff_color[j] == 0) {
                lib_sets[aff_color[i]][aff_gid[i]].insert(j);
            } else if (aff_color[j] != aff_color[i]) {
                if (is_friendly(aff_color[j])) lib_sets[aff_color[i]][aff_gid[i]].insert(j);
                else non_lib_sets[aff_color[i]][aff_gid[i]].insert(j);
            }
        }
    }

    // Note: a group may legitimately have zero liberties here - a protected
    // color (see calculate_legal_moves) can be left on the board at zero
    // liberties instead of being captured.
    GroupDict result;
    for (auto& [c, grps] : groups) {
        GroupLib entries;
        entries.reserve(grps.size());
        for (int idx = 0; idx < (int)grps.size(); idx++) {
            std::vector<int> libs(lib_sets[c][idx].begin(), lib_sets[c][idx].end());
            std::vector<int> non_libs(non_lib_sets[c][idx].begin(), non_lib_sets[c][idx].end());
            entries.push_back({grps[idx], std::move(libs), std::move(non_libs)});
        }
        result[c] = std::move(entries);
    }
    return result;
}

// ── Legal move calculation ────────────────────────────────────────────────────

// The ply_mod component of a HistoryManager (ply_mod, board) key: the real
// turn-cycle position for 'situational' (a repeat only "counts" when the
// player-to-move also matches), or a constant for 'positional' (any repeated
// board is a violation regardless of ply) - mirrors shared/boardState.ts's
// compareState(), which skips the ply-mod comparison entirely under
// 'positional'. Collapsing to a constant is sufficient (rather than needing a
// separate board-only keying scheme) because HistoryManager's key already
// embeds the full board content, so two positional-mode entries only collide
// when their boards are actually identical.
static int ko_ply_mod(KoRule ko_rule, int ply, int ltl) {
    return ko_rule == KoRule::Situational ? (ply % ltl) : 0;
}

// A turn may offer more than one stone color (TurnInfo.stones); legality and
// captures are computed separately for every offered color, since which color
// is placed can change both (protected/friendly are per-stone-color). Always
// returns a num_stones+1-wide table regardless of how many stones are offered
// this turn - unoffered stones are simply all-nullopt.
static LegalMovesData calculate_legal_moves(
    const std::vector<int>& board_in,
    const std::vector<std::vector<int>>& adj,
    const std::vector<TurnInfo>& turn_list,
    const std::unordered_map<uint64_t, int>& history_id_set,
    int len_history,
    const HistoryManager* hm,
    KoRule ko_rule,
    bool allow_suicide,
    int num_stones,
    const std::vector<std::vector<std::optional<int>>>& player_stone_place_limit,
    const std::vector<std::optional<int>>& global_stone_place_limit,
    const std::vector<std::vector<int>>& player_stone_place_cnt)
{
    // Private mutable copy: this function speculatively mutates a working
    // board below and must never touch the caller's live array before a move
    // is actually committed via make_move().
    std::vector<int> board = board_in;
    int N = static_cast<int>(board.size());
    int ltl = static_cast<int>(turn_list.size());
    int turn_idx = (len_history - 1) % ltl;
    const TurnInfo& turn_info = turn_list[turn_idx];

    std::vector<int> offered_stones;
    for (int s = 1; s <= num_stones; s++)
        if (turn_info.stones[s - 1] == 1) offered_stones.push_back(s);

    // A stone the mover has already placed as many times as their per-player
    // limit allows, or that's already been placed (by anyone) as many times as
    // its global limit allows, is treated exactly as if it were never offered
    // this turn. The global count isn't separately tracked - it's derived by
    // summing player_stone_place_cnt[stone-1] across every player.
    {
        std::vector<int> filtered;
        for (int s : offered_stones) {
            auto player_limit = player_stone_place_limit[s - 1][turn_info.player - 1];
            if (player_limit.has_value() && player_stone_place_cnt[s - 1][turn_info.player - 1] >= *player_limit) continue;
            auto global_limit = global_stone_place_limit[s - 1];
            if (global_limit.has_value()) {
                int global_cnt = 0;
                for (int c : player_stone_place_cnt[s - 1]) global_cnt += c;
                if (global_cnt >= *global_limit) continue;
            }
            filtered.push_back(s);
        }
        offered_stones = std::move(filtered);
    }

    const auto& protected_stones = turn_info.is_protected;
    auto is_protected = [&](int stone) { return protected_stones[stone - 1] == 1; };
    const auto& friendly_stones = turn_info.friendly;
    auto is_friendly = [&](int stone) { return friendly_stones[stone - 1] == 1; };

    GroupDict group_dict = group_liberty(board, adj, N, friendly_stones);

    // Nodes captured by a pass: every non-protected zero-liberty group on the
    // board, regardless of color - this doesn't depend on which stone the
    // mover could have played.
    std::vector<int> pass_capture;
    for (auto& [color, groups] : group_dict) {
        if (is_protected(color)) continue;
        for (auto& g : groups)
            if (g.liberties.empty())
                for (int node : g.group) pass_capture.push_back(node);
    }

    LegalMovesByStone captures(num_stones + 1, LegalMoves(N, std::nullopt));
    std::vector<std::vector<int>> legals_for_stone(num_stones + 1);
    std::vector<std::vector<int>> legals_for_location(N);
    int place_legals = 0;

    for (int next_player : offered_stones) {
        // A group may be sitting at zero liberties because it was protected on
        // some earlier turn and was never actually removed. If it's no longer
        // protected and isn't this candidate's own color, it's captured by
        // whichever move actually gets played, regardless of position -
        // equivalent to (and cheaply derived from) pass_capture, minus this
        // candidate's own color's contribution.
        std::unordered_set<int> early_opp_capture;
        for (int n : pass_capture) if (board[n] != next_player) early_opp_capture.insert(n);

        std::vector<GroupEntry*> me;
        std::vector<std::pair<int, GroupEntry*>> other;
        for (auto& [color, groups] : group_dict) {
            for (auto& g : groups) {
                if (color == next_player) me.push_back(&g);
                else other.push_back({color, &g});
            }
        }

        // `me` groups whose only remaining liberty (if any) is illusory: one of
        // their occupied neighbors is in early_opp_capture, which will actually
        // be vacated once applied (see make_move) - so such a group isn't
        // really down to just its current raw liberty count, and connecting to
        // it is safe.
        std::unordered_set<GroupEntry*> early_self_liberation;
        for (auto* g : me) {
            for (int n : g->non_liberties) {
                if (early_opp_capture.count(n)) { early_self_liberation.insert(g); break; }
            }
        }

        for (int i = 0; i < N; i++) {
            if (board[i] != 0) continue;

            // whether the newly placed stone is adjacent to an empty node
            bool i_lib = false;
            for (int j = 0; j < N && !i_lib; j++)
                if (adj[i][j] && board[j] == 0) i_lib = true;

            std::vector<int> nb = board;
            nb[i] = next_player;

            // Theorem: if any opponent group is captured, the move must be
            // legal. (Protected-colored groups are never captured. If the
            // mover's own color is friendly this turn, the new stone doesn't
            // take away anyone's liberty, so no capture happens here at all.)
            bool capture = false;
            std::unordered_set<int> pos_captures;
            if (!is_friendly(next_player)) {
                for (auto& [color, gptr] : other) {
                    if (is_protected(color)) continue;
                    auto& libs = gptr->liberties;
                    bool i_in_libs = std::find(libs.begin(), libs.end(), i) != libs.end();
                    if (i_in_libs && libs.size() == 1) {
                        capture = true;
                        for (int node : gptr->group) { nb[node] = 0; pos_captures.insert(node); }
                    }
                }
            }

            if (!i_lib && !capture) {
                // the new stone may connect previous friendly groups; if any
                // such group has >1 liberty, the move is still legal - or
                // exactly 1 (about to be filled by this placement), if that
                // group is in early_self_liberation.
                bool ok = false;
                for (auto* g : me) {
                    auto& libs = g->liberties;
                    if (std::find(libs.begin(), libs.end(), i) == libs.end()) continue;
                    if (libs.size() > 1) { ok = true; break; }
                    if (libs.size() == 1 && early_self_liberation.count(g)) { ok = true; break; }
                }
                if (!ok) {
                    if (is_protected(next_player)) {
                        // The mover's own color can't be removed either: the
                        // placed stone (and any connected group) simply stays
                        // on the board at zero liberties, legal regardless of
                        // allow_suicide - nothing to capture.
                    } else if (!allow_suicide) {
                        continue;
                    } else {
                        // Suicide: the new stone and any friendly group(s) it
                        // connects to (per the check above, none have another
                        // liberty) form one zero-liberty group - captured
                        // immediately, mirroring the opponent-capture case above.
                        nb[i] = 0;
                        pos_captures.insert(i);
                        for (auto* g : me) {
                            auto& libs = g->liberties;
                            if (std::find(libs.begin(), libs.end(), i) == libs.end()) continue;
                            for (int node : g->group) { nb[node] = 0; pos_captures.insert(node); }
                        }
                    }
                }
            }

            // Ko check: speculative lookup only - do not store the candidate board.
            auto new_id_opt = hm->lookup(ko_ply_mod(ko_rule, len_history, ltl), nb);
            if (new_id_opt.has_value() && history_id_set.count(*new_id_opt)) continue;

            // A leftover zero-liberty `me` group might additionally be freed by
            // THIS candidate's own captures - e.g. one of its occupied
            // neighbors is an opponent group this specific placement captures.
            // If none of its neighbors were freed (by this move's own captures
            // or by early_opp_capture), it's still dead and is captured too.
            std::unordered_set<int> early_self_captures;
            if (!is_protected(next_player)) {
                for (auto* g : me) {
                    if (!g->liberties.empty()) continue;
                    bool freed = false;
                    for (int n : g->non_liberties) {
                        if (pos_captures.count(n) || early_opp_capture.count(n)) { freed = true; break; }
                    }
                    if (!freed) for (int node : g->group) early_self_captures.insert(node);
                }
            }

            std::unordered_set<int> all_captures;
            all_captures.insert(pos_captures.begin(), pos_captures.end());
            all_captures.insert(early_self_captures.begin(), early_self_captures.end());
            all_captures.insert(early_opp_capture.begin(), early_opp_capture.end());

            captures[next_player][i] = std::vector<int>(all_captures.begin(), all_captures.end());
            legals_for_stone[next_player].push_back(i);
            legals_for_location[i].push_back(next_player);
            place_legals++;
        }
    }

    return LegalMovesData{ std::move(pass_capture), std::move(captures),
                            std::move(legals_for_stone), std::move(legals_for_location), place_legals };
}

// ── BoardState ────────────────────────────────────────────────────────────────

void BoardState::after_move(MoveInfo move_info, std::vector<std::vector<int>> new_player_stone_place_cnt,
                             std::vector<int> new_capture_count) {
    int ply = static_cast<int>(history_ids_.size());
    int ltl = static_cast<int>(turn_list.size());
    uint64_t board_id = hm_->store_board(ko_ply_mod(ko_rule, ply, ltl), board);
    history_ids_.push_back(board_id);
    history_id_set_[board_id]++;
    // Takes move_info.all_passed/new_player_stone_place_cnt as parameters (rather than reading
    // them back via all_passed()/player_stone_place_cnt()) since the new HistoryEntry isn't
    // interned yet - it's only stored once, below, alongside score.
    LegalMovesData legal_moves = compute_legal_moves(move_info.all_passed, new_player_stone_place_cnt);
    // count_score() is pure/board-only (like TS's territory flood-fill), so
    // capture_count - a running total, not derivable from the board - is
    // merged in from the caller, the same way new_player_stone_place_cnt is.
    ScoreData sd = count_score();
    sd.capture_count = std::move(new_capture_count);
    HistoryEntry entry{std::move(move_info), std::move(legal_moves), std::move(sd),
                        std::move(new_player_stone_place_cnt)};
    history_entry_ids_.push_back(hm_->store_history_entry(std::move(entry)));
    refresh_winners();
}

LegalMovesData BoardState::compute_legal_moves(bool all_passed, const std::vector<std::vector<int>>& player_stone_place_cnt) const {
    if (all_passed) {
        return LegalMovesData{
            {}, LegalMovesByStone(num_stones + 1, LegalMoves(N, std::nullopt)),
            std::vector<std::vector<int>>(num_stones + 1), std::vector<std::vector<int>>(N), 0,
        };
    }
    return calculate_legal_moves(
        board, *adj, turn_list, history_id_set_,
        static_cast<int>(history_ids_.size()), hm_, ko_rule, allow_suicide, num_stones,
        player_stone_place_limit, global_stone_place_limit, player_stone_place_cnt);
}

// Territory is found by flood-filling each maximal connected region of empty
// nodes (same multi-frontier BFS pattern as group_liberty): a region belongs
// to a stone type only if every node bordering it is that same type; regions
// bordering zero or several distinct types are neutral (dame) and score nobody.
ScoreData BoardState::count_score() const {
    ScoreData sd;
    for (int s = 1; s <= num_stones; s++) {
        sd.stone_count[s] = static_cast<int>(std::count(board.begin(), board.end(), s));
        sd.territory[s] = 0;
    }
    sd.territory_owner.assign(N, 0);

    const auto& adj_ = *adj;
    std::vector<bool> visited(N, false);
    for (int i = 0; i < N; i++) {
        if (board[i] != 0 || visited[i]) continue;
        std::vector<int> region = {i};
        visited[i] = true;
        std::unordered_set<int> border_stones;
        std::vector<int> stack = {i};
        while (!stack.empty()) {
            std::vector<int> nxt;
            for (int node : stack) {
                const auto& row = adj_[node];
                for (int j = 0; j < N; j++) {
                    if (!row[j]) continue;
                    if (board[j] != 0) { border_stones.insert(board[j]); continue; }
                    if (!visited[j]) { visited[j] = true; region.push_back(j); nxt.push_back(j); }
                }
            }
            stack = std::move(nxt);
        }
        if (border_stones.size() == 1) {
            int stone = *border_stones.begin();
            sd.territory[stone] += static_cast<int>(region.size());
            for (int node : region) sd.territory_owner[node] = stone;
        }
    }

    return sd;
}

std::unordered_map<int,int> BoardState::compute_points(const std::string& rule, const ScoreData& score) {
    std::unordered_map<int,int> points;
    for (auto& [s, stone_count] : score.stone_count) {
        auto it = score.territory.find(s);
        int territory = (it != score.territory.end()) ? it->second : 0;
        int pts;
        if (rule == "stone") pts = stone_count;
        else if (rule == "territoryonly" || rule == "territory") pts = territory;
        else pts = stone_count + territory;  // "area" (and unrecognized values)
        points[s] = pts;
    }
    return points;
}

std::vector<int> BoardState::compute_winners(const std::unordered_map<int,int>& points) const {
    auto resigned = resigned_players();
    auto is_resigned = [&](int p) { return std::find(resigned.begin(), resigned.end(), p) != resigned.end(); };

    std::unordered_map<int,float> player_count;
    for (int p = 1; p <= num_players; p++)
        if (!is_resigned(p)) player_count[p] = (p >= 1 && p <= (int)komi.size()) ? komi[p - 1] : 0.0f;
    if (score_rule == "territory") {
        const auto& cc = capture_count();
        for (int p = 1; p <= num_players; p++)
            if (player_count.count(p) && p >= 1 && p <= (int)cc.size()) player_count[p] += cc[p - 1];
    }
    for (auto& [stone, cnt] : points) {
        auto it = stone_to_player_map.find(stone);
        if (it == stone_to_player_map.end()) continue;
        for (int player : it->second)
            if (player_count.count(player)) player_count[player] += cnt;   // resigned players excluded above
    }
    // Flooring the max at 0 is safe here: komi is required to be >= 0 (see the
    // BoardState constructor assert), and stone/territory/capture counts are
    // never negative either, so every player_count value is already >= 0 -
    // this can never hide a genuine (negative) max the way it would if komi
    // were allowed to go negative.
    float max_cnt = 0.0f;
    for (auto& [p, c] : player_count) max_cnt = std::max(max_cnt, c);
    std::vector<int> result;
    for (int p = 1; p <= num_players; p++)
        if (player_count.count(p) && player_count[p] == max_cnt) result.push_back(p);
    return result;
}

void BoardState::refresh_winners() {
    winners = game_over() ? std::optional(compute_winners(compute_points(score_rule, score()))) : std::nullopt;
}

BoardState::BoardState(int num_stones_,
                       int num_players_,
                       std::vector<TurnInfo> turn_list_,
                       std::vector<std::vector<std::optional<int>>> player_stone_place_limit_,
                       std::vector<std::optional<int>> global_stone_place_limit_,
                       std::unordered_map<int, std::vector<int>> stone_to_player_map_,
                       bool forced_pass_only_,
                       std::string score_rule_,
                       std::vector<float> komi_,
                       KoRule ko_rule_,
                       bool allow_suicide_,
                       std::optional<int> max_plies_,
                       std::vector<int> initial_board,
                       const BoardConfig& bc)
    : num_stones(num_stones_),
      num_players(num_players_),
      turn_list(std::move(turn_list_)),
      player_stone_place_limit(std::move(player_stone_place_limit_)),
      global_stone_place_limit(std::move(global_stone_place_limit_)),
      stone_to_player_map(std::move(stone_to_player_map_)),
      forced_pass_only(forced_pass_only_),
      score_rule(std::move(score_rule_)),
      komi(std::move(komi_)),
      ko_rule(ko_rule_),
      allow_suicide(allow_suicide_),
      max_plies(max_plies_),
      board(std::move(initial_board)),
      adj(std::make_shared<std::vector<std::vector<int>>>(bc.adj)),
      N(bc.N)
{
    assert(num_stones > 0 && "num_stones must be > 0");
    assert(!turn_list.empty() && "turn_list must be non-empty");
    for (const auto& t : turn_list) {
        assert(t.player >= 1 && t.player <= num_players && "turn_list player out of range [1, num_players]");
        assert((int)t.stones.size() == num_stones && "turn_list stones length must equal num_stones");
        assert((int)t.is_protected.size() == num_stones && "turn_list is_protected length must equal num_stones");
        assert((int)t.friendly.size() == num_stones && "turn_list friendly length must equal num_stones");
        bool any_offered = false;
        for (int v : t.stones) { assert((v == 0 || v == 1) && "turn_list stones values must be 0 or 1"); if (v == 1) any_offered = true; }
        assert(any_offered && "turn_list stones must have at least one available stone");
        for (int v : t.is_protected) assert((v == 0 || v == 1) && "turn_list is_protected values must be 0 or 1");
        for (int v : t.friendly) assert((v == 0 || v == 1) && "turn_list friendly values must be 0 or 1");
    }
    assert((int)player_stone_place_limit.size() == num_stones && "player_stone_place_limit length must equal num_stones");
    for (auto& row : player_stone_place_limit) {
        assert((int)row.size() == num_players && "player_stone_place_limit sublist length must equal num_players");
        for (auto& v : row) assert((!v.has_value() || *v >= 0) && "player_stone_place_limit values must be nullopt or non-negative");
    }
    assert((int)global_stone_place_limit.size() == num_stones && "global_stone_place_limit length must equal num_stones");
    for (auto& v : global_stone_place_limit) assert((!v.has_value() || *v >= 0) && "global_stone_place_limit values must be nullopt or non-negative");
    assert((!max_plies.has_value() || *max_plies >= 1) && "max_plies must be nullopt or a positive integer");
    assert(std::all_of(komi.begin(), komi.end(), [](float k) { return k >= 0.0f; }) && "komi values must be >= 0");

    owned_hm_ = std::make_unique<HistoryManager>();
    hm_ = owned_hm_.get();
    next_turn = turn_list[0];
    after_move(
        MoveInfo{MoveType::NOMOVE, std::nullopt, std::nullopt, {}, 0, false},
        std::vector<std::vector<int>>(num_stones, std::vector<int>(num_players, 0)),
        std::vector<int>(num_players, 0));
}

bool BoardState::no_trad_legal() const {
    return legal_moves_data().place_legals == 0;
}

std::vector<std::pair<int,int>> BoardState::legal_move_list() const {
    std::vector<std::pair<int,int>> result;
    const auto& lfl = legal_moves_data().legals_for_location;
    for (int pos = 0; pos < (int)lfl.size(); pos++)
        for (int stone : lfl[pos]) result.push_back({pos, stone});
    return result;
}

std::vector<int> BoardState::resigned_players() const {
    std::vector<int> result;
    for (auto& [ply, players] : resigns_)
        for (int p : players) result.push_back(p);
    return result;
}

void BoardState::resign(int player) {
    auto rp = resigned_players();
    if (std::find(rp.begin(), rp.end(), player) != rp.end()) return;
    int ply = static_cast<int>(history_ids_.size()) - 1;
    resigns_[ply].push_back(player);
    // Resigning may end the game (too few non-resigned players remain) or
    // change who counts as a winner if it's already over - game_over()
    // checks resigned_players() live, so nothing needs stamping onto a move.
    refresh_winners();
}

void BoardState::advance_resigned() {
    while (!game_over()) {
        auto rp = resigned_players();
        if (std::find(rp.begin(), rp.end(), next_turn.player) == rp.end()) break;
        if (!make_move(std::nullopt)) break;
    }
}

bool BoardState::make_move(std::optional<int> k, std::optional<int> stone) {
    // No computation needed here: history_entry_ids_'s last entry is
    // never stale (see compute_legal_moves()), so there's nothing to refresh.
    if (game_over()) return false;
    // A resigned player may only pass, and always may (ignoring forced_pass_only).
    auto rp = resigned_players();
    bool resigned = std::find(rp.begin(), rp.end(), next_turn.player) != rp.end();
    if (resigned && k.has_value()) return false;

    int consecutive_passes = 0;
    if (!k.has_value()) {
        if (!resigned && forced_pass_only && !no_trad_legal()) return false;
        consecutive_passes = last_move().consecutive_passes + 1;
        if (consecutive_passes >= (int)turn_list.size()) {
            next_turn = turn_list[history_ids_.size() % turn_list.size()];
            MoveInfo move_info{MoveType::PASS, std::nullopt, std::nullopt, {}, consecutive_passes, true};
            after_move(std::move(move_info), player_stone_place_cnt(), capture_count());
            return true;
        }
    } else {
        if (no_trad_legal()) return false;
        if (!stone.has_value()) {
            std::vector<int> offered_idx;
            for (int i = 0; i < (int)next_turn.stones.size(); i++)
                if (next_turn.stones[i] == 1) offered_idx.push_back(i);
            if (offered_idx.size() != 1) return false;
            stone = offered_idx[0] + 1;
        }
        if (next_turn.stones[*stone - 1] != 1) return false;
        if (!legal_moves_data().captures[*stone][*k].has_value()) return false;
    }

    // A PLACE move's captures are fully precomputed (see calculate_legal_moves);
    // a (non-terminal) pass's captures are simply pass_capture.
    const std::vector<int>& captures = !k.has_value() ? legal_moves_data().pass_capture
                                                        : *legal_moves_data().captures[*stone][*k];
    std::vector<int> nb = board;
    if (k.has_value()) nb[*k] = *stone;
    for (int c : captures) nb[c] = 0;
    board = std::move(nb);
    // Must read next_turn.player (the mover) before it's reassigned below.
    auto new_cnt = player_stone_place_cnt();
    if (k.has_value()) new_cnt[*stone - 1][next_turn.player - 1]++;
    auto new_capture_cnt = capture_count();
    new_capture_cnt[next_turn.player - 1] += static_cast<int>(captures.size());
    next_turn = turn_list[history_ids_.size() % turn_list.size()];
    MoveType move_type = k.has_value() ? MoveType::PLACE : MoveType::PASS;
    // all_passed is always false here: this branch is only reached for a PLACE move or a
    // non-round-completing PASS (the round-completing case returns early above) - max_plies
    // ending the game is handled entirely by game_over()'s own live check, no stamping needed.
    MoveInfo move_info{move_type, k, k.has_value() ? stone : std::nullopt, captures,
                        k.has_value() ? 0 : consecutive_passes, false};
    after_move(std::move(move_info), std::move(new_cnt), std::move(new_capture_cnt));
    return true;
}

void BoardState::withdraw_move() {
    if ((int)history_ids_.size() <= 1) return;
    uint64_t removed_id = history_ids_.back();
    history_ids_.pop_back();
    if (--history_id_set_[removed_id] == 0) history_id_set_.erase(removed_id);
    history_entry_ids_.pop_back();
    int prev_ply = static_cast<int>(history_ids_.size()) - 1;
    board = hm_->board_of(history_ids_.back());
    next_turn = turn_list[prev_ply % turn_list.size()];
    // No recompute needed: history_entry_ids_'s entries never go stale (see
    // compute_legal_moves()) - game_over()'s resigned_players()/max_plies checks are
    // live/derived rather than stamped onto MoveInfo, and all_passed is a
    // per-ply-intrinsic fact that's never retroactively mutated, so the entry
    // this pop uncovers is already exactly correct. A resignation is permanent
    // (never un-resigned by withdraw_move) and game_over() reflects that live too.
    refresh_winners();
}

BoardState BoardState::make_copy_skeleton(const BoardState& src) {
    BoardState c;
    c.num_stones          = src.num_stones;
    c.num_players         = src.num_players;
    c.turn_list           = src.turn_list;
    c.player_stone_place_limit = src.player_stone_place_limit;
    c.global_stone_place_limit = src.global_stone_place_limit;
    c.stone_to_player_map = src.stone_to_player_map;
    c.forced_pass_only    = src.forced_pass_only;
    c.score_rule          = src.score_rule;
    c.komi                = src.komi;
    c.ko_rule              = src.ko_rule;
    c.allow_suicide       = src.allow_suicide;
    c.max_plies           = src.max_plies;
    c.board               = src.board;
    c.adj                 = src.adj;
    c.N                   = src.N;
    c.winners             = src.winners;
    c.resigns_             = src.resigns_;
    c.next_turn            = src.next_turn;
    // history_entry_ids_ is handled by the caller (copy / copy_with_hm),
    // mirroring how history_ids_ is rebuilt per copy-variant.
    return c;
}

BoardState BoardState::copy() const {
    BoardState c = make_copy_skeleton(*this);
    c.hm_            = hm_;
    c.history_ids_   = history_ids_;
    c.history_id_set_ = history_id_set_;
    // Same manager: the interned history-entry IDs stay valid, so just copy the list.
    c.history_entry_ids_ = history_entry_ids_;
    return c;
}

BoardState BoardState::copy_with_hm(HistoryManager* new_hm) const {
    BoardState c = make_copy_skeleton(*this);
    c.hm_ = new_hm;
    int ltl = static_cast<int>(turn_list.size());
    for (int i = 0; i < (int)history_ids_.size(); i++) {
        uint64_t new_id = new_hm->store_board(ko_ply_mod(ko_rule, i, ltl), hm_->board_of(history_ids_[i]));
        c.history_ids_.push_back(new_id);
        c.history_id_set_[new_id]++;
    }
    // Re-store the interned history entries into the new manager, mirroring the
    // board re-interning above (IDs are manager-specific so they must be rebuilt).
    c.history_entry_ids_.reserve(history_entry_ids_.size());
    for (uint64_t id : history_entry_ids_)
        c.history_entry_ids_.push_back(new_hm->store_history_entry(hm_->history_entry_of(id)));
    return c;
}
