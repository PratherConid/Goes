#include "game/board_state.h"
#include <cassert>
#include <algorithm>
#include <numeric>
#include <cmath>

// ── Reward ────────────────────────────────────────────────────────────────────

std::unordered_map<int,float> compute_player_rewards(
    const std::unordered_map<int,int>& stone_count,
    const std::unordered_map<int,int>& stone_to_player_map)
{
    std::unordered_map<int,int> player_stones;
    int total = 0;
    for (auto& [stone, cnt] : stone_count) {
        auto it = stone_to_player_map.find(stone);
        if (it == stone_to_player_map.end()) continue;
        player_stones[it->second] += cnt;
        total += cnt;
    }
    for (auto& [stone, player] : stone_to_player_map)
        player_stones.emplace(player, 0);

    int P = static_cast<int>(player_stones.size());

    std::unordered_map<int,float> rewards;
    for (auto& [p, my_cnt] : player_stones) {
        if (P <= 1) { rewards[p] = 0.0f; continue; }

        float sf = (total > 0) ? (static_cast<float>(my_cnt) / total) : 0.0f;

        int rank_diff = 0;
        for (auto& [q, their_cnt] : player_stones) {
            if (q == p) continue;
            if (their_cnt < my_cnt)      rank_diff++;
            else if (their_cnt > my_cnt) rank_diff--;
        }
        float rank_reward  = static_cast<float>(rank_diff) * 2.0f / (2 * P - 1);
        float stone_reward = std::tanh(sf - 1.0f / P) / (2 * P - 1);
        rewards[p] = rank_reward + stone_reward;
    }
    return rewards;
}

// ── HistoryManager ────────────────────────────────────────────────────────────

std::string HistoryManager::make_key(int ply_mod, const std::vector<int>& board) {
    std::string key(board.size() + 1, '\0');
    key[0] = static_cast<char>(ply_mod + 1);
    for (int i = 0; i < (int)board.size(); i++)
        key[i + 1] = static_cast<char>(board[i] + 1);
    return key;
}

uint64_t HistoryManager::intern(int ply_mod, const std::vector<int>& board) {
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

// ── Group / liberty computation ───────────────────────────────────────────────

GroupDict group_liberty(const std::vector<int>& board,
                        const std::vector<std::vector<int>>& adj,
                        int N) {
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

    std::unordered_map<int, std::vector<std::unordered_set<int>>> lib_sets;
    for (auto& [c, grps] : groups) lib_sets[c].resize(grps.size());
    for (int i = 0; i < N; i++) {
        if (aff_color[i] != 0) continue;
        const auto& row = adj[i];
        for (int j = 0; j < N; j++) {
            if (row[j] && aff_color[j] != 0)
                lib_sets[aff_color[j]][aff_gid[j]].insert(i);
        }
    }

    GroupDict result;
    for (auto& [c, grps] : groups) {
        GroupLib entries;
        entries.reserve(grps.size());
        for (int idx = 0; idx < (int)grps.size(); idx++) {
            assert(!lib_sets[c][idx].empty() && "group has no liberties");
            std::vector<int> libs(lib_sets[c][idx].begin(), lib_sets[c][idx].end());
            entries.push_back({grps[idx], std::move(libs)});
        }
        result[c] = std::move(entries);
    }
    return result;
}

// ── Legal move calculation ────────────────────────────────────────────────────

static LegalMoves calculate_legal_moves(
    const std::vector<int>& board,
    const std::vector<std::vector<int>>& adj,
    const std::vector<int>& turn_stone_list,
    const std::unordered_map<uint64_t, int>& history_id_set,
    int len_history,
    const HistoryManager* hm)
{
    int N = static_cast<int>(board.size());
    LegalMoves legal(N, std::nullopt);
    auto gdict = group_liberty(board, adj, N);
    int ltl = static_cast<int>(turn_stone_list.size());
    int next_player = turn_stone_list[(len_history - 1) % ltl];

    std::vector<std::pair<std::vector<int>, std::vector<int>>> me, other;
    for (auto& [color, entries] : gdict) {
        if (color == next_player)
            for (auto& e : entries) me.push_back(e);
        else
            for (auto& e : entries) other.push_back(e);
    }

    for (int i = 0; i < N; i++) {
        if (board[i] != 0) continue;

        bool i_lib = false;
        for (int j = 0; j < N && !i_lib; j++)
            if (adj[i][j] && board[j] == 0) i_lib = true;

        std::vector<int> nb = board;
        nb[i] = next_player;

        bool capture = false;
        std::vector<int> captures;
        for (auto& [group, libs] : other) {
            bool i_in_libs = (std::find(libs.begin(), libs.end(), i) != libs.end());
            if (i_in_libs && libs.size() == 1) {
                capture = true;
                for (int node : group) { nb[node] = 0; captures.push_back(node); }
            }
        }

        if (!i_lib && !capture) {
            bool ok = false;
            for (auto& [group, libs] : me) {
                bool i_in_libs = (std::find(libs.begin(), libs.end(), i) != libs.end());
                if (i_in_libs && libs.size() > 1) { ok = true; break; }
            }
            if (!ok) continue;
        }

        // Ko check: speculative lookup only — do not intern the candidate board.
        auto new_id_opt = hm->lookup(len_history % ltl, nb);
        if (new_id_opt.has_value() && history_id_set.count(*new_id_opt)) continue;

        legal[i] = std::move(captures);
    }
    return legal;
}

// ── BoardState ────────────────────────────────────────────────────────────────

void BoardState::add_to_history_and_after_move() {
    int ply = static_cast<int>(history_ids_.size());
    int ltl = static_cast<int>(turn_stone_list.size());
    uint64_t id = hm_->intern(ply % ltl, board);
    history_ids_.push_back(id);
    history_id_set_[id]++;
    after_move();
    legal_move_history_.push_back(legal_moves_with_take);
}

void BoardState::after_move() {
    if (last_move().move_type == MoveType::GAMEOVER)
        legal_moves_with_take.assign(N, std::nullopt);
    else
        legal_moves_with_take = calculate_legal_moves(
            board, *adj, turn_stone_list, history_id_set_,
            static_cast<int>(history_ids_.size()), hm_);
    count_stones();
}

void BoardState::count_stones() {
    stone_count.clear();
    for (int s = 1; s <= num_stones; s++)
        stone_count[s] = static_cast<int>(std::count(board.begin(), board.end(), s));
    std::unordered_map<int,int> player_count;
    for (int p = 1; p <= num_players; p++) player_count[p] = 0;
    for (auto& [stone, cnt] : stone_count) {
        auto it = stone_to_player_map.find(stone);
        if (it != stone_to_player_map.end())
            player_count[it->second] += cnt;
    }
    int max_cnt = 0;
    for (auto& [p, c] : player_count) max_cnt = std::max(max_cnt, c);
    winners.clear();
    for (int p = 1; p <= num_players; p++)
        if (player_count[p] == max_cnt) winners.push_back(p);
}

BoardState::BoardState(int num_stones_,
                       int num_players_,
                       std::vector<int> tsl,
                       std::unordered_map<int,int> s2p,
                       bool forced_,
                       std::vector<int> initial_board,
                       const BoardConfig& bc)
    : num_stones(num_stones_),
      num_players(num_players_),
      turn_stone_list(std::move(tsl)),
      stone_to_player_map(std::move(s2p)),
      forced_pass_only(forced_),
      board(std::move(initial_board)),
      adj(std::make_shared<std::vector<std::vector<int>>>(bc.adj)),
      N(bc.N)
{
    assert(num_stones > 0 && "num_stones must be > 0");
    assert(!turn_stone_list.empty() && "turn_stone_list must be non-empty");
    for (int p : turn_stone_list)
        assert(p >= 1 && p <= num_stones && "turn_stone_list entry out of range [1, num_stones]");
    owned_hm_ = std::make_unique<HistoryManager>();
    hm_ = owned_hm_.get();
    next_player = turn_stone_list[0];
    last_moves_.push_back({MoveType::NOMOVE, std::nullopt, {}, {}});
    add_to_history_and_after_move();
}

bool BoardState::no_trad_legal() const {
    return std::all_of(legal_moves_with_take.begin(), legal_moves_with_take.end(),
                       [](const LegalMove& m) { return !m.has_value(); });
}

std::vector<int> BoardState::legal_move_list() const {
    std::vector<int> result;
    for (int i = 0; i < (int)legal_moves_with_take.size(); i++)
        if (legal_moves_with_take[i].has_value()) result.push_back(i);
    return result;
}

bool BoardState::make_move(std::optional<int> k) {
    if (last_move().move_type == MoveType::GAMEOVER) {
        after_move();
        return false;
    }
    int ltl = static_cast<int>(turn_stone_list.size());

    if (!k.has_value()) {  // pass
        if (forced_pass_only && !no_trad_legal()) return false;
        auto passed = last_move().passed_players;
        passed.insert(next_player);
        std::unordered_set<int> unique_stones(turn_stone_list.begin(), turn_stone_list.end());
        if (passed.size() >= unique_stones.size()) {
            next_player = turn_stone_list[static_cast<int>(history_ids_.size()) % ltl];
            last_moves_.push_back({MoveType::GAMEOVER, std::nullopt, {}, passed});
            add_to_history_and_after_move();
            return true;
        }
        next_player = turn_stone_list[static_cast<int>(history_ids_.size()) % ltl];
        last_moves_.push_back({MoveType::PASS, std::nullopt, {}, passed});
        add_to_history_and_after_move();
        return true;
    }

    int pos = k.value();
    if (no_trad_legal()) return false;
    if (pos < 0 || pos >= N) return false;
    const LegalMove& cap = legal_moves_with_take[pos];
    if (!cap.has_value()) return false;

    std::vector<int> nb = board;
    nb[pos] = next_player;
    for (int c : cap.value()) nb[c] = 0;
    board = std::move(nb);
    next_player = turn_stone_list[static_cast<int>(history_ids_.size()) % ltl];
    last_moves_.push_back({MoveType::PLACE, pos, cap.value(), {}});
    add_to_history_and_after_move();
    return true;
}

void BoardState::retract_move() {
    if ((int)history_ids_.size() <= 1) return;
    uint64_t removed_id = history_ids_.back();
    history_ids_.pop_back();
    if (--history_id_set_[removed_id] == 0) history_id_set_.erase(removed_id);
    last_moves_.pop_back();
    legal_move_history_.pop_back();
    int prev_ply = static_cast<int>(history_ids_.size()) - 1;
    int ltl = static_cast<int>(turn_stone_list.size());
    board = hm_->board_of(history_ids_.back());
    next_player = turn_stone_list[prev_ply % ltl];
    after_move();
}

void BoardState::set_ply(int ply) {
    int ltl = static_cast<int>(turn_stone_list.size());
    int extra = ply % ltl;
    std::vector<int> dummy(N, 0);
    for (int i = 0; i < extra; i++) {
        int p = static_cast<int>(history_ids_.size());
        uint64_t id = hm_->intern(p % ltl, dummy);
        history_ids_.push_back(id);
        history_id_set_[id]++;
    }
    next_player = turn_stone_list[(static_cast<int>(history_ids_.size()) - 1) % ltl];
    after_move();
}

BoardState BoardState::make_copy_skeleton(const BoardState& src) {
    BoardState c;
    c.num_stones          = src.num_stones;
    c.num_players         = src.num_players;
    c.turn_stone_list     = src.turn_stone_list;
    c.stone_to_player_map = src.stone_to_player_map;
    c.forced_pass_only    = src.forced_pass_only;
    c.next_player         = src.next_player;
    c.board               = src.board;
    c.adj                 = src.adj;
    c.N                   = src.N;
    c.stone_count         = src.stone_count;
    c.winners             = src.winners;
    c.last_moves_         = src.last_moves_;
    c.legal_moves_with_take.reserve(src.legal_moves_with_take.size());
    for (const auto& m : src.legal_moves_with_take)
        c.legal_moves_with_take.push_back(m);
    c.legal_move_history_.reserve(src.legal_move_history_.size());
    for (const auto& row : src.legal_move_history_) {
        LegalMoves row_copy;
        row_copy.reserve(row.size());
        for (const auto& m : row) row_copy.push_back(m);
        c.legal_move_history_.push_back(std::move(row_copy));
    }
    return c;
}

BoardState BoardState::copy() const {
    BoardState c = make_copy_skeleton(*this);
    c.hm_            = hm_;
    c.history_ids_   = history_ids_;
    c.history_id_set_ = history_id_set_;
    return c;
}

BoardState BoardState::copy_with_hm(HistoryManager* new_hm) const {
    BoardState c = make_copy_skeleton(*this);
    c.hm_ = new_hm;
    int ltl = static_cast<int>(turn_stone_list.size());
    for (int i = 0; i < (int)history_ids_.size(); i++) {
        uint64_t new_id = new_hm->intern(i % ltl, hm_->board_of(history_ids_[i]));
        c.history_ids_.push_back(new_id);
        c.history_id_set_[new_id]++;
    }
    return c;
}
