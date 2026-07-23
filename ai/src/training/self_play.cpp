#include "training/self_play.h"
#include "model/features.h"
#include <chrono>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <random>
#include <cstdint>

static std::mt19937 rng(std::random_device{}());

using json = nlohmann::json;

bool weak_equal(const GameConfig& a, const GameConfig& b) {
    if (a.board_type != b.board_type) return false;
    if (a.board_args != b.board_args) return false;
    if (a.num_stones != b.num_stones) return false;
    if (a.num_players != b.num_players) return false;
    if (a.turn_list.size() != b.turn_list.size()) return false;
    for (size_t i = 0; i < a.turn_list.size(); i++) {
        const auto& ta = a.turn_list[i];
        const auto& tb = b.turn_list[i];
        if (ta.player != tb.player || ta.stones != tb.stones ||
            ta.is_protected != tb.is_protected || ta.friendly != tb.friendly)
            return false;
    }
    // Compare as sorted (stone,player) pair sets - order within one stone's
    // player list doesn't matter (matches the old model_tag()'s own flattening).
    auto flatten = [](const std::unordered_map<int, std::vector<int>>& m) {
        std::vector<std::pair<int,int>> pairs;
        for (auto& [stone, players] : m)
            for (int p : players) pairs.push_back({stone, p});
        std::sort(pairs.begin(), pairs.end());
        return pairs;
    };
    if (flatten(a.stone_to_player_map) != flatten(b.stone_to_player_map)) return false;
    if (a.forced_pass_only != b.forced_pass_only) return false;
    if (a.allow_suicide != b.allow_suicide) return false;
    if (a.score_rule != b.score_rule) return false;
    if (a.komi != b.komi) return false;
    if (a.ko_rule != b.ko_rule) return false;
    return true;
}

bool strong_equal(const GameConfig& a, const GameConfig& b) {
    return a.to_json() == b.to_json();
}

nlohmann::json GameConfig::to_json() const {
    json j;
    j["boardType"]  = board_type;
    j["boardArgs"]  = board_args;
    j["numStones"]  = num_stones;
    j["numPlayers"] = num_players;

    json tl = json::array();
    for (auto& t : turn_list) {
        json ti;
        ti["player"]    = t.player;
        ti["stones"]    = t.stones;
        ti["protected"] = t.is_protected;
        ti["friendly"]  = t.friendly;
        tl.push_back(std::move(ti));
    }
    j["turnList"] = tl;

    json pspl = json::array();
    for (auto& row : player_stone_place_limit) {
        json r = json::array();
        for (auto& v : row) r.push_back(v.has_value() ? json(*v) : json(nullptr));
        pspl.push_back(std::move(r));
    }
    j["playerStonePlaceLimit"] = pspl;

    json gspl = json::array();
    for (auto& v : global_stone_place_limit) gspl.push_back(v.has_value() ? json(*v) : json(nullptr));
    j["globalStonePlaceLimit"] = gspl;

    json s2p;
    for (auto& [k, v] : stone_to_player_map) s2p[std::to_string(k)] = v;
    j["stoneToPlayerMap"] = s2p;

    j["forcedPassOnly"] = forced_pass_only;
    j["scoreRule"]      = score_rule;
    j["komi"]           = komi;
    j["koRule"]          = (ko_rule == KoRule::Positional) ? "positional" : "situational";
    j["allowSuicide"]   = allow_suicide;
    j["maxPlies"]       = max_plies.has_value() ? json(*max_plies) : json(nullptr);
    return j;
}

// ── Config JSON parsing (matches shared/types.ts's GameConfig.toJSON() shape) ──

static std::vector<TurnInfo> parse_turn_list(const json& j) {
    std::vector<TurnInfo> out;
    for (auto& t : j) {
        TurnInfo ti;
        ti.player = t["player"].get<int>();
        ti.stones = t["stones"].get<std::vector<int>>();
        ti.is_protected = t.value("protected", std::vector<int>(ti.stones.size(), 0));
        ti.friendly     = t.value("friendly",  std::vector<int>(ti.stones.size(), 0));
        out.push_back(std::move(ti));
    }
    return out;
}

static std::unordered_map<int, std::vector<int>> parse_stone_to_player_map(const json& j) {
    std::unordered_map<int, std::vector<int>> out;
    for (auto& [k, v] : j.items()) out[std::stoi(k)] = v.get<std::vector<int>>();
    return out;
}

static std::vector<std::vector<std::optional<int>>> parse_player_stone_place_limit(
    const json& j, int num_stones, int num_players)
{
    if (j.is_null())
        return std::vector<std::vector<std::optional<int>>>(
            num_stones, std::vector<std::optional<int>>(num_players, std::nullopt));
    std::vector<std::vector<std::optional<int>>> out;
    for (auto& row : j) {
        std::vector<std::optional<int>> r;
        for (auto& v : row) r.push_back(v.is_null() ? std::nullopt : std::optional<int>(v.get<int>()));
        out.push_back(std::move(r));
    }
    return out;
}

static std::vector<std::optional<int>> parse_global_stone_place_limit(const json& j, int num_stones) {
    if (j.is_null()) return std::vector<std::optional<int>>(num_stones, std::nullopt);
    std::vector<std::optional<int>> out;
    for (auto& v : j) out.push_back(v.is_null() ? std::nullopt : std::optional<int>(v.get<int>()));
    return out;
}

static std::optional<int> parse_max_plies(const json& j) {
    return j.is_null() ? std::nullopt : std::optional<int>(j.get<int>());
}

GameConfig parse_game_cfg(const json& cfg) {
    GameConfig game_cfg;
    game_cfg.board_type = cfg["boardType"].get<std::string>();
    game_cfg.board_args = cfg["boardArgs"].get<std::vector<int>>();
    game_cfg.num_stones  = cfg["numStones"].get<int>();
    game_cfg.num_players = cfg["numPlayers"].get<int>();
    game_cfg.turn_list   = parse_turn_list(cfg["turnList"]);
    game_cfg.player_stone_place_limit = parse_player_stone_place_limit(
        cfg.value("playerStonePlaceLimit", json(nullptr)), game_cfg.num_stones, game_cfg.num_players);
    game_cfg.global_stone_place_limit = parse_global_stone_place_limit(
        cfg.value("globalStonePlaceLimit", json(nullptr)), game_cfg.num_stones);
    game_cfg.stone_to_player_map = parse_stone_to_player_map(cfg["stoneToPlayerMap"]);
    game_cfg.forced_pass_only = cfg.value("forcedPassOnly", true);
    game_cfg.score_rule       = cfg.value("scoreRule", std::string("area"));
    game_cfg.komi             = cfg.value("komi", std::vector<float>());
    game_cfg.ko_rule          = (cfg.value("koRule", std::string("situational")) == "positional")
                                     ? KoRule::Positional : KoRule::Situational;
    game_cfg.allow_suicide    = cfg.value("allowSuicide", false);
    game_cfg.max_plies        = parse_max_plies(cfg.value("maxPlies", json(nullptr)));
    return game_cfg;
}

std::vector<std::vector<std::optional<int>>> resolve_player_stone_place_limit(
    const std::vector<std::vector<std::optional<int>>>& limit, int num_stones, int num_players)
{
    if (!limit.empty()) return limit;
    return std::vector<std::vector<std::optional<int>>>(
        num_stones, std::vector<std::optional<int>>(num_players, std::nullopt));
}

std::vector<std::optional<int>> resolve_global_stone_place_limit(
    const std::vector<std::optional<int>>& limit, int num_stones)
{
    if (!limit.empty()) return limit;
    return std::vector<std::optional<int>>(num_stones, std::nullopt);
}

// Number of bits needed to binary-encode any integer in [0, c] - used by
// compute_input_descr(), below, to size the liberty/playerStoneBudget/
// globalStoneBudget feature-block channel counts.
static int bits_for(int c) {
    int b = 0;
    while ((1 << b) <= c) b++;
    return b;
}

nlohmann::json compute_input_descr(const GameConfig& cfg, int N) {
    int ns     = cfg.num_stones;
    int np     = cfg.num_players;
    int tl_len = (int)cfg.turn_list.size();
    int lib_bits = bits_for(N);

    json blocks = json::array();
    int total = 0;

    blocks.push_back(json::array({"stoneOccupancy", ns})); total += ns + 1;
    blocks.push_back(json::array({"legalPlace", ns}));      total += ns;
    blocks.push_back(json::array({"liberty", lib_bits}));    total += lib_bits;
    blocks.push_back(json::array({"groupSize"}));              total += 1;
    blocks.push_back(json::array({"plyMod", tl_len}));          total += tl_len;
    blocks.push_back(json::array({"consectivePassOneHot", tl_len + 1})); total += tl_len + 1;

    auto pspl = resolve_player_stone_place_limit(cfg.player_stone_place_limit, ns, np);
    json player_budget_bits = json::array();
    for (auto& row : pspl) {
        json r = json::array();
        for (auto& lim : row) {
            int bits = lim.has_value() ? bits_for(*lim) : 0;
            r.push_back(bits);
            total += bits;
        }
        player_budget_bits.push_back(std::move(r));
    }
    blocks.push_back(json::array({"playerStoneBudget", player_budget_bits}));

    auto gspl = resolve_global_stone_place_limit(cfg.global_stone_place_limit, ns);
    json global_budget_bits = json::array();
    for (auto& lim : gspl) {
        int bits = lim.has_value() ? bits_for(*lim) : 0;
        global_budget_bits.push_back(bits);
        total += bits;
    }
    blocks.push_back(json::array({"globalStoneBudget", global_budget_bits}));

    return {{"blocks", blocks}, {"totalDims", total}};
}

BoardState new_state(const GameConfig& cfg, const BoardConfig& bc) {
    auto pspl = resolve_player_stone_place_limit(cfg.player_stone_place_limit, cfg.num_stones, cfg.num_players);
    auto gspl = resolve_global_stone_place_limit(cfg.global_stone_place_limit, cfg.num_stones);
    auto komi = cfg.komi;
    if (komi.empty())
        komi = std::vector<float>(cfg.num_players, 0.0f);

    BoardState state(
        cfg.num_stones, cfg.num_players,
        cfg.turn_list, pspl, gspl, cfg.stone_to_player_map,
        cfg.forced_pass_only, cfg.score_rule, komi, cfg.ko_rule, cfg.allow_suicide, cfg.max_plies,
        std::vector<int>(bc.N, 0),
        bc);
    // linear_move_bound combines with (rather than being overridden by) a
    // fixed cfg.max_plies - take whichever bound is tighter.
    if (cfg.linear_move_bound.has_value()) {
        float k1 = cfg.linear_move_bound->first;
        float k2 = cfg.linear_move_bound->second;
        std::uniform_real_distribution<float> dist(k1, k2);
        int sampled = static_cast<int>(dist(rng) * bc.N);
        state.max_plies = state.max_plies.has_value() ? std::min(*state.max_plies, sampled) : sampled;
    }
    return state;
}

nlohmann::json PlyResult::to_json() const {
    json p;
    p["move"]   = move;
    p["stone"]  = stone;
    p["policy"] = policy;
    return p;
}

PlyResult parse_ply_result(const json& j) {
    PlyResult pr;
    pr.move   = j["move"].get<int>();
    pr.stone  = j["stone"].get<int>();
    pr.policy = j["policy"].get<std::vector<float>>();
    return pr;
}

// Converts a stone-type-indexed array (0 = none, 1..num_stones) to an int64 tensor.
static torch::Tensor to_int64_tensor(const std::vector<int>& stones) {
    auto t = torch::zeros({(int64_t)stones.size()}, torch::kInt64);
    auto a = t.accessor<int64_t, 1>();
    for (size_t i = 0; i < stones.size(); i++) a[i] = stones[i];
    return t;
}

GameRecord trajectory_and_result_to_record(
    const std::vector<PlyResult>& trajectory,
    const std::vector<int>& board,
    const std::vector<int>& territory_owner_stone)
{
    auto stone_owner     = to_int64_tensor(board);
    auto territory_owner = to_int64_tensor(territory_owner_stone);

    std::vector<torch::Tensor> feats, masks, policies, history_feats;
    feats.reserve(trajectory.size());
    masks.reserve(trajectory.size());
    policies.reserve(trajectory.size());
    for (auto& pr : trajectory) {
        auto p_tensor = torch::zeros({(int)pr.policy.size()}, torch::kFloat32);
        auto pa = p_tensor.accessor<float, 1>();
        for (int i = 0; i < (int)pr.policy.size(); i++) pa[i] = pr.policy[i];
        feats.push_back(pr.features);
        masks.push_back(pr.legal_mask);
        policies.push_back(p_tensor);
    }

    GameRecord record{
        torch::stack(feats, 0),     // (P, N, F)
        torch::stack(masks, 0),     // (P, numStones*N+1)
        torch::stack(policies, 0),  // (P, numStones*N+1)
        stone_owner,                 // (N,)
        territory_owner,             // (N,)
    };

    // history_features is only populated when the run passed a history_descr (Transformer only) -
    // check the first ply as a stand-in for the whole trajectory (history_descr is either always
    // passed for a whole run, or never).
    if (!trajectory.empty() && trajectory[0].history_features.defined()) {
        history_feats.reserve(trajectory.size());
        for (auto& pr : trajectory) history_feats.push_back(pr.history_features);
        record.history_features = torch::stack(history_feats, 0);  // (P, N, F_hist)
    }

    return record;
}

GameRecord trajectory_to_record(
    const std::vector<PlyResult>& trajectory,
    const GameConfig& cfg,
    const BoardConfig& bc,
    const nlohmann::json& descr,
    const nlohmann::json* history_descr)
{
    BoardState state = new_state(cfg, bc);
    // Replaying a sequence that already completed successfully once - a
    // freshly-sampled linear_move_bound draw could be shorter than this
    // trajectory and prematurely trip game_over(), rejecting legitimate
    // remaining moves. Disable the cap entirely for the replay.
    state.max_plies = std::nullopt;

    std::vector<PlyResult> replayed;
    replayed.reserve(trajectory.size());
    for (auto& pr : trajectory) {
        auto [ft, mask] = board_to_features(state, torch::kCPU, descr);
        PlyResult full = pr;
        full.features   = ft;
        full.legal_mask = mask;
        if (history_descr) full.history_features = board_to_features(state, torch::kCPU, *history_descr).features;
        replayed.push_back(std::move(full));

        std::optional<int> k, stone_opt;
        if (pr.stone != 0) {            // stone==0 means pass (PlyResult's own convention)
            stone_opt = pr.stone;
            k         = pr.move % bc.N; // move is stone-major; position is the low N-range
        }
        bool ok = state.make_move(k, stone_opt);
        assert(ok && "trajectory_to_record: replay move rejected - trajectory/config/input_descr mismatch");
    }
    return trajectory_and_result_to_record(replayed, state.board, state.score().territory_owner);
}

// Not thread-safe: uses the file-scope std::mt19937 rng to seed each MCTS instance.
std::pair<std::vector<PlyResult>, MCTSTiming> generate_one_ply_per_game(
    Evaluator& evaluator,
    const std::vector<BoardState*>& states,
    const nlohmann::json& descr,
    int num_simulations,
    int temperature_threshold,
    float c_puct,
    int verbosity,
    const nlohmann::json* history_descr)
{
    MCTS mcts(evaluator, c_puct, rng());

    std::vector<float> temps;
    temps.reserve(states.size());
    for (auto* s : states)
        temps.push_back(s->ply_count() < temperature_threshold ? 1.0f : 0.0f);

    auto t_search0 = std::chrono::high_resolution_clock::now();
    auto [results, timing] = mcts.search_batch(
        states, num_simulations,
        NoiseConfig{/*add_noise=*/true, 0.3f, 0.25f}, temps);
    double search_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_search0).count();
    timing.search = search_ms / 1000.0;

    std::vector<PlyResult> ply_results;
    ply_results.reserve(states.size());
    for (int j = 0; j < (int)states.size(); j++) {
        auto* s = states[j];
        auto& [policy, move] = results[j];
        auto [ft, mask] = board_to_features(*s, torch::kCPU, descr);
        torch::Tensor hist_ft;
        if (history_descr) hist_ft = board_to_features(*s, torch::kCPU, *history_descr).features;
        int N = s->N, ns = s->num_stones;
        // Decode the flat action (stone-major, matching MCTS::select()'s
        // layout) into (stone, pos) - not knowable before the move, since a
        // turn can now offer several stones.
        std::optional<int> k, stone_opt;
        if (move != ns * N) {
            stone_opt = move / N + 1;
            k         = move % N;
        }

        if (verbosity >= 2) {
            std::string move_str = (move == ns * N) ? "pass" : std::to_string(move);
            std::cout << "  slot=" << j << " ply=" << s->ply_count()
                      << " stone=" << stone_opt.value_or(0) << " move=" << move_str
                      << " p=" << policy[move] << std::endl;
        }

        bool ok = s->make_move(k, stone_opt);
        if (!ok)
            std::cerr << "MCTS returned illegal move: slot=" << j
                      << " ply=" << s->ply_count() << " move=" << move << std::endl;

        ply_results.push_back({ft, mask, hist_ft, std::move(policy), move, stone_opt.value_or(0)});
    }

    return {std::move(ply_results), timing};
}
