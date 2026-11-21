#include "game/game_config.h"
#include "game/cleg_parser.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

using json = nlohmann::json;

bool weak_equal(const GameConfig& a, const GameConfig& b) {
    if (a.board_descr != b.board_descr) return false;
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
    // player list doesn't matter.
    auto flatten = [](const std::unordered_map<int, std::vector<int>>& m) {
        std::vector<std::pair<int,int>> pairs;
        for (auto& [stone, players] : m)
            for (int p : players) pairs.push_back({stone, p});
        std::sort(pairs.begin(), pairs.end());
        return pairs;
    };
    if (flatten(a.stone_to_player_map) != flatten(b.stone_to_player_map)) return false;
    // Which place limits are set decides the input's budget channels (compute_input_descr(),
    // model/input_descr.h); their values may differ.
    auto set_entries = [](const std::vector<std::vector<std::optional<int>>>& limits) {
        std::vector<std::vector<bool>> set;
        for (auto& row : limits) {
            auto& r = set.emplace_back();
            for (auto& v : row) r.push_back(v.has_value());
        }
        return set;
    };
    if (set_entries(resolve_player_stone_place_limit(a.player_stone_place_limit, a.num_stones, a.num_players)) !=
        set_entries(resolve_player_stone_place_limit(b.player_stone_place_limit, b.num_stones, b.num_players)))
        return false;
    if (set_entries({resolve_global_stone_place_limit(a.global_stone_place_limit, a.num_stones)}) !=
        set_entries({resolve_global_stone_place_limit(b.global_stone_place_limit, b.num_stones)}))
        return false;
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
    j["boardDescr"] = board_descr;
    j["boardSeed"]  = board_seed.has_value() ? json(*board_seed) : json(nullptr);
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

// ── Config JSON parsing (matches shared/gameConfig.ts's GameConfig.toJSON() shape) ──

static std::vector<TurnInfo> parse_turn_list(const json& j) {
    std::vector<TurnInfo> out;
    for (auto& t : j) {
        TurnInfo ti;
        ti.player = t.at("player").get<int>();
        ti.stones = t.at("stones").get<std::vector<int>>();
        ti.is_protected = t.at("protected").get<std::vector<int>>();
        ti.friendly     = t.at("friendly").get<std::vector<int>>();
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

// Same range check as shared/gameConfig.ts's GameConfig constructor - get<uint32_t>() alone would
// silently wrap a negative or oversized value.
static std::optional<uint32_t> parse_board_seed(const json& j) {
    if (j.is_null()) return std::nullopt;
    if (!j.is_number_unsigned() || j.get<uint64_t>() > UINT32_MAX)
        throw std::runtime_error("boardSeed must be null or an integer in [0, 2^32), got " + j.dump());
    return j.get<uint32_t>();
}

// Required keys are read via .at(), not operator[]: on a const json, operator[] with a missing key
// is undefined behavior (an assert compiled out under NDEBUG), not an exception.
GameConfig parse_game_cfg(const json& cfg) {
    GameConfig game_cfg;
    // fromJSON()'s parseCleg, kept in toJSON()'s unparseCleg form: two formattings of one program
    // compare equal in weak_equal()/strong_equal().
    game_cfg.board_descr = unparse_cleg(*parse_cleg(cfg.at("boardDescr").get<std::string>()));
    game_cfg.board_seed  = parse_board_seed(cfg.value("boardSeed", json(nullptr)));
    game_cfg.num_stones  = cfg.at("numStones").get<int>();
    game_cfg.num_players = cfg.at("numPlayers").get<int>();
    game_cfg.turn_list   = parse_turn_list(cfg.at("turnList"));
    game_cfg.player_stone_place_limit = parse_player_stone_place_limit(
        cfg.value("playerStonePlaceLimit", json(nullptr)), game_cfg.num_stones, game_cfg.num_players);
    game_cfg.global_stone_place_limit = parse_global_stone_place_limit(
        cfg.value("globalStonePlaceLimit", json(nullptr)), game_cfg.num_stones);
    game_cfg.stone_to_player_map = parse_stone_to_player_map(cfg.at("stoneToPlayerMap"));
    game_cfg.forced_pass_only = cfg.at("forcedPassOnly").get<bool>();
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

BoardState new_state(const GameConfig& cfg, std::shared_ptr<const BoardConfig> bc) {
    auto pspl = resolve_player_stone_place_limit(cfg.player_stone_place_limit, cfg.num_stones, cfg.num_players);
    auto gspl = resolve_global_stone_place_limit(cfg.global_stone_place_limit, cfg.num_stones);
    auto komi = cfg.komi;
    if (komi.empty())
        komi = std::vector<float>(cfg.num_players, 0.0f);

    return BoardState(
        cfg.num_stones, cfg.num_players,
        cfg.turn_list, pspl, gspl, cfg.stone_to_player_map,
        cfg.forced_pass_only, cfg.score_rule, komi, cfg.ko_rule, cfg.allow_suicide, cfg.max_plies,
        std::vector<int>(bc->N, 0),
        bc);
}
