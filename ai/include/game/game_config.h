#pragma once
#include "game/board_config.h"
#include "game/board_state.h"
#include "nlohmann/json.hpp"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

struct GameConfig {
    // Mirrors shared/gameConfig.ts's GameConfig.boardDescr, but held as cleg source text (its
    // toJSON() form), not a parsed ClegProgram - re-parsed on demand (game/cleg_parser.h's
    // parse_cleg) wherever a board is needed, since nothing here keeps the AST around.
    std::string board_descr;
    // Mirrors shared/gameConfig.ts's GameConfig.boardSeed; nullopt = null. server.cpp requires it on
    // a /move request's config. train.cpp ignores it and draws its own seed per run.
    std::optional<uint32_t> board_seed;
    int num_stones;
    int num_players;
    std::vector<TurnInfo> turn_list;
    // Mirrors shared/gameConfig.ts's GameConfig.playerStonePlaceLimit; nullopt = null. If left
    // empty, new_state() defaults to all-unlimited, sized to num_stones x num_players.
    std::vector<std::vector<std::optional<int>>> player_stone_place_limit;
    // Mirrors shared/gameConfig.ts's GameConfig.globalStonePlaceLimit; nullopt = null. If left
    // empty, new_state() defaults to all-unlimited, sized to num_stones.
    std::vector<std::optional<int>> global_stone_place_limit;
    std::unordered_map<int, std::vector<int>> stone_to_player_map;
    bool forced_pass_only = true;
    // Mirrors shared/gameConfig.ts's GameConfig.scoreRule, as a plain string.
    std::string score_rule = "area";
    // Mirrors shared/gameConfig.ts's GameConfig.komi. If left empty, new_state() defaults to
    // all-zero, sized to num_players.
    std::vector<float> komi;
    // Mirrors shared/gameConfig.ts's GameConfig.koRule.
    KoRule ko_rule = KoRule::Situational;
    // Mirrors shared/gameConfig.ts's GameConfig.allowSuicide.
    bool allow_suicide = false;
    // Mirrors shared/gameConfig.ts's GameConfig.maxPlies; nullopt = null.
    std::optional<int> max_plies;

    // Mirrors shared/gameConfig.ts's GameConfig.toJSON(), minus `players` (the engine doesn't
    // track PlayerInfo). Used for train.cpp's checkpoint config.json.
    nlohmann::json to_json() const;
};

// True iff a and b agree on the fields that determine which trained model fits a ruleset:
// board_descr (exact text), num_stones, num_players, turn_list, stone_to_player_map (as a sorted
// (stone,player) set), which place limits are set (not their values), forced_pass_only,
// allow_suicide, score_rule, komi and ko_rule. board_seed is excluded because training draws its
// own board seed; max_plies is excluded too. Used by server.cpp to find the checkpoint directory
// for a /move request's config.
bool weak_equal(const GameConfig& a, const GameConfig& b);

// True iff a and b are identical in every field (== GameConfig::to_json()
// equality). Used by train.cpp's --resume/--retrain validation, where the game
// rules must match the checkpoint's exactly, not just on the weak_equal subset.
bool strong_equal(const GameConfig& a, const GameConfig& b);

// Mirrors shared/gameConfig.ts's GameConfig.fromJSON(), minus `players`. Shared by server.cpp's
// /move handler (its request `config` object) and train.cpp's --game-config file.
GameConfig parse_game_cfg(const nlohmann::json& cfg);

// Resolves an empty player_stone_place_limit/global_stone_place_limit
// ("not set" - see GameConfig's own field doc comments, above) to an
// explicit all-unlimited array of the right shape. Shared by new_state()
// and any caller (e.g. compute_input_descr(), model/input_descr.h) that
// needs the resolved shape before a BoardState exists.
std::vector<std::vector<std::optional<int>>> resolve_player_stone_place_limit(
    const std::vector<std::vector<std::optional<int>>>& limit, int num_stones, int num_players);
std::vector<std::optional<int>> resolve_global_stone_place_limit(
    const std::vector<std::optional<int>>& limit, int num_stones);

// Create a fresh starting board state for a game.
BoardState new_state(const GameConfig& cfg, std::shared_ptr<const BoardConfig> bc);
