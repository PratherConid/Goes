// HTTP inference server.
//
// Usage: goes_server [--checkpoint-dir PATH] [--port 8765] [--sims 200]
//                    [--temperature 0.0] [--cpu]
// Also reads GOES_CHECKPOINT_DIR, GOES_NUM_SIMS, and GOES_TEMPERATURE env vars.
//
// POST /move - JSON body fields:
//   config            object with game configuration:
//     boardType         "rect"|"rectd"|"cub"|"hcub"|"tri"|"twsq"|"gtsq"
//     boardArgs         integer dimensions matching the board type
//     numStones         int
//     numPlayers        int
//     turnList          [{player, stones, protected, friendly}] - see shared/types.ts's TurnInfo
//     stoneToPlayerMap  {stone: player[]} object - stone -> set of players it scores for
//     playerStonePlaceLimit  (int|null)[][] (optional, default all-null) - [stone-1][player-1]
//     globalStonePlaceLimit  (int|null)[] (optional, default all-null) - [stone-1]
//     maxPlies          int|null (optional, default null)
//     forcedPassOnly    bool
//     allowSuicide      bool (optional, default false) - whether a move that leaves
//                        the mover's own group with zero liberties is legal (self-
//                        captures that group immediately, rather than being rejected);
//                        like the other config fields, part of weak_equal() (self_play.h)
//                        - see parse_game_cfg() and find_checkpoint_dir(), below.
//     scoreRule         "stone"|"territoryonly"|"area"|"territory" (optional, default
//                        "area") - see BoardState::compute_points(); affects the
//                        terminal/estimated reward MCTS backs up (compute_player_rewards(),
//                        estimate_player_rewards()), so - like the other config fields -
//                        it's part of weak_equal() and selects which checkpoint is
//                        loaded. "territory" (real-world Japanese-style
//                        scoring) additionally folds each player's
//                        BoardState::capture_count() (stones captured so far) into both
//                        reward formulas and compute_winners(), the same way komi is.
//     komi              float[] (optional, default all-zero) - per-player scoring
//                        handicap added before winner determination and before both
//                        reward formulas above (see BoardState::komi); part of
//                        weak_equal().
//     koRule            "positional"|"situational" (optional, default "situational") -
//                        superko variant (see BoardState::ko_rule); part of
//                        weak_equal().
//   moves             array - full move history; each entry is EITHER a legacy
//                      int|null (null = pass, int = board index; stone auto-picked
//                      if the turn offers exactly one) OR an object
//                      {"pos": int|null, "stone": int|null} for turns offering
//                      several stones - see shared/types.ts's ReplayMove.
//   resigns           array (optional, default none) - [[ply, [player, ...]], ...],
//                      the players (1-indexed) who resigned at each ply - same shape as
//                      shared/types.ts's FinishedGame.toJSON()'s `resigns` field
//                      ([...Map.entries()]). Replayed interleaved with `moves` (see
//                      replay_tail()) since a resigned player may always pass even under
//                      forcedPassOnly, but resignation must be applied at the correct ply -
//                      applying it before replaying a player's own earlier real placements
//                      would incorrectly reject those too.
//   board             int[]        - current stone array (length N); used to verify
//                                    the replayed state matches the client's state
//   session_id        string|null  - opaque token returned by a previous response;
//                                    omit or pass "" / null for a new session
//   num_simulations   int (optional, overrides server default)
//   temperature       float (optional, overrides server default; 0 = argmax
//                      visit count, >0 = sample from visit distribution)
//
//   Returns: {move, stone, policy, value, session_id}
//     move       int|null   - chosen board index, or null for pass
//     stone      int|null   - chosen stone color, or null for pass
//     policy     float[]    - MCTS visit distribution over all actions (length num_stones*N+1)
//     value      float      - estimated value for the current player
//     session_id string     - token to include in the next request for this game
//
//   The server maintains a cached BoardState per session_id. On each request it
//   finds the longest common prefix of the stored and incoming move lists, then
//   withdraws/advances the cached state rather than replaying from scratch when
//   that is cheaper (stored_len < 2 * lcp_len). Otherwise it replays all moves
//   from an empty board and issues a fresh session_id.
//
// GET /health - returns {"status":"ok","loaded_tags":[...],"device":...}
//
// Checkpoint directories are opaque hash names (see train.cpp's
// handle_checkpoint_dir()), not derivable from a request's config - the
// server instead scans ckpt_dir's subdirectories once (cached in
// ServerState::checkpoint_index; directories created after that first scan
// need a server restart to be picked up) and matches a request's GameConfig
// against each directory's stored one via weak_equal() (self_play.h) - see
// find_checkpoint_dir(), below. Models are then loaded lazily on first
// request for each matched directory and cached by that directory name.
#include "game/board_config.h"
#include "game/board_state.h"
#include "model/any_model.h"
#include "model/features.h"
#include "model/model_config.h"
#include "mcts/mcts.h"
#include "training/self_play.h"
#include <torch/torch.h>
// Single-header libraries in third_party/
// (OpenSSL is opt-in - not defining CPPHTTPLIB_OPENSSL_SUPPORT keeps it plain HTTP)
#include "httplib.h"
#include "nlohmann/json.hpp"

#include <iostream>
#include <string>
#include <optional>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <algorithm>
#include <map>
#include <memory>
#include <fstream>
#include <cstdlib>
#include <random>
#include <cstdio>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ── Checkpoint helpers ────────────────────────────────────────────────────────

// Returns the latest checkpoint, preferring CNN over UNet over Transformer over GNN when multiple are present.
static std::optional<fs::path> latest_checkpoint(const fs::path& dir) {
    if (!fs::exists(dir)) return std::nullopt;
    std::vector<fs::path> cnn_ckpts, unet_ckpts, transformer_ckpts, gnn_ckpts;
    for (auto& e : fs::directory_iterator(dir)) {
        if (e.path().extension() != ".pt") continue;
        auto name = e.path().filename().string();
        if      (name.rfind("cnn_", 0) == 0)  cnn_ckpts.push_back(e.path());
        else if (name.rfind("unet_", 0) == 0) unet_ckpts.push_back(e.path());
        else if (name.rfind("transformer_", 0) == 0) transformer_ckpts.push_back(e.path());
        else if (name.rfind("gnn_", 0) == 0)  gnn_ckpts.push_back(e.path());
    }
    if (!cnn_ckpts.empty()) { std::sort(cnn_ckpts.begin(), cnn_ckpts.end()); return cnn_ckpts.back(); }
    if (!unet_ckpts.empty()) { std::sort(unet_ckpts.begin(), unet_ckpts.end()); return unet_ckpts.back(); }
    if (!transformer_ckpts.empty()) { std::sort(transformer_ckpts.begin(), transformer_ckpts.end()); return transformer_ckpts.back(); }
    if (!gnn_ckpts.empty()) { std::sort(gnn_ckpts.begin(), gnn_ckpts.end()); return gnn_ckpts.back(); }
    return std::nullopt;
}

// ── Session ID ────────────────────────────────────────────────────────────────

static std::string make_session_id() {
    static std::mt19937_64 rng(std::random_device{}());
    uint64_t v = rng();
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)v);
    return std::string(buf);
}

// ── Per-game session state ─────────────────────────────────────────────────────
// The server keeps a live BoardState for each active session so it can advance
// the state by replaying only new moves rather than reconstructing from scratch.

// pos == -1 means pass; stone == 0 means "unspecified" (auto-pick if the turn
// offers exactly one stone - see BoardState::make_move).
struct MoveRef {
    int pos;
    int stone;
    bool operator==(const MoveRef& o) const { return pos == o.pos && stone == o.stone; }
};

struct SessionState {
    std::string tag;                     // model/config tag - must match incoming request
    std::unique_ptr<BoardState> state;   // board state after all replayed moves
    std::vector<MoveRef> moves;          // moves applied so far
};

// ── Server state ──────────────────────────────────────────────────────────────

struct ServerState {
    std::unordered_map<std::string, AnyModel> models;
    std::unordered_map<std::string, std::unique_ptr<SessionState>> sessions;
    torch::Device device{torch::kCPU};
    int default_sims = 200;
    float default_temperature = 0.0f;  // 0 = argmax visit count; >0 = sample
    std::string ckpt_dir;
    // (checkpoint directory name, its stored GameConfig) for every
    // subdirectory of ckpt_dir - populated lazily, once, on the first /move
    // request (see find_checkpoint_dir()). Checkpoint directories created
    // after that first scan are not picked up without a server restart.
    //
    // nullopt vs. an empty vector are deliberately distinct states: nullopt
    // means "not scanned yet" (find_checkpoint_dir() should scan now), while
    // a present-but-empty vector means "scanned, and ckpt_dir genuinely had
    // no checkpoint subdirectories at that time" (do NOT scan again - a
    // plain vector using .empty() as the "should I scan?" signal couldn't
    // tell these apart, and would keep re-scanning after every request until
    // the first checkpoint appeared, then silently stop - an inconsistent,
    // partial live-refresh nobody asked for).
    std::optional<std::vector<std::pair<std::string, GameConfig>>> checkpoint_index;
};

// ── Request helpers ───────────────────────────────────────────────────────────

static BoardConfig build_bc(const json& cfg) {
    std::string kind   = cfg["boardType"].get<std::string>();
    std::vector<int> v = cfg["boardArgs"].get<std::vector<int>>();
    return build_board_config(kind, v);
}

// Each entry is either a legacy int|null, or {"pos": int|null, "stone": int|null}.
static MoveRef parse_move_ref(const json& m) {
    if (m.is_object()) {
        int pos   = (!m.contains("pos")   || m["pos"].is_null())   ? -1 : m["pos"].get<int>();
        int stone = (!m.contains("stone") || m["stone"].is_null()) ? 0  : m["stone"].get<int>();
        return {pos, stone};
    }
    return {m.is_null() ? -1 : m.get<int>(), 0};
}

// [[ply, [player, player, ...]], ...] - mirrors shared/types.ts's FinishedGame.toJSON()'s
// `resigns` shape ([...Map.entries()]). Absent "resigns" in the request means no resignations -
// see the /move doc comment above and replay_tail() below.
static std::map<int, std::vector<int>> parse_resigns(const json& j) {
    std::map<int, std::vector<int>> result;
    for (const auto& entry : j)
        result[entry[0].get<int>()] = entry[1].get<std::vector<int>>();
    return result;
}

// Replays moves[from..] onto `state`, interleaving resignations at the correct ply - mirrors
// shared/boardState.ts's BoardState.fromFinishedGame() exactly. A resigned player may always
// pass, even under forced_pass_only, but resign() must be called at the ply it actually happened:
// calling it any earlier would make that same player's own earlier real placements (made before
// they resigned) fail the "resigned players may only pass" check during replay too.
static void replay_tail(BoardState& state, const std::vector<MoveRef>& moves, size_t from,
                         const std::map<int, std::vector<int>>& resigns) {
    for (size_t i = from; i < moves.size(); i++) {
        auto rit = resigns.find(static_cast<int>(i));
        if (rit != resigns.end()) for (int p : rit->second) state.resign(p);
        const auto& mv = moves[i];
        std::optional<int> k     = (mv.pos < 0)    ? std::nullopt : std::optional<int>(mv.pos);
        std::optional<int> stone = (mv.stone == 0) ? std::nullopt : std::optional<int>(mv.stone);
        if (!state.make_move(k, stone))
            throw std::runtime_error("Failed to apply move " + std::to_string(mv.pos));
    }
    auto rit = resigns.find(static_cast<int>(moves.size()));
    if (rit != resigns.end()) for (int p : rit->second) state.resign(p);
}

// Finds the single <arch>_config.json in a checkpoint directory, without
// assuming which <arch> - mirrors train.cpp's identically-named helper.
static std::optional<fs::path> find_config_json(const fs::path& dir) {
    const std::string suffix = "_config.json";
    for (auto& e : fs::directory_iterator(dir)) {
        auto name = e.path().filename().string();
        if (name.size() > suffix.size() &&
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
            return e.path();
    }
    return std::nullopt;
}

// Finds which checkpoint directory (now an opaque hash name, not derivable
// from request_cfg the way the old model_tag()-named directories were)
// satisfies a /move request's game config. Scans ss.ckpt_dir's subdirectories
// and caches (directory name, parsed GameConfig) in ss.checkpoint_index the
// first time this is called (see that field's own doc comment for the
// no-live-refresh caveat), then returns the first entry that weak_equal-
// matches request_cfg (training/self_play.h) - i.e. agrees on every field
// the old model_tag() encoded into its directory name.
static const std::string& find_checkpoint_dir(ServerState& ss, const GameConfig& request_cfg) {
    if (!ss.checkpoint_index.has_value()) {
        ss.checkpoint_index.emplace();
        if (fs::exists(ss.ckpt_dir)) {
            for (auto& e : fs::directory_iterator(ss.ckpt_dir)) {
                if (!e.is_directory()) continue;
                auto cfg_path = find_config_json(e.path());
                if (!cfg_path.has_value()) continue;
                json cfg_json = json::parse(std::ifstream(*cfg_path));
                ss.checkpoint_index->emplace_back(e.path().filename().string(), parse_game_cfg(cfg_json));
            }
        }
    }
    for (auto& [dir, cfg] : *ss.checkpoint_index)
        if (weak_equal(cfg, request_cfg)) return dir;
    throw std::runtime_error("No checkpoint found matching the requested config");
}

// Load (or return cached) model for the given tag.
// bc is needed to construct UNet/CNN (which require grid dimensions at build time).
static AnyModel& load_model(ServerState& ss, const std::string& tag,
                             const BoardConfig& bc) {
    auto it = ss.models.find(tag);
    if (it != ss.models.end()) return it->second;

    fs::path subdir = fs::path(ss.ckpt_dir) / tag;
    auto latest = latest_checkpoint(subdir);
    if (!latest.has_value()) {
        std::cerr << "[inference] No checkpoint found for tag=" << tag
                  << " (looked in " << subdir << ")\n";
        throw std::runtime_error("No checkpoint found for config: " + subdir.string());
    }

    const std::string fname = latest.value().filename().string();
    std::string arch = (fname.rfind("cnn_", 0) == 0)  ? "cnn"
                      : (fname.rfind("unet_", 0) == 0) ? "unet"
                      : (fname.rfind("transformer_", 0) == 0) ? "transformer"
                                                        : "gnn";
    fs::path json_path = latest.value().parent_path() / (arch + "_config.json");
    if (!fs::exists(json_path)) {
        std::cerr << "[inference] Config JSON missing: " << json_path << "\n";
        throw std::runtime_error("Config JSON missing: " + json_path.string());
    }

    // featureDim/inputDescr persisted directly by train.cpp rather than
    // recomputed here - see compute_input_descr()'s doc comment
    // (training/self_play.h) for why.
    json cfg_json                          = json::parse(std::ifstream(json_path));
    GameConfig game_cfg                    = parse_game_cfg(cfg_json);
    std::unique_ptr<ModelConfig> model_cfg = parse_model_config(cfg_json);

    // model_cfg's dynamic type always matches arch (both come from the same
    // checkpoint file: arch from the .pt filename prefix, model_cfg's type
    // from that same file's sibling _config.json's modelType key), so the
    // static_casts below are safe downcasts, not a real runtime type check.
    AnyModel model_any = [&]() -> AnyModel {
        if (arch == "cnn") {
            return CNN(bc, static_cast<const CNNConfig&>(*model_cfg), game_cfg.num_players, game_cfg.num_stones);
        } else if (arch == "unet") {
            return UNet(bc, static_cast<const UNetConfig&>(*model_cfg), game_cfg.num_players, game_cfg.num_stones);
        } else if (arch == "transformer") {
            return Transformer(bc, static_cast<const TransformerConfig&>(*model_cfg), game_cfg.num_players, game_cfg.num_stones);
        } else {
            // adj_norms is only needed to size the GNN's neighbor-count embedding
            // table (max_degree); compute it locally rather than threading it
            // through load_model's signature for architectures that don't use it.
            auto adj_norms = compute_adj_norms(bc, torch::kCPU);
            return MessagePassingGNN(static_cast<const GNNConfig&>(*model_cfg), game_cfg.num_players, game_cfg.num_stones, adj_norms);
        }
    }();

    std::visit([&](auto& m) {
        torch::load(m, latest.value().string());
        m->to(ss.device);
        m->eval();
    }, model_any);

    std::cout << "[inference] Loaded " << latest.value().filename()
              << " for tag=" << tag
              << " (in_dim=" << model_cfg->feature_dim << ", model=" << arch << ")\n";

    auto [ins, ok] = ss.models.emplace(tag, std::move(model_any));
    return ins->second;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    std::string ckpt_dir = "ai/checkpoints";
    int port = 8765;
    int default_sims = 200;
    float default_temperature = 0.0f;
    bool use_cpu = false;

    // Simple arg parsing
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--checkpoint-dir") ckpt_dir = argv[++i];
        else if (a == "--port")        port = std::stoi(argv[++i]);
        else if (a == "--sims")        default_sims = std::stoi(argv[++i]);
        else if (a == "--temperature") default_temperature = std::stof(argv[++i]);
        else if (a == "--cpu")         use_cpu = true;
    }

    // Environment variable overrides
    if (const char* e = std::getenv("GOES_CHECKPOINT_DIR")) ckpt_dir = e;
    if (const char* e = std::getenv("GOES_NUM_SIMS"))      default_sims = std::stoi(e);
    if (const char* e = std::getenv("GOES_TEMPERATURE"))   default_temperature = std::stof(e);

    ServerState ss;
    ss.device      = (torch::cuda::is_available() && !use_cpu) ? torch::kCUDA : torch::kCPU;
    ss.default_sims = default_sims;
    ss.default_temperature = default_temperature;
    ss.ckpt_dir    = ckpt_dir;
    std::cout << "[inference] Checkpoint dir: " << ckpt_dir
              << "  device: " << ss.device << "\n"
              << "[inference] Models loaded lazily on first request per game config\n";

    httplib::Server svr;

    svr.Post("/move", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            auto j = json::parse(req.body);
            const auto& cfg = j["config"];

            std::string session_id;
            if (j.contains("session_id") && j["session_id"].is_string())
                session_id = j["session_id"].get<std::string>();

            GameConfig game_cfg     = parse_game_cfg(cfg);
            const std::string& tag  = find_checkpoint_dir(ss, game_cfg);
            auto bc                 = build_bc(cfg);
            auto adj_norms          = compute_adj_norms(bc, ss.device);
            auto& model_v           = load_model(ss, tag, bc);
            auto evaluator          = make_evaluator(model_v, adj_norms);

            // Moves list from client: each entry is a legacy int|null or a
            // {"pos","stone"} object - see parse_move_ref().
            std::vector<MoveRef> req_moves;
            if (j.contains("moves")) {
                for (const auto& m : j["moves"])
                    req_moves.push_back(parse_move_ref(m));
            }

            // Ply -> resigned players, applied interleaved with move replay - see parse_resigns().
            std::map<int, std::vector<int>> req_resigns;
            if (j.contains("resigns")) req_resigns = parse_resigns(j["resigns"]);

            // Current board sent by client - used to verify our replayed state matches.
            std::vector<int> req_board = j["board"].get<std::vector<int>>();

            // ── Find or create session ────────────────────────────────────────

            SessionState* sess = nullptr;
            std::string result_session_id;

            auto sit = ss.sessions.find(session_id);
            if (!session_id.empty() && sit != ss.sessions.end() && sit->second->tag == tag) {
                sess = sit->second.get();
            }

            if (sess) {
                // Find the longest common prefix of the stored and requested move lists.
                const auto& stored = sess->moves;
                size_t lcp = 0;
                while (lcp < stored.size() && lcp < req_moves.size() && stored[lcp] == req_moves[lcp])
                    ++lcp;

                // Use withdraw+forward when cheaper than a full rebuild:
                //   withdraw+forward cost ≈ (stored.size()-lcp) + (req.size()-lcp)
                //   full rebuild cost    ≈ req.size()
                // withdraw+forward wins when stored.size() < 2*lcp.
                if (stored.size() < 2 * lcp) {
                    // Withdraw to the common prefix. Resignations recorded at earlier plies are
                    // never undone (permanent on both the TS client and this session's state), so
                    // no un-resigning is needed here - see replay_tail()'s doc comment.
                    size_t to_withdraw = stored.size() - lcp;
                    for (size_t i = 0; i < to_withdraw; i++)
                        sess->state->withdraw_move();
                    // Forward-play the request's tail beyond the common prefix.
                    replay_tail(*sess->state, req_moves, lcp, req_resigns);
                    sess->moves = req_moves;
                    result_session_id = session_id;
                } else {
                    sess = nullptr;  // full rebuild is cheaper; fall through
                }
            }

            if (!sess) {
                // Slow path: replay all moves from an empty board.
                auto fresh = std::make_unique<BoardState>(new_state(game_cfg, bc));
                replay_tail(*fresh, req_moves, 0, req_resigns);

                result_session_id = make_session_id();
                auto new_sess = std::make_unique<SessionState>();
                new_sess->tag   = tag;
                new_sess->state = std::move(fresh);
                new_sess->moves = req_moves;
                ss.sessions[result_session_id] = std::move(new_sess);
                sess = ss.sessions[result_session_id].get();
            }

            // Verify the replayed board matches what the client sent.
            if (sess->state->board != req_board)
                throw std::runtime_error("Board state mismatch: replayed board does not match provided board");

            // ── Run MCTS ─────────────────────────────────────────────────────

            int num_sims = j.value("num_simulations", ss.default_sims);
            float temperature = j.value("temperature", ss.default_temperature);
            MCTS mcts(evaluator, 1.0f);
            auto [results, _timing] = mcts.search_batch(
                {sess->state.get()}, num_sims, NoiseConfig{/*add_noise=*/false, 0.3f, 0.25f},
                {temperature});
            auto& [policy_vec, move_idx] = results[0];
            auto [_pol_t, ownership_t] = evaluator.evaluate_batch({sess->state.get()});
            auto reward_t = estimate_player_rewards(ownership_t, sess->state->score_rule,
                                                    sess->state->stone_to_player_map, sess->state->num_players,
                                                    sess->state->komi, sess->state->capture_count());
            // Turn ownership (who is moving) is independent of stone_to_player_map
            // (scoring-only, and now potentially multi-valued) - next_turn.player
            // is the sole source of truth, mirrors shared/boardState.ts.
            int pid = sess->state->next_turn.player;
            float value = reward_t[0][pid - 1].item<float>();

            // Decode the flat, stone-major action index (see MCTS::select()) into
            // a (pos, stone) pair for the response.
            int ns = sess->state->num_stones;
            json resp;
            if (move_idx == ns * bc.N) {
                resp["move"]  = nullptr;
                resp["stone"] = nullptr;
            } else {
                resp["move"]  = move_idx % bc.N;
                resp["stone"] = move_idx / bc.N + 1;
            }
            resp["policy"]     = policy_vec;
            resp["value"]      = value;
            resp["session_id"] = result_session_id;

            res.set_content(resp.dump(), "application/json");
        } catch (const std::exception& ex) {
            res.status = 500;
            res.set_content(std::string("{\"error\":\"") + ex.what() + "\"}", "application/json");
        }
    });

    svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        json resp;
        resp["status"]  = "ok";
        resp["device"]  = (ss.device == torch::kCUDA) ? "cuda" : "cpu";
        std::vector<std::string> tags;
        for (auto& [tag, _] : ss.models) tags.push_back(tag);
        resp["loaded_tags"] = tags;
        res.set_content(resp.dump(), "application/json");
    });

    std::cout << "[inference] Listening on 0.0.0.0:" << port << "\n";
    svr.listen("0.0.0.0", port);
    return 0;
}
