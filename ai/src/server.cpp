// HTTP inference server.
//
// Usage: goes_server [--checkpoint-dir ai/checkpoints] [--port 8765] [--sims 200]
//                    [--temperature 0.0] [--cpu] [--help]
// The GOES_CHECKPOINT_DIR, GOES_NUM_SIMS and GOES_TEMPERATURE env vars, when set, override
// --checkpoint-dir, --sims and --temperature (they're applied after the flags).
//
// POST /move - JSON body fields:
//   config            object - shared/gameConfig.ts's GameConfig.toJSON() shape (see
//                      parse_game_cfg()):
//     boardDescr        string - cleg source text, e.g. "rectB(9, 9);"
//     boardSeed         int in [0, 2^32) - required here; not part of weak_equal()
//     numStones         int
//     numPlayers        int
//     turnList          [{player, stones, protected, friendly}] - see shared/types.ts's TurnInfo
//     stoneToPlayerMap  {stone: player[]} object - stone -> set of players it scores for
//     playerStonePlaceLimit  (int|null)[][] (optional, default all-null) - [stone-1][player-1]
//     globalStonePlaceLimit  (int|null)[] (optional, default all-null) - [stone-1]
//     maxPlies          int|null (optional, default null)
//     forcedPassOnly    bool
//     allowSuicide      bool (optional, default false)
//     scoreRule         "stone"|"territoryonly"|"area"|"territory" (optional, default
//                        "area") - also the totals the search's utility compares
//                        (mcts/utility.h)
//     komi              float[] (optional, default all-zero)
//     koRule            "positional"|"situational" (optional, default "situational")
//   moves             {"pos": int|null, "stone": int|null}[] - full move history as
//                      shared/types.ts's ReplayMove (pos null = pass; stone null =
//                      auto-picked if the turn offers exactly one) - see parse_move_ref().
//   resigns           array (optional, default none) - shared/gameConfig.ts's
//                      FinishedGame.toJSON()'s `resigns` shape, [[ply, [player, ...]], ...];
//                      replayed interleaved with `moves` (see replay_tail()).
//   board             int[]        - current stone array (length N); used to verify
//                                    the replayed state matches the client's state
//   session_id        string|null  - opaque token returned by a previous response;
//                                    omit or pass "" / null for a new session
//   num_simulations   int (optional, overrides server default; must be >= 1)
//   temperature       float (optional, overrides server default; 0 = the search's best
//                      move, >0 = sample from its play-selection values at that temperature)
//
//   Returns: {move, stone, policy, value, session_id}
//     move       int|null   - chosen board index, or null for pass
//     stone      int|null   - chosen stone color, or null for pass
//     policy     float[]    - the search's policy over all actions (length num_stones*N+1)
//     value      float      - the search's utility for the player to move, in [-1, 1]-ish
//                             (mcts/utility.h)
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
// fresh_checkpoint_dir()), not derivable from a request's config - the
// server instead scans ckpt_dir's subdirectories once (cached in
// ServerState::checkpoint_index; directories created after that first scan
// need a server restart to be picked up) and matches a request's GameConfig
// against each directory's stored one via weak_equal() (self_play.h) - see
// find_checkpoint_dir(), below. Models are then loaded lazily on first
// request for each matched directory and cached by that directory name.
#include "game/board_config.h"
#include "game/cleg_eval.h"
#include "game/cleg_parser.h"
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
#include <cctype>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ── Checkpoint helpers ────────────────────────────────────────────────────────

// The <arch>_<iter>.pt (train.cpp's iter_name()) directly in `dir` with the highest iteration.
static std::optional<fs::path> latest_model_file(const fs::path& dir, const std::string& arch) {
    const std::string prefix = arch + "_", suffix = ".pt";
    std::optional<fs::path> latest;
    long long latest_iter = -1;
    for (auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        auto name = e.path().filename().string();
        if (name.size() <= prefix.size() + suffix.size() || name.rfind(prefix, 0) != 0 ||
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)
            continue;
        std::string iter = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
        if (!std::all_of(iter.begin(), iter.end(), [](unsigned char c) { return std::isdigit(c); })) continue;
        long long n = std::stoll(iter);
        if (n > latest_iter) { latest_iter = n; latest = e.path(); }
    }
    return latest;
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
    std::string tag;                     // checkpoint directory name - must match the incoming request's
    std::unique_ptr<BoardState> state;   // board state after all replayed moves
    std::vector<MoveRef> moves;          // moves applied so far
    std::shared_ptr<const GameInput> game_input;  // built with the session (make_game_input())
};

// ── Server state ──────────────────────────────────────────────────────────────

struct ServerState {
    std::unordered_map<std::string, AnyModel> models;
    std::unordered_map<std::string, std::unique_ptr<SessionState>> sessions;
    torch::Device device{torch::kCPU};
    int default_sims = 200;
    // 0 = the search's best move; >0 = sample from its play-selection values at that temperature.
    float default_temperature = 0.0f;
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

// One `moves` entry, a {"pos": int|null, "stone": int|null} object (a missing field counts as null).
static MoveRef parse_move_ref(const json& m) {
    if (!m.is_object())
        throw std::runtime_error("each moves entry must be a {pos, stone} object, got " + m.dump());
    int pos   = (!m.contains("pos")   || m["pos"].is_null())   ? -1 : m["pos"].get<int>();
    int stone = (!m.contains("stone") || m["stone"].is_null()) ? 0  : m["stone"].get<int>();
    return {pos, stone};
}

// Mirrors shared/gameConfig.ts's FinishedGame.fromJSON()'s `resigns` parsing, into a std::map.
static std::map<int, std::vector<int>> parse_resigns(const json& j) {
    std::map<int, std::vector<int>> result;
    for (const auto& entry : j)
        result[entry[0].get<int>()] = entry[1].get<std::vector<int>>();
    return result;
}

// Mirrors shared/boardState.ts's BoardState.fromFinishedGame()'s replay loop, but onto an existing
// `state` from moves[from] onward, and throws on a rejected move.
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

// The name of the first checkpoint directory whose stored GameConfig weak_equal()s request_cfg
// (training/self_play.h); throws if none does. The first call scans ss.ckpt_dir's subdirectories
// into ss.checkpoint_index (see that field for the no-live-refresh caveat), skipping with a warning
// any whose config can't be read or parsed, and stores the index only once the scan has finished.
static const std::string& find_checkpoint_dir(ServerState& ss, const GameConfig& request_cfg) {
    if (!ss.checkpoint_index.has_value()) {
        std::vector<std::pair<std::string, GameConfig>> index;
        if (fs::exists(ss.ckpt_dir)) {
            for (auto& e : fs::directory_iterator(ss.ckpt_dir)) {
                if (!e.is_directory()) continue;
                auto cfg_path = find_config_json(e.path());
                if (!cfg_path.has_value()) continue;
                try {
                    json cfg_json = json::parse(std::ifstream(*cfg_path));
                    index.emplace_back(e.path().filename().string(), parse_game_cfg(cfg_json));
                } catch (const std::exception& ex) {
                    std::cerr << "[inference] Skipping checkpoint " << e.path().filename().string()
                              << " - unreadable config " << cfg_path->filename() << ": " << ex.what() << "\n";
                }
            }
        }
        ss.checkpoint_index = std::move(index);
    }
    for (auto& [dir, cfg] : *ss.checkpoint_index)
        if (weak_equal(cfg, request_cfg)) return dir;
    throw std::runtime_error("No checkpoint found matching the requested config");
}

// Loads (or returns the cached) model for checkpoint directory `tag`: the newest model file of the
// architecture its <arch>_config.json names (train.cpp's checkpoint directory layout), built for
// the request's board `bc`.
static AnyModel& load_model(ServerState& ss, const std::string& tag,
                             const BoardConfig& bc) {
    auto it = ss.models.find(tag);
    if (it != ss.models.end()) return it->second;

    fs::path dir = fs::path(ss.ckpt_dir) / tag;
    auto cfg_path = find_config_json(dir);
    if (!cfg_path) {
        std::cerr << "[inference] No <arch>_config.json in " << dir << "\n";
        throw std::runtime_error("No <arch>_config.json in checkpoint directory " + dir.string());
    }

    // featureDim/inputDescr persisted directly by train.cpp rather than
    // recomputed here - see compute_input_descr()'s doc comment
    // (training/self_play.h) for why.
    json cfg_json                          = json::parse(std::ifstream(*cfg_path));
    GameConfig game_cfg                    = parse_game_cfg(cfg_json);
    std::unique_ptr<ModelConfig> model_cfg = parse_model_config(cfg_json);
    const std::string& arch                = model_cfg->model_type;

    auto model_path = latest_model_file(dir, arch);
    if (!model_path) {
        std::cerr << "[inference] No " << arch << "_<iter>.pt model file in " << dir << "\n";
        throw std::runtime_error("No " + arch + "_<iter>.pt model file in checkpoint directory " + dir.string());
    }

    AnyModel model_any = build_model(bc, *model_cfg, game_cfg.num_stones, game_cfg.num_players);
    std::visit([&](auto& m) {
        torch::load(m, model_path->string());
        m->to(ss.device);
        m->eval();
    }, model_any);

    std::cout << "[inference] Loaded " << model_path->filename()
              << " for tag=" << tag
              << " (in_dim=" << model_cfg->feature_dim << ", model=" << arch << ")\n";

    auto [ins, ok] = ss.models.emplace(tag, std::move(model_any));
    return ins->second;
}

// ── Main ──────────────────────────────────────────────────────────────────────

static void print_usage(std::ostream& out, const char* prog) {
    out << "Usage: " << prog << " [options]\n"
        << "  --checkpoint-dir PATH  Directory of goes_train checkpoint directories (default: ai/checkpoints)\n"
        << "  --port N               HTTP port (default: 8765)\n"
        << "  --sims N               Default search visits per move, >= 1 (default: 200)\n"
        << "  --temperature F        Default move temperature, 0 = the search's best move (default: 0)\n"
        << "  --cpu                  Force CPU even if CUDA is available\n"
        << "GOES_CHECKPOINT_DIR, GOES_NUM_SIMS and GOES_TEMPERATURE, when set, override\n"
        << "--checkpoint-dir, --sims and --temperature.\n";
}

int main(int argc, char* argv[]) {
    std::string ckpt_dir = "ai/checkpoints";
    int port = 8765;
    int default_sims = 200;
    float default_temperature = 0.0f;
    bool use_cpu = false;

    auto need = [&](int& i) -> std::string {
        if (i + 1 >= argc) {
            std::cerr << argv[i] << " requires a value\n";
            print_usage(std::cerr, argv[0]);
            std::exit(1);
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") { print_usage(std::cout, argv[0]); return 0; }
        else if (a == "--checkpoint-dir") ckpt_dir = need(i);
        else if (a == "--port")           port = std::stoi(need(i));
        else if (a == "--sims")           default_sims = std::stoi(need(i));
        else if (a == "--temperature")    default_temperature = std::stof(need(i));
        else if (a == "--cpu")            use_cpu = true;
        else {
            std::cerr << "Unknown argument: " << a << "\n";
            print_usage(std::cerr, argv[0]);
            return 1;
        }
    }

    // Applied after the flags, so they take precedence.
    if (const char* e = std::getenv("GOES_CHECKPOINT_DIR")) ckpt_dir = e;
    if (const char* e = std::getenv("GOES_NUM_SIMS"))      default_sims = std::stoi(e);
    if (const char* e = std::getenv("GOES_TEMPERATURE"))   default_temperature = std::stof(e);
    if (default_sims < 1) {
        std::cerr << "--sims / GOES_NUM_SIMS must be >= 1 (got " << default_sims << ")\n";
        return 1;
    }

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

            // Checked before any session/model work, so a rejected request changes nothing.
            int num_sims = j.value("num_simulations", ss.default_sims);
            if (num_sims < 1)
                throw std::runtime_error("num_simulations must be >= 1, got " + std::to_string(num_sims));

            GameConfig game_cfg     = parse_game_cfg(cfg);
            if (!game_cfg.board_seed)
                throw std::runtime_error("config.boardSeed is required");
            const std::string& tag  = find_checkpoint_dir(ss, game_cfg);
            auto bc                 = std::make_shared<const BoardConfig>(
                                          build_board_from_cleg(*parse_cleg(game_cfg.board_descr), *game_cfg.board_seed));
            auto& model_v           = load_model(ss, tag, *bc);
            auto evaluator          = make_evaluator(model_v);

            // Full move history, one {"pos","stone"} object per ply - see parse_move_ref().
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
                    // no un-resigning is needed here.
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
                new_sess->game_input = make_game_input(*fresh, input_descr_of(model_v));
                new_sess->state = std::move(fresh);
                new_sess->moves = req_moves;
                ss.sessions[result_session_id] = std::move(new_sess);
                sess = ss.sessions[result_session_id].get();
            }

            // Verify the replayed board matches what the client sent.
            if (sess->state->board != req_board)
                throw std::runtime_error("Board state mismatch: replayed board does not match provided board");

            // ── Run MCTS ─────────────────────────────────────────────────────

            if (sess->state->game_over())
                throw std::runtime_error("the game is already over");
            // KataGo's play-mode search settings: no root noise, a small root FPU reduction, and the
            // move chosen from the LCB-adjusted play-selection values at the request's temperature.
            RootSettings root;
            root.visits = num_sims;
            root.fpu_reduction_max = 0.1f;
            root.move_temperature = j.value("temperature", ss.default_temperature);
            MCTS mcts(SearchParams::server(), std::random_device{}());
            SearchResult result = mcts.search(*sess->state, root, [&](const BoardState& s) {
                return evaluate_rows(evaluator, {&s}, sess->game_input)[0];
            });
            // The mover is next_turn.player (stone_to_player_map is scoring-only - see
            // shared/types.ts's TurnInfo).
            float value = result.utility[sess->state->next_turn.player - 1];

            // Decode the flat, stone-major action index into a (pos, stone) pair.
            int ns = sess->state->num_stones;
            int move_idx = result.move;
            json resp;
            if (move_idx == ns * bc->N) {
                resp["move"]  = nullptr;
                resp["stone"] = nullptr;
            } else {
                resp["move"]  = move_idx % bc->N;
                resp["stone"] = move_idx / bc->N + 1;
            }
            resp["policy"]     = result.policy_target;
            resp["value"]      = value;
            resp["session_id"] = result_session_id;

            res.set_content(resp.dump(), "application/json");
        } catch (const std::exception& ex) {
            res.status = 500;
            // dump() escapes the message's quotes and backslashes (e.g. a Windows path's).
            json err;
            err["error"] = ex.what();
            res.set_content(err.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
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
