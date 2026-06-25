// HTTP inference server.
//
// Usage: goes_server [--checkpoint-dir PATH] [--port 8765] [--sims 200]
//                    [--temperature 0.0] [--cpu]
// Also reads GOES_CHECKPOINT_DIR, GOES_NUM_SIMS, and GOES_TEMPERATURE env vars.
//
// POST /move - JSON body fields:
//   board_type        "rect"|"rectd"|"cub"|"hcub"|"tri"|"twsq"|"gtsq"
//   board_args        integer dimensions matching the board type
//   num_stones        int
//   num_players       int
//   turn_stone_list   int[]
//   stone_to_player_map  {stone: player} object
//   forced_pass_only  bool
//   moves             (int|null)[]  - full move history; null = pass, int = board index
//   board             int[]        - current stone array (length N); used to verify
//                                    the replayed state matches the client's state
//   session_id        string|null  - opaque token returned by a previous response;
//                                    omit or pass "" / null for a new session
//   num_simulations   int (optional, overrides server default)
//   temperature       float (optional, overrides server default; 0 = argmax
//                      visit count, >0 = sample from visit distribution)
//
//   Returns: {move, policy, value, session_id}
//     move       int|null   - chosen board index, or null for pass
//     policy     float[]    - MCTS visit distribution over all actions (length N+1)
//     value      float      - estimated value for the current player
//     session_id string     - token to include in the next request for this game
//
//   The server maintains a cached BoardState per session_id. On each request it
//   finds the longest common prefix of the stored and incoming move lists, then
//   retracts/advances the cached state rather than replaying from scratch when
//   that is cheaper (stored_len < 2 * lcp_len). Otherwise it replays all moves
//   from an empty board and issues a fresh session_id.
//
// GET /health - returns {"status":"ok","loaded_tags":[...],"device":...}
//
// Models are loaded lazily on first request for each game config and cached.
#include "game/board_config.h"
#include "game/board_state.h"
#include "model/any_model.h"
#include "model/features.h"
#include "mcts/mcts.h"
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

// Returns the latest checkpoint, preferring CNN over GNN when both are present.
static std::optional<fs::path> latest_checkpoint(const fs::path& dir) {
    if (!fs::exists(dir)) return std::nullopt;
    std::vector<fs::path> cnn_ckpts, gnn_ckpts;
    for (auto& e : fs::directory_iterator(dir)) {
        if (e.path().extension() != ".pt") continue;
        auto name = e.path().filename().string();
        if      (name.rfind("cnn_", 0) == 0) cnn_ckpts.push_back(e.path());
        else if (name.rfind("gnn_", 0) == 0) gnn_ckpts.push_back(e.path());
    }
    if (!cnn_ckpts.empty()) { std::sort(cnn_ckpts.begin(), cnn_ckpts.end()); return cnn_ckpts.back(); }
    if (!gnn_ckpts.empty()) { std::sort(gnn_ckpts.begin(), gnn_ckpts.end()); return gnn_ckpts.back(); }
    return std::nullopt;
}

// ── Board factory (mirrors build_board in train.cpp) ─────────────────────────

static BoardConfig build_board(const std::string& kind, const std::vector<int>& v) {
    if (kind == "rect")  return rectangular_board(v[0], v[1]);
    if (kind == "rectd") return rectangular_diagonal_board(v[0], v[1], v[2]);
    if (kind == "cub")   return cubical_board(v[0], v[1], v[2]);
    if (kind == "hcub")  return hypercube_board(v[0], v[1], v[2], v[3]);
    if (kind == "tri")   return triangular_board(v[0]);
    if (kind == "twsq")  return twisted_square_board(v[0], v[1], v[2]);
    if (kind == "gtsq")  return glue_twisted_square_board(v[0], v[1], v[2]);
    throw std::runtime_error("Unknown board type: " + kind);
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

struct SessionState {
    std::string tag;                     // model/config tag - must match incoming request
    std::unique_ptr<BoardState> state;   // board state after all replayed moves
    std::vector<int> moves;              // moves applied so far (-1 = pass, ≥0 = board index)
};

// ── Server state ──────────────────────────────────────────────────────────────

struct ServerState {
    std::unordered_map<std::string, AnyModel> models;
    std::unordered_map<std::string, std::unique_ptr<SessionState>> sessions;
    torch::Device device{torch::kCPU};
    int default_sims = 200;
    float default_temperature = 0.0f;  // 0 = argmax visit count; >0 = sample
    std::string ckpt_dir;
};

// ── Request helpers ───────────────────────────────────────────────────────────

static BoardConfig build_bc(const json& j) {
    std::string kind   = j["board_type"].get<std::string>();
    std::vector<int> v = j["board_args"].get<std::vector<int>>();
    return build_board(kind, v);
}

static BoardState build_state(const json& j, const BoardConfig& bc) {
    std::vector<int> board    = j["board"].get<std::vector<int>>();
    int num_stones            = j["num_stones"].get<int>();
    int num_players           = j["num_players"].get<int>();
    std::vector<int> tsl      = j["turn_stone_list"].get<std::vector<int>>();
    bool forced               = j.value("forced_pass_only", true);
    int ply                   = j.value("ply", 0);

    std::unordered_map<int,int> s2p;
    for (auto& [k, v] : j["stone_to_player_map"].items())
        s2p[std::stoi(k)] = v.get<int>();

    BoardState state(num_stones, num_players, tsl, s2p, forced, board, bc);
    state.set_ply(ply);
    return state;
}

// Compute the checkpoint subdirectory tag from a /move request.
// Must match model_tag() in train.cpp.
static std::string request_tag(const json& j) {
    std::string kind   = j["board_type"].get<std::string>();
    std::vector<int> v = j["board_args"].get<std::vector<int>>();
    int num_stones     = j["num_stones"].get<int>();
    int num_players    = j["num_players"].get<int>();
    auto tsl           = j["turn_stone_list"].get<std::vector<int>>();
    bool fp            = j.value("forced_pass_only", false);

    std::string s = kind;
    for (int x : v) s += '-' + std::to_string(x);
    s += "_s" + std::to_string(num_stones);
    s += "_p" + std::to_string(num_players);
    s += "_tsl";
    for (int i = 0; i < (int)tsl.size(); i++) { if (i) s += '.'; s += std::to_string(tsl[i]); }
    s += "_s2p";
    std::map<int,int> s2p;
    for (auto& [k, val] : j["stone_to_player_map"].items()) s2p[std::stoi(k)] = val.get<int>();
    bool first = true;
    for (auto& [k, val] : s2p) {
        if (!first) s += '.'; first = false;
        s += std::to_string(k) + 'k' + std::to_string(val);
    }
    if (fp) s += "_fp";
    return s;
}

// Load (or return cached) model for the given tag.
// bc is needed to construct ConvNN (which requires grid dimensions at build time).
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
    bool use_cnn = (fname.rfind("cnn_", 0) == 0);
    fs::path json_path = latest.value().parent_path() / ((use_cnn ? "cnn" : "gnn") + std::string("_config.json"));
    if (!fs::exists(json_path)) {
        std::cerr << "[inference] Config JSON missing: " << json_path << "\n";
        throw std::runtime_error("Config JSON missing: " + json_path.string());
    }

    json cfg        = json::parse(std::ifstream(json_path));
    // num_stones + 4 + turn_stone_list.size()
    int in_dim      = cfg["num_stones"].get<int>() + 4 + (int)cfg["turn_stone_list"].size();
    int num_players = cfg.value("num_players", 2);

    AnyModel model_any = [&]() -> AnyModel {
        if (use_cnn) {
            int cnn_hidden = cfg.value("cnn_hidden_dim", 32);
            return ConvNN(bc, in_dim, cnn_hidden, num_players);
        } else {
            int hidden_dim = cfg.contains("gnn_hidden_dim") ? cfg["gnn_hidden_dim"].get<int>()
                                                             : cfg.value("hidden_dim", 128);
            int num_layers = cfg.value("num_layers", 8);
            return MessagePassingGNN(in_dim, hidden_dim, num_layers, num_players);
        }
    }();

    std::visit([&](auto& m) {
        torch::load(m, latest.value().string());
        m->to(ss.device);
        m->eval();
    }, model_any);

    std::cout << "[inference] Loaded " << latest.value().filename()
              << " for tag=" << tag
              << " (in_dim=" << in_dim << ", model=" << (use_cnn ? "CNN" : "GNN") << ")\n";

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

            std::string session_id;
            if (j.contains("session_id") && j["session_id"].is_string())
                session_id = j["session_id"].get<std::string>();
            std::string tag        = request_tag(j);
            auto bc                = build_bc(j);
            auto adj_norms         = compute_adj_norms(bc, ss.device);
            auto& model_v          = load_model(ss, tag, bc);
            auto evaluator         = make_evaluator(model_v, adj_norms);

            // Moves list from client: null element → pass (-1), integer → board index.
            std::vector<int> req_moves;
            if (j.contains("moves")) {
                for (const auto& m : j["moves"])
                    req_moves.push_back(m.is_null() ? -1 : m.get<int>());
            }

            // Current board sent by client - used to verify our replayed state matches.
            std::vector<int> req_board = j["board"].get<std::vector<int>>();

            // Game config fields needed to reconstruct state from scratch.
            int num_stones  = j["num_stones"].get<int>();
            int num_players = j["num_players"].get<int>();
            auto tsl        = j["turn_stone_list"].get<std::vector<int>>();
            bool forced     = j.value("forced_pass_only", true);
            std::unordered_map<int,int> s2p;
            for (auto& [k, v] : j["stone_to_player_map"].items())
                s2p[std::stoi(k)] = v.get<int>();

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

                // Use retract+forward when cheaper than a full rebuild:
                //   retract+forward cost ≈ (stored.size()-lcp) + (req.size()-lcp)
                //   full rebuild cost    ≈ req.size()
                // retract+forward wins when stored.size() < 2*lcp.
                if (stored.size() < 2 * lcp) {
                    // Retract to the common prefix.
                    size_t to_retract = stored.size() - lcp;
                    for (size_t i = 0; i < to_retract; i++)
                        sess->state->retract_move();
                    // Forward-play the request's tail beyond the common prefix.
                    for (size_t i = lcp; i < req_moves.size(); i++) {
                        int mv = req_moves[i];
                        std::optional<int> k = (mv < 0) ? std::nullopt : std::optional<int>(mv);
                        if (!sess->state->make_move(k))
                            throw std::runtime_error("Failed to apply move " + std::to_string(mv));
                    }
                    sess->moves = req_moves;
                    result_session_id = session_id;
                } else {
                    sess = nullptr;  // full rebuild is cheaper; fall through
                }
            }

            if (!sess) {
                // Slow path: replay all moves from an empty board.
                auto fresh = std::make_unique<BoardState>(
                    num_stones, num_players, tsl, s2p, forced,
                    std::vector<int>(bc.N, 0), bc);
                for (int mv : req_moves) {
                    std::optional<int> k = (mv < 0) ? std::nullopt : std::optional<int>(mv);
                    if (!fresh->make_move(k))
                        throw std::runtime_error("Failed to replay move " + std::to_string(mv));
                }

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
            auto [_pol_t, value_t] = evaluator.evaluate_batch({sess->state.get()});
            auto s2p_it = sess->state->stone_to_player_map.find(sess->state->next_player);
            int pid = (s2p_it != sess->state->stone_to_player_map.end()) ? s2p_it->second : 1;
            float value = value_t[0][pid - 1].item<float>();

            json resp;
            if (move_idx == bc.N) resp["move"] = nullptr;
            else resp["move"] = move_idx;
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
