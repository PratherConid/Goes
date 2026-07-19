// Main training script.
//
// Usage: goes_train --game-config <path> [--iterations 100] ...
// Run with --help for the full option list.
#include "game/board_config.h"
#include "model/any_model.h"
#include "model/features.h"
#include "model/model_config.h"
#include "training/self_play.h"
#include "training/replay_buffer.h"
#include "util/sha256.h"
#include <torch/torch.h>
#include <iostream>
#include <string>
#include <vector>
#include <optional>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <random>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <cassert>
#include <cmath>
#include <memory>
#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace fs = std::filesystem;

// ── CLI argument parsing ──────────────────────────────────────────────────────

struct Args {
    // Path to a GameConfig JSON file (shared/types.ts's GameConfig.toJSON()
    // wire shape - same as server.cpp's /move request `config` object).
    // Required: there's no safe universal default, since forced_pass_only
    // must be false for training (see the assert in main()) but the
    // browser-facing public/game_presets/*_fpo.json presets all set it true.
    std::string game_config_path;
    int gnn_hidden_dim    = 128;
    int unet_hidden_dim   = 16;
    int cnn_hidden_dim    = 64;
    int num_layers        = 9;
    int iterations        = 200;
    int self_play_games   = 10;
    int gamegen_batch_size = 25;
    int num_simulations   = 200;
    float train_fraction  = 0.1f;
    int batch_size        = 128;
    int buffer_size       = 2048;
    float lr              = 1e-3f;
    float l2              = 1e-4f;
    float c_puct          = 1.0f;
    int save_every        = 10;
    std::string checkpoint_dir = "ai/checkpoints";
    std::string net_arch  = "auto";
    bool cpu              = false;
    int verbosity         = 1;
    std::optional<std::pair<float,float>> linear_move_bound;
    // Opaque checkpoint-directory hash (see handle_checkpoint_dir) to resume
    // from - empty means start a fresh run instead. Directories are no longer
    // named deterministically from --game-config, so resuming requires
    // explicitly naming which one.
    std::string resume_tag;
};

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --game-config PATH        (required) Path to a GameConfig JSON file (same shape\n"
              << "                            as shared/types.ts's GameConfig.toJSON(), e.g. a file\n"
              << "                            under public/game_presets/) - forcedPassOnly must be\n"
              << "                            false (see the assert in main())\n"
              << "  --gnn-hidden-dim N        GNN hidden dimension (default: 128)\n"
              << "  --unet-hidden-dim N       UNet hidden dimension (default: 16)\n"
              << "  --cnn-hidden-dim N        CNN hidden dimension (default: 64)\n"
              << "  --num-layers N            GNN message-passing layers (default: 9)\n"
              << "  --iterations N            Training iterations (default: 200)\n"
              << "  --self-play-games N       Games to complete before each training step (default: 10)\n"
              << "  --gamegen-batch-size N    Games generated in parallel (default: 10)\n"
              << "  --num-simulations N       MCTS simulations per move (default: 200)\n"
              << "  --train-fraction F        Train on F * current buffer size randomly selected game\n"
              << "                            states per iteration, rounded up w.r.t. batch size (default: 0.1)\n"
              << "  --batch-size N            Training batch size (default: 128)\n"
              << "  --buffer-size N           Replay buffer capacity in number of games (default: 2048)\n"
              << "  --lr F                    Learning rate (default: 0.001)\n"
              << "  --l2 F                    Weight decay (default: 0.0001)\n"
              << "  --c-puct F                MCTS exploration constant (default: 1.0)\n"
              << "  --save-every N            Save checkpoint every N iterations (default: 10)\n"
              << "  --checkpoint-dir PATH     Checkpoint directory (default: ai/checkpoints)\n"
              << "  --resume TAG              Resume from ai/checkpoints/TAG (an existing hash-named\n"
              << "                            directory printed by a previous run) instead of starting\n"
              << "                            a fresh one - errors if TAG doesn't exist or its saved\n"
              << "                            config doesn't exactly match the current --game-config/\n"
              << "                            architecture flags\n"
              << "  --net-arch auto|cnn|unet|gnn  Network architecture (default: auto)\n"
              << "  --cpu                     Force CPU even if CUDA is available\n"
              << "  --verbosity N             0=silent, 1=per-game, >=2=per-ply (default: 1)\n"
              << "  --linear-move-bound K1 K2 End games after Uniform(K1,K2)*N plies, resampled per game\n"
              << "                            (no shared/types.ts analog - a self-play-only sampling\n"
              << "                            knob for BoardState::max_plies, so it stays a CLI flag\n"
              << "                            rather than part of --game-config)\n";
}

static Args parse_args(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") { print_usage(argv[0]); std::exit(0); }
        else if (a == "--game-config")     args.game_config_path = argv[++i];
        else if (a == "--gnn-hidden-dim")  args.gnn_hidden_dim  = std::stoi(argv[++i]);
        else if (a == "--unet-hidden-dim") args.unet_hidden_dim = std::stoi(argv[++i]);
        else if (a == "--cnn-hidden-dim")  args.cnn_hidden_dim  = std::stoi(argv[++i]);
        else if (a == "--num-layers")      args.num_layers      = std::stoi(argv[++i]);
        else if (a == "--iterations")      args.iterations      = std::stoi(argv[++i]);
        else if (a == "--self-play-games") args.self_play_games = std::stoi(argv[++i]);
        else if (a == "--gamegen-batch-size") args.gamegen_batch_size = std::stoi(argv[++i]);
        else if (a == "--num-simulations") args.num_simulations = std::stoi(argv[++i]);
        else if (a == "--train-fraction")  args.train_fraction  = std::stof(argv[++i]);
        else if (a == "--batch-size")      args.batch_size      = std::stoi(argv[++i]);
        else if (a == "--buffer-size")     args.buffer_size     = std::stoi(argv[++i]);
        else if (a == "--lr")              args.lr              = std::stof(argv[++i]);
        else if (a == "--l2")              args.l2              = std::stof(argv[++i]);
        else if (a == "--c-puct")          args.c_puct          = std::stof(argv[++i]);
        else if (a == "--save-every")      args.save_every      = std::stoi(argv[++i]);
        else if (a == "--checkpoint-dir")  args.checkpoint_dir  = argv[++i];
        else if (a == "--resume")          args.resume_tag      = argv[++i];
        else if (a == "--net-arch")        args.net_arch        = argv[++i];
        else if (a == "--cpu")             args.cpu             = true;
        else if (a == "--verbosity")       args.verbosity       = std::stoi(argv[++i]);
        else if (a == "--linear-move-bound") {
            if (i + 2 >= argc) {
                std::cerr << "--linear-move-bound requires two values: K1 K2\n"; std::exit(1);
            }
            float k1 = std::stof(argv[++i]);
            float k2 = std::stof(argv[++i]);
            if (k1 > k2) {
                std::cerr << "--linear-move-bound: K1 must be <= K2 (got " << k1 << " " << k2 << ")\n";
                std::exit(1);
            }
            args.linear_move_bound = {k1, k2};
        }
        else { std::cerr << "Unknown argument: " << a << "\n"; std::exit(1); }
    }
    if (args.game_config_path.empty()) {
        std::cerr << "--game-config is required\n"; print_usage(argv[0]); std::exit(1);
    }
    return args;
}

// ── Model factory ─────────────────────────────────────────────────────────────

// Returns "cnn", "unet", or "gnn" — the architecture that will actually be used.
static std::string effective_arch(const Args& args, const std::string& board_kind) {
    bool grid2d_supported = (board_kind == "rect" || board_kind == "rectd" || board_kind == "tri" ||
                          board_kind == "twsq" || board_kind == "gtsq");
    if (args.net_arch == "cnn") {
        if (!grid2d_supported) {
            std::cerr << "Error: --net-arch cnn is not supported for board type '" << board_kind
                      << "'. CNN requires a 2D grid embedding (rect/rectd/tri/twsq/gtsq).\n";
            std::exit(1);
        }
        return "cnn";
    }
    if (args.net_arch == "unet") {
        if (!grid2d_supported) {
            std::cerr << "Error: --net-arch unet is not supported for board type '" << board_kind
                      << "'. UNet requires a 2D grid embedding (rect/rectd/tri/twsq/gtsq).\n";
            std::exit(1);
        }
        return "unet";
    }
    if (args.net_arch == "gnn") return "gnn";
    if (args.net_arch == "auto") return grid2d_supported ? "cnn" : "gnn";
    std::cerr << "Error: unknown --net-arch '" << args.net_arch
              << "'. Valid options: auto, cnn, unet, gnn.\n";
    std::exit(1);
}

// cfg's dynamic type always matches cfg.model_type (both come from the same
// construction site in main(), or the same parse_model_config() call in
// server.cpp's load_model()), so the static_casts below are safe downcasts,
// not a real runtime type check.
static AnyModel build_model(const BoardConfig& bc, const ModelConfig& cfg, const GameConfig& game_cfg) {
    if (cfg.model_type == "cnn")
        return CNN(bc, static_cast<const CNNConfig&>(cfg), game_cfg.num_players, game_cfg.num_stones);
    if (cfg.model_type == "unet")
        return UNet(bc, static_cast<const UNetConfig&>(cfg), game_cfg.num_players, game_cfg.num_stones);
    // adj_norms is only needed to size the GNN's neighbor-count embedding
    // table (max_degree); compute it locally rather than threading it
    // through build_model's signature for architectures that don't use it.
    auto adj_norms = compute_adj_norms(bc, torch::kCPU);
    return MessagePassingGNN(static_cast<const GNNConfig&>(cfg), game_cfg.num_players, game_cfg.num_stones, adj_norms);
}

// ── Checkpoint utilities ──────────────────────────────────────────────────────

static std::optional<fs::path> latest_checkpoint(const fs::path& dir,
                                                   const std::string& arch) {
    if (!fs::exists(dir)) return std::nullopt;
    std::string prefix = arch + "_";
    std::vector<fs::path> ckpts;
    for (auto& e : fs::directory_iterator(dir)) {
        auto name = e.path().filename().string();
        if (name.rfind(prefix, 0) == 0 && e.path().extension() == ".pt")
            ckpts.push_back(e.path());
    }
    if (ckpts.empty()) return std::nullopt;
    std::sort(ckpts.begin(), ckpts.end());
    return ckpts.back();
}

static int iteration_from_model_path(const fs::path& p) {
    // unet_000042.pt / gnn_000042.pt → 42
    std::string stem = p.stem().string();
    auto pos = stem.rfind('_');
    if (pos == std::string::npos) return 0;
    return std::stoi(stem.substr(pos + 1));
}

// Finds the single <arch>_config.json in a checkpoint directory, without
// assuming which <arch> - used when resuming, where the original run's
// architecture isn't known ahead of time (that's exactly one of the things
// being validated against, see handle_checkpoint_dir).
static std::optional<fs::path> find_config_json(const fs::path& dir) {
    if (!fs::exists(dir)) return std::nullopt;
    const std::string suffix = "_config.json";
    for (auto& e : fs::directory_iterator(dir)) {
        auto name = e.path().filename().string();
        if (name.size() > suffix.size() &&
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
            return e.path();
    }
    return std::nullopt;
}

// Determines this run's checkpoint directory:
// - no --resume: hashes model_cfg+game_cfg+a timestamp into a fresh, unique
//   directory name (see sha256_hex, util/sha256.h) and creates it - errors
//   (exits) if that path somehow already exists.
// - --resume TAG: errors (exits) if ai/checkpoints/TAG doesn't exist, has no
//   checkpoint config to validate against, or that config doesn't
//   strong_equal-match the current game_cfg/model_cfg exactly - otherwise
//   returns the existing directory as-is for resume() to load from.
static fs::path handle_checkpoint_dir(const Args& args, const GameConfig& game_cfg,
                                       const ModelConfig& model_cfg)
{
    fs::path ckpt_dir;
    if (args.resume_tag.empty()) {
        auto timestamp = std::to_string(
            std::chrono::system_clock::now().time_since_epoch().count());
        std::string hash = sha256_hex(model_cfg.to_json().dump() + game_cfg.to_json().dump() + timestamp);
        ckpt_dir = fs::path(args.checkpoint_dir) / hash;
        if (fs::exists(ckpt_dir)) {
            std::cerr << "Error: checkpoint directory already exists (hash collision?): " << ckpt_dir << "\n";
            std::exit(1);
        }
        fs::create_directories(ckpt_dir);
        std::cout << "New checkpoint directory: " << hash
                  << " (pass --resume " << hash << " to continue this run later)\n";
    } else {
        ckpt_dir = fs::path(args.checkpoint_dir) / args.resume_tag;
        if (!fs::exists(ckpt_dir)) {
            std::cerr << "Error: --resume directory does not exist: " << ckpt_dir << "\n";
            std::exit(1);
        }
        auto cfg_path = find_config_json(ckpt_dir);
        if (!cfg_path.has_value()) {
            std::cerr << "Error: --resume directory has no checkpoint config to validate against: "
                      << ckpt_dir << "\n";
            std::exit(1);
        }
        json existing = json::parse(std::ifstream(*cfg_path));
        GameConfig existing_game_cfg = parse_game_cfg(existing);
        auto existing_model_cfg = parse_model_config(existing);
        if (!strong_equal(game_cfg, existing_game_cfg) || !strong_equal(model_cfg, *existing_model_cfg)) {
            // Same combined game+model shape the checkpoint's own config.json is
            // saved as (see main()'s checkpoint block, below) - so "current" is
            // directly comparable to `existing`, already in that same shape.
            json current = game_cfg.to_json();
            current.update(model_cfg.to_json());
            // json::diff() (RFC 6902 JSON Patch: source=current, target=existing)
            // shows only what differs rather than two full configs to eyeball.
            std::cerr << "Error: --resume config mismatch - current --game-config/model flags don't "
                          "exactly match " << *cfg_path << "\n"
                      << "Diff (current -> to be resumed):\n" << json::diff(current, existing).dump(2) << "\n";
            std::exit(1);
        }
    }
    return ckpt_dir;
}

static int iteration_from_traj_path(const fs::path& p) {
    // cnn_000009_traj.json -> 9 (iteration_from_model_path expects a bare
    // "<arch>_XXXXXX" stem and can't parse the "_traj" suffix directly).
    std::string stem = p.stem().string();  // "cnn_000009_traj"
    const std::string suffix = "_traj";
    if (stem.size() > suffix.size() &&
        stem.compare(stem.size() - suffix.size(), suffix.size(), suffix) == 0)
        stem = stem.substr(0, stem.size() - suffix.size());  // "cnn_000009"
    auto pos = stem.rfind('_');
    if (pos == std::string::npos) return 0;
    return std::stoi(stem.substr(pos + 1));
}

// Iteration numbers of every <arch>_XXXXXX_traj.json in dir, descending (most
// recent first) - mirrors latest_checkpoint()'s directory-scan style.
static std::vector<int> trajectory_iterations_desc(const fs::path& dir, const std::string& arch) {
    std::vector<int> iters;
    if (!fs::exists(dir)) return iters;
    std::string prefix = arch + "_";
    std::string suffix = "_traj.json";
    for (auto& e : fs::directory_iterator(dir)) {
        auto name = e.path().filename().string();
        if (name.rfind(prefix, 0) == 0 && name.size() > suffix.size() &&
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
            iters.push_back(iteration_from_traj_path(e.path()));
    }
    std::sort(iters.rbegin(), iters.rend());
    return iters;
}

// Loads the latest checkpoint's weights (if any) into model_var, and
// reconstructs up to target_buffer_size games of ReplayBuffer state from
// historical <arch>_XXXXXX_traj.json dumps (most recent games first) so a
// resumed run doesn't start training against an empty buffer. Returns the
// iteration to resume from (0 if no checkpoint exists, in which case
// `buffer` is left untouched/empty).
static int resume(const fs::path& ckpt_dir, const ModelConfig& model_cfg, AnyModel& model_var,
                   const GameConfig& game_cfg, const BoardConfig& bc,
                   int target_buffer_size, ReplayBuffer& buffer)
{
    auto latest = latest_checkpoint(ckpt_dir, model_cfg.model_type);
    if (!latest.has_value()) return 0;

    std::cout << "Resuming from " << latest.value() << std::endl;
    std::visit([&](auto& m) { torch::load(m, latest.value().string()); }, model_var);
    int start_iter = iteration_from_model_path(latest.value()) + 1;

    // Walk trajectory files newest-first; within each file, walk its games
    // newest-first too (traj_store.push_back() order is chronological) - so
    // recent_games ends up ordered most-recent-game-first overall.
    std::vector<std::vector<PlyResult>> recent_games;
    for (int it : trajectory_iterations_desc(ckpt_dir, model_cfg.model_type)) {
        if ((int)recent_games.size() >= target_buffer_size) break;
        std::ostringstream toss;
        toss << model_cfg.model_type << "_" << std::setfill('0') << std::setw(6) << it << "_traj.json";
        std::ifstream f(ckpt_dir / toss.str());
        if (!f) continue;
        json trajs; f >> trajs;
        for (auto git = trajs.rbegin(); git != trajs.rend(); ++git) {
            if ((int)recent_games.size() >= target_buffer_size) break;
            std::vector<PlyResult> game;
            game.reserve(git->size());
            for (auto& p : *git) game.push_back(parse_ply_result(p));
            recent_games.push_back(std::move(game));
        }
    }

    // Add oldest-first (reverse of recent_games' most-recent-first order) so
    // ReplayBuffer's FIFO eviction/insertion-order assumptions hold exactly
    // as if these games had been added live, in their original sequence.
    for (auto git = recent_games.rbegin(); git != recent_games.rend(); ++git)
        buffer.add(trajectory_to_record(*git, game_cfg, bc, model_cfg.input_descr));

    if (!recent_games.empty())
        std::cout << "  Reconstructed replay buffer: " << recent_games.size() << " games" << std::endl;
    return start_iter;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    auto args = parse_args(argc, argv);

    std::ifstream cfg_file(args.game_config_path);
    if (!cfg_file) {
        std::cerr << "Cannot open --game-config file: " << args.game_config_path << "\n";
        return 1;
    }
    json cfg_json;
    cfg_file >> cfg_json;
    GameConfig game_cfg = parse_game_cfg(cfg_json);
    // linear_move_bound has no shared/types.ts analog - it's a self-play-only
    // sampling knob for BoardState::max_plies, so it stays a plain CLI flag
    // rather than part of --game-config.
    game_cfg.linear_move_bound = args.linear_move_bound;

    // Requires forced_pass_only=False. When forced_pass_only is enabled, a player
    // may only pass when no traditional placement is legal. In this case, players
    // will be forced to kill their own groups, and the game only ends when both
    // players have no legal moves simultaneously, which closely depends on the full
    // history of the game. The model receives only per-node features derived from the
    // current board, so it cannot function correctly in this case,
    // This restriction will be lifted once history-aware models are implemented.
    assert(!game_cfg.forced_pass_only &&
           "forced_pass_only=true is not supported: model lacks history-aware features");

    torch::Device device = (torch::cuda::is_available() && !args.cpu)
        ? torch::kCUDA : torch::kCPU;
    std::cout << "Device: " << device << std::endl;

    auto bc = build_board_config(game_cfg.board_type, game_cfg.board_args);
    std::cout << "Board: " << game_cfg.board_type;
    for (int a : game_cfg.board_args) std::cout << " " << a;
    std::cout << "  N=" << bc.N << std::endl;

    const std::string arch = effective_arch(args, game_cfg.board_type);
    nlohmann::json input_descr = compute_input_descr(game_cfg, bc.N);
    int in_dim = input_descr.at("totalDims").get<int>();
    int hidden_dim = (arch == "cnn")  ? args.cnn_hidden_dim
                    : (arch == "unet") ? args.unet_hidden_dim
                                        : args.gnn_hidden_dim;
    std::unique_ptr<ModelConfig> model_cfg;
    if (arch == "cnn")       model_cfg = std::make_unique<CNNConfig>(in_dim, hidden_dim, input_descr);
    else if (arch == "unet") model_cfg = std::make_unique<UNetConfig>(in_dim, hidden_dim, input_descr);
    else                     model_cfg = std::make_unique<GNNConfig>(in_dim, hidden_dim, args.num_layers, input_descr);
    auto model_var = build_model(bc, *model_cfg, game_cfg);
    std::visit([&](auto& m) { m->to(device); }, model_var);

    // Resume from checkpoint (weights + replay buffer state)
    fs::path ckpt_dir = handle_checkpoint_dir(args, game_cfg, *model_cfg);
    ReplayBuffer buffer(args.buffer_size);
    int start_iter = resume(ckpt_dir, *model_cfg, model_var, game_cfg, bc,
                             args.buffer_size, buffer);

    auto optimizer = torch::optim::Adam(
        std::visit([](auto& m) { return m->parameters(); }, model_var),
        torch::optim::AdamOptions(args.lr).weight_decay(args.l2));

    std::mt19937 rng(42);

    auto adj_norms = compute_adj_norms(bc, device);

    auto evaluator = make_evaluator(model_var, adj_norms);

    // ── Game pool ─────────────────────────────────────────────────────────────
    // Pool of gamegen_batch_size in-progress games. Slots are replenished
    // immediately when a game ends, so the batch is always full.
    // The pool persists across training iterations. Each game's own
    // max_plies (rolled by new_state() from game_cfg.linear_move_bound) is
    // carried on its BoardState, so no separate list is needed here.
    std::vector<BoardState> pool;
    pool.reserve(args.gamegen_batch_size);
    for (int i = 0; i < args.gamegen_batch_size; i++)
        pool.push_back(new_state(game_cfg, bc));
    std::vector<std::vector<PlyResult>> trajectories(args.gamegen_batch_size);

    // Accumulates completed game trajectories between checkpoints; dumped and
    // cleared each time a checkpoint is saved.
    std::vector<std::vector<PlyResult>> traj_store;
    traj_store.reserve(args.gamegen_batch_size * args.save_every);

    int ply_iter = 0;
    for (int iter = start_iter; iter < args.iterations; iter++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::visit([](auto& m) { m->eval(); }, model_var);

        // ── Self-play ────────────────────────────────────────────────────────
        if (args.verbosity >= 1)
            std::cout << "[iter " << std::setw(4) << iter << "] self-play ("
                      << args.self_play_games << " games, batch=" << args.gamegen_batch_size << ") ..." << std::endl;

        int games_this_iter = 0;
        while (games_this_iter < args.self_play_games) {
            std::vector<BoardState*> ptrs;
            ptrs.reserve(args.gamegen_batch_size);
            for (auto& s : pool) ptrs.push_back(&s);

            auto t_ply0 = std::chrono::high_resolution_clock::now();
            auto [ply_results, timing] = generate_one_ply_per_game(
                evaluator, ptrs, model_cfg->input_descr,
                args.num_simulations, /*temperature_threshold=*/static_cast<int>(2 * std::sqrt(bc.N)) + 3, args.c_puct,
                args.verbosity);
            double total_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t_ply0).count();
            if (args.verbosity >= 1) {
                std::cout << std::fixed << std::setprecision(0)
                          << "  ply iter " << ply_iter << ": generate=" << total_ms << "ms"
                          << "  search=" << timing.search * 1000.0 << "ms"
                          << "  simulate=" << timing.simulate * 1000.0 << "ms"
                          << "  teardown=" << timing.teardown * 1000.0 << "ms"
                          << "  eval=" << timing.eval * 1000.0 << "ms"
                          << "  select=" << timing.select * 1000.0 << "ms"
                          << std::defaultfloat << std::endl;
            }
            ++ply_iter;

            for (int slot = 0; slot < args.gamegen_batch_size; slot++) {
                trajectories[slot].push_back(std::move(ply_results[slot]));

                bool done = pool[slot].game_over();
                if (!done) continue;

                auto record = trajectory_and_result_to_record(
                    trajectories[slot], pool[slot].board, pool[slot].score().territory_owner);

                if (args.verbosity >= 1) {
                    std::cout << "  game " << (games_this_iter + 1)
                              << "/" << args.self_play_games
                              << "  plies=" << trajectories[slot].size()
                              << "  winners=[";
                    // done implies game_over(), so winners should always be set here.
                    if (pool[slot].winners.has_value())
                        for (int w : pool[slot].winners.value()) std::cout << w << ",";
                    else
                        std::cout << "error, winner has not been computed";
                    std::cout << "]" << std::endl;
                }

                traj_store.push_back(trajectories[slot]);

                buffer.add(std::move(record));
                games_this_iter++;

                // Replenish slot immediately with a fresh game (new_state()
                // rolls this game's own max_plies from game_cfg.linear_move_bound)
                pool[slot] = new_state(game_cfg, bc);
                trajectories[slot].clear();
            }
        }

        if (buffer.size() < args.batch_size) {
            std::cout << "[iter " << iter << "] buffer too small ("
                      << buffer.size() << "), skipping train step" << std::endl;
            continue;
        }

        // ── Training ─────────────────────────────────────────────────────────
        std::visit([](auto& m) { m->train(); }, model_var);
        double total_loss = 0, total_pol = 0, total_stone = 0, total_territory = 0, total_point = 0;

        int train_steps = static_cast<int>(
            std::ceil(args.train_fraction * buffer.size() / args.batch_size));
        int num_stones = game_cfg.num_stones;

        for (int step = 0; step < train_steps; step++) {
            auto [x_, mask_, p_tgt, so_tgt_, to_tgt_] = buffer.sample(args.batch_size, rng);
            torch::Tensor x      = x_.to(device);
            torch::Tensor mask   = mask_.to(device);
            p_tgt = p_tgt.to(device);
            torch::Tensor so_tgt = so_tgt_.to(device);
            torch::Tensor to_tgt = to_tgt_.to(device);

            auto [policy, ownership] = std::visit(
                [&](auto& m) -> std::pair<torch::Tensor, torch::Tensor> {
                    using M = std::decay_t<decltype(m)>;
                    if constexpr (std::is_same_v<M, MessagePassingGNN>)
                        return m->forward(x, adj_norms, mask);
                    else
                        return m->forward(x, mask);
                }, model_var);

            // Policy loss: cross-entropy against MCTS visit distribution, scaled down by
            // log(action count) - cross-entropy over num_stones*N+1 actions grows with
            // both board size and stone count, so this keeps the loss magnitude
            // comparable across differently sized action spaces.
            auto log_policy  = torch::log(policy.clamp_min(1e-8f));
            auto policy_loss = -(p_tgt * log_policy).sum(-1).mean() / std::log(static_cast<float>(bc.N * num_stones));

            // ownership: (B, 2, N, num_stones+1) - index 0 = stone estimate, index 1 = territory estimate
            auto stone_est     = ownership.select(1, 0);   // (B, N, num_stones+1)
            auto territory_est = ownership.select(1, 1);   // (B, N, num_stones+1)

            // Ownership loss: per-location MSE between predicted and actual stone/territory
            // ownership distributions (channel 0 = none, channels 1..num_stones = that stone type).
            // Summed (not averaged) over the num_stones+1 channels, then averaged over locations
            // and batch only - torch::mse_loss's default per-element mean would additionally
            // divide by (num_stones+1), making the loss too small to carry much gradient signal.
            auto actual_stone_owner     = torch::one_hot(so_tgt, num_stones + 1).to(torch::kFloat32);  // (B,N,S+1)
            auto actual_territory_owner = torch::one_hot(to_tgt, num_stones + 1).to(torch::kFloat32);  // (B,N,S+1)
            auto stone_loss     = (stone_est - actual_stone_owner).pow(2).sum(-1).mean();
            auto territory_loss = (territory_est - actual_territory_owner).pow(2).sum(-1).mean();

            // Point loss: raw per-stone-type point total (no rank adjustment, no player
            // aggregation) - actual vs. the model's own expected total under the game's
            // scoring rule. Analogous to the pre-ownership-refactor scalar value loss, but
            // supervises raw points instead of the rank-adjusted reward. Reuses
            // estimate_stone_points() for both sides: the model's prediction from
            // `ownership`, and the ground truth by treating the one-hot stone/territory
            // owner tensors as an ownership-shaped input. The raw point difference is
            // scaled by (num_stones / board size) before squaring - unscaled, it can be as
            // large as N, dwarfing the per-location stone_loss/territory_loss terms.
            auto actual_ownership = torch::stack({actual_stone_owner, actual_territory_owner}, 1); // (B,2,N,S+1)
            auto predicted_points = estimate_stone_points(ownership, game_cfg.score_rule);        // (B,num_stones)
            auto actual_points    = estimate_stone_points(actual_ownership, game_cfg.score_rule);  // (B,num_stones)
            auto point_diff = (predicted_points - actual_points) / bc.N * num_stones;              // (B,num_stones)
            auto point_loss = point_diff.pow(2).sum(-1).mean();

            auto loss = policy_loss + stone_loss + territory_loss + point_loss;

            optimizer.zero_grad();
            loss.backward();
            torch::nn::utils::clip_grad_norm_(
                std::visit([](auto& m) { return m->parameters(); }, model_var), 1.0);
            optimizer.step();

            total_loss     += loss.item<double>();
            total_pol      += policy_loss.item<double>();
            total_stone    += stone_loss.item<double>();
            total_territory += territory_loss.item<double>();
            total_point    += point_loss.item<double>();
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        int n = train_steps;
        std::cout << "[iter " << std::setw(4) << iter << "] "
                  << "loss=" << std::fixed << std::setprecision(4) << total_loss/n
                  << "  policy="   << total_pol/n
                  << "  stone="    << total_stone/n
                  << "  territory=" << total_territory/n
                  << "  point="    << total_point/n
                  << "  buf="    << buffer.size()
                  << "  time="   << std::setprecision(1) << elapsed << "s" << std::endl;

        // ── Checkpoint ───────────────────────────────────────────────────────
        if ((iter + 1) % args.save_every == 0 || iter == args.iterations - 1) {
            std::ostringstream oss;
            oss << arch << "_" << std::setfill('0') << std::setw(6) << iter << ".pt";
            fs::path ckpt_path = ckpt_dir / oss.str();
            std::visit([&](auto& m) { torch::save(m, ckpt_path.string()); }, model_var);
            fs::path json_path = ckpt_dir / (arch + "_config.json");
            {
                // featureDim/inputDescr persisted directly (rather than
                // re-derived from player_stone_place_limit/global_stone_place_limit)
                // so server.cpp's load_model() doesn't need to round-trip the
                // full limit structure through the checkpoint JSON - see
                // compute_input_descr()'s doc comment (training/self_play.h).
                // model_cfg is the same one build_model() constructed the
                // network from, above.
                json cfg_json = game_cfg.to_json();
                cfg_json.update(model_cfg->to_json());
                std::ofstream(json_path) << cfg_json.dump(2) << "\n";
            }
            std::ostringstream toss;
            toss << arch << "_" << std::setfill('0') << std::setw(6) << iter << "_traj" << ".json";
            fs::path traj_path = ckpt_dir / toss.str();
            {
                json trajs = json::array();
                for (auto& traj : traj_store) {
                    json t = json::array();
                    for (auto& ply : traj) t.push_back(ply.to_json());
                    trajs.push_back(std::move(t));
                }
                std::ofstream(traj_path) << trajs.dump() << "\n";
            }
            traj_store.clear();
            std::cout << "  Saved " << ckpt_path << std::endl;
        }
    }

    return 0;
}
