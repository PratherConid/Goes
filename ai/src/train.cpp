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
    // requires --net-arch transformer (see the assert in main()) but the
    // browser-facing public/game_presets/*_fpo.json presets all set it true.
    std::string game_config_path;
    int gnn_hidden_dim    = 128;
    int unet_hidden_dim   = 16;
    int cnn_hidden_dim    = 64;
    int cnn_conv_size     = 5;
    int transformer_hidden_dim = 128;
    int num_layers        = 9;
    int num_attn_layers   = 3;  // Transformer's cross-attention/history self-attention depth
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
    // Opaque checkpoint-directory hash to retrain from - see
    // validate_retrain_source_dir/main()'s replay phase. Mutually exclusive
    // with resume_tag (enforced at parse time in parse_args): unlike
    // --resume, this trains a FRESH model (never loads the source's weights)
    // against the source directory's existing _traj.json games, replayed in
    // order into a NEW checkpoint directory, before continuing with live
    // self-play for any remaining iterations.
    std::string retrain_tag;
};

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --game-config PATH        (required) Path to a GameConfig JSON file (same shape\n"
              << "                            as shared/types.ts's GameConfig.toJSON(), e.g. a file\n"
              << "                            under public/game_presets/) - forcedPassOnly requires\n"
              << "                            --net-arch transformer (see the assert in main())\n"
              << "  --gnn-hidden-dim N        GNN hidden dimension (default: 128)\n"
              << "  --unet-hidden-dim N       UNet hidden dimension (default: 16)\n"
              << "  --cnn-hidden-dim N        CNN hidden dimension (default: 64)\n"
              << "  --cnn-conv-size N         CNN convolution kernel size - must be odd and > 1\n"
              << "                            (default: 5)\n"
              << "  --transformer-hidden-dim N  Transformer hidden dimension (default: 128)\n"
              << "  --num-layers N            GNN message-passing layers (default: 9)\n"
              << "  --num-attn-layers N       Transformer history self-attention/cross-attention\n"
              << "                            layers (default: 3)\n"
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
              << "  --retrain TAG             Train a FRESH model against ai/checkpoints/TAG's existing\n"
              << "                            self-play data instead of generating new games: replays\n"
              << "                            its _traj.json files in order into a new checkpoint\n"
              << "                            directory, then continues with live self-play for any\n"
              << "                            remaining --iterations - errors if TAG doesn't exist or\n"
              << "                            its saved game config doesn't exactly match --game-config.\n"
              << "                            Mutually exclusive with --resume\n"
              << "  --net-arch auto|cnn|unet|gnn|transformer  Network architecture (default: auto)\n"
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
        else if (a == "--cnn-conv-size") {
            args.cnn_conv_size = std::stoi(argv[++i]);
            if (args.cnn_conv_size <= 1 || args.cnn_conv_size % 2 == 0) {
                std::cerr << "--cnn-conv-size must be an odd integer > 1 (got " << args.cnn_conv_size << ")\n";
                std::exit(1);
            }
        }
        else if (a == "--transformer-hidden-dim") args.transformer_hidden_dim = std::stoi(argv[++i]);
        else if (a == "--num-layers")      args.num_layers      = std::stoi(argv[++i]);
        else if (a == "--num-attn-layers") args.num_attn_layers = std::stoi(argv[++i]);
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
        else if (a == "--resume") {
            if (!args.retrain_tag.empty()) {
                std::cerr << "--resume and --retrain are mutually exclusive\n"; std::exit(1);
            }
            args.resume_tag = argv[++i];
        }
        else if (a == "--retrain") {
            if (!args.resume_tag.empty()) {
                std::cerr << "--resume and --retrain are mutually exclusive\n"; std::exit(1);
            }
            args.retrain_tag = argv[++i];
        }
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

// Returns "cnn", "unet", "gnn", or "transformer" — the architecture that will actually be used.
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
    // Topology-agnostic by design (flattens features into one MLP, no adjacency/shape assumption),
    // so no board-type gate here - unlike cnn/unet above.
    if (args.net_arch == "transformer") return "transformer";
    if (args.net_arch == "auto") return grid2d_supported ? "cnn" : "gnn";
    std::cerr << "Error: unknown --net-arch '" << args.net_arch
              << "'. Valid options: auto, cnn, unet, gnn, transformer.\n";
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
    if (cfg.model_type == "transformer")
        return Transformer(bc, static_cast<const TransformerConfig&>(cfg), game_cfg.num_players, game_cfg.num_stones);
    // adj_norms is only needed to size the GNN's neighbor-count embedding
    // table (max_degree); compute it locally rather than threading it
    // through build_model's signature for architectures that don't use it.
    auto adj_norms = compute_adj_norms(bc, torch::kCPU);
    return MessagePassingGNN(static_cast<const GNNConfig&>(cfg), game_cfg.num_players, game_cfg.num_stones, adj_norms);
}

// Returns a pointer to TransformerConfig::history_descr when cfg is a transformer config, else
// nullptr - used to thread the Transformer's minimal per-ply history descriptor into self-play
// calls (generate_one_ply_per_game()/trajectory_to_record()) without adding a parameter anywhere
// else, since TransformerConfig's own persisted field is the single source of truth. Same safe-
// downcast reasoning as build_model()'s static_casts above.
static const nlohmann::json* history_descr_ptr(const ModelConfig& cfg) {
    return cfg.model_type == "transformer"
        ? &static_cast<const TransformerConfig&>(cfg).history_descr
        : nullptr;
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

// Hashes model_cfg+game_cfg+a timestamp into a fresh, unique directory name
// (see sha256_hex, util/sha256.h) and creates it - errors (exits) if that
// path somehow already exists. Used for a plain fresh run, and for
// --retrain's output directory (always distinct from its source - see
// main()'s --retrain handling).
static fs::path fresh_checkpoint_dir(const Args& args, const GameConfig& game_cfg,
                                      const ModelConfig& model_cfg)
{
    auto timestamp = std::to_string(
        std::chrono::system_clock::now().time_since_epoch().count());
    std::string hash = sha256_hex(model_cfg.to_json().dump() + game_cfg.to_json().dump() + timestamp);
    fs::path ckpt_dir = fs::path(args.checkpoint_dir) / hash;
    if (fs::exists(ckpt_dir)) {
        std::cerr << "Error: checkpoint directory already exists (hash collision?): " << ckpt_dir << "\n";
        std::exit(1);
    }
    fs::create_directories(ckpt_dir);
    std::cout << "New checkpoint directory: " << hash
              << " (pass --resume " << hash << " to continue this run later)\n";
    return ckpt_dir;
}

// Errors (exits) if `dir` has no checkpoint config to validate against, or
// that config doesn't strong_equal-match the current game_cfg/model_cfg
// exactly. Used for --resume, where the model architecture must match too
// (weights get loaded directly into it) - see validate_retrain_source_dir
// for --retrain's separately/more-loosely-checked case.
static void validate_checkpoint_dir_config(const fs::path& dir, const GameConfig& game_cfg,
                                            const ModelConfig& model_cfg)
{
    auto cfg_path = find_config_json(dir);
    if (!cfg_path.has_value()) {
        std::cerr << "Error: --resume directory has no checkpoint config to validate against: "
                  << dir << "\n";
        std::exit(1);
    }
    json existing = json::parse(std::ifstream(*cfg_path));
    GameConfig existing_game_cfg = parse_game_cfg(existing);
    auto existing_model_cfg = parse_model_config(existing);
    if (!strong_equal(game_cfg, existing_game_cfg) || !strong_equal(model_cfg, *existing_model_cfg)) {
        // Same combined game+model shape the checkpoint's own config.json is
        // saved as (see save_checkpoint(), below) - so "current" is directly
        // comparable to `existing`, already in that same shape.
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

// Errors (exits) if `dir` has no checkpoint config to validate against, or
// its saved GameConfig doesn't strong_equal-match the current one. Deliberately
// does NOT check model config (architecture/hidden_dim/etc.) - unlike
// --resume, --retrain never loads the source directory's weights, so only
// the game rules need to match for its _traj.json games to replay
// meaningfully into the current (freshly-initialized) model.
static void validate_retrain_source_dir(const fs::path& dir, const GameConfig& game_cfg) {
    auto cfg_path = find_config_json(dir);
    if (!cfg_path.has_value()) {
        std::cerr << "Error: --retrain directory has no checkpoint config to validate against: "
                  << dir << "\n";
        std::exit(1);
    }
    json existing = json::parse(std::ifstream(*cfg_path));
    GameConfig existing_game_cfg = parse_game_cfg(existing);
    if (!strong_equal(game_cfg, existing_game_cfg)) {
        std::cerr << "Error: --retrain config mismatch - current --game-config doesn't exactly match "
                  << *cfg_path << "\n"
                  << "Diff (current -> to be retrained from):\n"
                  << json::diff(game_cfg.to_json(), existing_game_cfg.to_json()).dump(2) << "\n";
        std::exit(1);
    }
}

// Determines this run's checkpoint directory for the non-retrain case:
// - no --resume: a fresh directory (fresh_checkpoint_dir).
// - --resume TAG: errors (exits) if ai/checkpoints/TAG doesn't exist;
//   otherwise validates it (validate_checkpoint_dir_config) and returns the
//   existing directory as-is for resume() to load from.
static fs::path handle_checkpoint_dir(const Args& args, const GameConfig& game_cfg,
                                       const ModelConfig& model_cfg)
{
    if (args.resume_tag.empty()) return fresh_checkpoint_dir(args, game_cfg, model_cfg);
    fs::path ckpt_dir = fs::path(args.checkpoint_dir) / args.resume_tag;
    if (!fs::exists(ckpt_dir)) {
        std::cerr << "Error: --resume directory does not exist: " << ckpt_dir << "\n";
        std::exit(1);
    }
    validate_checkpoint_dir_config(ckpt_dir, game_cfg, model_cfg);
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
        buffer.add(trajectory_to_record(*git, game_cfg, bc, model_cfg.input_descr, history_descr_ptr(model_cfg)));

    if (!recent_games.empty())
        std::cout << "  Reconstructed replay buffer: " << recent_games.size() << " games" << std::endl;
    return start_iter;
}

// ── Training step ─────────────────────────────────────────────────────────────

// Runs one iteration's worth of backprop against whatever is currently in
// `buffer` (sampling train_fraction*buffer.size()/batch_size batches,
// rounded up), and prints the same "[iter ...] loss=..." summary line the
// live self-play loop always has. `t0` is purely for the printed elapsed
// time - the live loop passes a start time from before self-play so
// generation+training are reported together, while a replay-phase caller can
// pass a start time from just before this call to report training time
// alone; this function doesn't care which.
//
// Returns false (and prints "buffer too small, skipping train step" instead)
// exactly when buffer.size() < batch_size - callers must skip
// checkpoint-saving too in that case, matching the original inline code's
// `continue` past the whole rest of the iteration.
struct Losses {
    torch::Tensor total, policy, stone, territory, point;
};

// Shared loss math for every architecture - factored out so the training loop's per-architecture
// std::visit branch only differs in how (policy, ownership, p_tgt, so_tgt, to_tgt) are obtained,
// not in how they're turned into a loss.
static Losses compute_losses(const torch::Tensor& policy, const torch::Tensor& ownership,
                              const torch::Tensor& p_tgt, const torch::Tensor& so_tgt,
                              const torch::Tensor& to_tgt, const std::string& score_rule,
                              int num_stones, int N)
{
    // Policy loss: cross-entropy against MCTS visit distribution, scaled down by
    // log(action count) - cross-entropy over num_stones*N+1 actions grows with
    // both board size and stone count, so this keeps the loss magnitude
    // comparable across differently sized action spaces.
    auto log_policy  = torch::log(policy.clamp_min(1e-8f));
    auto policy_loss = -(p_tgt * log_policy).sum(-1).mean() / std::log(static_cast<float>(N * num_stones));

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
    auto predicted_points = estimate_stone_points(ownership, score_rule);        // (B,num_stones)
    auto actual_points    = estimate_stone_points(actual_ownership, score_rule);  // (B,num_stones)
    auto point_diff = (predicted_points - actual_points) / N * num_stones;              // (B,num_stones)
    auto point_loss = point_diff.pow(2).sum(-1).mean();

    auto loss = policy_loss + stone_loss + territory_loss + point_loss;
    return {loss, policy_loss, stone_loss, territory_loss, point_loss};
}

static bool run_training_iteration(
    int iter, AnyModel& model_var, torch::optim::Adam& optimizer, ReplayBuffer& buffer,
    std::mt19937& rng, const GameConfig& game_cfg, const BoardConfig& bc,
    const AdjNorms& adj_norms, torch::Device device,
    float train_fraction, int batch_size,
    std::chrono::high_resolution_clock::time_point t0)
{
    if (buffer.size() < batch_size) {
        std::cout << "[iter " << iter << "] buffer too small ("
                  << buffer.size() << "), skipping train step" << std::endl;
        return false;
    }

    std::visit([](auto& m) { m->train(); }, model_var);
    double total_loss = 0, total_pol = 0, total_stone = 0, total_territory = 0, total_point = 0;

    int train_steps = static_cast<int>(
        std::ceil(train_fraction * buffer.size() / batch_size));
    int num_stones = game_cfg.num_stones;

    for (int step = 0; step < train_steps; step++) {
        torch::Tensor policy, ownership, p_tgt, so_tgt, to_tgt;

        std::visit([&](auto& m) {
            using M = std::decay_t<decltype(m)>;
            if constexpr (std::is_same_v<M, Transformer>) {
                auto hb = buffer.sample_with_history(batch_size, rng);
                std::tie(policy, ownership) = m->forward(hb.hist_features.to(device), hb.hist_mask.to(device),
                                                          hb.cur_features.to(device), hb.legal_mask.to(device));
                p_tgt = hb.policy_target.to(device);
                so_tgt = hb.stone_owner.to(device);
                to_tgt = hb.territory_owner.to(device);
            } else {
                auto [x_, mask_, p_, so_, to_] = buffer.sample(batch_size, rng);
                torch::Tensor x = x_.to(device), mask = mask_.to(device);
                p_tgt = p_.to(device); so_tgt = so_.to(device); to_tgt = to_.to(device);
                if constexpr (std::is_same_v<M, MessagePassingGNN>)
                    std::tie(policy, ownership) = m->forward(x, adj_norms, mask);
                else
                    std::tie(policy, ownership) = m->forward(x, mask);
            }
        }, model_var);

        auto losses = compute_losses(policy, ownership, p_tgt, so_tgt, to_tgt,
                                      game_cfg.score_rule, num_stones, bc.N);

        optimizer.zero_grad();
        losses.total.backward();
        torch::nn::utils::clip_grad_norm_(
            std::visit([](auto& m) { return m->parameters(); }, model_var), 1.0);
        optimizer.step();

        total_loss     += losses.total.item<double>();
        total_pol      += losses.policy.item<double>();
        total_stone    += losses.stone.item<double>();
        total_territory += losses.territory.item<double>();
        total_point    += losses.point.item<double>();
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
    return true;
}

// Writes <arch>_XXXXXX.pt (weights), <arch>_config.json (game_cfg+model_cfg,
// same combined shape validate_checkpoint_dir_config()/validate_retrain_source_dir()
// read back), and <arch>_XXXXXX_traj.json (games_to_dump, JSON-array-of-arrays
// via PlyResult::to_json()) into ckpt_dir for iteration `iter`. Shared by the
// live self-play loop (games_to_dump = its accumulated traj_store) and
// --retrain's replay phase (games_to_dump = the current span's historical
// games, re-dumped into the new output directory so a later --resume from it
// doesn't start with an empty buffer).
static void save_checkpoint(
    const fs::path& ckpt_dir, const std::string& arch, int iter,
    AnyModel& model_var, const GameConfig& game_cfg, const ModelConfig& model_cfg,
    const std::vector<std::vector<PlyResult>>& games_to_dump)
{
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
        // network from.
        json cfg_json = game_cfg.to_json();
        cfg_json.update(model_cfg.to_json());
        std::ofstream(json_path) << cfg_json.dump(2) << "\n";
    }
    std::ostringstream toss;
    toss << arch << "_" << std::setfill('0') << std::setw(6) << iter << "_traj" << ".json";
    fs::path traj_path = ckpt_dir / toss.str();
    {
        json trajs = json::array();
        for (auto& traj : games_to_dump) {
            json t = json::array();
            for (auto& ply : traj) t.push_back(ply.to_json());
            trajs.push_back(std::move(t));
        }
        std::ofstream(traj_path) << trajs.dump() << "\n";
    }
    std::cout << "  Saved " << ckpt_path << std::endl;
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

    auto bc = build_board_config(game_cfg.board_type, game_cfg.board_args);
    std::cout << "Board: " << game_cfg.board_type;
    for (int a : game_cfg.board_args) std::cout << " " << a;
    std::cout << "  N=" << bc.N << std::endl;

    const std::string arch = effective_arch(args, game_cfg.board_type);

    // Requires forced_pass_only=False, unless --net-arch transformer. When forced_pass_only is
    // enabled, a player may only pass when no traditional placement is legal. In this case, players
    // will be forced to kill their own groups, and the game only ends when both players have no
    // legal moves simultaneously, which closely depends on the full history of the game. CNN/UNet/GNN
    // receive only per-node features derived from the current board, so they cannot function
    // correctly in this case - only the history-aware transformer architecture can.
    assert((!game_cfg.forced_pass_only || arch == "transformer") &&
           "forced_pass_only=true requires --net-arch transformer");

    torch::Device device = (torch::cuda::is_available() && !args.cpu)
        ? torch::kCUDA : torch::kCPU;
    std::cout << "Device: " << device << std::endl;

    nlohmann::json input_descr = compute_input_descr(game_cfg, bc.N);
    int in_dim = input_descr.at("totalDims").get<int>();

    // The transformer's separate, much narrower per-ply HISTORY descriptor (plyMod +
    // stoneOccupancy only) - built directly here rather than via any shared function/filter on
    // compute_input_descr(), since it has exactly one call site (TransformerConfig construction,
    // just below) and no reuse to justify one. Never used for the current ply (which uses
    // input_descr above, like every other architecture) and never used by CNN/UNet/GNN.
    nlohmann::json history_descr;
    if (arch == "transformer") {
        int tl_len = (int)game_cfg.turn_list.size();
        int ns = game_cfg.num_stones;
        history_descr = {
            {"blocks", json::array({json::array({"plyMod", tl_len}), json::array({"stoneOccupancy", ns})})},
            {"totalDims", tl_len + ns + 1}
        };
    }
    int hidden_dim = (arch == "cnn")  ? args.cnn_hidden_dim
                    : (arch == "unet") ? args.unet_hidden_dim
                    : (arch == "transformer") ? args.transformer_hidden_dim
                                        : args.gnn_hidden_dim;
    std::unique_ptr<ModelConfig> model_cfg;
    if (arch == "cnn")       model_cfg = std::make_unique<CNNConfig>(in_dim, hidden_dim, input_descr, args.cnn_conv_size);
    else if (arch == "unet") model_cfg = std::make_unique<UNetConfig>(in_dim, hidden_dim, input_descr);
    else if (arch == "transformer") model_cfg = std::make_unique<TransformerConfig>(in_dim, hidden_dim, args.num_attn_layers, input_descr, history_descr);
    else                     model_cfg = std::make_unique<GNNConfig>(in_dim, hidden_dim, args.num_layers, input_descr);
    auto model_var = build_model(bc, *model_cfg, game_cfg);
    std::visit([&](auto& m) { m->to(device); }, model_var);

    // Resume from checkpoint (weights + replay buffer state), or set up
    // --retrain's source (read-only, existing) + destination (fresh) split.
    fs::path ckpt_dir;
    fs::path retrain_source_dir;
    ReplayBuffer buffer(args.buffer_size);
    int start_iter = 0;
    if (!args.retrain_tag.empty()) {
        retrain_source_dir = fs::path(args.checkpoint_dir) / args.retrain_tag;
        if (!fs::exists(retrain_source_dir)) {
            std::cerr << "Error: --retrain directory does not exist: " << retrain_source_dir << "\n";
            return 1;
        }
        validate_retrain_source_dir(retrain_source_dir, game_cfg);
        // Never write into retrain_source_dir - always a fresh directory, so
        // the source run's own checkpoints/trajectories are never touched.
        ckpt_dir = fresh_checkpoint_dir(args, game_cfg, *model_cfg);
    } else {
        ckpt_dir = handle_checkpoint_dir(args, game_cfg, *model_cfg);
        start_iter = resume(ckpt_dir, *model_cfg, model_var, game_cfg, bc,
                             args.buffer_size, buffer);
    }

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

    int iter = start_iter;

    // ── --retrain replay phase ───────────────────────────────────────────────
    // Replays retrain_source_dir's existing _traj.json files, in ascending
    // (chronological) order, into the freshly-initialized model_var/buffer -
    // no self-play here, just the games that were already recorded. Each
    // file's games are added to the buffer, then exactly as many training
    // iterations run as the source file originally spanned (its own iteration
    // number minus the previous file's), before a checkpoint is saved into
    // ckpt_dir (never retrain_source_dir) so the new output directory stays
    // self-contained for a later --resume. Once every source file is
    // consumed (or args.iterations is reached first), execution falls
    // through into the live self-play loop below exactly as an ordinary run
    // would, continuing from wherever `iter` ended up.
    if (!args.retrain_tag.empty()) {
        // trajectory_iterations_desc returns newest-first (mirrors resume()'s
        // own use of it); reversed here for the ascending order this phase needs.
        std::vector<int> iters_desc = trajectory_iterations_desc(retrain_source_dir, model_cfg->model_type);
        int prev_it = -1;
        bool hit_iteration_cap = false;
        for (auto it_rit = iters_desc.rbegin(); it_rit != iters_desc.rend() && !hit_iteration_cap; ++it_rit) {
            int it = *it_rit;
            std::ostringstream toss;
            toss << model_cfg->model_type << "_" << std::setfill('0') << std::setw(6) << it << "_traj.json";
            std::ifstream f(retrain_source_dir / toss.str());
            if (!f) { prev_it = it; continue; }
            json trajs; f >> trajs;

            std::vector<std::vector<PlyResult>> span_games;
            span_games.reserve(trajs.size());
            for (auto& gj : trajs) {
                std::vector<PlyResult> game;
                game.reserve(gj.size());
                for (auto& p : gj) game.push_back(parse_ply_result(p));
                buffer.add(trajectory_to_record(game, game_cfg, bc, model_cfg->input_descr, history_descr_ptr(*model_cfg)));
                span_games.push_back(std::move(game));
            }

            int k = it - prev_it;  // # original iterations this file spans
            bool did_train = false;
            for (int j = 0; j < k; j++) {
                if (iter >= args.iterations) { hit_iteration_cap = true; break; }
                auto t0 = std::chrono::high_resolution_clock::now();
                did_train = run_training_iteration(iter, model_var, optimizer, buffer, rng,
                                                    game_cfg, bc, adj_norms, device,
                                                    args.train_fraction, args.batch_size, t0);
                iter++;
            }
            if (!hit_iteration_cap && did_train)
                save_checkpoint(ckpt_dir, arch, iter - 1, model_var, game_cfg, *model_cfg, span_games);

            prev_it = it;
        }
    }

    int ply_iter = 0;
    for (; iter < args.iterations; iter++) {
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
                args.verbosity, history_descr_ptr(*model_cfg));
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
                    auto& score = pool[slot].score();
                    // stone_count/territory are stone-indexed maps (ScoreData) that
                    // omit a stone type entirely if it never appears on the board -
                    // .count()/.at() below default those absent entries to 0.
                    auto stone_at = [](const std::unordered_map<int,int>& m, int s) {
                        auto it = m.find(s);
                        return it != m.end() ? it->second : 0;
                    };

                    std::cout << "  game " << (games_this_iter + 1)
                              << "/" << args.self_play_games
                              << "  plies=" << trajectories[slot].size()
                              << "  stones=[";
                    for (int s = 1; s <= pool[slot].num_stones; s++) std::cout << stone_at(score.stone_count, s) << ",";
                    std::cout << "]  territories=[";
                    for (int s = 1; s <= pool[slot].num_stones; s++) std::cout << stone_at(score.territory, s) << ",";
                    std::cout << "]  winners=[";
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

        bool did_train = run_training_iteration(iter, model_var, optimizer, buffer, rng,
                                                 game_cfg, bc, adj_norms, device,
                                                 args.train_fraction, args.batch_size, t0);
        if (!did_train) continue;

        // ── Checkpoint ───────────────────────────────────────────────────────
        if ((iter + 1) % args.save_every == 0 || iter == args.iterations - 1) {
            save_checkpoint(ckpt_dir, arch, iter, model_var, game_cfg, *model_cfg, traj_store);
            traj_store.clear();
        }
    }

    return 0;
}
