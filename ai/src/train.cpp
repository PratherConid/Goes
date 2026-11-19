// Main training script: KataGo's synchronous self-play loop (python/selfplay/synchronous_loop.sh) -
// each iteration plays a batch of self-play games with the latest model, adds their rows to the
// training window, trains for a number of samples tied to the new data, and exports a new model.
// See ai/Readme.md's "Training".
//
// Usage: goes_train --game-config <path> [--iterations 200] ...
// Run with --help for the full option list.
#include "game/board_config.h"
#include "game/cleg_eval.h"
#include "game/cleg_parser.h"
#include "model/any_model.h"
#include "model/features.h"
#include "model/model_config.h"
#include "training/self_play.h"
#include "training/replay_buffer.h"
#include "util/sha256.h"
#include <torch/torch.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "nlohmann/json.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;
using Clock = std::chrono::high_resolution_clock;

// ── CLI argument parsing ──────────────────────────────────────────────────────

struct Args {
    // A GameConfig JSON file (shared/gameConfig.ts's GameConfig.toJSON() wire shape - the same as
    // server.cpp's /move request `config`), e.g. one under public/game_presets/.
    std::string game_config_path;
    int gnn_hidden_dim = 128;
    int unet_hidden_dim = 16;
    int cnn_hidden_dim = 64;
    int cnn_conv_size = 3;
    int transformer_hidden_dim = 128;
    int num_layers = 9;
    int num_attn_layers = 8;
    // Loop sizes and training-window settings default to KataGo's synchronous_loop.sh values.
    int iterations = 200;
    int saves_every = 8;
    int games_per_iteration = 500;
    int gamegen_batch_size = 128;
    int nn_server_threads = 1;
    int nn_max_batch = 128;
    int full_visits = 600;
    int cheap_visits = 100;
    float cheap_prob = 0.75f;
    int batch_size = 128;
    float lr = 1e-3f;
    float l2 = 1e-4f;
    bool lr_warmup = true;
    float c_puct = 1.1f;
    int64_t min_rows = 100000;
    double taper_window_scale = 50000;
    double taper_window_exponent = 0.65;
    double expand_window_per_row = 0.4;
    // KataGo's is 8: Goes has no symmetry augmentation, so each reuse of a row is an identical input.
    double max_train_per_data = 4;
    int64_t max_train_samples_per_iteration = 500000;
    double swa_scale = 8;
    int64_t swa_period_samples = 80000;
    std::string checkpoint_dir = "ai/checkpoints";
    std::string net_arch = "auto";
    bool cpu = false;
    int verbosity = 1;
    std::optional<std::pair<float, float>> linear_move_bound;
    std::string resume_tag;
    std::string retrain_tag;
};

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --game-config PATH        (required) GameConfig JSON file (shared/gameConfig.ts's\n"
              << "                            GameConfig.toJSON() shape, e.g. under public/game_presets/) -\n"
              << "                            forcedPassOnly requires --net-arch transformer\n"
              << "  --net-arch auto|cnn|unet|gnn|transformer  Network architecture (default: auto)\n"
              << "  --gnn-hidden-dim N        GNN hidden dimension (default: 128)\n"
              << "  --unet-hidden-dim N       UNet hidden dimension (default: 16)\n"
              << "  --cnn-hidden-dim N        CNN hidden dimension (default: 64)\n"
              << "  --cnn-conv-size N         CNN kernel size, odd and > 1 (default: 3)\n"
              << "  --transformer-hidden-dim N  Transformer hidden dimension (default: 128)\n"
              << "  --num-layers N            GNN message-passing layers (default: 9)\n"
              << "  --num-attn-layers N       Transformer cross-attention layers (default: 8)\n"
              << "  --iterations N            Self-play/train/export cycles (default: 200)\n"
              << "  --saves-every N           Keep the exported model file of every Nth iteration and of\n"
              << "                            iteration --iterations (default: 8)\n"
              << "  --games-per-iteration N   Self-play games finished per cycle (default: 500)\n"
              << "  --gamegen-batch-size N    Games played concurrently, one thread each (default: 128)\n"
              << "  --nn-server-threads N     Threads evaluating the games' network requests (default: 1)\n"
              << "  --nn-max-batch N          Most network requests evaluated as one batch (default: 128)\n"
              << "  --full-visits N           Visits of a full (recorded) search (default: 600)\n"
              << "  --cheap-visits N          Visits of a cheap (unrecorded) search (default: 100)\n"
              << "  --cheap-prob F            Probability a move gets a cheap search (default: 0.75)\n"
              << "  --c-puct F                PUCT exploration constant (default: 1.1)\n"
              << "  --batch-size N            Training batch size (default: 128)\n"
              << "  --lr F                    Adam learning rate after warmup (default: 0.001)\n"
              << "  --l2 F                    Adam weight decay (default: 0.0001)\n"
              << "  --no-lr-warmup            Skip the learning-rate warmup over the first 2M samples\n"
              << "  --min-rows N              Rows needed before training starts (default: 100000)\n"
              << "  --taper-window-scale F    Training-window power-law scale (default: 50000)\n"
              << "  --taper-window-exponent F Training-window power-law exponent (default: 0.65)\n"
              << "  --expand-window-per-row F Training-window growth per new row (default: 0.4)\n"
              << "  --max-train-per-data F    Training samples per new data row (default: 4)\n"
              << "  --max-train-samples-per-iteration N  Cap on samples per cycle (default: 500000)\n"
              << "  --swa-scale F             Weight-average window, in snapshots (default: 8)\n"
              << "  --swa-period-samples N    Samples between weight-average snapshots (default: 80000;\n"
              << "                            0 disables it, exporting the raw model)\n"
              << "  --checkpoint-dir PATH     Checkpoint directory (default: ai/checkpoints)\n"
              << "  --resume TAG              Continue the run in <checkpoint-dir>/TAG - errors if its saved\n"
              << "                            config doesn't exactly match --game-config/architecture flags\n"
              << "  --retrain TAG             Train a fresh model on <checkpoint-dir>/TAG's recorded games,\n"
              << "                            in order, into a new directory, then continue with self-play\n"
              << "                            for any remaining --iterations (exclusive with --resume)\n"
              << "  --linear-move-bound K1 K2 End games after Uniform(K1,K2)*N plies, resampled per game\n"
              << "  --cpu                     Force CPU even if CUDA is available\n"
              << "  --verbosity N             0=silent, 1=per-game, >=2=network batch stats (default: 1)\n";
}

static Args parse_args(int argc, char* argv[]) {
    Args args;
    auto need = [&](int& i) -> std::string {
        if (i + 1 >= argc) { std::cerr << argv[i] << " requires a value\n"; std::exit(1); }
        return argv[++i];
    };
    auto positive = [](const std::string& flag, double v) {
        if (v <= 0) { std::cerr << flag << " must be > 0 (got " << v << ")\n"; std::exit(1); }
    };
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") { print_usage(argv[0]); std::exit(0); }
        else if (a == "--game-config") args.game_config_path = need(i);
        else if (a == "--net-arch") args.net_arch = need(i);
        else if (a == "--gnn-hidden-dim") args.gnn_hidden_dim = std::stoi(need(i));
        else if (a == "--unet-hidden-dim") args.unet_hidden_dim = std::stoi(need(i));
        else if (a == "--cnn-hidden-dim") args.cnn_hidden_dim = std::stoi(need(i));
        else if (a == "--cnn-conv-size") {
            args.cnn_conv_size = std::stoi(need(i));
            if (args.cnn_conv_size <= 1 || args.cnn_conv_size % 2 == 0) {
                std::cerr << "--cnn-conv-size must be an odd integer > 1 (got " << args.cnn_conv_size << ")\n";
                std::exit(1);
            }
        }
        else if (a == "--transformer-hidden-dim") args.transformer_hidden_dim = std::stoi(need(i));
        else if (a == "--num-layers") args.num_layers = std::stoi(need(i));
        else if (a == "--num-attn-layers") args.num_attn_layers = std::stoi(need(i));
        else if (a == "--iterations") args.iterations = std::stoi(need(i));
        else if (a == "--saves-every") { args.saves_every = std::stoi(need(i)); positive(a, args.saves_every); }
        else if (a == "--games-per-iteration") { args.games_per_iteration = std::stoi(need(i)); positive(a, args.games_per_iteration); }
        else if (a == "--gamegen-batch-size") { args.gamegen_batch_size = std::stoi(need(i)); positive(a, args.gamegen_batch_size); }
        else if (a == "--nn-server-threads") { args.nn_server_threads = std::stoi(need(i)); positive(a, args.nn_server_threads); }
        else if (a == "--nn-max-batch") { args.nn_max_batch = std::stoi(need(i)); positive(a, args.nn_max_batch); }
        else if (a == "--full-visits") { args.full_visits = std::stoi(need(i)); positive(a, args.full_visits); }
        else if (a == "--cheap-visits") { args.cheap_visits = std::stoi(need(i)); positive(a, args.cheap_visits); }
        else if (a == "--cheap-prob") {
            args.cheap_prob = std::stof(need(i));
            if (args.cheap_prob < 0 || args.cheap_prob >= 1) { std::cerr << "--cheap-prob must be in [0, 1)\n"; std::exit(1); }
        }
        else if (a == "--c-puct") args.c_puct = std::stof(need(i));
        else if (a == "--batch-size") { args.batch_size = std::stoi(need(i)); positive(a, args.batch_size); }
        else if (a == "--lr") args.lr = std::stof(need(i));
        else if (a == "--l2") args.l2 = std::stof(need(i));
        else if (a == "--no-lr-warmup") args.lr_warmup = false;
        else if (a == "--min-rows") { args.min_rows = std::stoll(need(i)); positive(a, (double)args.min_rows); }
        else if (a == "--taper-window-scale") { args.taper_window_scale = std::stod(need(i)); positive(a, args.taper_window_scale); }
        else if (a == "--taper-window-exponent") { args.taper_window_exponent = std::stod(need(i)); positive(a, args.taper_window_exponent); }
        else if (a == "--expand-window-per-row") args.expand_window_per_row = std::stod(need(i));
        else if (a == "--max-train-per-data") { args.max_train_per_data = std::stod(need(i)); positive(a, args.max_train_per_data); }
        else if (a == "--max-train-samples-per-iteration") {
            args.max_train_samples_per_iteration = std::stoll(need(i));
            positive(a, (double)args.max_train_samples_per_iteration);
        }
        else if (a == "--swa-scale") { args.swa_scale = std::stod(need(i)); positive(a, args.swa_scale); }
        else if (a == "--swa-period-samples") args.swa_period_samples = std::stoll(need(i));
        else if (a == "--checkpoint-dir") args.checkpoint_dir = need(i);
        else if (a == "--resume") args.resume_tag = need(i);
        else if (a == "--retrain") args.retrain_tag = need(i);
        else if (a == "--cpu") args.cpu = true;
        else if (a == "--verbosity") args.verbosity = std::stoi(need(i));
        else if (a == "--linear-move-bound") {
            if (i + 2 >= argc) { std::cerr << "--linear-move-bound requires two values: K1 K2\n"; std::exit(1); }
            float k1 = std::stof(argv[++i]), k2 = std::stof(argv[++i]);
            if (k1 > k2) { std::cerr << "--linear-move-bound: K1 must be <= K2\n"; std::exit(1); }
            args.linear_move_bound = {k1, k2};
        }
        else { std::cerr << "Unknown argument: " << a << "\n"; std::exit(1); }
    }
    if (args.game_config_path.empty()) { std::cerr << "--game-config is required\n"; print_usage(argv[0]); std::exit(1); }
    if (!args.resume_tag.empty() && !args.retrain_tag.empty()) {
        std::cerr << "--resume and --retrain are mutually exclusive\n";
        std::exit(1);
    }
    return args;
}

// ── Model factory ─────────────────────────────────────────────────────────────

// The architecture actually used: CNN/UNet need a 2D integer embedding (emb_dim == 2), which "auto"
// prefers; otherwise the GNN. The Transformer is topology-agnostic.
static std::string effective_arch(const Args& args, const std::string& board_descr, unsigned emb_dim) {
    bool grid2d = emb_dim == 2;
    if (args.net_arch == "cnn" || args.net_arch == "unet") {
        if (!grid2d) {
            std::cerr << "Error: --net-arch " << args.net_arch << " is not supported for board description '"
                      << board_descr << "' (emb_dim=" << emb_dim << "): it needs a 2D grid embedding.\n";
            std::exit(1);
        }
        return args.net_arch;
    }
    if (args.net_arch == "gnn" || args.net_arch == "transformer") return args.net_arch;
    if (args.net_arch == "auto") return grid2d ? "cnn" : "gnn";
    std::cerr << "Error: unknown --net-arch '" << args.net_arch << "'. Valid: auto, cnn, unet, gnn, transformer.\n";
    std::exit(1);
}

static AnyModel build_model(const BoardConfig& bc, const ModelConfig& cfg, const GameConfig& game_cfg) {
    if (cfg.model_type == "cnn")
        return CNN(bc, static_cast<const CNNConfig&>(cfg), game_cfg.num_stones, game_cfg.num_players);
    if (cfg.model_type == "unet")
        return UNet(bc, static_cast<const UNetConfig&>(cfg), game_cfg.num_stones, game_cfg.num_players);
    if (cfg.model_type == "transformer")
        return Transformer(bc, static_cast<const TransformerConfig&>(cfg), game_cfg.num_stones, game_cfg.num_players);
    // Only needed to size the GNN's neighbor-count embedding table.
    auto adj_norms = compute_adj_norms(bc, torch::kCPU);
    return MessagePassingGNN(static_cast<const GNNConfig&>(cfg), game_cfg.num_stones, game_cfg.num_players, adj_norms);
}

// TransformerConfig::history_descr for a Transformer run, else nullptr.
static const json* history_descr_ptr(const ModelConfig& cfg) {
    return cfg.model_type == "transformer" ? &static_cast<const TransformerConfig&>(cfg).history_descr : nullptr;
}

static std::vector<torch::Tensor> parameters_of(AnyModel& m) {
    return std::visit([](auto& mm) { return mm->parameters(); }, m);
}

// An independent eval-mode copy of src's weights, through the same serialization checkpoints use.
static AnyModel clone_model(AnyModel& src, const BoardConfig& bc, const ModelConfig& cfg,
                            const GameConfig& game_cfg, torch::Device device) {
    std::ostringstream oss;
    std::visit([&](auto& m) { torch::save(m, oss); }, src);
    AnyModel fresh = build_model(bc, cfg, game_cfg);
    std::istringstream iss(oss.str());
    std::visit([&](auto& m) { torch::load(m, iss); m->to(device); m->eval(); }, fresh);
    return fresh;
}

// ── Training state ────────────────────────────────────────────────────────────

// Everything besides weights and optimizer state that a resumed run needs - KataGo's train_state.
struct TrainState {
    int iteration = 0;            // last completed iteration
    int64_t total_games = 0;
    int64_t total_rows = 0;       // every row ever generated
    int64_t random_rows = 0;      // rows generated by the uniform evaluator, before the first model
    int64_t global_samples = 0;   // training samples taken
    double bucket = 0;            // KataGo's train bucket: samples the new data entitles training to
    bool bucket_started = false;
    int64_t swa_accum = 0;        // samples since the last weight-average snapshot
    bool swa_started = false;     // whether the weight average has taken its first snapshot
    int exported_iteration = -1;  // iteration of the newest exported model; -1 = none yet

    json to_json() const {
        return {{"iteration", iteration}, {"totalGames", total_games}, {"totalRows", total_rows},
                {"randomRows", random_rows}, {"globalSamples", global_samples}, {"bucket", bucket},
                {"bucketStarted", bucket_started}, {"swaAccum", swa_accum}, {"swaStarted", swa_started},
                {"exportedIteration", exported_iteration}};
    }
    static TrainState from_json(const json& j) {
        TrainState s;
        s.iteration = j.at("iteration");
        s.total_games = j.at("totalGames");
        s.total_rows = j.at("totalRows");
        s.random_rows = j.at("randomRows");
        s.global_samples = j.at("globalSamples");
        s.bucket = j.at("bucket");
        s.bucket_started = j.at("bucketStarted");
        s.swa_accum = j.at("swaAccum");
        s.swa_started = j.at("swaStarted");
        s.exported_iteration = j.at("exportedIteration");
        return s;
    }
};

// KataGo's power-law training window (python/shuffle.py's compute_desired_num_rows()): rows from
// before the first model count only up to min_rows, and past that the window grows sublinearly -
// with slope expand_window_per_row at first, flattening per taper_window_exponent - from min_rows.
// 0 while there aren't min_rows usable rows yet (no training).
static int64_t desired_window_rows(const TrainState& st, const Args& args) {
    double usable = (double)std::min(st.random_rows, args.min_rows) + (double)(st.total_rows - st.random_rows);
    if (usable < (double)args.min_rows) return 0;
    double s = args.taper_window_scale, e = args.taper_window_exponent;
    double x = usable - (double)args.min_rows + s;
    double scaled = (std::pow(x, e) - std::pow(s, e)) / (e * std::pow(s, e - 1.0));
    return std::max<int64_t>(args.min_rows, (int64_t)(scaled * args.expand_window_per_row + (double)args.min_rows));
}

// KataGo's learning-rate warmup over the first 2M training samples.
static double lr_warmup_scale(int64_t samples) {
    static const std::pair<int64_t, double> table[] = {
        {250000, 1.0 / 20}, {500000, 1.0 / 14}, {750000, 1.0 / 10}, {1000000, 1.0 / 7},
        {1250000, 1.0 / 5}, {1500000, 1.0 / 3}, {1750000, 1.0 / 2}, {2000000, 1.0 / 1.4}};
    for (auto& [limit, scale] : table) if (samples < limit) return scale;
    return 1.0;
}

// ── Checkpoint utilities ──────────────────────────────────────────────────────

// Checkpoint directory layout:
//   <arch>_config.json        game config + model config (what server.cpp matches /move requests on)
//   <arch>_<iter>.pt          the model exported at the end of that iteration, kept every
//                             --saves-every iterations and at iteration --iterations (what server.cpp
//                             uses)
//   <arch>_<iter>_traj.json   that iteration's self-play games (SelfPlayRunner's trajectory format)
//   trainer/                  what --resume needs: model.pt, weight_average.pt, optimizer.pt, state.json

static std::string iter_name(const std::string& arch, int iter, const std::string& suffix) {
    std::ostringstream oss;
    oss << arch << "_" << std::setfill('0') << std::setw(6) << iter << suffix;
    return oss.str();
}

static std::optional<fs::path> find_config_json(const fs::path& dir) {
    if (!fs::exists(dir)) return std::nullopt;
    const std::string suffix = "_config.json";
    for (auto& e : fs::directory_iterator(dir)) {
        auto name = e.path().filename().string();
        if (name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
            return e.path();
    }
    return std::nullopt;
}

// A fresh directory named by a hash of the configs and a timestamp.
static fs::path fresh_checkpoint_dir(const Args& args, const GameConfig& game_cfg, const ModelConfig& model_cfg) {
    auto timestamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    std::string hash = sha256_hex(model_cfg.to_json().dump() + game_cfg.to_json().dump() + timestamp);
    fs::path dir = fs::path(args.checkpoint_dir) / hash;
    if (fs::exists(dir)) { std::cerr << "Error: checkpoint directory already exists: " << dir << "\n"; std::exit(1); }
    fs::create_directories(dir / "trainer");
    std::cout << "New checkpoint directory: " << hash << " (pass --resume " << hash << " to continue it)\n";
    return dir;
}

// Exits unless `dir`'s saved config matches: the game config always, the model config too when
// check_model (--resume loads weights into the current architecture; --retrain trains a fresh one).
static void validate_dir_config(const fs::path& dir, const GameConfig& game_cfg, const ModelConfig& model_cfg,
                                bool check_model, const std::string& flag) {
    auto cfg_path = find_config_json(dir);
    if (!cfg_path) { std::cerr << "Error: " << flag << " directory has no config to validate against: " << dir << "\n"; std::exit(1); }
    json existing = json::parse(std::ifstream(*cfg_path));
    GameConfig existing_game = parse_game_cfg(existing);
    bool ok = strong_equal(game_cfg, existing_game);
    if (check_model) ok = ok && strong_equal(model_cfg, *parse_model_config(existing));
    if (!ok) {
        json current = game_cfg.to_json();
        if (check_model) current.update(model_cfg.to_json());
        else existing = existing_game.to_json();
        std::cerr << "Error: " << flag << " config mismatch with " << *cfg_path << "\n"
                  << "Diff (current -> saved):\n" << json::diff(current, existing).dump(2) << "\n";
        std::exit(1);
    }
}

// Iterations with a saved <arch>_<iter>_traj.json in `dir`, ascending.
static std::vector<int> trajectory_iterations(const fs::path& dir, const std::string& arch) {
    std::vector<int> iters;
    const std::string prefix = arch + "_", suffix = "_traj.json";
    for (auto& e : fs::directory_iterator(dir)) {
        auto name = e.path().filename().string();
        if (name.rfind(prefix, 0) == 0 && name.size() > prefix.size() + suffix.size() &&
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
            iters.push_back(std::stoi(name.substr(prefix.size(), name.size() - prefix.size() - suffix.size())));
    }
    std::sort(iters.begin(), iters.end());
    return iters;
}

static json load_trajectories(const fs::path& dir, const std::string& arch, int iter) {
    std::ifstream f(dir / iter_name(arch, iter, "_traj.json"));
    json j;
    f >> j;
    return j;
}

static void save_config_json(const fs::path& dir, const std::string& arch, const GameConfig& game_cfg,
                             const ModelConfig& model_cfg) {
    json cfg_json = game_cfg.to_json();
    cfg_json.update(model_cfg.to_json());
    std::ofstream(dir / (arch + "_config.json")) << cfg_json.dump(2) << "\n";
}

// ── Training ──────────────────────────────────────────────────────────────────

struct Losses {
    torch::Tensor total, policy, stone, territory, point, rank;
};

static Losses compute_losses(const ModelOutput& out, const Batch& b, const std::string& score_rule,
                             int num_stones, int N) {
    // Policy: cross-entropy against the search's policy target, scaled by 1/log(action count) so
    // its magnitude is comparable across action-space sizes.
    auto log_policy = torch::log(out.policy.clamp_min(1e-8f));
    auto policy_loss = -(b.policy_target * log_policy).sum(-1).mean() / std::log(static_cast<float>(N * num_stones + 1));

    // Ownership: per-location squared error of the stone/territory distributions (channel 0 =
    // none), summed over channels and averaged over locations and batch.
    auto stone_est = out.ownership.select(1, 0);
    auto territory_est = out.ownership.select(1, 1);
    auto actual_stone = torch::one_hot(b.stone_owner, num_stones + 1).to(torch::kFloat32);
    auto actual_territory = torch::one_hot(b.territory_owner, num_stones + 1).to(torch::kFloat32);
    auto stone_loss = (stone_est - actual_stone).pow(2).sum(-1).mean();
    auto territory_loss = (territory_est - actual_territory).pow(2).sum(-1).mean();

    // Points: expected vs actual per-stone-type totals under the scoring rule, scaled by
    // num_stones/N so they don't dwarf the per-location terms.
    auto actual_ownership = torch::stack({actual_stone, actual_territory}, 1);
    auto point_diff = (estimate_stone_points(out.ownership, score_rule)
                       - estimate_stone_points(actual_ownership, score_rule)) / N * num_stones;
    auto point_loss = point_diff.pow(2).sum(-1).mean();

    // Rank: cross-entropy of each pair's predicted outcome against its blended target, scaled by
    // 1/log 3 so a uniform guess against a one-hot target scores 1.
    auto rank_loss = -(b.rank_target * torch::log(out.rank.clamp_min(1e-8f))).sum(-1).mean() / std::log(3.0f);

    auto total = policy_loss + stone_loss + territory_loss + point_loss + rank_loss;
    return {total, policy_loss, stone_loss, territory_loss, point_loss, rank_loss};
}

// The trained model, its weight average, and the optimizer - the state that persists across iterations.
struct Trainer {
    AnyModel model;
    std::optional<AnyModel> average;  // KataGo's SWA: an exponential moving average of snapshots
    std::unique_ptr<torch::optim::Adam> optimizer;
};

// KataGo's SWA snapshot: average += (model - average) / swa_scale; the first snapshot copies.
static void snapshot_average(Trainer& t, TrainState& st, const Args& args, const BoardConfig& bc,
                             const ModelConfig& cfg, const GameConfig& game_cfg, torch::Device device) {
    if (!st.swa_started || !t.average) {
        t.average = clone_model(t.model, bc, cfg, game_cfg, device);
        st.swa_started = true;
        return;
    }
    torch::NoGradGuard ng;
    auto src = parameters_of(t.model), dst = parameters_of(*t.average);
    for (size_t i = 0; i < src.size(); i++) dst[i].add_(src[i].detach() - dst[i], 1.0 / args.swa_scale);
}

// Trains while the bucket holds at least one batch, up to max_train_samples_per_iteration samples,
// sampling uniformly from the newest `window` rows. Returns the number of steps taken.
static int train_on_bucket(Trainer& t, TrainState& st, ReplayBuffer& buffer, int64_t window, const Args& args,
                           const GameConfig& game_cfg, const BoardConfig& bc, const ModelConfig& model_cfg,
                           const AdjNorms& adj_norms, torch::Device device, std::mt19937& rng, int iter) {
    auto t0 = Clock::now();
    std::visit([](auto& m) { m->train(); }, t.model);
    double sums[6] = {0, 0, 0, 0, 0, 0};
    int steps = 0;
    int64_t samples = 0;
    while (st.bucket >= args.batch_size && samples + args.batch_size <= args.max_train_samples_per_iteration) {
        double lr = args.lr * (args.lr_warmup ? lr_warmup_scale(st.global_samples) : 1.0);
        for (auto& group : t.optimizer->param_groups())
            static_cast<torch::optim::AdamOptions&>(group.options()).lr(lr);

        ModelOutput out;
        Batch b;
        std::visit([&](auto& m) {
            using M = std::decay_t<decltype(m)>;
            if constexpr (std::is_same_v<M, Transformer>) {
                auto hb = buffer.sample_with_history(args.batch_size, window, rng);
                b = hb.current;
                out = m->forward(hb.hist_features.to(device), hb.hist_mask.to(device),
                                 b.features.to(device), b.legal_mask.to(device));
            } else {
                b = buffer.sample(args.batch_size, window, rng);
                if constexpr (std::is_same_v<M, MessagePassingGNN>)
                    out = m->forward(b.features.to(device), adj_norms, b.legal_mask.to(device));
                else
                    out = m->forward(b.features.to(device), b.legal_mask.to(device));
            }
        }, t.model);
        b.policy_target = b.policy_target.to(device);
        b.stone_owner = b.stone_owner.to(device);
        b.territory_owner = b.territory_owner.to(device);
        b.rank_target = b.rank_target.to(device);

        auto losses = compute_losses(out, b, game_cfg.score_rule, game_cfg.num_stones, bc.N);
        t.optimizer->zero_grad();
        losses.total.backward();
        torch::nn::utils::clip_grad_norm_(parameters_of(t.model), 1.0);
        t.optimizer->step();

        int64_t n = b.policy_target.size(0);
        st.bucket -= (double)n;
        st.global_samples += n;
        samples += n;
        steps++;
        if (args.swa_period_samples > 0) {
            st.swa_accum += n;
            if (st.swa_accum >= args.swa_period_samples) {
                st.swa_accum = 0;
                snapshot_average(t, st, args, bc, model_cfg, game_cfg, device);
            }
        }
        torch::Tensor parts[6] = {losses.total, losses.policy, losses.stone, losses.territory, losses.point, losses.rank};
        for (int k = 0; k < 6; k++) sums[k] += parts[k].item<double>();
    }
    if (steps > 0 && args.verbosity >= 1) {
        auto avg = [&](int k) { return sums[k] / steps; };
        std::cout << "[iter " << std::setw(4) << iter << "] train: " << samples << " samples ("
                  << st.global_samples << " total), window=" << window
                  << std::fixed << std::setprecision(4)
                  << "  loss=" << avg(0) << "  policy=" << avg(1) << "  stone=" << avg(2)
                  << "  territory=" << avg(3) << "  point=" << avg(4) << "  rank=" << avg(5)
                  << std::setprecision(1) << "  time=" << std::chrono::duration<double>(Clock::now() - t0).count() << "s"
                  << std::defaultfloat << std::endl;
    }
    return steps;
}

// Adds the bucket allowance for `new_rows` new rows (one pass over all the data the first time
// training is possible - KataGo's first cycle trains about one sample per existing row), then trains.
static void train_iteration(Trainer& t, TrainState& st, ReplayBuffer& buffer, int64_t new_rows, const Args& args,
                            const GameConfig& game_cfg, const BoardConfig& bc, const ModelConfig& model_cfg,
                            const AdjNorms& adj_norms, torch::Device device, std::mt19937& rng, int iter) {
    int64_t window = desired_window_rows(st, args);
    if (window == 0) {
        if (args.verbosity >= 1)
            std::cout << "[iter " << std::setw(4) << iter << "] " << st.total_rows << " rows, " << args.min_rows
                      << " needed before training" << std::endl;
        return;
    }
    double cap = (double)args.max_train_samples_per_iteration;
    if (!st.bucket_started) {
        st.bucket = std::min((double)st.total_rows, cap);
        st.bucket_started = true;
    } else {
        st.bucket = std::min(st.bucket + new_rows * args.max_train_per_data, cap);
    }
    buffer.trim(window);
    train_on_bucket(t, st, buffer, window, args, game_cfg, bc, model_cfg, adj_norms, device, rng, iter);
}

// Whether iteration `iter`'s export is kept as <arch>_<iter>.pt.
static bool keeps_model_file(int iter, const Args& args) {
    return iter % args.saves_every == 0 || iter == args.iterations;
}

// Exports the model self-play and the server should use - the weight average once it exists - and
// saves everything --resume needs under trainer/. The export is also kept as <arch>_<iter>.pt when
// keep_model_file.
static void export_and_save(Trainer& t, TrainState& st, const fs::path& dir, const std::string& arch, int iter,
                            bool keep_model_file) {
    if (keep_model_file) {
        AnyModel& exported = t.average ? *t.average : t.model;
        fs::path path = dir / iter_name(arch, iter, ".pt");
        std::visit([&](auto& m) { torch::save(m, path.string()); }, exported);
    }
    st.exported_iteration = iter;
    fs::create_directories(dir / "trainer");
    std::visit([&](auto& m) { torch::save(m, (dir / "trainer" / "model.pt").string()); }, t.model);
    if (t.average) std::visit([&](auto& m) { torch::save(m, (dir / "trainer" / "weight_average.pt").string()); }, *t.average);
    torch::save(*t.optimizer, (dir / "trainer" / "optimizer.pt").string());
    std::ofstream(dir / "trainer" / "state.json") << st.to_json().dump(2) << "\n";
}

static void print_game_result(int game_num, int total_games, int model_id, const BoardState& state) {
    auto& score = state.score();
    auto at = [](const std::unordered_map<int, int>& m, int s) { auto it = m.find(s); return it != m.end() ? it->second : 0; };
    std::cout << "  game " << game_num << "/" << total_games << "  players=[";
    for (int p = 0; p < state.num_players; p++) std::cout << model_id << ",";
    std::cout << "]  plies=" << state.ply_count() << "  stones=[";
    for (int s = 1; s <= state.num_stones; s++) std::cout << at(score.stone_count, s) << ",";
    std::cout << "]  territories=[";
    for (int s = 1; s <= state.num_stones; s++) std::cout << at(score.territory, s) << ",";
    std::cout << "]  winners=[";
    if (state.winners) for (int w : *state.winners) std::cout << w << ",";
    std::cout << "]" << std::endl;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    auto args = parse_args(argc, argv);

    std::ifstream cfg_file(args.game_config_path);
    if (!cfg_file) { std::cerr << "Cannot open --game-config file: " << args.game_config_path << "\n"; return 1; }
    json cfg_json;
    cfg_file >> cfg_json;
    GameConfig game_cfg = parse_game_cfg(cfg_json);
    // A self-play sampling knob for BoardState::max_plies, not part of the game's rules.
    game_cfg.linear_move_bound = args.linear_move_bound;

    // One board per run, from a fresh random seed (game_cfg.board_seed is ignored).
    const uint32_t board_seed = std::random_device{}();
    auto bc = build_board_from_cleg(*parse_cleg(game_cfg.board_descr), board_seed);
    std::cout << "Board:\n" << game_cfg.board_descr << "  N=" << bc.N << "  seed=" << board_seed << std::endl;
    const std::string arch = effective_arch(args, game_cfg.board_descr, bc.emb_dim);

    // With forced_pass_only, the game's end depends on the full move history, which only the
    // history-aware Transformer sees.
    if (game_cfg.forced_pass_only && arch != "transformer") {
        std::cerr << "Error: forced_pass_only=true requires --net-arch transformer (got '" << arch << "').\n";
        return 1;
    }

    torch::Device device = (torch::cuda::is_available() && !args.cpu) ? torch::kCUDA : torch::kCPU;
    std::cout << "Device: " << device << std::endl;

    json input_descr = compute_input_descr(game_cfg, bc.N);
    int in_dim = input_descr.at("totalDims").get<int>();
    // The Transformer's per-ply history input: just the turn position and stones.
    json history_descr;
    if (arch == "transformer") {
        int tl_len = (int)game_cfg.turn_list.size(), ns = game_cfg.num_stones;
        history_descr = {{"blocks", json::array({json::array({"plyMod", tl_len}), json::array({"stoneOccupancy", ns})})},
                         {"totalDims", tl_len + ns + 1}};
    }
    int hidden_dim = arch == "cnn" ? args.cnn_hidden_dim : arch == "unet" ? args.unet_hidden_dim
                   : arch == "transformer" ? args.transformer_hidden_dim : args.gnn_hidden_dim;
    std::unique_ptr<ModelConfig> model_cfg;
    if (arch == "cnn") model_cfg = std::make_unique<CNNConfig>(in_dim, hidden_dim, input_descr, args.cnn_conv_size);
    else if (arch == "unet") model_cfg = std::make_unique<UNetConfig>(in_dim, hidden_dim, input_descr);
    else if (arch == "transformer")
        model_cfg = std::make_unique<TransformerConfig>(in_dim, hidden_dim, args.num_attn_layers, input_descr, history_descr);
    else model_cfg = std::make_unique<GNNConfig>(in_dim, hidden_dim, args.num_layers, input_descr);
    const json* hist_descr = history_descr_ptr(*model_cfg);

    Trainer trainer{build_model(bc, *model_cfg, game_cfg), std::nullopt, nullptr};
    std::visit([&](auto& m) { m->to(device); }, trainer.model);
    trainer.optimizer = std::make_unique<torch::optim::Adam>(
        parameters_of(trainer.model), torch::optim::AdamOptions(args.lr).weight_decay(args.l2));

    TrainState st;
    ReplayBuffer buffer;
    fs::path ckpt_dir;
    auto adj_norms = compute_adj_norms(bc, device);
    std::mt19937 rng(std::random_device{}());

    if (!args.resume_tag.empty()) {
        ckpt_dir = fs::path(args.checkpoint_dir) / args.resume_tag;
        if (!fs::exists(ckpt_dir / "trainer" / "state.json")) {
            std::cerr << "Error: --resume directory has no trainer state: " << ckpt_dir << "\n";
            return 1;
        }
        validate_dir_config(ckpt_dir, game_cfg, *model_cfg, /*check_model=*/true, "--resume");
        st = TrainState::from_json(json::parse(std::ifstream(ckpt_dir / "trainer" / "state.json")));
        // Weights and optimizer state exist once the first model has been exported.
        if (fs::exists(ckpt_dir / "trainer" / "model.pt")) {
            std::visit([&](auto& m) { torch::load(m, (ckpt_dir / "trainer" / "model.pt").string()); m->to(device); },
                       trainer.model);
            torch::load(*trainer.optimizer, (ckpt_dir / "trainer" / "optimizer.pt").string());
        }
        if (fs::exists(ckpt_dir / "trainer" / "weight_average.pt")) {
            trainer.average = build_model(bc, *model_cfg, game_cfg);
            std::visit([&](auto& m) {
                torch::load(m, (ckpt_dir / "trainer" / "weight_average.pt").string());
                m->to(device);
                m->eval();
            }, *trainer.average);
        }
        // Refill the window from the newest recorded games.
        int64_t window = std::max<int64_t>(desired_window_rows(st, args), args.min_rows);
        std::vector<GameRecord> records;
        int64_t rows = 0;
        auto iters = trajectory_iterations(ckpt_dir, arch);
        for (auto it = iters.rbegin(); it != iters.rend() && rows < window; ++it) {
            json games = load_trajectories(ckpt_dir, arch, *it);
            for (auto g = games.rbegin(); g != games.rend() && rows < window; ++g) {
                records.push_back(replay_trajectory(*g, game_cfg, bc, model_cfg->input_descr, hist_descr));
                rows += records.back().num_rows();
            }
        }
        for (auto r = records.rbegin(); r != records.rend(); ++r) buffer.add(std::move(*r));
        std::cout << "Resuming after iteration " << st.iteration << ": " << st.total_games << " games, "
                  << st.total_rows << " rows (" << buffer.stored_rows() << " reloaded)" << std::endl;
    } else {
        ckpt_dir = fresh_checkpoint_dir(args, game_cfg, *model_cfg);
    }
    save_config_json(ckpt_dir, arch, game_cfg, *model_cfg);

    // ── --retrain: train a fresh model on another run's recorded games, file by file ──
    if (!args.retrain_tag.empty()) {
        fs::path src = fs::path(args.checkpoint_dir) / args.retrain_tag;
        if (!fs::exists(src)) { std::cerr << "Error: --retrain directory does not exist: " << src << "\n"; return 1; }
        validate_dir_config(src, game_cfg, *model_cfg, /*check_model=*/false, "--retrain");
        for (int it : trajectory_iterations(src, arch)) {
            if (it > args.iterations) break;
            json games = load_trajectories(src, arch, it);
            int64_t new_rows = 0;
            for (auto& g : games) {
                GameRecord r = replay_trajectory(g, game_cfg, bc, model_cfg->input_descr, hist_descr);
                int64_t rows = r.num_rows();
                new_rows += rows;
                if (g.value("model", -1) < 0) st.random_rows += rows;
                buffer.add(std::move(r));
                st.total_games++;
            }
            st.total_rows += new_rows;
            std::ofstream(ckpt_dir / iter_name(arch, it, "_traj.json")) << games.dump() << "\n";
            train_iteration(trainer, st, buffer, new_rows, args, game_cfg, bc, *model_cfg, adj_norms, device, rng, it);
            st.iteration = it;
            if (st.bucket_started) export_and_save(trainer, st, ckpt_dir, arch, it, keeps_model_file(it, args));
        }
    }

    // ── Self-play loop ───────────────────────────────────────────────────────
    SearchParams search_params = SearchParams::self_play();
    search_params.cpuct = args.c_puct;
    SelfPlayParams sp_params;
    sp_params.full_visits = args.full_visits;
    sp_params.cheap_visits = std::min(args.cheap_visits, args.full_visits);
    sp_params.reduced_visits_min = std::min(sp_params.reduced_visits_min, args.full_visits);
    sp_params.cheap_prob = args.cheap_prob;
    SelfPlayRunner runner(game_cfg, bc, model_cfg->input_descr, hist_descr, sp_params, search_params,
                          args.gamegen_batch_size, args.nn_server_threads, args.nn_max_batch,
                          std::random_device{}());

    // The model self-play plays with: a copy of the newest export, taken from the trainer (which holds
    // it whether or not its <arch>_<iter>.pt was kept), or the uniform evaluator before any.
    std::optional<AnyModel> selfplay_model;
    auto load_selfplay_model = [&]() {
        selfplay_model.reset();
        if (st.exported_iteration < 0) return;
        selfplay_model = clone_model(trainer.average ? *trainer.average : trainer.model, bc, *model_cfg, game_cfg, device);
    };
    load_selfplay_model();

    for (int iter = st.iteration + 1; iter <= args.iterations; iter++) {
        auto t0 = Clock::now();
        Evaluator evaluator = selfplay_model ? make_evaluator(*selfplay_model, adj_norms)
                                             : make_uniform_evaluator(bc.N, game_cfg.num_stones, game_cfg.num_players);
        int model_id = selfplay_model ? st.exported_iteration : -1;
        if (args.verbosity >= 1)
            std::cout << "[iter " << std::setw(4) << iter << "] self-play with "
                      << (selfplay_model ? "model " + std::to_string(model_id) : std::string("uniform evaluator"))
                      << " (" << args.games_per_iteration << " games, " << args.gamegen_batch_size << " at a time)" << std::endl;

        json traj = json::array();
        int64_t new_rows = 0;
        int games = 0;
        auto finished = runner.play(evaluator, args.games_per_iteration);
        NNServer::Stats nn = runner.take_nn_stats();
        for (auto& g : finished) {
            games++;
            if (args.verbosity >= 1) print_game_result(games, (int)finished.size(), model_id, g.final_state);
            int64_t rows = g.record.num_rows();
            new_rows += rows;
            if (!selfplay_model) st.random_rows += rows;
            g.trajectory["model"] = model_id;
            traj.push_back(std::move(g.trajectory));
            buffer.add(std::move(g.record));
        }
        if (args.verbosity >= 2 && nn.batches > 0)
            std::cout << std::fixed << std::setprecision(1) << "  network: " << nn.rows << " rows in " << nn.batches
                      << " batches (avg " << static_cast<double>(nn.rows) / nn.batches << "), "
                      << nn.seconds << "s evaluating" << std::defaultfloat << std::endl;
        st.total_games += games;
        st.total_rows += new_rows;
        std::ofstream(ckpt_dir / iter_name(arch, iter, "_traj.json")) << traj.dump() << "\n";
        if (args.verbosity >= 1)
            std::cout << "[iter " << std::setw(4) << iter << "] " << games << " games, " << new_rows << " new rows ("
                      << st.total_rows << " total), " << std::fixed << std::setprecision(1)
                      << std::chrono::duration<double>(Clock::now() - t0).count() << "s" << std::defaultfloat << std::endl;

        train_iteration(trainer, st, buffer, new_rows, args, game_cfg, bc, *model_cfg, adj_norms, device, rng, iter);
        st.iteration = iter;
        if (st.bucket_started) {
            export_and_save(trainer, st, ckpt_dir, arch, iter, keeps_model_file(iter, args));
            load_selfplay_model();
        } else {
            std::ofstream(ckpt_dir / "trainer" / "state.json") << st.to_json().dump(2) << "\n";
        }
    }
    return 0;
}
