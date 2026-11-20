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
#include "game/game_config.h"
#include "model/any_model.h"
#include "model/features.h"
#include "model/input_descr.h"
#include "model/model_config.h"
#include "training/checkpoint_files.h"
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
#include <map>
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
    std::vector<int> selfplay_gpus;  // empty: self-play evaluates on the training device
    int nn_max_batch = 128;
    int full_visits = 600;
    int cheap_visits = 100;
    float cheap_prob = 0.75f;
    int batch_size = 128;
    float lr = 1e-3f;
    float l2 = 1e-4f;
    bool lr_warmup = true;
    float c_puct = 1.1f;
    int64_t min_rows = 2048;  // KataGo's is 100000.
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
    std::optional<std::vector<unsigned>> side_lens;  // default: bc.emb.side_lens
    std::optional<int> n_bound;                      // overrides num_cells(side_lens)
    std::optional<int> conn_px;                      // the connImg block's cube side; CNN/UNet only
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
              << "  --side-lens L1,L2,...     A standard box for boards of several sizes, which every network's\n"
              << "                            input and output are zero-extended to, containing the board's own\n"
              << "                            side lengths (default: the board's own; any for a board without an\n"
              << "                            embedding)\n"
              << "  --n-bound N               The node bound sizing the liberty/groupSize features, >= the\n"
              << "                            board's node count (default: the product of the side lengths;\n"
              << "                            this or --side-lens is required for a board without an embedding)\n"
              << "  --conn-px N               CNN/UNet only: add the connImg features - each grid point's cube of\n"
              << "                            N pixels per side of the board's edges, drawn anti-aliased on the\n"
              << "                            N-times-finer grid (default: 0, none)\n"
              << "  --iterations N            Self-play/train/export cycles (default: 200)\n"
              << "  --saves-every N           Keep the exported model file of every Nth iteration and of\n"
              << "                            iteration --iterations (default: 8)\n"
              << "  --games-per-iteration N   Self-play games finished per cycle (default: 500)\n"
              << "  --gamegen-batch-size N    Games played concurrently, one thread each (default: 128)\n"
              << "  --nn-server-threads N     Threads evaluating the games' network requests (default: 1)\n"
              << "  --nn-max-batch N          Most network requests evaluated as one batch (default: 128)\n"
              << "  --selfplay-gpus LIST      CUDA devices for self-play's network evaluation, e.g. 0,1 - network\n"
              << "                            server thread i uses the (i mod count)-th, each GPU with its own\n"
              << "                            copy of the model (default: the training device)\n"
              << "  --full-visits N           Visits of a full (recorded) search (default: 600)\n"
              << "  --cheap-visits N          Visits of a cheap (unrecorded) search (default: 100)\n"
              << "  --cheap-prob F            Probability a move gets a cheap search (default: 0.75)\n"
              << "  --c-puct F                PUCT exploration constant (default: 1.1)\n"
              << "  --batch-size N            Training batch size (default: 128)\n"
              << "  --lr F                    Adam learning rate after warmup (default: 0.001)\n"
              << "  --l2 F                    Adam weight decay (default: 0.0001)\n"
              << "  --no-lr-warmup            Skip the learning-rate warmup over the first 2M samples\n"
              << "  --min-rows N              Rows needed before training starts (default: 2048)\n"
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
              << "                            (including --side-lens/--n-bound/--conn-px)\n"
              << "  --retrain TAG             Train a fresh model on <checkpoint-dir>/TAG's recorded games,\n"
              << "                            in order, into a new directory, then continue with self-play\n"
              << "                            for any remaining --iterations (exclusive with --resume)\n"
              << "  --linear-move-bound K1 K2 End games after Uniform(K1,K2)*N plies, resampled per game\n"
              << "  --cpu                     Force CPU even if CUDA is available\n"
              << "  --verbosity N             0=silent, 1=per-game, >=2=network batch stats (default: 1)\n";
}

// The comma-separated integers of `flag`'s `value`, each >= `min`; exits on anything else.
static std::vector<int> parse_int_list(const std::string& flag, const std::string& value, int min) {
    std::vector<int> out;
    std::stringstream list(value);
    std::string item;
    while (std::getline(list, item, ',')) {
        size_t used = 0;
        int v = min - 1;
        try { v = std::stoi(item, &used); } catch (const std::exception&) {}
        if (v < min || used != item.size()) {
            std::cerr << flag << " expects comma-separated integers >= " << min << ", got '" << item << "'\n";
            std::exit(1);
        }
        out.push_back(v);
    }
    if (out.empty()) { std::cerr << flag << " needs at least one value\n"; std::exit(1); }
    return out;
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
        else if (a == "--side-lens") {
            auto lens = parse_int_list(a, need(i), 1);
            args.side_lens.emplace(lens.begin(), lens.end());
        }
        else if (a == "--n-bound") { args.n_bound = std::stoi(need(i)); positive(a, *args.n_bound); }
        else if (a == "--conn-px") {
            args.conn_px = std::stoi(need(i));
            if (*args.conn_px < 0) { std::cerr << "--conn-px must be >= 0 (got " << *args.conn_px << ")\n"; std::exit(1); }
        }
        else if (a == "--iterations") args.iterations = std::stoi(need(i));
        else if (a == "--saves-every") { args.saves_every = std::stoi(need(i)); positive(a, args.saves_every); }
        else if (a == "--games-per-iteration") { args.games_per_iteration = std::stoi(need(i)); positive(a, args.games_per_iteration); }
        else if (a == "--gamegen-batch-size") { args.gamegen_batch_size = std::stoi(need(i)); positive(a, args.gamegen_batch_size); }
        else if (a == "--nn-server-threads") { args.nn_server_threads = std::stoi(need(i)); positive(a, args.nn_server_threads); }
        else if (a == "--nn-max-batch") { args.nn_max_batch = std::stoi(need(i)); positive(a, args.nn_max_batch); }
        else if (a == "--selfplay-gpus") args.selfplay_gpus = parse_int_list(a, need(i), 0);
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

// Throws std::runtime_error with `parts` streamed together - main() prints it as "Error: <parts>".
template <class... Parts>
[[noreturn]] static void fail(const Parts&... parts) {
    std::ostringstream msg;
    (msg << ... << parts);
    throw std::runtime_error(msg.str());
}

// ── Model factory ─────────────────────────────────────────────────────────────

// The architecture actually used: CNN/UNet need a 2D integer embedding (emb_dim == 2), which "auto"
// prefers; otherwise the GNN. The Transformer is topology-agnostic. Throws if the game config or
// --conn-px doesn't fit it.
static std::string effective_arch(const Args& args, const GameConfig& game_cfg, unsigned emb_dim) {
    bool grid2d = emb_dim == 2;
    std::string arch;
    if (args.net_arch == "cnn" || args.net_arch == "unet") {
        if (!grid2d)
            fail("--net-arch ", args.net_arch, " is not supported for board description '", game_cfg.board_descr,
                 "' (emb_dim=", emb_dim, "): it needs a 2D grid embedding.");
        arch = args.net_arch;
    } else if (args.net_arch == "gnn" || args.net_arch == "transformer") {
        arch = args.net_arch;
    } else if (args.net_arch == "auto") {
        arch = grid2d ? "cnn" : "gnn";
    } else {
        fail("unknown --net-arch '", args.net_arch, "'. Valid: auto, cnn, unet, gnn, transformer.");
    }
    // With forced_pass_only, the game's end depends on the full move history, which only the
    // history-aware Transformer sees.
    if (game_cfg.forced_pass_only && arch != "transformer")
        fail("forced_pass_only=true requires --net-arch transformer (got '", arch, "').");
    if (args.conn_px && (arch == "gnn" || arch == "transformer"))
        fail("--conn-px only applies to the CNN/UNet (got '", arch, "')");
    return arch;
}

// The slot-space box every network's input and output are zero-extended to, and the node bound
// sizing the liberty/groupSize features (see compute_input_descr()).
struct InputBounds {
    std::vector<unsigned> side_lens;
    int n_bound;
};

// --side-lens/--n-bound, defaulting to the reference board's own; throws unless they contain it.
static InputBounds input_bounds(const Args& args, const BoardConfig& bc) {
    std::vector<unsigned> side_lens = args.side_lens.value_or(bc.emb.side_lens);
    if (bc.emb.emb_dim() != 0 && !fits_in(bc.emb, side_lens))
        fail("--side-lens ", json(side_lens).dump(), " doesn't contain the board's side lengths ",
             json(bc.emb.side_lens).dump());
    if (!args.n_bound && side_lens.empty())
        fail("a board without an embedding needs --n-bound or --side-lens (an upper bound on its node count)");
    const int n_bound = args.n_bound.value_or(num_cells(side_lens));
    if (n_bound < bc.N)
        fail("nBound ", n_bound, (args.n_bound ? " (--n-bound)" : " (the product of the side lengths)"),
             " is below the board's node count ", bc.N);
    std::cout << "sideLens=" << json(side_lens).dump() << "  nBound=" << n_bound << std::endl;
    return {side_lens, n_bound};
}

// Self-play's network evaluation devices; network server thread i uses the (i mod count)-th.
static std::vector<torch::Device> selfplay_devices(const Args& args, torch::Device device) {
    if (args.selfplay_gpus.empty()) return {device};
    if (!device.is_cuda()) fail("--selfplay-gpus needs CUDA (and no --cpu)");
    std::vector<torch::Device> devices;
    int count = static_cast<int>(torch::cuda::device_count());
    for (int gpu : args.selfplay_gpus) {
        if (gpu >= count) fail("--selfplay-gpus: device ", gpu, " does not exist (", count, " visible)");
        devices.emplace_back(torch::kCUDA, static_cast<torch::DeviceIndex>(gpu));
    }
    if (args.nn_server_threads < (int)devices.size())
        fail("--nn-server-threads (", args.nn_server_threads, ") is less than the number of --selfplay-gpus (",
             devices.size(), "), so some would go unused");
    std::cout << "Self-play devices:";
    for (auto& d : devices) std::cout << " " << d;
    std::cout << std::endl;
    return devices;
}

// A fresh run's model config for architecture `arch`, on the input descriptor compute_input_descr()
// builds.
static std::unique_ptr<ModelConfig> make_model_config(const Args& args, const std::string& arch,
                                                      const GameConfig& game_cfg, unsigned emb_dim,
                                                      const InputBounds& bounds) {
    json input_descr = compute_input_descr(game_cfg, emb_dim, bounds.side_lens, bounds.n_bound, args.conn_px.value_or(0));
    int in_dim = input_descr.at("totalDims").get<int>();
    if (arch == "cnn") return std::make_unique<CNNConfig>(in_dim, args.cnn_hidden_dim, input_descr, args.cnn_conv_size);
    if (arch == "unet") return std::make_unique<UNetConfig>(in_dim, args.unet_hidden_dim, input_descr);
    if (arch == "transformer")
        return std::make_unique<TransformerConfig>(in_dim, args.transformer_hidden_dim, args.num_attn_layers,
                                                   input_descr, compute_history_descr(game_cfg));
    return std::make_unique<GNNConfig>(in_dim, args.gnn_hidden_dim, args.num_layers, input_descr);
}

// TransformerConfig::history_descr for a Transformer run, else nullptr.
static const json* history_descr_ptr(const ModelConfig& cfg) {
    return cfg.model_type == "transformer" ? &static_cast<const TransformerConfig&>(cfg).history_descr : nullptr;
}

static std::vector<torch::Tensor> parameters_of(AnyModel& m) {
    return std::visit([](auto& mm) { return mm->parameters(); }, m);
}

// An independent eval-mode copy of src's weights, through the same serialization checkpoints use.
static AnyModel clone_model(AnyModel& src, const ModelConfig& cfg, const GameConfig& game_cfg,
                            torch::Device device) {
    std::ostringstream oss;
    std::visit([&](auto& m) { torch::save(m, oss); }, src);
    std::istringstream iss(oss.str());
    return load_eval_model(cfg, game_cfg.num_stones, game_cfg.num_players, iss, device);
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

// Checkpoint directory layout: see training/checkpoint_files.h.

// A fresh directory named by a hash of the configs and a timestamp.
static fs::path fresh_checkpoint_dir(const Args& args, const GameConfig& game_cfg, const ModelConfig& model_cfg) {
    auto timestamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    std::string hash = sha256_hex(model_cfg.to_json().dump() + game_cfg.to_json().dump() + timestamp);
    fs::path dir = fs::path(args.checkpoint_dir) / hash;
    if (fs::exists(dir)) fail("checkpoint directory already exists: ", dir);
    fs::create_directories(dir / "trainer");
    std::cout << "New checkpoint directory: " << hash << " (pass --resume " << hash << " to continue it)\n";
    return dir;
}

// Throws unless `dir`'s saved config matches: the game config always, the model config too when
// check_model (--resume loads weights into the current architecture; --retrain trains a fresh one).
static void validate_dir_config(const fs::path& dir, const GameConfig& game_cfg, const ModelConfig& model_cfg,
                                bool check_model, const std::string& flag) {
    auto cfg_path = find_config_json(dir);
    if (!cfg_path) fail(flag, " directory has no config to validate against: ", dir);
    json existing = json::parse(std::ifstream(*cfg_path));
    GameConfig existing_game = parse_game_cfg(existing);
    bool ok = strong_equal(game_cfg, existing_game);
    if (check_model) ok = ok && strong_equal(model_cfg, *parse_model_config(existing));
    if (!ok) {
        json current = game_cfg.to_json();
        if (check_model) current.update(model_cfg.to_json());
        else existing = existing_game.to_json();
        fail(flag, " config mismatch with ", *cfg_path, "\nDiff (current -> saved):\n",
             json::diff(current, existing).dump(2));
    }
}

static json load_trajectories(const fs::path& dir, const std::string& arch, int iter) {
    std::ifstream f(dir / iter_file_name(arch, iter, kTrajectorySuffix));
    json j;
    f >> j;
    return j;
}

static void save_config_json(const fs::path& dir, const GameConfig& game_cfg, const ModelConfig& model_cfg) {
    json cfg_json = game_cfg.to_json();
    cfg_json.update(model_cfg.to_json());
    std::ofstream(dir / config_file_name(model_cfg.model_type)) << cfg_json.dump(2) << "\n";
}

static void save_train_state(const fs::path& dir, const TrainState& st) {
    std::ofstream(dir / "trainer" / "state.json") << st.to_json().dump(2) << "\n";
}

// ── Training ──────────────────────────────────────────────────────────────────

struct Losses {
    torch::Tensor total, policy, stone, territory, point, rank;
};

static Losses compute_losses(const ModelOutput& out, const Batch& b, const std::string& score_rule,
                             int num_stones) {
    // Every row is in slot space; n is each row's own node count, and slots no node occupies are
    // left out of every per-location term.
    auto valid = b.valid.to(out.ownership.device());
    auto n = valid.sum(1);  // (B,)

    // Policy: cross-entropy against the search's policy target, scaled by 1/log(action count) so
    // its magnitude is comparable across action-space sizes.
    auto log_policy = torch::log(out.policy.clamp_min(1e-8f));
    auto policy_loss = (-(b.policy_target * log_policy).sum(-1) / torch::log(n * num_stones + 1)).mean();

    // Ownership: per-location squared error of the stone/territory distributions (channel 0 =
    // none), summed over channels and averaged over locations and batch.
    auto stone_est = out.ownership.select(1, 0);
    auto territory_est = out.ownership.select(1, 1);
    auto actual_stone = torch::one_hot(b.stone_owner, num_stones + 1).to(torch::kFloat32);
    auto actual_territory = torch::one_hot(b.territory_owner, num_stones + 1).to(torch::kFloat32);
    auto location_mean = [&](const torch::Tensor& per_slot) { return ((per_slot * valid).sum(1) / n).mean(); };
    auto stone_loss = location_mean((stone_est - actual_stone).pow(2).sum(-1));
    auto territory_loss = location_mean((territory_est - actual_territory).pow(2).sum(-1));

    // Points: expected vs actual per-stone-type totals under the scoring rule, scaled by
    // num_stones/N so they don't dwarf the per-location terms. (The actual ownership is "none" at
    // slots no node occupies, so it scores nothing there.)
    auto actual_ownership = torch::stack({actual_stone, actual_territory}, 1);
    auto predicted_ownership = out.ownership * valid.view({valid.size(0), 1, -1, 1});
    auto point_diff = (estimate_stone_points(predicted_ownership, score_rule)
                       - estimate_stone_points(actual_ownership, score_rule)) / n.unsqueeze(1) * num_stones;
    auto point_loss = point_diff.pow(2).sum(-1).mean();

    // Rank: cross-entropy of each pair's predicted outcome against its blended target, scaled by
    // 1/log 3 so a uniform guess against a one-hot target scores 1.
    auto rank_loss = -(b.rank_target * torch::log(out.rank.clamp_min(1e-8f))).sum(-1).mean() / std::log(3.0f);

    auto total = policy_loss + stone_loss + territory_loss + point_loss + rank_loss;
    return {total, policy_loss, stone_loss, territory_loss, point_loss, rank_loss};
}

// The training side of a run: the settings every training step reads, and the state that persists
// across iterations - the trained model, its weight average, the optimizer, the training window and
// the TrainState.
struct Trainer {
    const Args& args;
    const GameConfig& game_cfg;
    const ModelConfig& model_cfg;
    torch::Device device;
    AnyModel model;
    std::optional<AnyModel> average;  // KataGo's SWA: an exponential moving average of snapshots
    std::unique_ptr<torch::optim::Adam> optimizer;
    TrainState st;
    ReplayBuffer buffer;
    std::mt19937 rng{std::random_device{}()};

    // A freshly initialized model on `device`, with a fresh optimizer.
    Trainer(const Args& args_, const GameConfig& game_cfg_, const ModelConfig& model_cfg_, torch::Device device_)
        : args(args_), game_cfg(game_cfg_), model_cfg(model_cfg_), device(device_),
          model(build_model(model_cfg_, game_cfg_.num_stones, game_cfg_.num_players)) {
        std::visit([&](auto& m) { m->to(device); }, model);
        optimizer = std::make_unique<torch::optim::Adam>(
            parameters_of(model), torch::optim::AdamOptions(args.lr).weight_decay(args.l2));
    }

    const std::string& arch() const { return model_cfg.model_type; }
    // The model self-play and the server should use: the weight average once it exists.
    AnyModel& exported() { return average ? *average : model; }
};

// KataGo's SWA snapshot: average += (model - average) / swa_scale; the first snapshot copies.
static void snapshot_average(Trainer& t) {
    if (!t.st.swa_started || !t.average) {
        t.average = clone_model(t.model, t.model_cfg, t.game_cfg, t.device);
        t.st.swa_started = true;
        return;
    }
    torch::NoGradGuard ng;
    auto src = parameters_of(t.model), dst = parameters_of(*t.average);
    for (size_t i = 0; i < src.size(); i++) dst[i].add_(src[i].detach() - dst[i], 1.0 / t.args.swa_scale);
}

// Trains while the bucket holds at least one batch, up to max_train_samples_per_iteration samples,
// sampling uniformly from the newest `window` rows. Returns the number of steps taken.
static int train_on_bucket(Trainer& t, int64_t window, int iter) {
    const Args& args = t.args;
    TrainState& st = t.st;
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
                auto hb = t.buffer.sample_with_history(args.batch_size, window, t.rng);
                b = hb.current;
                out = m->forward(hb.hist_features.to(t.device), hb.hist_mask.to(t.device),
                                 b.features.to(t.device), b.legal_mask.to(t.device));
            } else {
                b = t.buffer.sample(args.batch_size, window, t.rng);
                if constexpr (std::is_same_v<M, MessagePassingGNN>)
                    out = m->forward(b.features.to(t.device), b.games, b.legal_mask.to(t.device));
                else
                    out = m->forward(b.features.to(t.device), b.valid.to(t.device), b.legal_mask.to(t.device));
            }
        }, t.model);
        b.policy_target = b.policy_target.to(t.device);
        b.stone_owner = b.stone_owner.to(t.device);
        b.territory_owner = b.territory_owner.to(t.device);
        b.rank_target = b.rank_target.to(t.device);

        auto losses = compute_losses(out, b, t.game_cfg.score_rule, t.game_cfg.num_stones);
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
                snapshot_average(t);
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
static void train_iteration(Trainer& t, int64_t new_rows, int iter) {
    const Args& args = t.args;
    TrainState& st = t.st;
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
    t.buffer.trim(window);
    train_on_bucket(t, window, iter);
}

// Whether iteration `iter`'s export is kept as <arch>_<iter>.pt.
static bool keeps_model_file(int iter, const Args& args) {
    return iter % args.saves_every == 0 || iter == args.iterations;
}

// Exports Trainer::exported() and saves everything --resume needs under trainer/. The export is also
// kept as <arch>_<iter>.pt when keeps_model_file().
static void export_and_save(Trainer& t, const fs::path& dir, int iter) {
    if (keeps_model_file(iter, t.args)) {
        fs::path path = dir / iter_file_name(t.arch(), iter, kModelSuffix);
        std::visit([&](auto& m) { torch::save(m, path.string()); }, t.exported());
    }
    t.st.exported_iteration = iter;
    fs::create_directories(dir / "trainer");
    std::visit([&](auto& m) { torch::save(m, (dir / "trainer" / "model.pt").string()); }, t.model);
    if (t.average) std::visit([&](auto& m) { torch::save(m, (dir / "trainer" / "weight_average.pt").string()); }, *t.average);
    torch::save(*t.optimizer, (dir / "trainer" / "optimizer.pt").string());
    save_train_state(dir, t.st);
}

// Completes iteration `iter`: export_and_save() once training has started, else just the state.
// Returns whether it exported.
static bool end_iteration(Trainer& t, const fs::path& dir, int iter) {
    t.st.iteration = iter;
    if (!t.st.bucket_started) {
        save_train_state(dir, t.st);
        return false;
    }
    export_and_save(t, dir, iter);
    return true;
}

static void print_game_result(int game_num, int total_games, int model_id, const BoardState& state) {
    auto& score = state.score();
    auto at = [](const std::unordered_map<int, int>& m, int s) { auto it = m.find(s); return it != m.end() ? it->second : 0; };
    std::cout << "  game " << game_num << "/" << total_games << "  model=" << model_id
              << "  plies=" << state.ply_count() << "  stones=[";
    for (int s = 1; s <= state.num_stones; s++) std::cout << at(score.stone_count, s) << ",";
    std::cout << "]  territories=[";
    for (int s = 1; s <= state.num_stones; s++) std::cout << at(score.territory, s) << ",";
    std::cout << "]  winners=[";
    if (state.winners) for (int w : *state.winners) std::cout << w << ",";
    std::cout << "]" << std::endl;
}

// ── Run steps ─────────────────────────────────────────────────────────────────

static GameConfig load_game_config(const std::string& path) {
    std::ifstream cfg_file(path);
    if (!cfg_file) { std::cerr << "Cannot open --game-config file: " << path << "\n"; std::exit(1); }
    json cfg_json;
    cfg_file >> cfg_json;
    return parse_game_cfg(cfg_json);
}

// A reference board from a fresh random seed (game_cfg.board_seed is ignored), which fixes the
// architecture and the side length/node bound defaults; self-play gives each game thread a board of
// its own (SelfPlayRunner).
static std::shared_ptr<const BoardConfig> reference_board(const ClegProgram& program, const GameConfig& game_cfg) {
    const uint32_t board_seed = std::random_device{}();
    auto bc = std::make_shared<const BoardConfig>(build_board_from_cleg(program, board_seed));
    std::cout << "Board:\n" << game_cfg.board_descr << "  N=" << bc->N << "  seed=" << board_seed << std::endl;
    return bc;
}

// Rebuilds recorded games' GameRecords (replay_trajectory()), building each board seed's board once -
// a self-play thread's games all share its board.
struct GameReplayer {
    const ClegProgram& program;
    const GameConfig& game_cfg;
    const ModelConfig& model_cfg;
    std::map<uint32_t, std::shared_ptr<const BoardConfig>> boards;

    GameRecord operator()(const json& trajectory) {
        const auto board_seed = trajectory.at("boardSeed").get<uint32_t>();
        auto& board = boards[board_seed];
        if (!board) board = std::make_shared<const BoardConfig>(build_board_from_cleg(program, board_seed));
        return replay_trajectory(trajectory, game_cfg, board, model_cfg.input_descr, history_descr_ptr(model_cfg));
    }
};

// --resume: restores `t` from <checkpoint-dir>/<tag>'s trainer/ and refills its window from the
// newest recorded games. Returns that directory.
static fs::path resume(Trainer& t, GameReplayer& replay) {
    const Args& args = t.args;
    const fs::path dir = fs::path(args.checkpoint_dir) / args.resume_tag;
    const fs::path trainer_dir = dir / "trainer";
    if (!fs::exists(trainer_dir / "state.json")) fail("--resume directory has no trainer state: ", dir);
    validate_dir_config(dir, t.game_cfg, t.model_cfg, /*check_model=*/true, "--resume");
    t.st = TrainState::from_json(json::parse(std::ifstream(trainer_dir / "state.json")));
    // Weights and optimizer state exist once the first model has been exported.
    if (fs::exists(trainer_dir / "model.pt")) {
        load_weights(t.model, (trainer_dir / "model.pt").string(), t.device);
        torch::load(*t.optimizer, (trainer_dir / "optimizer.pt").string());
    }
    if (fs::exists(trainer_dir / "weight_average.pt"))
        t.average = load_eval_model(t.model_cfg, t.game_cfg.num_stones, t.game_cfg.num_players,
                                    (trainer_dir / "weight_average.pt").string(), t.device);
    // Refill the window from the newest recorded games.
    int64_t window = std::max<int64_t>(desired_window_rows(t.st, args), args.min_rows);
    std::vector<GameRecord> records;
    int64_t rows = 0;
    auto iters = iter_files(dir, t.arch(), kTrajectorySuffix);
    for (auto it = iters.rbegin(); it != iters.rend() && rows < window; ++it) {
        json games = load_trajectories(dir, t.arch(), *it);
        for (auto g = games.rbegin(); g != games.rend() && rows < window; ++g) {
            records.push_back(replay(*g));
            rows += records.back().num_rows();
        }
    }
    for (auto r = records.rbegin(); r != records.rend(); ++r) t.buffer.add(std::move(*r));
    std::cout << "Resuming after iteration " << t.st.iteration << ": " << t.st.total_games << " games, "
              << t.st.total_rows << " rows (" << t.buffer.stored_rows() << " reloaded)" << std::endl;
    return dir;
}

// --retrain: trains `t` on <checkpoint-dir>/<tag>'s recorded games, file by file, copying each into
// `dir`.
static void retrain(Trainer& t, GameReplayer& replay, const fs::path& dir) {
    const Args& args = t.args;
    const fs::path src = fs::path(args.checkpoint_dir) / args.retrain_tag;
    if (!fs::exists(src)) fail("--retrain directory does not exist: ", src);
    validate_dir_config(src, t.game_cfg, t.model_cfg, /*check_model=*/false, "--retrain");
    for (int it : iter_files(src, t.arch(), kTrajectorySuffix)) {
        if (it > args.iterations) break;
        json games = load_trajectories(src, t.arch(), it);
        int64_t new_rows = 0;
        for (auto& g : games) {
            GameRecord r = replay(g);
            int64_t rows = r.num_rows();
            new_rows += rows;
            if (g.value("model", -1) < 0) t.st.random_rows += rows;
            t.buffer.add(std::move(r));
            t.st.total_games++;
        }
        t.st.total_rows += new_rows;
        std::ofstream(dir / iter_file_name(t.arch(), it, kTrajectorySuffix)) << games.dump() << "\n";
        train_iteration(t, new_rows, it);
        end_iteration(t, dir, it);
    }
}

// The models self-play plays with: one copy of Trainer::exported() per self-play device (held
// whether or not its <arch>_<iter>.pt was kept), or none - the uniform evaluator - before any export.
static std::vector<AnyModel> selfplay_models(Trainer& t, const std::vector<torch::Device>& devices) {
    std::vector<AnyModel> models;
    if (t.st.exported_iteration < 0) return models;
    for (auto& d : devices) models.push_back(clone_model(t.exported(), t.model_cfg, t.game_cfg, d));
    return models;
}

// The self-play loop: plays, trains and exports each remaining iteration up to --iterations.
static void self_play(Trainer& t, const fs::path& dir, const std::vector<torch::Device>& devices) {
    const Args& args = t.args;
    TrainState& st = t.st;
    SearchParams search_params = SearchParams::self_play();
    search_params.cpuct = args.c_puct;
    SelfPlayParams sp_params;
    sp_params.full_visits = args.full_visits;
    sp_params.cheap_visits = std::min(args.cheap_visits, args.full_visits);
    sp_params.reduced_visits_min = std::min(sp_params.reduced_visits_min, args.full_visits);
    sp_params.cheap_prob = args.cheap_prob;
    sp_params.linear_move_bound = args.linear_move_bound;
    SelfPlayRunner runner(t.game_cfg, t.model_cfg.input_descr, history_descr_ptr(t.model_cfg), sp_params,
                          search_params, args.gamegen_batch_size, args.nn_server_threads, args.nn_max_batch,
                          std::random_device{}());
    std::vector<AnyModel> models = selfplay_models(t, devices);

    for (int iter = st.iteration + 1; iter <= args.iterations; iter++) {
        auto t0 = Clock::now();
        bool has_model = !models.empty();
        std::vector<Evaluator> evaluators;
        for (auto& m : models) evaluators.push_back(make_evaluator(m));
        if (!has_model)
            evaluators.push_back(make_uniform_evaluator(t.game_cfg.num_stones, t.game_cfg.num_players));
        int model_id = has_model ? st.exported_iteration : -1;
        if (args.verbosity >= 1)
            std::cout << "[iter " << std::setw(4) << iter << "] self-play with "
                      << (has_model ? "model " + std::to_string(model_id) : std::string("uniform evaluator"))
                      << " (" << args.games_per_iteration << " games, " << args.gamegen_batch_size << " at a time)" << std::endl;

        json traj = json::array();
        int64_t new_rows = 0;
        int games = 0;
        auto finished = runner.play(evaluators, args.games_per_iteration);
        NNServer::Stats nn = runner.take_nn_stats();
        for (auto& g : finished) {
            games++;
            if (args.verbosity >= 1) print_game_result(games, (int)finished.size(), model_id, g.final_state);
            int64_t rows = g.record.num_rows();
            new_rows += rows;
            if (!has_model) st.random_rows += rows;
            g.trajectory["model"] = model_id;
            traj.push_back(std::move(g.trajectory));
            t.buffer.add(std::move(g.record));
        }
        if (args.verbosity >= 2 && nn.batches > 0)
            std::cout << std::fixed << std::setprecision(1) << "  network: " << nn.rows << " rows in " << nn.batches
                      << " batches (avg " << static_cast<double>(nn.rows) / nn.batches << "), "
                      << nn.seconds << "s evaluating" << std::defaultfloat << std::endl;
        st.total_games += games;
        st.total_rows += new_rows;
        std::ofstream(dir / iter_file_name(t.arch(), iter, kTrajectorySuffix)) << traj.dump() << "\n";
        if (args.verbosity >= 1)
            std::cout << "[iter " << std::setw(4) << iter << "] " << games << " games, " << new_rows << " new rows ("
                      << st.total_rows << " total), " << std::fixed << std::setprecision(1)
                      << std::chrono::duration<double>(Clock::now() - t0).count() << "s" << std::defaultfloat << std::endl;

        train_iteration(t, new_rows, iter);
        if (end_iteration(t, dir, iter)) models = selfplay_models(t, devices);
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

static void run(const Args& args) {
    const GameConfig game_cfg = load_game_config(args.game_config_path);
    const auto program = parse_cleg(game_cfg.board_descr);
    const auto board = reference_board(*program, game_cfg);
    // BoardState's constructor checks the game config. None of its checks depend on the board, so
    // this also covers the self-play threads' boards, whose threads couldn't report the error.
    new_state(game_cfg, board);
    const unsigned emb_dim = board->emb.emb_dim();
    const std::string arch = effective_arch(args, game_cfg, emb_dim);
    const InputBounds bounds = input_bounds(args, *board);

    torch::Device device = (torch::cuda::is_available() && !args.cpu) ? torch::kCUDA : torch::kCPU;
    std::cout << "Device: " << device << std::endl;
    const std::vector<torch::Device> devices = selfplay_devices(args, device);

    const auto model_cfg = make_model_config(args, arch, game_cfg, emb_dim, bounds);
    Trainer trainer(args, game_cfg, *model_cfg, device);
    GameReplayer replay{*program, game_cfg, *model_cfg, {}};
    const fs::path dir = args.resume_tag.empty() ? fresh_checkpoint_dir(args, game_cfg, *model_cfg)
                                                 : resume(trainer, replay);
    save_config_json(dir, game_cfg, *model_cfg);
    if (!args.retrain_tag.empty()) retrain(trainer, replay, dir);
    self_play(trainer, dir, devices);
}

int main(int argc, char* argv[]) {
    // libtorch's CPU ops would otherwise give each calling thread (e.g. every NN server thread) its
    // own OpenMP team of OMP_NUM_THREADS workers, whose spin-waiting starves the self-play threads.
    torch::set_num_threads(1);
    try {
        run(parse_args(argc, argv));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
