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
#include <map>
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
    int cnn_conv_size     = 3;
    int transformer_hidden_dim = 128;
    int num_layers        = 9;
    int num_attn_layers   = 8;  // Transformer's cross-attention stack depth
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
    // Controls only the self-play trajectory (_traj.json) dump cadence now - see main()'s
    // save_trajectories() call sites. Model weights are no longer saved on this cadence; they're
    // saved only as tournament winners (see num_tournament_models et al., below).
    int save_every        = 10;
    // Target size of the active self-play model set (see ModelSnapshots, train.cpp) - each
    // self-play player is independently randomly assigned one of the currently-active snapshots
    // when their game is created. Grows directly (each iteration's freshly-trained challenger is
    // promoted straight in - see ChallengerPool/main()'s per-iteration training step) until it
    // reaches this size, then only changes via tournament promotion (see tournament_every et al.).
    int num_selfplay_models = 1;
    // Number of participants in each tournament: the current active_models set plus randomly
    // sampled challengers, filling up to this many total - must be > num_selfplay_models (validated
    // after parsing) so every tournament has at least one wildcard slot.
    int num_tournament_models = 6;
    // Training iterations between tournaments - see main()'s tournament block. Fresh model
    // snapshots captured in between accumulate as challengers until the next tournament (or until
    // active_models is still growing towards num_selfplay_models capacity, in which case they join
    // active_models directly instead - see main()).
    int tournament_every = 16;
    // Games played per tournament, distributed across participants via a balanced random sequence
    // spanning the whole tournament (see assign_by_sequence()).
    int num_tournament_games = 512;
    // MCTS simulations per ply during tournament games - independent of (and typically much lower
    // than) --num-simulations, since tournament games exist purely to rank models, not to produce
    // training data.
    int tournament_num_simulations = 128;
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
              << "                            (default: 3)\n"
              << "  --transformer-hidden-dim N  Transformer hidden dimension (default: 128)\n"
              << "  --num-layers N            GNN message-passing layers (default: 9)\n"
              << "  --num-attn-layers N       Transformer cross-attention layers (default: 8)\n"
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
              << "  --save-every N            Dump self-play trajectories every N iterations - no\n"
              << "                            longer controls model checkpointing, see\n"
              << "                            --num-tournament-models (default: 10)\n"
              << "  --num-selfplay-models N   Target size of the active self-play model set - each\n"
              << "                            player is independently randomly assigned one active\n"
              << "                            snapshot per game (default: 1)\n"
              << "  --num-tournament-models N Participants per tournament (active models + random\n"
              << "                            challengers) - must be > --num-selfplay-models\n"
              << "                            (default: 6)\n"
              << "  --tournament-every N      Training iterations between tournaments (default: 16)\n"
              << "  --num-tournament-games N  Games played per tournament (default: 512)\n"
              << "  --tournament-num-simulations N  MCTS simulations per ply during tournament\n"
              << "                            games (default: 128)\n"
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
        else if (a == "--num-selfplay-models") {
            args.num_selfplay_models = std::stoi(argv[++i]);
            if (args.num_selfplay_models < 1) {
                std::cerr << "--num-selfplay-models must be >= 1 (got " << args.num_selfplay_models << ")\n";
                std::exit(1);
            }
        }
        else if (a == "--num-tournament-models") {
            args.num_tournament_models = std::stoi(argv[++i]);
            if (args.num_tournament_models < 1) {
                std::cerr << "--num-tournament-models must be >= 1 (got " << args.num_tournament_models << ")\n";
                std::exit(1);
            }
        }
        else if (a == "--tournament-every") {
            args.tournament_every = std::stoi(argv[++i]);
            if (args.tournament_every < 1) {
                std::cerr << "--tournament-every must be >= 1 (got " << args.tournament_every << ")\n";
                std::exit(1);
            }
        }
        else if (a == "--num-tournament-games") {
            args.num_tournament_games = std::stoi(argv[++i]);
            if (args.num_tournament_games < 1) {
                std::cerr << "--num-tournament-games must be >= 1 (got " << args.num_tournament_games << ")\n";
                std::exit(1);
            }
        }
        else if (a == "--tournament-num-simulations") {
            args.tournament_num_simulations = std::stoi(argv[++i]);
            if (args.tournament_num_simulations < 1) {
                std::cerr << "--tournament-num-simulations must be >= 1 (got "
                          << args.tournament_num_simulations << ")\n";
                std::exit(1);
            }
        }
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
    if (args.num_tournament_models <= args.num_selfplay_models) {
        std::cerr << "--num-tournament-models must be > --num-selfplay-models (got "
                  << args.num_tournament_models << " <= " << args.num_selfplay_models << ")\n";
        std::exit(1);
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

// ── Multi-model self-play & tournament-based model selection ───────────────────

// id -> frozen model snapshot; id is the training iteration whose training step produced it (or,
// for the very first seed, start_iter - 1 - see main()). Entries are never mutated in place - each
// training step operates on a fresh clone (see ChallengerPool, below, and main()'s per-iteration
// training step), so every stored entry stays a valid, freely shallow-copyable snapshot for as
// long as it's kept around. Used both for the active self-play model set (active_models, a plain
// ModelSnapshots) and, wrapped in ChallengerPool below, the pool of not-yet-promoted challengers.
using ModelSnapshots = std::vector<std::pair<int, AnyModel>>;

// Ordered list of challenger model snapshots awaiting either further training or tournament
// promotion. Entries at index < trained_boundary_ have already been used once as a training base
// (see pick_untrained_challenger()/mark_picked_trained()) - a challenger is only ever trained once
// in its lifetime, producing a child that's appended at the tail (push_back()) and is itself
// untrained. Because every mutation either appends at the tail or advances the boundary by exactly
// one, "trained = prefix, untrained = suffix" holds at all times, so a single index fully captures
// trained/untrained status - no per-id set needed.
class ChallengerPool {
public:
    // Resets to `models` (each considered untrained) - used at run startup and after every
    // tournament ("initialize challengers with active models").
    void reset(ModelSnapshots models) { entries_ = std::move(models); trained_boundary_ = 0; }

    // The oldest not-yet-trained entry, or nullptr if every current entry has already been used as
    // a training base (shouldn't happen given the replenishment invariant in main()'s per-iteration
    // training step, but guarded defensively rather than assumed). The returned pointer is only
    // valid until the next push_back() (which may reallocate) - callers must extract what they
    // need (clone the model, note the id) before calling push_back().
    std::pair<int, AnyModel>* pick_untrained_challenger() {
        return trained_boundary_ < (int)entries_.size() ? &entries_[trained_boundary_] : nullptr;
    }

    // Marks the current pick_untrained_challenger() entry trained (advances the boundary by one) -
    // call only after successfully training a clone of it.
    void mark_picked_trained() { trained_boundary_++; }

    // Appends a fresh, untrained entry at the tail - preserves the prefix/suffix invariant above.
    void push_back(int id, AnyModel model) { entries_.push_back({id, std::move(model)}); }

    // Indices of wildcard-eligible entries for a tournament: every entry whose id isn't currently
    // in active_models - excludes exactly the entries reset() just copied in from active_models
    // and that haven't been trained since, which would otherwise be redundant, bit-identical
    // tournament participants.
    std::vector<int> eligible_wildcard_indices(const ModelSnapshots& active_models) const {
        std::vector<int> idx;
        for (int i = 0; i < (int)entries_.size(); i++) {
            int id = entries_[i].first;
            bool is_active = std::any_of(active_models.begin(), active_models.end(),
                                          [&](auto& e) { return e.first == id; });
            if (!is_active) idx.push_back(i);
        }
        return idx;
    }

    ModelSnapshots& entries() { return entries_; }

private:
    ModelSnapshots entries_;
    int trained_boundary_ = 0;
};

// Produces an independent, frozen copy of src - not just another reference to the same
// shared_ptr-backed torch module (which a plain AnyModel copy would be), since a snapshot must
// stay unaffected by further training on src (seed_model during setup, or a picked challenger
// still undergoing its one training step - see main()'s per-iteration training step). Goes through
// the exact same torch::save/torch::load (via std::visit) already used for checkpoints, just
// round-tripped through an in-memory stream instead of a file - build_model() constructs the fresh
// AnyModel of the right architecture to load into. Only needed when independently cloning
// something that either is or might still become subject to further training - copying an
// already-frozen ModelSnapshots entry (active_models/challengers/participants) is safe as a plain
// (shallow, shared_ptr-aliasing) AnyModel copy, since nothing ever trains those further (only
// clones of them, produced via this function, ever get trained).
static AnyModel clone_model(const AnyModel& src, const BoardConfig& bc, const ModelConfig& cfg,
                             const GameConfig& game_cfg, torch::Device device) {
    std::ostringstream oss;
    std::visit([&](auto& m) { torch::save(m, oss); }, src);
    AnyModel fresh = build_model(bc, cfg, game_cfg);
    std::istringstream iss(oss.str());
    std::visit([&](auto& m) { torch::load(m, iss); m->to(device); m->eval(); }, fresh);
    return fresh;
}

// Uniformly random id from `models` - a lookup key into `models` itself and into an evaluators
// map built from the same vector via build_evaluators().
static int pick_random_model_id(const ModelSnapshots& models, std::mt19937& rng) {
    std::uniform_int_distribution<size_t> dist(0, models.size() - 1);
    return models[dist(rng)].first;
}

// Randomly (re)assigns every player of `state` to one of `models`' current entries - used both for
// a brand-new self-play game (every player) and, via refresh_player() below, for a single player
// whose previously-assigned model is no longer active.
static void assign_random_models(BoardState& state, const ModelSnapshots& models, std::mt19937& rng) {
    for (int p = 0; p < state.num_players; p++)
        state.player_model_id[p] = pick_random_model_id(models, rng);
}

// After active_models changes (a tournament can replace more than one entry at once, unlike the
// old per-iteration FIFO eviction), `state` may still have a player referencing an id that's no
// longer present - if so, that would otherwise fail MCTS::evaluate_batch()'s lookup once the
// evaluators map is rebuilt to match the new set. Reassigns exactly that player to a fresh random
// pick instead, keeping `state`'s assignments always a subset of `models`.
static void refresh_player(BoardState& state, const ModelSnapshots& models, std::mt19937& rng) {
    for (int p = 0; p < state.num_players; p++) {
        int id = state.player_model_id[p];
        bool still_present = std::any_of(models.begin(), models.end(),
                                          [&](auto& e) { return e.first == id; });
        if (!still_present) state.player_model_id[p] = pick_random_model_id(models, rng);
    }
}

// (Re)builds the full evaluators map from `models` - called whenever active_models changes (not
// per ply; active_models only changes a few times per training iteration at most, and many
// ply-generation calls happen within one iteration's self-play phase against an unchanged set).
static std::map<int, Evaluator> build_evaluators(ModelSnapshots& models, const AdjNorms& adj_norms) {
    std::map<int, Evaluator> evaluators;
    for (auto& [id, model] : models)
        evaluators[id] = make_evaluator(model, adj_norms);
    return evaluators;
}

// Assigns every slot across all of `states` the next ids from a sequence built by concatenating
// enough independent random shuffles of `participants`' ids - each containing every participant
// exactly once - to cover every slot `states` has (states.size() * num_players, assuming every
// state shares the same num_players). Consumption order alone balances how often each participant
// plays across the whole tournament: called exactly once, up front, with `states` = every game the
// tournament will run (see run_tournament_games()) - not per-batch-slot, so the whole tournament's
// slot sequence is one coherent list rather than being decided batch-by-batch as games start.
static void assign_by_sequence(const std::vector<BoardState*>& states, const ModelSnapshots& participants,
                                std::mt19937& rng) {
    std::vector<int> ids;
    ids.reserve(participants.size());
    for (auto& [id, m] : participants) ids.push_back(id);

    int total_slots = 0;
    for (auto* state : states) total_slots += state->num_players;
    std::vector<int> sequence;
    sequence.reserve(total_slots);
    while ((int)sequence.size() < total_slots) {
        std::shuffle(ids.begin(), ids.end(), rng);
        sequence.insert(sequence.end(), ids.begin(), ids.end());
    }

    int cursor = 0;
    for (auto* state : states)
        for (int p = 0; p < state->num_players; p++) state->player_model_id[p] = sequence[cursor++];
}

// Runs `num_games` tournament games (batched `gamegen_batch_size` at a time) among `participants`,
// discarding trajectories (tournament games are for ranking models only, never trained on) and
// accumulating each participant's total reward + games played into reward_sum/game_count (both
// assumed already zero-initialized for every id in participants by the caller). Reuses
// generate_one_ply_per_game() unchanged - the same MCTS/model-routing machinery self-play uses -
// simply dropping the returned PlyResults each ply instead of recording them.
static void run_tournament_games(
    ModelSnapshots& participants, std::map<int, Evaluator>& evaluators,
    const GameConfig& game_cfg, const BoardConfig& bc, const nlohmann::json& input_descr,
    const nlohmann::json* history_descr, int num_games, int gamegen_batch_size,
    int num_simulations, float c_puct, int verbosity, std::mt19937& rng,
    std::unordered_map<int,float>& reward_sum, std::unordered_map<int,int>& game_count)
{
    // Every game the tournament will run, pre-created and assigned up front (see
    // assign_by_sequence()) - the active batch below draws from this list (moved, not re-created)
    // both for its initial fill and every replenishment.
    std::vector<BoardState> all_games;
    all_games.reserve(num_games);
    for (int i = 0; i < num_games; i++) all_games.push_back(new_state(game_cfg, bc));

    std::vector<BoardState*> all_ptrs;
    all_ptrs.reserve(num_games);
    for (auto& s : all_games) all_ptrs.push_back(&s);
    assign_by_sequence(all_ptrs, participants, rng);

    // Active batch, capped at gamegen_batch_size for compute reasons.
    int batch_n = std::min(gamegen_batch_size, num_games);
    std::vector<BoardState> tpool;
    tpool.reserve(batch_n);
    int next_game = 0;
    for (int i = 0; i < batch_n; i++) tpool.push_back(std::move(all_games[next_game++]));
    int temperature_threshold = static_cast<int>(2 * std::sqrt(bc.N)) + 3;

    int games_done = 0, tournament_ply_iter = 0;
    while (games_done < num_games) {
        std::vector<BoardState*> ptrs;
        ptrs.reserve(tpool.size());
        for (auto& s : tpool) ptrs.push_back(&s);

        auto t_ply0 = std::chrono::high_resolution_clock::now();
        auto [ply_results, timing] = generate_one_ply_per_game(
            evaluators, ptrs, input_descr, num_simulations,
            temperature_threshold, c_puct, verbosity, history_descr);
        double total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_ply0).count();
        if (verbosity >= 1) {
            // Mirrors the live loop's own per-ply timing print (train.cpp's main()),
            // "tournament iter" instead of "ply iter" to distinguish it in the log.
            std::cout << std::fixed << std::setprecision(0)
                      << "  tournament iter " << tournament_ply_iter << ": generate=" << total_ms << "ms"
                      << "  search=" << timing.search * 1000.0 << "ms"
                      << "  simulate=" << timing.simulate * 1000.0 << "ms"
                      << "  teardown=" << timing.teardown * 1000.0 << "ms"
                      << "  eval=" << timing.eval * 1000.0 << "ms"
                      << "  select=" << timing.select * 1000.0 << "ms"
                      << std::defaultfloat << std::endl;
        }
        ++tournament_ply_iter;

        for (int slot = 0; slot < (int)tpool.size(); slot++) {
            if (!tpool[slot].game_over()) continue;

            auto rewards = compute_player_rewards(
                BoardState::compute_points(tpool[slot].score_rule, tpool[slot].score()),
                tpool[slot].stone_to_player_map, tpool[slot].komi,
                tpool[slot].score_rule, tpool[slot].capture_count());
            for (int p = 0; p < tpool[slot].num_players; p++) {
                int model_id = tpool[slot].player_model_id[p];
                auto it = rewards.find(p + 1);
                if (it == rewards.end()) continue;  // e.g. a player none of this turn's stones scores for
                reward_sum[model_id] += it->second;
                game_count[model_id] += 1;
            }

            if (verbosity >= 1) {
                // Mirrors the live loop's own per-self-play-game print (train.cpp's main()),
                // "Tournament"-prefixed to distinguish it in the log.
                auto& score = tpool[slot].score();
                auto stone_at = [](const std::unordered_map<int,int>& m, int s) {
                    auto it = m.find(s);
                    return it != m.end() ? it->second : 0;
                };
                std::cout << "  Tournament game " << (games_done + 1)
                          << "/" << num_games
                          << "  players=[";
                for (int id : tpool[slot].player_model_id) std::cout << id << ",";
                std::cout << "]  plies=" << tpool[slot].ply_count()
                          << "  stones=[";
                for (int s = 1; s <= tpool[slot].num_stones; s++) std::cout << stone_at(score.stone_count, s) << ",";
                std::cout << "]  territories=[";
                for (int s = 1; s <= tpool[slot].num_stones; s++) std::cout << stone_at(score.territory, s) << ",";
                std::cout << "]  winners=[";
                if (tpool[slot].winners.has_value())
                    for (int w : tpool[slot].winners.value()) std::cout << w << ",";
                else
                    std::cout << "error, winner has not been computed";
                std::cout << "]" << std::endl;
            }

            games_done++;

            if (next_game < num_games) {
                tpool[slot] = std::move(all_games[next_game++]);
            } else if (slot != (int)tpool.size() - 1) {
                tpool[slot] = std::move(tpool.back());
                tpool.pop_back();
                slot--;  // re-visit the swapped-in state at this index
            } else {
                // slot is already the last index - swapping it with itself would be a
                // self-move-assignment, so just drop it instead.
                tpool.pop_back();
            }
        }
    }
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

// Writes <arch>_config.json (game_cfg+model_cfg, same combined shape
// validate_checkpoint_dir_config()/validate_retrain_source_dir() read back) into ckpt_dir. Called
// once, early in main() (both fresh-run and --resume paths - idempotent on --resume, since its
// content is already validated to match) rather than tied to any particular checkpoint event, so
// it's guaranteed to exist well before the first model weights or trajectories are ever saved.
static void save_config_json(const fs::path& ckpt_dir, const std::string& arch,
                              const GameConfig& game_cfg, const ModelConfig& model_cfg)
{
    // featureDim/inputDescr persisted directly (rather than re-derived from
    // player_stone_place_limit/global_stone_place_limit) so server.cpp's load_model() doesn't need
    // to round-trip the full limit structure through the checkpoint JSON - see
    // compute_input_descr()'s doc comment (training/self_play.h). model_cfg is the same one
    // build_model() constructed the network from.
    json cfg_json = game_cfg.to_json();
    cfg_json.update(model_cfg.to_json());
    std::ofstream(ckpt_dir / (arch + "_config.json")) << cfg_json.dump(2) << "\n";
}

// Writes <arch>_XXXXXX.pt (weights) into ckpt_dir, tagged with `id` - the live self-play loop
// passes a tournament winner's own capture iteration (may be well before the current training
// iteration - see main()'s tournament block); --retrain's replay phase passes seed_model's own
// current iteration directly (that phase has no tournament concept). Either way,
// iteration_from_model_path()/latest_checkpoint() read `id` back the same way regardless of which
// case produced it.
static void save_model_weights(const fs::path& ckpt_dir, const std::string& arch, int id,
                                AnyModel& model, const GameConfig&, const ModelConfig&)
{
    std::ostringstream oss;
    oss << arch << "_" << std::setfill('0') << std::setw(6) << id << ".pt";
    fs::path ckpt_path = ckpt_dir / oss.str();
    std::visit([&](auto& m) { torch::save(m, ckpt_path.string()); }, model);
    std::cout << "  Saved " << ckpt_path << std::endl;
}

// Writes <arch>_XXXXXX_traj.json (games_to_dump, JSON-array-of-arrays via PlyResult::to_json())
// into ckpt_dir for iteration `iter` - shared by the live self-play loop's --save-every cadence
// (games_to_dump = its accumulated traj_store) and --retrain's replay phase (games_to_dump = the
// current span's historical games, re-dumped into the new output directory so a later --resume
// from it doesn't start with an empty buffer).
static void save_trajectories(const fs::path& ckpt_dir, const std::string& arch, int iter,
                               const std::vector<std::vector<PlyResult>>& games_to_dump)
{
    std::ostringstream toss;
    toss << arch << "_" << std::setfill('0') << std::setw(6) << iter << "_traj" << ".json";
    fs::path traj_path = ckpt_dir / toss.str();
    json trajs = json::array();
    for (auto& traj : games_to_dump) {
        json t = json::array();
        for (auto& ply : traj) t.push_back(ply.to_json());
        trajs.push_back(std::move(t));
    }
    std::ofstream(traj_path) << trajs.dump() << "\n";
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
    //
    // A plain assert() would be compiled out entirely under NDEBUG (this project's Release builds),
    // silently letting a stateless architecture train against forced_pass_only=true games - a real
    // correctness bug, not just a debug-time sanity check - so this must be a live, unconditional
    // runtime check instead.
    if (game_cfg.forced_pass_only && arch != "transformer") {
        std::cerr << "Error: forced_pass_only=true requires --net-arch transformer (got '" << arch
                  << "') - CNN/UNet/GNN aren't history-aware and can't function correctly under "
                     "forced_pass_only rules.\n";
        return 1;
    }

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
    // Only ever directly trained by --retrain's replay phase, below (its own persistent
    // retrain_optimizer) - once past that phase (or immediately, for a plain fresh/--resume run
    // with no --retrain), this seeds active_models/challengers and is never referenced again. The
    // live loop trains individual challenger clones instead (see ChallengerPool, above, and the
    // live loop's per-iteration training step) - there is no single continuously-trained model.
    auto seed_model = build_model(bc, *model_cfg, game_cfg);
    std::visit([&](auto& m) { m->to(device); }, seed_model);

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
        start_iter = resume(ckpt_dir, *model_cfg, seed_model, game_cfg, bc,
                             args.buffer_size, buffer);
    }
    // Written unconditionally/early (rather than tied to any particular checkpoint event) so it's
    // guaranteed to exist from the very start of the run, well before the first tournament winner
    // or --save-every trajectory dump - see save_config_json()'s doc comment.
    save_config_json(ckpt_dir, arch, game_cfg, *model_cfg);

    // Only used by --retrain's replay phase, below - the live loop constructs a fresh optimizer
    // per training step instead (see ChallengerPool's per-iteration training step), since each
    // challenger is only ever trained once in its lifetime.
    auto retrain_optimizer = torch::optim::Adam(
        std::visit([](auto& m) { return m->parameters(); }, seed_model),
        torch::optim::AdamOptions(args.lr).weight_decay(args.l2));

    std::mt19937 rng(42);

    auto adj_norms = compute_adj_norms(bc, device);

    // Active self-play model set, grown directly (see the live loop below) until it reaches
    // num_selfplay_models capacity, then only ever changed wholesale via tournament promotion -
    // seeded with a single snapshot of seed_model, which uniformly covers a fresh run, a
    // --resume'd run, and the post---retrain fall-through (none of those paths need special-casing
    // here, since this only ever looks at whatever seed_model currently holds).
    // Tagged start_iter - 1 (matching the id of the checkpoint it was just loaded from, on
    // --resume - see resume()'s "start_iter = id + 1"; -1 for a fresh run, a sentinel that never
    // collides with any real iteration) rather than start_iter itself - the live loop's first pass
    // (iter == start_iter) captures its own post-backprop snapshot tagged `iter`, i.e. start_iter;
    // reusing that same id for the seed would collide in the evaluators map (std::map silently
    // drops the second insert for a repeated key), silently losing one of two distinct models.
    ModelSnapshots active_models;
    active_models.push_back({start_iter - 1, clone_model(seed_model, bc, *model_cfg, game_cfg, device)});
    auto evaluators = build_evaluators(active_models, adj_norms);

    // Not-yet-promoted challenger snapshots, seeded with the same single model as active_models -
    // see ChallengerPool's doc comment (above) for how trained/untrained status is tracked, and the
    // live loop for how this grows/gets consumed and how a tournament reseeds it.
    ChallengerPool challengers;
    challengers.reset(active_models);

    // ── Game pool ─────────────────────────────────────────────────────────────
    // Pool of gamegen_batch_size in-progress games. Slots are replenished
    // immediately when a game ends, so the batch is always full.
    // The pool persists across training iterations. Each game's own
    // max_plies (rolled by new_state() from game_cfg.linear_move_bound) is
    // carried on its BoardState, so no separate list is needed here.
    std::vector<BoardState> pool;
    pool.reserve(args.gamegen_batch_size);
    for (int i = 0; i < args.gamegen_batch_size; i++) {
        pool.push_back(new_state(game_cfg, bc));
        assign_random_models(pool.back(), active_models, rng);
    }
    std::vector<std::vector<PlyResult>> trajectories(args.gamegen_batch_size);

    // Accumulates completed game trajectories between checkpoints; dumped and
    // cleared each time a checkpoint is saved.
    std::vector<std::vector<PlyResult>> traj_store;
    traj_store.reserve(args.gamegen_batch_size * args.save_every);

    int iter = start_iter;

    // ── --retrain replay phase ───────────────────────────────────────────────
    // Replays retrain_source_dir's existing _traj.json files, in ascending
    // (chronological) order, into the freshly-initialized seed_model/buffer -
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
                did_train = run_training_iteration(iter, seed_model, retrain_optimizer, buffer, rng,
                                                    game_cfg, bc, adj_norms, device,
                                                    args.train_fraction, args.batch_size, t0);
                iter++;
            }
            if (!hit_iteration_cap && did_train) {
                // No tournament concept here (this phase never runs self-play/MCTS) - seed_model
                // itself is the only thing to save, tagged by its own current iteration.
                save_model_weights(ckpt_dir, arch, iter - 1, seed_model, game_cfg, *model_cfg);
                save_trajectories(ckpt_dir, arch, iter - 1, span_games);
            }

            prev_it = it;
        }
    }

    int ply_iter = 0;
    for (; iter < args.iterations; iter++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        // No top-of-iteration eval-mode toggle needed here (unlike the old single-model_var
        // design) - every active_models/challengers entry is already in eval mode from the moment
        // it's stored, via clone_model()'s own convention.

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
                evaluators, ptrs, model_cfg->input_descr,
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
                assign_random_models(pool[slot], active_models, rng);
                trajectories[slot].clear();
            }
        }

        // ── Training ─────────────────────────────────────────────────────────
        auto* picked = challengers.pick_untrained_challenger();
        if (!picked) {
            std::cout << "[iter " << iter << "] no untrained challenger available, skipping train step" << std::endl;
            continue;
        }
        AnyModel trainee = clone_model(picked->second, bc, *model_cfg, game_cfg, device);
        auto trainee_optimizer = torch::optim::Adam(
            std::visit([](auto& m) { return m->parameters(); }, trainee),
            torch::optim::AdamOptions(args.lr).weight_decay(args.l2));

        bool did_train = run_training_iteration(iter, trainee, trainee_optimizer, buffer, rng,
                                                 game_cfg, bc, adj_norms, device,
                                                 args.train_fraction, args.batch_size, t0);
        if (!did_train) continue;

        // The picked entry is retired (never trained again - see ChallengerPool's doc comment);
        // trainee is its trained child, appended fresh at the tail. While active_models hasn't yet
        // reached capacity, the child is ALSO promoted directly (nothing to displace yet, so no
        // reason to gate it behind a tournament) - keeps the untrained-challenger pool from running
        // dry during growth (it would start with only the single seed entry otherwise).
        challengers.mark_picked_trained();
        std::visit([](auto& m) { m->eval(); }, trainee);  // run_training_iteration leaves .train() mode set

        if ((int)active_models.size() < args.num_selfplay_models) {
            active_models.push_back({iter, trainee});  // shallow copy - safe, see ModelSnapshots
            challengers.push_back(iter, std::move(trainee));
            evaluators = build_evaluators(active_models, adj_norms);
        } else {
            challengers.push_back(iter, std::move(trainee));
        }

        // ── Tournament ───────────────────────────────────────────────────────
        // iter == args.iterations - 1 mirrors --save-every's own end-of-run flush below, so
        // whatever's accumulated in challengers always gets one last shot before the run ends.
        if ((iter + 1) % args.tournament_every == 0 || iter == args.iterations - 1) {
            std::vector<int> eligible = challengers.eligible_wildcard_indices(active_models);
            int wildcards_needed = args.num_tournament_models - (int)active_models.size();
            if ((int)eligible.size() < wildcards_needed) {
                // Not enough eligible wildcards yet to fill a full num_tournament_models tournament
                // with all-distinct participants - skip this round entirely (challengers keeps
                // accumulating untouched) rather than running an under-sized tournament.
                if (args.verbosity >= 1)
                    std::cout << "[iter " << std::setw(4) << iter << "] tournament: only "
                              << eligible.size() << " eligible challengers available (" << wildcards_needed
                              << " needed) - skipping this tournament" << std::endl;
            } else {
                // Distinct wildcard challengers: shuffle the eligible indices and take the first
                // wildcards_needed - each used at most once, so participants never contains the
                // same challenger twice.
                std::shuffle(eligible.begin(), eligible.end(), rng);
                ModelSnapshots participants = active_models;
                for (int k = 0; k < wildcards_needed; k++)
                    participants.push_back(challengers.entries()[eligible[k]]);

                auto tournament_evaluators = build_evaluators(participants, adj_norms);

                if (args.verbosity >= 1)
                    std::cout << "[iter " << std::setw(4) << iter << "] tournament (" << participants.size()
                              << " models, " << args.num_tournament_games << " games, batch="
                              << args.gamegen_batch_size << ") ..." << std::endl;

                std::unordered_map<int,float> reward_sum;
                std::unordered_map<int,int> game_count;
                for (auto& [id, m] : participants) { reward_sum[id] = 0.0f; game_count[id] = 0; }

                run_tournament_games(participants, tournament_evaluators, game_cfg, bc,
                                      model_cfg->input_descr, history_descr_ptr(*model_cfg),
                                      args.num_tournament_games, args.gamegen_batch_size,
                                      args.tournament_num_simulations, args.c_puct, args.verbosity, rng,
                                      reward_sum, game_count);

                // Standings: participants with >=1 game, ranked by avg reward descending;
                // participants with zero games (pathological - only possible if
                // num_tournament_games is far too small relative to num_tournament_models) are
                // kept separately, unranked.
                std::vector<std::pair<int,float>> ranked;
                std::vector<int> unranked;
                for (auto& [id, m] : participants) {
                    int cnt = game_count[id];
                    if (cnt > 0) ranked.push_back({id, reward_sum[id] / cnt});
                    else unranked.push_back(id);
                }
                std::sort(ranked.begin(), ranked.end(),
                          [](auto& a, auto& b) { return a.second > b.second; });

                int top_k = std::min(args.num_selfplay_models, (int)participants.size());
                std::vector<int> winner_ids;
                for (auto& [id, avg] : ranked) { if ((int)winner_ids.size() >= top_k) break; winner_ids.push_back(id); }
                for (int id : unranked) { if ((int)winner_ids.size() >= top_k) break; winner_ids.push_back(id); }
                auto is_winner = [&](int id) {
                    return std::find(winner_ids.begin(), winner_ids.end(), id) != winner_ids.end();
                };
                // A winner is either a genuinely new challenger being promoted for the first time,
                // or an incumbent active model that simply prevailed again - distinguished for both
                // the standings print and the save-gating below, since only the former needs
                // saving. old_active_ids (active_models' ids immediately before this tournament's
                // promotion) is the right test here, not challengers membership - a reseed
                // duplicates active_models into challengers, so a prevailing incumbent can
                // legitimately appear there too.
                std::vector<int> old_active_ids;
                for (auto& [id, m] : active_models) old_active_ids.push_back(id);
                auto is_new = [&](int id) {
                    return std::find(old_active_ids.begin(), old_active_ids.end(), id) == old_active_ids.end();
                };
                auto winner_tag = [&](int id) {
                    if (!is_winner(id)) return "";
                    return is_new(id) ? "  [promoted]" : "  [prevailed]";
                };

                if (args.verbosity >= 1) {
                    std::cout << "[iter " << std::setw(4) << iter << "] tournament standings:" << std::endl;
                    int rank = 1;
                    for (auto& [id, avg] : ranked) {
                        std::cout << "  #" << rank++ << " model " << id
                                  << "  avg_reward=" << std::fixed << std::setprecision(4) << avg
                                  << "  games=" << game_count[id]
                                  << winner_tag(id)
                                  << std::defaultfloat << std::endl;
                    }
                    for (int id : unranked)
                        std::cout << "  model " << id << "  (no games played)"
                                  << winner_tag(id) << std::endl;
                }

                // Promote: winner_ids become the new active_models. Only save the ones genuinely
                // new (is_new) - an incumbent that simply prevailed again was already saved the
                // tournament it was first promoted, so re-saving it here would just duplicate that
                // file.
                ModelSnapshots new_active;
                new_active.reserve(winner_ids.size());
                for (int id : winner_ids) {
                    auto pit = std::find_if(participants.begin(), participants.end(),
                                             [&](auto& e) { return e.first == id; });
                    if (is_new(id)) save_model_weights(ckpt_dir, arch, id, pit->second, game_cfg, *model_cfg);
                    new_active.push_back(*pit);
                }
                active_models = std::move(new_active);

                for (auto& state : pool) refresh_player(state, active_models, rng);
                evaluators = build_evaluators(active_models, adj_norms);
                // Reseed (not clear): challengers becomes the new active_models, all untrained -
                // see ChallengerPool's doc comment and Readme's "Tournament-Based Model Selection".
                challengers.reset(active_models);
            }
        }

        // ── Trajectory checkpoint ───────────────────────────────────────────────
        if ((iter + 1) % args.save_every == 0 || iter == args.iterations - 1) {
            save_trajectories(ckpt_dir, arch, iter, traj_store);
            traj_store.clear();
        }
    }

    return 0;
}
