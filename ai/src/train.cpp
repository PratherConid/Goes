// Main training script.
//
// Usage: goes_train [--board rect 9 9] [--iterations 100] ...
// Run with --help for the full option list.
#include "game/board_config.h"
#include "model/any_model.h"
#include "model/features.h"
#include "training/self_play.h"
#include "training/replay_buffer.h"
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
#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace fs = std::filesystem;

// ── CLI argument parsing ──────────────────────────────────────────────────────

struct Args {
    std::vector<std::string> board = {"rect", "9", "9"};
    int num_stones        = 2;
    int num_players       = 2;
    bool forced_pass_only = false;
    int gnn_hidden_dim    = 128;
    int cnn_hidden_dim    = 32;
    int num_layers        = 9;
    int iterations        = 200;
    int self_play_games   = 10;
    int gamegen_batch_size = 25;
    int num_simulations   = 200;
    int train_steps       = 64;
    int batch_size        = 128;
    int buffer_size       = 100000;
    float lr              = 1e-3f;
    float l2              = 1e-4f;
    float c_puct          = 1.0f;
    int save_every        = 10;
    std::string checkpoint_dir = "ai/checkpoints";
    bool cpu              = false;
    int verbosity         = 1;
    std::optional<float> linear_move_bound;
};

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --board <type> [args...]  Board type and dimensions (default: rect 9 9)\n"
              << "  --num-stones N            Number of stone types (default: 2)\n"
              << "  --num-players N           Number of players (default: 2)\n"
              << "  --forced-pass-only        Enable forced-pass-only mode\n"
              << "  --gnn-hidden-dim N        GNN hidden dimension (default: 128)\n"
              << "  --cnn-hidden-dim N        CNN hidden dimension (default: 32)\n"
              << "  --num-layers N            GNN message-passing layers (default: 9)\n"
              << "  --iterations N            Training iterations (default: 200)\n"
              << "  --self-play-games N       Games to complete before each training step (default: 10)\n"
              << "  --gamegen-batch-size N    Games generated in parallel (default: 10)\n"
              << "  --num-simulations N       MCTS simulations per move (default: 200)\n"
              << "  --train-steps N           Gradient steps per iteration (default: 32)\n"
              << "  --batch-size N            Training batch size (default: 128)\n"
              << "  --buffer-size N           Replay buffer capacity (default: 100000)\n"
              << "  --lr F                    Learning rate (default: 0.001)\n"
              << "  --l2 F                    Weight decay (default: 0.0001)\n"
              << "  --c-puct F                MCTS exploration constant (default: 1.0)\n"
              << "  --save-every N            Save checkpoint every N iterations (default: 10)\n"
              << "  --checkpoint-dir PATH     Checkpoint directory (default: ai/checkpoints)\n"
              << "  --cpu                     Force CPU even if CUDA is available\n"
              << "  --verbosity N             0=silent, 1=per-game, >=2=per-ply (default: 1)\n"
              << "  --linear-move-bound F     End games after k*N plies\n";
}

static Args parse_args(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") { print_usage(argv[0]); std::exit(0); }
        else if (a == "--board") {
            args.board.clear();
            i++;
            while (i < argc && argv[i][0] != '-') args.board.push_back(argv[i++]);
            i--;
        }
        else if (a == "--num-stones")      args.num_stones      = std::stoi(argv[++i]);
        else if (a == "--num-players")     args.num_players     = std::stoi(argv[++i]);
        else if (a == "--forced-pass-only") args.forced_pass_only = true;
        else if (a == "--no-forced-pass-only") args.forced_pass_only = false;
        else if (a == "--gnn-hidden-dim")  args.gnn_hidden_dim  = std::stoi(argv[++i]);
        else if (a == "--cnn-hidden-dim")  args.cnn_hidden_dim  = std::stoi(argv[++i]);
        else if (a == "--num-layers")      args.num_layers      = std::stoi(argv[++i]);
        else if (a == "--iterations")      args.iterations      = std::stoi(argv[++i]);
        else if (a == "--self-play-games") args.self_play_games = std::stoi(argv[++i]);
        else if (a == "--gamegen-batch-size") args.gamegen_batch_size = std::stoi(argv[++i]);
        else if (a == "--num-simulations") args.num_simulations = std::stoi(argv[++i]);
        else if (a == "--train-steps")     args.train_steps     = std::stoi(argv[++i]);
        else if (a == "--batch-size")      args.batch_size      = std::stoi(argv[++i]);
        else if (a == "--buffer-size")     args.buffer_size     = std::stoi(argv[++i]);
        else if (a == "--lr")              args.lr              = std::stof(argv[++i]);
        else if (a == "--l2")              args.l2              = std::stof(argv[++i]);
        else if (a == "--c-puct")          args.c_puct          = std::stof(argv[++i]);
        else if (a == "--save-every")      args.save_every      = std::stoi(argv[++i]);
        else if (a == "--checkpoint-dir")  args.checkpoint_dir  = argv[++i];
        else if (a == "--cpu")             args.cpu             = true;
        else if (a == "--verbosity")       args.verbosity       = std::stoi(argv[++i]);
        else if (a == "--linear-move-bound") args.linear_move_bound = std::stof(argv[++i]);
        else { std::cerr << "Unknown argument: " << a << "\n"; std::exit(1); }
    }
    return args;
}

// ── Board factory ─────────────────────────────────────────────────────────────

static BoardConfig build_board(const std::vector<std::string>& board_args) {
    assert(!board_args.empty());
    const std::string& kind = board_args[0];
    auto ints = [&](int from) {
        std::vector<int> v;
        for (int i = from; i < (int)board_args.size(); i++) v.push_back(std::stoi(board_args[i]));
        return v;
    };
    if (kind == "rect")  { auto v = ints(1); return rectangular_board(v[0], v[1]); }
    if (kind == "rectd") { auto v = ints(1); return rectangular_diagonal_board(v[0], v[1], v[2]); }
    if (kind == "cub")   { auto v = ints(1); return cubical_board(v[0], v[1], v[2]); }
    if (kind == "hcub")  { auto v = ints(1); return hypercube_board(v[0], v[1], v[2], v[3]); }
    if (kind == "tri")   { auto v = ints(1); return triangular_board(v[0]); }
    if (kind == "twsq")  { auto v = ints(1); return twisted_square_board(v[0], v[1], v[2]); }
    if (kind == "gtsq")  { auto v = ints(1); return glue_twisted_square_board(v[0], v[1], v[2]); }
    std::cerr << "Unknown board type: " << kind << "\n"; std::exit(1);
}

// ── Model factory ─────────────────────────────────────────────────────────────

static AnyModel build_model(const Args& args, const BoardConfig& bc) {
    int in_dim = 2 * args.num_stones + 4;
    const std::string& kind = args.board[0];
    // 2-D board types → ConvNN
    if (kind == "rect" || kind == "rectd" || kind == "tri" ||
        kind == "twsq" || kind == "gtsq")
        return ConvNN(bc, in_dim, args.cnn_hidden_dim, args.num_players);
    // Higher-dimensional board types → GNN
    if (kind == "cub" || kind == "hcub")
        return MessagePassingGNN(in_dim, args.gnn_hidden_dim, args.num_layers, args.num_players);
    std::cerr << "Unknown board type: " << kind << "\n"; std::exit(1);
}

// ── Checkpoint utilities ──────────────────────────────────────────────────────

static std::optional<fs::path> latest_checkpoint(const fs::path& dir) {
    if (!fs::exists(dir)) return std::nullopt;
    std::vector<fs::path> ckpts;
    for (auto& e : fs::directory_iterator(dir)) {
        auto name = e.path().filename().string();
        if (name.rfind("ckpt_", 0) == 0 && e.path().extension() == ".pt")
            ckpts.push_back(e.path());
    }
    if (ckpts.empty()) return std::nullopt;
    std::sort(ckpts.begin(), ckpts.end());
    return ckpts.back();
}

static int iteration_from_path(const fs::path& p) {
    // ckpt_000042.pt → 42
    std::string stem = p.stem().string();
    auto pos = stem.rfind('_');
    if (pos == std::string::npos) return 0;
    return std::stoi(stem.substr(pos + 1));
}

// Build a subdirectory name from game config only (no architecture).
// Must match request_tag() in server.cpp so the server can locate the right folder.
// Example: rect-9-9_s2_p2_tsl1.2_s2p1k1.2k2
static std::string model_tag(const Args& args) {
    std::string s;
    for (auto& b : args.board) { if (!s.empty()) s += '-'; s += b; }
    s += "_s" + std::to_string(args.num_stones);
    s += "_p" + std::to_string(args.num_players);
    s += "_tsl";
    for (int i = 1; i <= args.num_stones; i++) {
        if (i > 1) s += '.';
        s += std::to_string(i);
    }
    s += "_s2p";
    for (int i = 1; i <= args.num_stones; i++) {
        if (i > 1) s += '.';
        s += std::to_string(i) + 'k' + std::to_string(((i - 1) % args.num_players) + 1);
    }
    if (args.forced_pass_only) s += "_fp";
    return s;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    auto args = parse_args(argc, argv);

    // Requires forced_pass_only=False. When forced_pass_only is enabled, a player
    // may only pass when no traditional placement is legal. In this case, players
    // will be forced to kill their own groups, and the game only ends when both
    // players have no legal moves simultaneously, which closely depends on the full
    // history of the game. The model receives only per-node features derived from the
    // current board, so it cannot function correctly in this case,
    // and the policy head's pass logit becomes meaningless. This restriction
    // will be lifted once history-aware features are added to the input.
    assert(!args.forced_pass_only &&
           "forced_pass_only=true is not supported: model lacks history-aware features");

    torch::Device device = (torch::cuda::is_available() && !args.cpu)
        ? torch::kCUDA : torch::kCPU;
    std::cout << "Device: " << device << std::endl;

    auto bc = build_board(args.board);
    std::cout << "Board:";
    for (auto& s : args.board) std::cout << " " << s;
    std::cout << "  N=" << bc.N << std::endl;

    GameConfig game_cfg;
    game_cfg.num_stones = args.num_stones;
    game_cfg.num_players = args.num_players;
    for (int s = 1; s <= args.num_stones; s++) game_cfg.turn_stone_list.push_back(s);
    for (int s = 1; s <= args.num_stones; s++)
        game_cfg.stone_to_player_map[s] = ((s - 1) % args.num_players) + 1;
    game_cfg.forced_pass_only = args.forced_pass_only;

    auto model_var = build_model(args, bc);
    std::visit([&](auto& m) { m->to(device); }, model_var);

    // Resume from checkpoint
    fs::path ckpt_dir = fs::path(args.checkpoint_dir) / model_tag(args);
    fs::create_directories(ckpt_dir);
    int start_iter = 0;
    auto latest = latest_checkpoint(ckpt_dir);
    if (latest.has_value()) {
        std::cout << "Resuming from " << latest.value() << std::endl;
        std::visit([&](auto& m) { torch::load(m, latest.value().string()); }, model_var);
        start_iter = iteration_from_path(latest.value()) + 1;
    }

    auto optimizer = torch::optim::Adam(
        std::visit([](auto& m) { return m->parameters(); }, model_var),
        torch::optim::AdamOptions(args.lr).weight_decay(args.l2));

    ReplayBuffer buffer(args.buffer_size);
    std::mt19937 rng(42);

    auto adj_norms = compute_adj_norms(bc, device);

    auto evaluator = make_evaluator(model_var, adj_norms);

    std::optional<int> max_plies;
    if (args.linear_move_bound.has_value())
        max_plies = static_cast<int>(args.linear_move_bound.value() * bc.N);

    // ── Game pool ─────────────────────────────────────────────────────────────
    // Pool of gamegen_batch_size in-progress games. Slots are replenished
    // immediately when a game ends, so the batch is always full.
    // The pool persists across training iterations.
    std::vector<BoardState> pool;
    pool.reserve(args.gamegen_batch_size);
    for (int i = 0; i < args.gamegen_batch_size; i++)
        pool.push_back(new_state(game_cfg, bc));
    std::vector<std::vector<PlyResult>> trajectories(args.gamegen_batch_size);

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

            auto [ply_results, _timing] = generate_one_ply_per_game(
                evaluator, ptrs, device,
                args.num_simulations, /*temperature_threshold=*/bc.N / 3, args.c_puct,
                args.verbosity, max_plies);

            for (int slot = 0; slot < args.gamegen_batch_size; slot++) {
                trajectories[slot].push_back(std::move(ply_results[slot]));

                bool done = pool[slot].game_over() ||
                            (max_plies.has_value() &&
                             pool[slot].ply_count() >= max_plies.value());
                if (!done) continue;

                auto records = trajectory_to_records(
                    trajectories[slot], pool[slot].stone_count, game_cfg.stone_to_player_map,
                    game_cfg.num_players);

                if (args.verbosity >= 1) {
                    std::cout << "  game " << (games_this_iter + 1)
                              << "/" << args.self_play_games
                              << "  plies=" << records.size()
                              << "  winners=[";
                    for (int w : pool[slot].winners) std::cout << w << ",";
                    std::cout << "]" << std::endl;
                }

                buffer.add(std::move(records));
                games_this_iter++;

                // Replenish slot immediately with a fresh game
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
        double total_loss = 0, total_pol = 0, total_val = 0;

        for (int step = 0; step < args.train_steps; step++) {
            auto [x, mask, p_tgt, v_tgt] = buffer.sample(args.batch_size, rng);
            x     = x.to(device);
            mask  = mask.to(device);
            p_tgt = p_tgt.to(device);
            v_tgt = v_tgt.to(device);

            auto [policy, value] = std::visit(
                [&](auto& m) -> std::pair<torch::Tensor, torch::Tensor> {
                    using M = std::decay_t<decltype(m)>;
                    if constexpr (std::is_same_v<M, MessagePassingGNN>)
                        return m->forward(x, adj_norms, mask);
                    else
                        return m->forward(x, mask);
                }, model_var);

            // Policy loss: cross-entropy against MCTS visit distribution
            auto log_policy  = torch::log(policy.clamp_min(1e-8f));
            auto policy_loss = -(p_tgt * log_policy).sum(-1).mean();
            // Value loss: MSE
            auto value_loss  = torch::mse_loss(value, v_tgt);
            auto loss        = policy_loss + value_loss;

            optimizer.zero_grad();
            loss.backward();
            torch::nn::utils::clip_grad_norm_(
                std::visit([](auto& m) { return m->parameters(); }, model_var), 1.0);
            optimizer.step();

            total_loss += loss.item<double>();
            total_pol  += policy_loss.item<double>();
            total_val  += value_loss.item<double>();
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        int n = args.train_steps;
        std::cout << "[iter " << std::setw(4) << iter << "] "
                  << "loss=" << std::fixed << std::setprecision(4) << total_loss/n
                  << "  policy=" << total_pol/n
                  << "  value="  << total_val/n
                  << "  buf="    << buffer.size()
                  << "  time="   << std::setprecision(1) << elapsed << "s" << std::endl;

        // ── Checkpoint ───────────────────────────────────────────────────────
        if ((iter + 1) % args.save_every == 0 || iter == args.iterations - 1) {
            std::ostringstream oss;
            oss << "ckpt_" << std::setfill('0') << std::setw(6) << iter << ".pt";
            fs::path ckpt_path = ckpt_dir / oss.str();
            std::visit([&](auto& m) { torch::save(m, ckpt_path.string()); }, model_var);
            fs::path json_path = ckpt_path; json_path.replace_extension(".json");
            {
                json cfg;
                cfg["board"]             = args.board;
                cfg["num_stones"]        = args.num_stones;
                cfg["num_players"]       = args.num_players;
                cfg["forced_pass_only"]  = args.forced_pass_only;
                cfg["turn_stone_list"]   = game_cfg.turn_stone_list;
                json s2p_j;
                for (auto& [k, v] : game_cfg.stone_to_player_map) s2p_j[std::to_string(k)] = v;
                cfg["stone_to_player_map"] = s2p_j;
                cfg["gnn_hidden_dim"]    = args.gnn_hidden_dim;
                cfg["cnn_hidden_dim"]    = args.cnn_hidden_dim;
                cfg["num_layers"]        = args.num_layers;
                cfg["num_players"]       = args.num_players;
                std::ofstream(json_path) << cfg.dump(2) << "\n";
            }
            std::cout << "  Saved " << ckpt_path << std::endl;
        }
    }

    return 0;
}
