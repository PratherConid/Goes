#include "training/self_play.h"
#include "game/cleg_eval.h"
#include "game/cleg_parser.h"
#include "model/features.h"
#include "model/rank_pairs.h"
#include <algorithm>
#include <random>
#include <set>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <mutex>

using json = nlohmann::json;

// ── Value targets and data weighting ─────────────────────────────────────────

// KataGo's three TD horizons: c in nowFactor = 1/(1 + N*c), N the board's node count.
static constexpr float kTDHorizonCoeffs[3] = {0.176f, 0.056f, 0.016f};

static float td_now_factor(float coeff, int N) { return 1.0f / (1.0f + static_cast<float>(N) * coeff); }

// KataGo's fillValueTDTargets(): searched ply t's target is
//   sum_{k=t}^{T-1} f(1-f)^(k-t) V_k  +  (1-f)^(T-t) Z
// over the searched plies' search-averaged pair outcomes V_k and the final outcome Z.
static std::vector<float> td_target(const std::vector<const std::vector<float>*>& values,
                                    const std::vector<float>& final_probs, size_t t, float f) {
    std::vector<float> out(final_probs.size(), 0.0f);
    double left = 1.0;
    for (size_t k = t; k < values.size(); k++) {
        double w = left * f;
        left *= 1.0 - f;
        for (size_t c = 0; c < out.size(); c++) out[c] += static_cast<float>(w * (*values[k])[c]);
    }
    for (size_t c = 0; c < out.size(); c++) out[c] += static_cast<float>(left * final_probs[c]);
    return out;
}

// KL(target || predicted), summed over entries: entries with target <= 1e-30 contribute nothing and
// predicted is floored at 1e-100.
static double kl_divergence(const std::vector<float>& target, const std::vector<float>& predicted) {
    double kl = 0.0;
    for (size_t i = 0; i < target.size(); i++)
        if (target[i] > 1e-30f)
            kl += target[i] * (std::log(static_cast<double>(target[i])) -
                               std::log(std::max(static_cast<double>(predicted[i]), 1e-100)));
    return kl;
}

// Mean over pairs of KL(target || predicted) between 3-way pair-outcome distributions, capped at 1 -
// the multi-pair form of KataGo's valueSurpriseKL() over its single win/loss/no-result distribution.
static float pair_outcome_kl(const std::vector<float>& target, const std::vector<float>& predicted) {
    size_t num_pairs = target.size() / 3;
    if (num_pairs == 0) return 0.0f;
    return static_cast<float>(std::min(1.0, std::max(0.0, kl_divergence(target, predicted) / num_pairs)));
}

// KL(target || prior) over actions - KataGo's policy surprise.
static float policy_kl(const std::vector<float>& target, const std::vector<float>& prior) {
    return static_cast<float>(std::max(0.0, kl_divergence(target, prior)));
}

// KataGo's end-of-game data weighting: redistributes a game's total target weight so that a
// policy_surprise_data_weight share follows each ply's policy surprise (including cheap plies whose
// surprise exceeds 1.5x the game's average) and a value_surprise_data_weight share follows each
// ply's value surprise - KL between the network's pair outcomes and the forward-looking result
// smoothed with the shortest TD horizon - then rounds each weight stochastically to a row count.
static void resolve_row_counts(std::vector<PlyRecord>& plies, const std::vector<float>& final_probs, int N,
                               const SelfPlayParams& params, std::mt19937_64& rng) {
    std::vector<PlyRecord*> searched;
    for (auto& p : plies) if (p.searched) searched.push_back(&p);
    size_t T = searched.size();

    std::vector<float> value_surprise(T, 0.0f);
    {
        float f = td_now_factor(kTDHorizonCoeffs[2], N);
        std::vector<float> running = final_probs;
        for (size_t i = T; i-- > 0;) {
            const auto& v = searched[i]->search_pair_probs;
            for (size_t c = 0; c < running.size(); c++) running[c] += f * (v[c] - running[c]);
            value_surprise[i] = pair_outcome_kl(running, searched[i]->nn_pair_probs);
        }
    }

    std::vector<float> weight(T);
    double sum_w = 0.0, sum_ps = 0.0, sum_vs = 0.0;
    for (size_t i = 0; i < T; i++) {
        weight[i] = searched[i]->target_weight;
        sum_w += weight[i];
        sum_ps += searched[i]->policy_surprise * weight[i];
        sum_vs += value_surprise[i] * weight[i];
    }
    if (sum_w >= 1.0) {
        double avg_ps = sum_ps / sum_w, avg_vs = sum_vs / sum_w;
        double vs_weight = params.value_surprise_data_weight;
        if (avg_vs < 0.010) vs_weight *= avg_vs / 0.010;
        double threshold = avg_ps * 1.5;
        std::vector<double> ps_prop(T), vs_prop(T);
        double ps_total = 0.0, vs_total = 0.0;
        for (size_t i = 0; i < T; i++) {
            double w = weight[i], ps = searched[i]->policy_surprise;
            ps_prop[i] = w * ps + (1.0 - w) * std::max(0.0, ps - threshold);
            vs_prop[i] = w * value_surprise[i];
            ps_total += ps_prop[i];
            vs_total += vs_prop[i];
        }
        ps_total = std::max(ps_total, 1e-10);
        vs_total = std::max(vs_total, 1e-10);
        for (size_t i = 0; i < T; i++)
            weight[i] = static_cast<float>((1.0 - params.policy_surprise_data_weight - vs_weight) * weight[i]
                + params.policy_surprise_data_weight * ps_prop[i] * sum_w / ps_total
                + vs_weight * vs_prop[i] * sum_w / vs_total);
    }

    std::uniform_real_distribution<float> unif(0.0f, 1.0f);
    for (size_t i = 0; i < T; i++) {
        float w = std::max(0.0f, weight[i]);
        float floored = std::floor(w);
        searched[i]->row_count = static_cast<int>(floored) + (unif(rng) < w - floored ? 1 : 0);
    }
}

GameRecord build_game_record(const std::vector<PlyRecord>& plies, const BoardState& final_state,
                             std::shared_ptr<const GameInput> game) {
    int N = final_state.N;
    auto final_eval = exact_pair_eval(final_state);
    int num_pairs = num_rank_pairs(final_state.num_players);

    std::vector<const std::vector<float>*> values;
    std::vector<size_t> searched_index_of(plies.size(), 0);
    for (size_t i = 0; i < plies.size(); i++)
        if (plies[i].searched) { searched_index_of[i] = values.size(); values.push_back(&plies[i].search_pair_probs); }

    GameRecord record;
    std::vector<torch::Tensor> feats, masks, policies, ranks;
    for (size_t i = 0; i < plies.size(); i++) {
        const PlyRecord& p = plies[i];
        if (!p.searched || p.row_count <= 0) continue;
        // KataGo's four value targets (the final outcome and three TD horizons) each get equal loss
        // weight; cross-entropy is linear in its target, so one head trained on their mean gets
        // exactly that loss.
        std::vector<float> target = final_eval.probs;
        for (float coeff : kTDHorizonCoeffs) {
            auto td = td_target(values, final_eval.probs, searched_index_of[i], td_now_factor(coeff, N));
            for (size_t c = 0; c < target.size(); c++) target[c] += td[c];
        }
        for (float& x : target) x *= 0.25f;
        feats.push_back(p.features.to(torch::kFloat16));
        masks.push_back(p.legal_mask);
        policies.push_back(torch::tensor(p.policy_target, torch::kFloat32));
        ranks.push_back(torch::tensor(target, torch::kFloat32).view({num_pairs, 3}));
        record.row_count.push_back(p.row_count);
        record.row_ply.push_back(static_cast<int>(i));
    }
    if (!feats.empty()) {
        record.features = torch::stack(feats, 0);
        record.legal_mask = torch::stack(masks, 0);
        record.policy_target = torch::stack(policies, 0);
        record.rank_target = torch::stack(ranks, 0);
    }
    auto to_int64 = [](const std::vector<int>& v) {
        auto t = torch::empty({(int64_t)v.size()}, torch::kInt64);
        auto a = t.accessor<int64_t, 1>();
        for (size_t i = 0; i < v.size(); i++) a[i] = v[i];
        return t;
    };
    record.stone_owner = to_int64(final_state.board);
    record.territory_owner = to_int64(final_state.score().territory_owner);
    if (!plies.empty() && plies[0].history_features.defined()) {
        std::vector<torch::Tensor> hist;
        for (auto& p : plies) hist.push_back(p.history_features.to(torch::kFloat16));
        record.history_features = torch::stack(hist, 0);
    }
    record.game = std::move(game);
    return record;
}

// ── Self-play ─────────────────────────────────────────────────────────────────

SelfPlayRunner::SelfPlayRunner(const GameConfig& cfg, nlohmann::json input_descr,
                               const nlohmann::json* history_descr, SelfPlayParams params,
                               SearchParams search_params, int num_games, int nn_server_threads,
                               int nn_max_batch, uint64_t seed)
    : cfg_(cfg), input_descr_(std::move(input_descr)), history_descr_(history_descr),
      params_(params), search_params_(search_params), nn_(nn_server_threads, nn_max_batch)
{
    std::mt19937_64 seeder(seed);
    auto program = parse_cleg(cfg_.board_descr);
    std::set<uint32_t> used;
    std::vector<std::pair<uint32_t, std::shared_ptr<const BoardConfig>>> boards;
    while ((int)boards.size() < num_games) {
        auto board_seed = static_cast<uint32_t>(seeder());
        if (!used.insert(board_seed).second) continue;
        auto bc = std::make_shared<const BoardConfig>(build_board_from_cleg(*program, board_seed));
        try {
            compute_node_slots(*bc, input_descr_);
        } catch (const std::runtime_error& e) {
            throw std::runtime_error("self-play board seed " + std::to_string(board_seed) + ": " + e.what());
        }
        boards.emplace_back(board_seed, std::move(bc));
    }
    threads_.reserve(num_games);
    for (auto& b : boards)
        threads_.emplace_back([this, s = seeder(), board_seed = b.first, bc = b.second] { worker(s, board_seed, bc); });
}

SelfPlayRunner::~SelfPlayRunner() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    workers_cv_.notify_all();
    for (auto& t : threads_) t.join();
}

std::vector<FinishedGame> SelfPlayRunner::play(const std::vector<Evaluator>& evaluators, int num_games) {
    // Every thread is paused at a ply boundary here, so no network request is pending.
    nn_.set_evaluators(evaluators);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        finished_.clear();
        target_games_ = num_games;
        running_ = true;
    }
    workers_cv_.notify_all();
    std::unique_lock<std::mutex> lock(mutex_);
    main_cv_.wait(lock, [&] { return !running_ && paused_ == (int)threads_.size(); });
    return std::move(finished_);
}

void SelfPlayRunner::worker(uint64_t seed, uint32_t board_seed, std::shared_ptr<const BoardConfig> board) {
    std::mt19937_64 rng(seed);
    MCTS mcts(search_params_, rng());
    Slot slot = new_slot(board, board_seed, rng);
    while (true) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (!running_ && !stopping_) {
                paused_++;
                main_cv_.notify_all();
                workers_cv_.wait(lock, [&] { return running_ || stopping_; });
                paused_--;
            }
            if (stopping_) return;
        }
        play_ply(slot, mcts, rng);
        if (slot.state.game_over()) {
            FinishedGame game = finish(slot, rng);
            slot = new_slot(board, board_seed, rng);
            std::lock_guard<std::mutex> lock(mutex_);
            finished_.push_back(std::move(game));
            if ((int)finished_.size() >= target_games_) running_ = false;
        }
    }
}

// A fresh game whose first floor(Exp(1) * policy_init_area_prop * N) moves are sampled from the raw
// policy (KataGo's initGamesWithPolicy) for opening variety.
SelfPlayRunner::Slot SelfPlayRunner::new_slot(std::shared_ptr<const BoardConfig> board, uint32_t board_seed,
                                              std::mt19937_64& rng) const {
    Slot s{new_state(cfg_, std::move(board)), {}, 0, {}, board_seed};
    s.game_input = make_game_input(s.state, input_descr_);
    int N = s.state.N;
    if (params_.linear_move_bound) {
        std::uniform_real_distribution<float> dist(params_.linear_move_bound->first, params_.linear_move_bound->second);
        int sampled = static_cast<int>(dist(rng) * N);
        s.state.max_plies = s.state.max_plies ? std::min(*s.state.max_plies, sampled) : sampled;
    }
    std::exponential_distribution<double> exp1(1.0);
    s.opening_moves_left = static_cast<int>(std::floor(exp1(rng) * params_.policy_init_area_prop * N));
    return s;
}

FinishedGame SelfPlayRunner::finish(Slot& slot, std::mt19937_64& rng) const {
    resolve_row_counts(slot.plies, exact_pair_eval(slot.state).probs, slot.state.N, params_, rng);
    json plies = json::array();
    for (auto& p : slot.plies) {
        json j;
        j["move"] = p.move;
        if (p.searched) {
            j["search"] = p.search_pair_probs;
            j["weight"] = p.row_count;
            if (p.row_count > 0) j["policy"] = p.policy_target;
        }
        plies.push_back(std::move(j));
    }
    GameRecord record = build_game_record(slot.plies, slot.state, slot.game_input);
    return FinishedGame{std::move(record), json{{"boardSeed", slot.board_seed}, {"plies", std::move(plies)}},
                        std::move(slot.state)};
}

void SelfPlayRunner::play_ply(Slot& slot, MCTS& mcts, std::mt19937_64& rng) {
    int N = slot.state.N;
    auto history_of = [&](const BoardState& s) {
        return history_descr_ ? board_to_features(s, *history_descr_).features : torch::Tensor();
    };

    if (slot.opening_moves_left > 0) {
        NNRow row = nn_.evaluate(slot.state, slot.game_input);
        std::vector<int> legal = legal_actions(slot.state);
        std::vector<double> w(legal.size(), 0.0);
        double total = 0.0;
        for (size_t k = 0; k < legal.size(); k++)
            if (row.policy[legal[k]] > 0) {
                w[k] = std::pow(static_cast<double>(row.policy[legal[k]]), 1.0 / params_.policy_init_temperature);
                total += w[k];
            }
        if (total <= 0.0) std::fill(w.begin(), w.end(), 1.0);
        std::discrete_distribution<int> dist(w.begin(), w.end());
        PlyRecord rec;
        rec.move = legal[dist(rng)];
        rec.history_features = history_of(slot.state);
        apply_action(slot.state, rec.move);
        slot.plies.push_back(std::move(rec));
        slot.opening_moves_left--;
        return;
    }

    int ply = slot.state.ply_count();
    RootSettings rs;
    rs.move_temperature = interpolate_early(params_.temperature_halflife, params_.move_temperature_early,
                                            params_.move_temperature, ply, N);
    float weight = 1.0f;
    std::uniform_real_distribution<float> unif(0.0f, 1.0f);
    if (unif(rng) < params_.cheap_prob) {
        rs.visits = params_.cheap_visits;
        rs.fpu_reduction_max = search_params_.fpu_reduction_max;
        weight = 0.0f;
    } else {
        rs.visits = params_.full_visits;
        rs.noise = true;
        rs.policy_temperature = interpolate_early(params_.temperature_halflife, params_.root_policy_temperature_early,
                                                  params_.root_policy_temperature, ply, N);
        rs.forced_playouts_coeff = params_.forced_playouts_coeff;
        // KataGo's reduceVisits: once some player's search win term has stayed above the threshold
        // for the last `lookback` plies, visits move toward reduced_visits_min and the weight toward
        // reduced_visits_weight by the square of how far past it is.
        int lookback = params_.reduce_visits_lookback;
        if (params_.reduce_visits && (int)slot.recent_win.size() >= lookback) {
            float most_extreme = *std::min_element(slot.recent_win.end() - lookback, slot.recent_win.end());
            float through = std::min(most_extreme, 1.0f) - params_.reduce_visits_threshold;
            if (through > 0) {
                float r = through / (1.0f - params_.reduce_visits_threshold);
                r *= r;
                rs.visits = std::max(params_.reduced_visits_min, static_cast<int>(std::round(
                    rs.visits + r * (params_.reduced_visits_min - rs.visits))));
                weight += r * (params_.reduced_visits_weight - weight);
            }
        }
    }

    SearchResult res = mcts.search(slot.state, rs, [&](const BoardState& s) { return nn_.evaluate(s, slot.game_input); });
    PlyRecord rec;
    rec.move = res.move;
    rec.searched = true;
    auto ft = board_to_features(slot.state, input_descr_);
    rec.features = ft.features;
    rec.legal_mask = ft.legal_mask;
    rec.history_features = history_of(slot.state);
    rec.policy_surprise = policy_kl(res.policy_target, res.root_prior);
    rec.policy_target = std::move(res.policy_target);
    rec.search_pair_probs = std::move(res.pair_probs);
    rec.nn_pair_probs = std::move(res.nn_pair_probs);
    rec.target_weight = weight;
    slot.recent_win.push_back(*std::max_element(res.win.begin(), res.win.end()));
    apply_action(slot.state, rec.move);
    slot.plies.push_back(std::move(rec));
}

GameRecord replay_trajectory(const nlohmann::json& trajectory, const GameConfig& cfg,
                             std::shared_ptr<const BoardConfig> bc, const nlohmann::json& descr,
                             const nlohmann::json* history_descr) {
    BoardState state = new_state(cfg, bc);
    state.max_plies = std::nullopt;  // the recorded game already obeyed its own (possibly sampled) bound
    auto game = make_game_input(state, descr);
    std::vector<PlyRecord> plies;
    for (const auto& j : trajectory.at("plies")) {
        PlyRecord p;
        p.move = j.at("move").get<int>();
        p.searched = j.contains("search");
        if (history_descr) p.history_features = board_to_features(state, *history_descr).features;
        if (p.searched) {
            p.search_pair_probs = j.at("search").get<std::vector<float>>();
            p.row_count = j.at("weight").get<int>();
            if (p.row_count > 0) {
                p.policy_target = j.at("policy").get<std::vector<float>>();
                auto ft = board_to_features(state, descr);
                p.features = ft.features;
                p.legal_mask = ft.legal_mask;
            }
        }
        apply_action(state, p.move);
        plies.push_back(std::move(p));
    }
    return build_game_record(plies, state, std::move(game));
}
