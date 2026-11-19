#include "training/self_play.h"
#include "model/features.h"
#include "model/rank_head.h"
#include <algorithm>
#include <random>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <mutex>

using json = nlohmann::json;

bool weak_equal(const GameConfig& a, const GameConfig& b) {
    if (a.board_descr != b.board_descr) return false;
    if (a.num_stones != b.num_stones) return false;
    if (a.num_players != b.num_players) return false;
    if (a.turn_list.size() != b.turn_list.size()) return false;
    for (size_t i = 0; i < a.turn_list.size(); i++) {
        const auto& ta = a.turn_list[i];
        const auto& tb = b.turn_list[i];
        if (ta.player != tb.player || ta.stones != tb.stones ||
            ta.is_protected != tb.is_protected || ta.friendly != tb.friendly)
            return false;
    }
    // Compare as sorted (stone,player) pair sets - order within one stone's
    // player list doesn't matter.
    auto flatten = [](const std::unordered_map<int, std::vector<int>>& m) {
        std::vector<std::pair<int,int>> pairs;
        for (auto& [stone, players] : m)
            for (int p : players) pairs.push_back({stone, p});
        std::sort(pairs.begin(), pairs.end());
        return pairs;
    };
    if (flatten(a.stone_to_player_map) != flatten(b.stone_to_player_map)) return false;
    // Which place limits are set decides the input's budget channels (compute_input_descr()); their
    // values may differ.
    auto set_entries = [](const std::vector<std::vector<std::optional<int>>>& limits) {
        std::vector<std::vector<bool>> set;
        for (auto& row : limits) {
            auto& r = set.emplace_back();
            for (auto& v : row) r.push_back(v.has_value());
        }
        return set;
    };
    if (set_entries(resolve_player_stone_place_limit(a.player_stone_place_limit, a.num_stones, a.num_players)) !=
        set_entries(resolve_player_stone_place_limit(b.player_stone_place_limit, b.num_stones, b.num_players)))
        return false;
    if (set_entries({resolve_global_stone_place_limit(a.global_stone_place_limit, a.num_stones)}) !=
        set_entries({resolve_global_stone_place_limit(b.global_stone_place_limit, b.num_stones)}))
        return false;
    if (a.forced_pass_only != b.forced_pass_only) return false;
    if (a.allow_suicide != b.allow_suicide) return false;
    if (a.score_rule != b.score_rule) return false;
    if (a.komi != b.komi) return false;
    if (a.ko_rule != b.ko_rule) return false;
    return true;
}

bool strong_equal(const GameConfig& a, const GameConfig& b) {
    return a.to_json() == b.to_json();
}

nlohmann::json GameConfig::to_json() const {
    json j;
    j["boardDescr"] = board_descr;
    j["boardSeed"]  = board_seed.has_value() ? json(*board_seed) : json(nullptr);
    j["numStones"]  = num_stones;
    j["numPlayers"] = num_players;

    json tl = json::array();
    for (auto& t : turn_list) {
        json ti;
        ti["player"]    = t.player;
        ti["stones"]    = t.stones;
        ti["protected"] = t.is_protected;
        ti["friendly"]  = t.friendly;
        tl.push_back(std::move(ti));
    }
    j["turnList"] = tl;

    json pspl = json::array();
    for (auto& row : player_stone_place_limit) {
        json r = json::array();
        for (auto& v : row) r.push_back(v.has_value() ? json(*v) : json(nullptr));
        pspl.push_back(std::move(r));
    }
    j["playerStonePlaceLimit"] = pspl;

    json gspl = json::array();
    for (auto& v : global_stone_place_limit) gspl.push_back(v.has_value() ? json(*v) : json(nullptr));
    j["globalStonePlaceLimit"] = gspl;

    json s2p;
    for (auto& [k, v] : stone_to_player_map) s2p[std::to_string(k)] = v;
    j["stoneToPlayerMap"] = s2p;

    j["forcedPassOnly"] = forced_pass_only;
    j["scoreRule"]      = score_rule;
    j["komi"]           = komi;
    j["koRule"]          = (ko_rule == KoRule::Positional) ? "positional" : "situational";
    j["allowSuicide"]   = allow_suicide;
    j["maxPlies"]       = max_plies.has_value() ? json(*max_plies) : json(nullptr);
    return j;
}

// ── Config JSON parsing (matches shared/gameConfig.ts's GameConfig.toJSON() shape) ──

static std::vector<TurnInfo> parse_turn_list(const json& j) {
    std::vector<TurnInfo> out;
    for (auto& t : j) {
        TurnInfo ti;
        ti.player = t.at("player").get<int>();
        ti.stones = t.at("stones").get<std::vector<int>>();
        ti.is_protected = t.at("protected").get<std::vector<int>>();
        ti.friendly     = t.at("friendly").get<std::vector<int>>();
        out.push_back(std::move(ti));
    }
    return out;
}

static std::unordered_map<int, std::vector<int>> parse_stone_to_player_map(const json& j) {
    std::unordered_map<int, std::vector<int>> out;
    for (auto& [k, v] : j.items()) out[std::stoi(k)] = v.get<std::vector<int>>();
    return out;
}

static std::vector<std::vector<std::optional<int>>> parse_player_stone_place_limit(
    const json& j, int num_stones, int num_players)
{
    if (j.is_null())
        return std::vector<std::vector<std::optional<int>>>(
            num_stones, std::vector<std::optional<int>>(num_players, std::nullopt));
    std::vector<std::vector<std::optional<int>>> out;
    for (auto& row : j) {
        std::vector<std::optional<int>> r;
        for (auto& v : row) r.push_back(v.is_null() ? std::nullopt : std::optional<int>(v.get<int>()));
        out.push_back(std::move(r));
    }
    return out;
}

static std::vector<std::optional<int>> parse_global_stone_place_limit(const json& j, int num_stones) {
    if (j.is_null()) return std::vector<std::optional<int>>(num_stones, std::nullopt);
    std::vector<std::optional<int>> out;
    for (auto& v : j) out.push_back(v.is_null() ? std::nullopt : std::optional<int>(v.get<int>()));
    return out;
}

static std::optional<int> parse_max_plies(const json& j) {
    return j.is_null() ? std::nullopt : std::optional<int>(j.get<int>());
}

// Same range check as shared/gameConfig.ts's GameConfig constructor - get<uint32_t>() alone would
// silently wrap a negative or oversized value.
static std::optional<uint32_t> parse_board_seed(const json& j) {
    if (j.is_null()) return std::nullopt;
    if (!j.is_number_unsigned() || j.get<uint64_t>() > UINT32_MAX)
        throw std::runtime_error("boardSeed must be null or an integer in [0, 2^32), got " + j.dump());
    return j.get<uint32_t>();
}

// Required keys are read via .at(), not operator[]: on a const json, operator[] with a missing key
// is undefined behavior (an assert compiled out under NDEBUG), not an exception.
GameConfig parse_game_cfg(const json& cfg) {
    GameConfig game_cfg;
    game_cfg.board_descr = cfg.at("boardDescr").get<std::string>();
    game_cfg.board_seed  = parse_board_seed(cfg.value("boardSeed", json(nullptr)));
    game_cfg.num_stones  = cfg.at("numStones").get<int>();
    game_cfg.num_players = cfg.at("numPlayers").get<int>();
    game_cfg.turn_list   = parse_turn_list(cfg.at("turnList"));
    game_cfg.player_stone_place_limit = parse_player_stone_place_limit(
        cfg.value("playerStonePlaceLimit", json(nullptr)), game_cfg.num_stones, game_cfg.num_players);
    game_cfg.global_stone_place_limit = parse_global_stone_place_limit(
        cfg.value("globalStonePlaceLimit", json(nullptr)), game_cfg.num_stones);
    game_cfg.stone_to_player_map = parse_stone_to_player_map(cfg.at("stoneToPlayerMap"));
    game_cfg.forced_pass_only = cfg.at("forcedPassOnly").get<bool>();
    game_cfg.score_rule       = cfg.value("scoreRule", std::string("area"));
    game_cfg.komi             = cfg.value("komi", std::vector<float>());
    game_cfg.ko_rule          = (cfg.value("koRule", std::string("situational")) == "positional")
                                     ? KoRule::Positional : KoRule::Situational;
    game_cfg.allow_suicide    = cfg.value("allowSuicide", false);
    game_cfg.max_plies        = parse_max_plies(cfg.value("maxPlies", json(nullptr)));
    return game_cfg;
}

std::vector<std::vector<std::optional<int>>> resolve_player_stone_place_limit(
    const std::vector<std::vector<std::optional<int>>>& limit, int num_stones, int num_players)
{
    if (!limit.empty()) return limit;
    return std::vector<std::vector<std::optional<int>>>(
        num_stones, std::vector<std::optional<int>>(num_players, std::nullopt));
}

std::vector<std::optional<int>> resolve_global_stone_place_limit(
    const std::vector<std::optional<int>>& limit, int num_stones)
{
    if (!limit.empty()) return limit;
    return std::vector<std::optional<int>>(num_stones, std::nullopt);
}

// Number of bits needed to binary-encode any integer in [0, c] - used by
// compute_input_descr(), below, to size the liberty/playerStoneBudget/
// globalStoneBudget feature-block channel counts.
static int bits_for(int c) {
    int b = 0;
    while ((1 << b) <= c) b++;
    return b;
}

nlohmann::json compute_input_descr(const GameConfig& cfg, int N) {
    int ns     = cfg.num_stones;
    int np     = cfg.num_players;
    int tl_len = (int)cfg.turn_list.size();
    int lib_bits = bits_for(N);

    json blocks = json::array();
    int total = 0;

    blocks.push_back(json::array({"stoneOccupancy", ns})); total += ns + 1;
    blocks.push_back(json::array({"legalPlace", ns}));      total += ns;
    blocks.push_back(json::array({"liberty", lib_bits}));    total += lib_bits;
    blocks.push_back(json::array({"groupSize"}));              total += 1;
    blocks.push_back(json::array({"plyMod", tl_len}));          total += tl_len;
    blocks.push_back(json::array({"consectivePassOneHot", tl_len + 1})); total += tl_len + 1;

    auto pspl = resolve_player_stone_place_limit(cfg.player_stone_place_limit, ns, np);
    json player_budget_bits = json::array();
    for (auto& row : pspl) {
        json r = json::array();
        for (auto& lim : row) {
            int bits = lim.has_value() ? bits_for(*lim) : 0;
            r.push_back(bits);
            total += bits;
        }
        player_budget_bits.push_back(std::move(r));
    }
    blocks.push_back(json::array({"playerStoneBudget", player_budget_bits}));

    auto gspl = resolve_global_stone_place_limit(cfg.global_stone_place_limit, ns);
    json global_budget_bits = json::array();
    for (auto& lim : gspl) {
        int bits = lim.has_value() ? bits_for(*lim) : 0;
        global_budget_bits.push_back(bits);
        total += bits;
    }
    blocks.push_back(json::array({"globalStoneBudget", global_budget_bits}));

    return {{"blocks", blocks}, {"totalDims", total}};
}

BoardState new_state(const GameConfig& cfg, const BoardConfig& bc) {
    auto pspl = resolve_player_stone_place_limit(cfg.player_stone_place_limit, cfg.num_stones, cfg.num_players);
    auto gspl = resolve_global_stone_place_limit(cfg.global_stone_place_limit, cfg.num_stones);
    auto komi = cfg.komi;
    if (komi.empty())
        komi = std::vector<float>(cfg.num_players, 0.0f);

    return BoardState(
        cfg.num_stones, cfg.num_players,
        cfg.turn_list, pspl, gspl, cfg.stone_to_player_map,
        cfg.forced_pass_only, cfg.score_rule, komi, cfg.ko_rule, cfg.allow_suicide, cfg.max_plies,
        std::vector<int>(bc.N, 0),
        bc);
}


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

// Mean over pairs of KL(target || predicted) between 3-way pair-outcome distributions, capped at 1 -
// the multi-pair form of KataGo's valueSurpriseKL() over its single win/loss/no-result distribution.
static float pair_outcome_kl(const std::vector<float>& target, const std::vector<float>& predicted) {
    size_t num_pairs = target.size() / 3;
    if (num_pairs == 0) return 0.0f;
    double kl = 0.0;
    for (size_t i = 0; i < target.size(); i++)
        if (target[i] > 1e-30f)
            kl += target[i] * (std::log(static_cast<double>(target[i])) -
                               std::log(std::max(static_cast<double>(predicted[i]), 1e-100)));
    return static_cast<float>(std::min(1.0, std::max(0.0, kl / num_pairs)));
}

// KL(target || prior) over actions - KataGo's policy surprise.
static float policy_kl(const std::vector<float>& target, const std::vector<float>& prior) {
    double kl = 0.0;
    for (size_t a = 0; a < target.size(); a++)
        if (target[a] > 1e-30f)
            kl += target[a] * (std::log(static_cast<double>(target[a])) -
                               std::log(std::max(static_cast<double>(prior[a]), 1e-100)));
    return static_cast<float>(std::max(0.0, kl));
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

GameRecord build_game_record(const std::vector<PlyRecord>& plies, const BoardState& final_state) {
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
    return record;
}

// ── Self-play ─────────────────────────────────────────────────────────────────

SelfPlayRunner::SelfPlayRunner(const GameConfig& cfg, const BoardConfig& bc, nlohmann::json input_descr,
                               const nlohmann::json* history_descr, SelfPlayParams params,
                               SearchParams search_params, int num_games, int nn_server_threads,
                               int nn_max_batch, uint64_t seed)
    : cfg_(cfg), bc_(bc), input_descr_(std::move(input_descr)), history_descr_(history_descr),
      params_(params), search_params_(search_params), nn_(nn_server_threads, nn_max_batch)
{
    std::mt19937_64 seeder(seed);
    threads_.reserve(num_games);
    for (int i = 0; i < num_games; i++) threads_.emplace_back([this, s = seeder()] { worker(s); });
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

void SelfPlayRunner::worker(uint64_t seed) {
    std::mt19937_64 rng(seed);
    MCTS mcts(search_params_, rng());
    Slot slot = new_slot(rng);
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
            slot = new_slot(rng);
            std::lock_guard<std::mutex> lock(mutex_);
            finished_.push_back(std::move(game));
            if ((int)finished_.size() >= target_games_) running_ = false;
        }
    }
}

// A fresh game whose first floor(Exp(1) * policy_init_area_prop * N) moves are sampled from the raw
// policy (KataGo's initGamesWithPolicy) for opening variety.
SelfPlayRunner::Slot SelfPlayRunner::new_slot(std::mt19937_64& rng) const {
    Slot s{new_state(cfg_, bc_), {}, 0, {}};
    if (params_.linear_move_bound) {
        std::uniform_real_distribution<float> dist(params_.linear_move_bound->first, params_.linear_move_bound->second);
        int sampled = static_cast<int>(dist(rng) * bc_.N);
        s.state.max_plies = s.state.max_plies ? std::min(*s.state.max_plies, sampled) : sampled;
    }
    std::exponential_distribution<double> exp1(1.0);
    s.opening_moves_left = static_cast<int>(std::floor(exp1(rng) * params_.policy_init_area_prop * bc_.N));
    return s;
}

FinishedGame SelfPlayRunner::finish(Slot& slot, std::mt19937_64& rng) const {
    resolve_row_counts(slot.plies, exact_pair_eval(slot.state).probs, bc_.N, params_, rng);
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
    GameRecord record = build_game_record(slot.plies, slot.state);
    return FinishedGame{std::move(record), json{{"plies", std::move(plies)}}, std::move(slot.state)};
}

void SelfPlayRunner::play_ply(Slot& slot, MCTS& mcts, std::mt19937_64& rng) {
    int N = bc_.N;
    auto history_of = [&](const BoardState& s) {
        return history_descr_ ? board_to_features(s, *history_descr_).features : torch::Tensor();
    };

    if (slot.opening_moves_left > 0) {
        NNRow row = nn_.evaluate(slot.state);
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

    SearchResult res = mcts.search(slot.state, rs, [this](const BoardState& s) { return nn_.evaluate(s); });
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

GameRecord replay_trajectory(const nlohmann::json& trajectory, const GameConfig& cfg, const BoardConfig& bc,
                             const nlohmann::json& descr, const nlohmann::json* history_descr) {
    BoardState state = new_state(cfg, bc);
    state.max_plies = std::nullopt;  // the recorded game already obeyed its own (possibly sampled) bound
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
    return build_game_record(plies, state);
}
