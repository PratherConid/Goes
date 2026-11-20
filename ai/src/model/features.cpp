#include "model/features.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

int num_node_slots(const nlohmann::json& descr) {
    return descr.at("embDim").get<unsigned>() != 0 ? num_cells(descr.at("sideLens").get<std::vector<unsigned>>())
                                                   : descr.at("nBound").get<int>();
}

NodeSlots compute_node_slots(const BoardConfig& bc, const nlohmann::json& descr) {
    const auto side_lens = descr.at("sideLens").get<std::vector<unsigned>>();
    const int n_bound = descr.at("nBound").get<int>();
    if (bc.emb.emb_dim() != descr.at("embDim").get<unsigned>())
        throw std::runtime_error("the board's embedding dimension " + std::to_string(bc.emb.emb_dim()) +
                                 " differs from the model's embDim " + descr.at("embDim").dump());
    if (bc.N > n_bound)
        throw std::runtime_error("the board's " + std::to_string(bc.N) + " nodes exceed the model's nBound " +
                                 std::to_string(n_bound));
    if (bc.emb.emb_dim() != 0 && !fits_in(bc.emb, side_lens))
        throw std::runtime_error("the board's side lengths " + nlohmann::json(bc.emb.side_lens).dump() +
                                 " don't fit in the model's sideLens " + descr.at("sideLens").dump());
    const int num_slots = num_node_slots(descr);

    auto slot  = torch::empty({bc.N}, torch::kLong);
    auto valid = torch::zeros({num_slots}, torch::kFloat32);
    auto slot_a  = slot.accessor<int64_t, 1>();
    auto valid_a = valid.accessor<float, 1>();
    for (int i = 0; i < bc.N; i++) {
        int64_t s = bc.emb.emb_dim() == 0 ? i : 0, stride = 1;
        for (unsigned k = 0; k < bc.emb.emb_dim(); k++) {
            s += static_cast<int64_t>(bc.emb.pos[i][k]) * stride;
            stride *= side_lens[k];
        }
        slot_a[i]  = s;
        valid_a[s] = 1.0f;
    }
    return {slot, valid};
}

torch::Tensor slot_action_index(const NodeSlots& slots, int num_stones) {
    int64_t N = slots.slot.size(0), S = slots.valid.size(0);
    auto idx = torch::empty({num_stones * N + 1}, torch::kLong);
    for (int s = 0; s < num_stones; s++) idx.slice(0, s * N, (s + 1) * N).copy_(slots.slot + s * S);
    idx[num_stones * N] = num_stones * S;
    return idx;
}

torch::Tensor nodes_to_slots(const torch::Tensor& per_node, const NodeSlots& slots, int64_t dim) {
    auto shape = per_node.sizes().vec();
    shape[dim] = slots.valid.size(0);
    return torch::zeros(shape, per_node.options()).index_copy_(dim, slots.slot, per_node);
}

torch::Tensor actions_to_slots(const torch::Tensor& per_action, const GameInput& game) {
    int64_t num_stones = (game.action_index.size(0) - 1) / game.slots.slot.size(0);
    return torch::zeros({num_stones * game.slots.valid.size(0) + 1}, per_action.options())
        .index_copy_(0, game.action_index, per_action);
}

NeighborTiers neighbor_tiers(const BoardConfig& bc, const NodeSlots& slots) {
    const int N = bc.N;
    auto slot_a = slots.slot.accessor<int64_t, 1>();
    std::vector<int> by_slot(N);
    std::iota(by_slot.begin(), by_slot.end(), 0);
    std::sort(by_slot.begin(), by_slot.end(), [&](int a, int b) { return slot_a[a] < slot_a[b]; });

    std::array<std::vector<int32_t>, 3> src;
    auto deg = torch::zeros({3, slots.valid.size(0)}, torch::kLong);
    auto deg_a = deg.accessor<int64_t, 2>();
    std::vector<int> dist(N, -1), reached, frontier, next;
    for (int i : by_slot) {
        dist[i] = 0;
        reached.assign(1, i);
        frontier.assign(1, i);
        for (int d = 1; d <= 4; d++) {
            next.clear();
            for (int u : frontier)
                for (int v : bc.adlist[u])
                    if (dist[v] < 0) { dist[v] = d; next.push_back(v); reached.push_back(v); }
            frontier.swap(next);
        }
        for (int v : reached) {
            int t = dist[v] <= 1 ? 0 : dist[v] == 2 ? 1 : dist[v] == 4 ? 2 : -1;
            if (t < 0) continue;
            src[t].push_back(static_cast<int32_t>(slot_a[v]));
            deg_a[t][slot_a[i]]++;
        }
        for (int v : reached) dist[v] = -1;
    }
    NeighborTiers tiers;
    for (int t = 0; t < 3; t++) {
        tiers.src[t] = torch::empty({static_cast<int64_t>(src[t].size())}, torch::kInt32);
        std::copy(src[t].begin(), src[t].end(), tiers.src[t].data_ptr<int32_t>());
    }
    tiers.deg = deg;
    return tiers;
}

std::shared_ptr<const GameInput> make_game_input(const BoardState& state, const nlohmann::json& descr) {
    auto game = std::make_shared<GameInput>();
    game->slots = compute_node_slots(*state.bc, descr);
    game->action_index = slot_action_index(game->slots, state.num_stones);
    game->tiers = neighbor_tiers(*state.bc, game->slots);
    const auto& last = descr.at("blocks").back();
    if (last.at(0).get<std::string>() == "connImg")
        game->conn_img = render_conn_img(*state.bc, descr, last.at(1).get<int>());
    return game;
}

torch::Tensor render_conn_img(const BoardConfig& bc, const nlohmann::json& descr, int conn_px) {
    const auto side_lens = descr.at("sideLens").get<std::vector<unsigned>>();
    const size_t d = side_lens.size();
    const double px = conn_px;
    // The conn_px-scaled image, dimension 0 fastest.
    std::vector<int64_t> dims(d), stride(d);
    int64_t pixels = 1;
    for (size_t k = 0; k < d; k++) {
        dims[k] = static_cast<int64_t>(side_lens[k]) * conn_px;
        stride[k] = pixels;
        pixels *= dims[k];
    }
    std::vector<float> img(pixels, 0.0f);
    std::vector<double> a(d), b(d), frac(d);
    std::vector<int64_t> lo(d);
    for (int i = 0; i < bc.N; i++)
        for (int j : bc.adlist[i]) {
            if (j < i) continue;
            size_t m = 0;  // the line's longest axis
            for (size_t k = 0; k < d; k++) {
                a[k] = (bc.emb.pos[i][k] + 0.5) * px;
                b[k] = (bc.emb.pos[j][k] + 0.5) * px;
                if (std::abs(b[k] - a[k]) > std::abs(b[m] - a[m])) m = k;
            }
            if (a[m] == b[m]) continue;
            int64_t first = static_cast<int64_t>(std::ceil(std::min(a[m], b[m]) - 0.5));
            int64_t last  = static_cast<int64_t>(std::floor(std::max(a[m], b[m]) - 0.5));
            for (int64_t t = first; t <= last; t++) {
                double s = (t + 0.5 - a[m]) / (b[m] - a[m]);
                for (size_t k = 0; k < d; k++) {
                    if (k == m) continue;
                    double u = a[k] + s * (b[k] - a[k]) - 0.5;
                    lo[k] = static_cast<int64_t>(std::floor(u));
                    frac[k] = u - lo[k];
                }
                for (size_t corner = 0; corner < (size_t{1} << d); corner++) {
                    if (corner >> m & 1) continue;
                    double w = 1.0;
                    int64_t idx = t * stride[m];
                    bool inside = true;
                    for (size_t k = 0; k < d && inside; k++) {
                        if (k == m) continue;
                        bool up = corner >> k & 1;
                        int64_t c = lo[k] + up;
                        w *= up ? frac[k] : 1.0 - frac[k];
                        inside = c >= 0 && c < dims[k];
                        idx += c * stride[k];
                    }
                    if (inside && w > 0) img[idx] = std::max(img[idx], static_cast<float>(w));
                }
            }
        }

    int64_t S = num_cells(side_lens), C = 1;
    for (size_t k = 0; k < d; k++) C *= conn_px;
    auto out = torch::empty({S, C}, torch::kFloat32);
    auto out_a = out.accessor<float, 2>();
    for (int64_t s = 0; s < S; s++)
        for (int64_t r = 0; r < C; r++) {
            int64_t idx = 0, q = s, rr = r;
            for (size_t k = 0; k < d; k++) {
                idx += ((q % side_lens[k]) * conn_px + rr % conn_px) * stride[k];
                q /= side_lens[k];
                rr /= conn_px;
            }
            out_a[s][r] = img[idx];
        }
    return out;
}

torch::Tensor slot_features(const torch::Tensor& node_features, const GameInput& game) {
    auto x = nodes_to_slots(node_features, game.slots);
    if (game.conn_img.defined())
        x.narrow(1, x.size(1) - game.conn_img.size(1), game.conn_img.size(1)).copy_(game.conn_img);
    return x;
}

ModelInput slot_input(const BoardState& state, const nlohmann::json& descr, std::shared_ptr<const GameInput> game) {
    auto [features, legal_mask] = board_to_features(state, descr);
    auto x = slot_features(features, *game);
    auto mask = actions_to_slots(legal_mask, *game);
    return {std::move(x), std::move(mask), {}, std::move(game)};
}

torch::Tensor stack_valid(const std::vector<const ModelInput*>& inputs) {
    std::vector<torch::Tensor> rows;
    rows.reserve(inputs.size());
    for (const ModelInput* in : inputs) rows.push_back(in->game->slots.valid);
    return stack_cpu(rows);
}

// Soft, monotonic stand-in for "bit i of value": 1 at value=0, ramping down
// linearly to 0 at value=2^i and clamped at 0 beyond - i.e. max(0, 1 -
// value/2^i). Used (in place of raw binary bit-extraction) for every
// channel of the "liberty"/"playerStoneBudget"/"globalStoneBudget" blocks
// below, one channel per bit_index in [0, bits), so a small change in value
// always moves every channel by a small, continuous amount rather than
// flipping a bit discontinuously.
static float clamp_scale(int value, int bit_index) {
    return std::max(0.0f, 1.0f - static_cast<float>(value) / static_cast<float>(1 << bit_index));
}

// Computes the (N,F) feature tensor for `state` as of a given ply - shared by board_to_features()
// (ply = state.ply_count(), i.e. "now") and history_features_at_ply() (any past ply). Every block
// reads ply-indexed accessors (board_at(), turn_list[ply % len], legal_moves_data_at(),
// player_stone_place_cnt_at(), consecutive_passes_at()) rather than state's own "live" fields
// (state.board, state.next_turn, state.legal_moves_data(), ...) - those live fields are themselves
// exactly the ply_count()-indexed case of each accessor (e.g. next_turn ==
// turn_list[ply_count() % len], confirmed via make_move()'s own use of that formula). All of
// these accessors are O(1), backed by HistoryManager's per-ply interning - no BoardState mutation
// (withdraw_move()) is ever needed to reconstruct a past ply's features.
static torch::Tensor board_to_features_at_ply(const BoardState& state, int ply, const nlohmann::json& descr) {
    if (ply < 0 || ply > state.ply_count()) {
        std::cerr << "board_to_features_at_ply: ply " << ply << " out of bounds [0, "
                  << state.ply_count() << "]\n";
        throw std::out_of_range("board_to_features_at_ply: ply out of bounds");
    }

    int N = state.N;
    int ns = state.num_stones;
    int F = descr.at("totalDims").get<int>();
    int tl_len = (int)state.turn_list.size();

    auto features = torch::zeros({N, F}, torch::kFloat32);
    auto feat_a = features.accessor<float, 2>();

    const std::vector<int>& board = state.board_at(ply);
    const TurnInfo& turn_at_ply = state.turn_list[ply % tl_len];

    // group_liberty() is only needed to fill node_liberty/node_group_size, which only the
    // "liberty"/"groupSize" blocks read - skip the traversal entirely when descr requests neither
    // (e.g. the Transformer's minimal history descriptor never does), rather than always paying
    // for it and discarding the result.
    bool needs_liberty = false;
    for (auto& entry : descr.at("blocks")) {
        std::string name = entry.at(0).get<std::string>();
        if (name == "liberty" || name == "groupSize") { needs_liberty = true; break; }
    }

    std::vector<int> node_liberty(N, 0), node_group_size(N, 0);
    if (needs_liberty) {
        auto gdict = group_liberty(board, *state.adlist, N, turn_at_ply.friendly);
        for (auto& [color, entries] : gdict) {
            for (auto& e : entries) {
                int lib_count  = static_cast<int>(e.liberties.size());
                int group_size = static_cast<int>(e.group.size());
                for (int node : e.group) {
                    node_liberty[node]    = lib_count;
                    node_group_size[node] = group_size;
                }
            }
        }
    }

    auto& legal     = state.legal_moves_data_at(ply);
    auto& place_cnt = state.player_stone_place_cnt_at(ply);

    int off = 0;
    for (auto& entry : descr.at("blocks")) {
        std::string name = entry.at(0).get<std::string>();

        if (name == "stoneOccupancy") {
            for (int i = 0; i < N; i++) {
                int stone = board[i];
                if (stone > 0) feat_a[i][off + stone - 1] = 1.0f;
                else            feat_a[i][off + ns] = 1.0f;
            }
            off += ns + 1;
        } else if (name == "legalPlace") {
            for (int s = 1; s <= ns; s++)
                for (int i = 0; i < N; i++)
                    if (legal.captures[s][i].has_value())
                        feat_a[i][off + s - 1] = 1.0f;
            off += ns;
        } else if (name == "liberty") {
            int bits = entry.at(1).get<int>();
            for (int i = 0; i < N; i++) {
                int lib = node_liberty[i];
                for (int b = 0; b < bits; b++)
                    feat_a[i][off + b] = clamp_scale(lib, b);
            }
            off += bits;
        } else if (name == "groupSize") {
            float inv_n_bound = 1.0f / std::max(descr.at("nBound").get<int>(), 1);
            for (int i = 0; i < N; i++)
                feat_a[i][off] = node_group_size[i] * inv_n_bound;
            off += 1;
        } else if (name == "plyMod") {
            int block_tl_len = entry.at(1).get<int>();
            if (block_tl_len > 0) {
                int ply_mod = ply % block_tl_len;
                for (int i = 0; i < N; i++)
                    feat_a[i][off + ply_mod] = 1.0f;
            }
            off += block_tl_len;
        } else if (name == "consectivePassOneHot") {
            int bits = entry.at(1).get<int>();
            int cp = state.consecutive_passes_at(ply);
            for (int i = 0; i < N; i++)
                feat_a[i][off + cp] = 1.0f;
            off += bits;
        } else if (name == "playerStoneBudget") {
            auto& rows = entry.at(1);
            for (size_t s = 0; s < rows.size(); s++) {
                auto& row = rows[s];
                for (size_t p = 0; p < row.size(); p++) {
                    int bits = row[p].get<int>();
                    if (bits == 0) continue;
                    int lim = state.player_stone_place_limit[s][p].value();
                    int remaining = lim - place_cnt[s][p];
                    for (int b = 0; b < bits; b++) {
                        float v = clamp_scale(remaining, b);
                        for (int i = 0; i < N; i++) feat_a[i][off + b] = v;
                    }
                    off += bits;
                }
            }
        } else if (name == "globalStoneBudget") {
            auto& row = entry.at(1);
            for (size_t s = 0; s < row.size(); s++) {
                int bits = row[s].get<int>();
                if (bits == 0) continue;
                int lim = state.global_stone_place_limit[s].value();
                int placed = 0;
                for (int p = 0; p < state.num_players; p++) placed += place_cnt[s][p];
                int remaining = lim - placed;
                for (int b = 0; b < bits; b++) {
                    float v = clamp_scale(remaining, b);
                    for (int i = 0; i < N; i++) feat_a[i][off + b] = v;
                }
                off += bits;
            }
        } else if (name == "connImg") {
            // Board-derived and per slot rather than per node: slot_features() fills these channels
            // in from the game's GameInput after zero-extension, as the last ones.
            if (&entry != &descr.at("blocks").back())
                throw std::runtime_error("board_to_features_at_ply: connImg must be the last feature block");
            int channels = 1;
            for (unsigned k = 0; k < descr.at("embDim").get<unsigned>(); k++) channels *= entry.at(1).get<int>();
            off += channels;
        } else {
            throw std::runtime_error("board_to_features_at_ply: unknown feature block: " + name);
        }
    }
    assert(off == F && "board_to_features_at_ply offset accounting must match descr's totalDims");

    return features;
}

// Convert a BoardState to model input tensors (identical for all four architectures).
//
// Returns (CPU tensors)
//   features  : float32 (N, F)               per-node feature matrix
//   legal_mask: bool    (num_stones*N + 1,)   True = legal action (stone-major
//                                             flattened: index (stone-1)*N+pos;
//                                             last entry = pass) - slot_input()
//                                             zero-extends it to the slot-space
//                                             layout PolicyHead (heads.h) outputs.
//
// descr is the self-describing feature-block descriptor built once by
// compute_input_descr() (model/input_descr.h) at the start of a fresh
// training run and persisted verbatim thereafter (ModelConfig::input_descr,
// model_config.h) - shape {"blocks": [[name, ...args], ...], "totalDims": F,
// "embDim": emb_dim, "sideLens": side_lens, "nBound": n_bound}.
// See board_to_features_at_ply() above for the recognized block names/args
// and per-block fill logic (shared with this function).
//
// Features come from board_to_features_at_ply() at the current ply; the legal mask comes from
// legal_action_mask() rather than that shared function, since it depends on
// game_over()/no_trad_legal(), concepts that are only meaningful for the current position, not an
// arbitrary past ply.
FeatureTensors board_to_features(const BoardState& state, const nlohmann::json& descr) {
    return {board_to_features_at_ply(state, state.ply_count(), descr), legal_action_mask(state)};
}

std::vector<int> legal_actions(const BoardState& state) {
    int N = state.N;
    int ns = state.num_stones;
    std::vector<int> actions;
    if (state.game_over()) return actions;
    // As in BoardState::make_move(), a resigned player may only pass, and always may.
    auto resigned = state.resigned_players();
    if (std::find(resigned.begin(), resigned.end(), state.next_turn.player) != resigned.end()) {
        actions.push_back(ns * N);
        return actions;
    }
    const LegalMovesByStone& captures = state.legal_moves_data().captures;
    for (int s = 1; s <= ns; s++)
        for (int i = 0; i < N; i++)
            if (captures[s][i].has_value()) actions.push_back((s - 1) * N + i);
    if (!state.forced_pass_only || state.no_trad_legal()) actions.push_back(ns * N);
    return actions;
}

void apply_action(BoardState& state, int action) {
    int N = state.N;
    int ns = state.num_stones;
    bool ok = action == ns * N ? state.make_move(std::nullopt, std::nullopt)
                               : state.make_move(action % N, action / N + 1);
    if (!ok)
        throw std::runtime_error("apply_action: illegal action " + std::to_string(action) + " at ply " +
                                 std::to_string(state.ply_count()));
}

torch::Tensor legal_action_mask(const BoardState& state) {
    auto legal_mask = torch::zeros({state.num_stones * state.N + 1}, torch::kBool);
    auto mask_a = legal_mask.accessor<bool, 1>();
    for (int a : legal_actions(state)) mask_a[a] = true;
    return legal_mask;
}

torch::Tensor history_features_at_ply(const BoardState& state, int ply, const nlohmann::json& descr) {
    return board_to_features_at_ply(state, ply, descr);
}
