#include "model/input_descr.h"

using json = nlohmann::json;

// Number of bits needed to binary-encode any integer in [0, c] - used by
// compute_input_descr(), below, to size the liberty/playerStoneBudget/
// globalStoneBudget feature-block channel counts.
static int bits_for(int c) {
    int b = 0;
    while ((1 << b) <= c) b++;
    return b;
}

nlohmann::json compute_input_descr(const GameConfig& cfg, unsigned emb_dim, const std::vector<unsigned>& side_lens,
                                   int n_bound, int conn_px) {
    int ns     = cfg.num_stones;
    int np     = cfg.num_players;
    int tl_len = (int)cfg.turn_list.size();
    int lib_bits = bits_for(n_bound);

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

    if (conn_px > 0) {
        int channels = 1;
        for (unsigned k = 0; k < emb_dim; k++) channels *= conn_px;
        blocks.push_back(json::array({"connImg", conn_px})); total += channels;
    }

    return {{"blocks", blocks}, {"totalDims", total}, {"embDim", emb_dim}, {"sideLens", side_lens}, {"nBound", n_bound}};
}

nlohmann::json compute_history_descr(const GameConfig& cfg) {
    int tl_len = (int)cfg.turn_list.size(), ns = cfg.num_stones;
    return {{"blocks", json::array({json::array({"plyMod", tl_len}), json::array({"stoneOccupancy", ns})})},
            {"totalDims", tl_len + ns + 1}};
}
