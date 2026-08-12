#include "game/fractal.h"
#include "game/topology.h"
#include <cassert>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

// Mirrors shared/fractal.ts's computeFlakeGlue(): for each base edge (i, j), exhaustively finds the
// unique pair of other base edges - (m1, m2) within the copy at i, (n1, n2) within the copy at j -
// whose S_i/S_j-transformed endpoints coincide two-for-two (S_i(x) = r*x + c*verts[i]). `verts` here
// are plain doubles used ONLY as scratch data for this one-time search - unlike BoardConfig::embed
// elsewhere in this codebase, they are never required to be exact integers, and are never exposed as
// a board's own position data (see fractal.h's own top comment).
static std::vector<std::array<int, 6>> compute_flake_glue(
    const std::vector<std::vector<double>>& verts, const std::vector<std::pair<int, int>>& edges,
    double r, double c) {
    auto dist = [](const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0.0;
        for (size_t k = 0; k < a.size(); k++) { double d = a[k] - b[k]; s += d * d; }
        return std::sqrt(s);
    };
    auto transform = [&](int i, int m) {
        std::vector<double> t(verts[m].size());
        for (size_t k = 0; k < t.size(); k++) t[k] = r * verts[m][k] + c * verts[i][k];
        return t;
    };
    auto swapped = [](const std::pair<int, int>& e) { return std::pair<int, int>(e.second, e.first); };

    std::vector<std::array<int, 6>> glue;
    for (const auto& ij : edges) {
        int i = ij.first, j = ij.second;
        std::vector<std::array<int, 4>> matches;
        for (const auto& e1 : edges)
            for (const auto& mab : { e1, swapped(e1) })
                for (const auto& e2 : edges)
                    for (const auto& nab : { e2, swapped(e2) })
                        if (dist(transform(i, mab.first), transform(j, nab.first)) < 1e-9 &&
                            dist(transform(i, mab.second), transform(j, nab.second)) < 1e-9)
                            matches.push_back({ mab.first, mab.second, nab.first, nab.second });

        // Dedupe representations that differ only by swapping (ma,mb)<->(na,nb) together.
        std::set<std::array<int, 4>> seen;
        std::vector<std::array<int, 4>> canon;
        for (const auto& m : matches) {
            std::array<int, 4> mirror = { m[1], m[0], m[3], m[2] };
            if (seen.count(m) || seen.count(mirror)) continue;
            seen.insert(m);
            canon.push_back(m);
        }
        assert(canon.size() == 1 && "expected exactly one glue relation per base edge");
        glue.push_back({ i, j, canon[0][0], canon[0][1], canon[0][2], canon[0][3] });
    }
    return glue;
}

// Mirrors shared/fractal.ts's computeNodeGlue(): for each base edge (i, j), exhaustively finds the
// unique pair of vertices (m, p) - m within the copy at i, p within the copy at j - whose
// S_i/S_j-transformed positions coincide. Unlike compute_flake_glue()'s own edge-to-edge (2-point)
// search, this is a single-point search - regular_polygon_flake_board()'s own non-multiple-of-4 case
// (see board_config.h's own doc comment): node-merge copies share exactly one point per base edge,
// not a whole growing edge, so there is no second point to match and no chain to track.
static std::vector<std::array<int, 4>> compute_node_glue(
    const std::vector<std::vector<double>>& verts, const std::vector<std::pair<int, int>>& edges,
    double r, double c) {
    auto dist = [](const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0.0;
        for (size_t k = 0; k < a.size(); k++) { double d = a[k] - b[k]; s += d * d; }
        return std::sqrt(s);
    };
    auto transform = [&](int i, int m) {
        std::vector<double> t(verts[m].size());
        for (size_t k = 0; k < t.size(); k++) t[k] = r * verts[m][k] + c * verts[i][k];
        return t;
    };

    std::vector<std::array<int, 4>> glue;
    for (const auto& ij : edges) {
        int i = ij.first, j = ij.second;
        std::vector<std::pair<int, int>> matches;
        for (size_t m = 0; m < verts.size(); m++)
            for (size_t p = 0; p < verts.size(); p++)
                if (dist(transform(i, static_cast<int>(m)), transform(j, static_cast<int>(p))) < 1e-9)
                    matches.push_back({ static_cast<int>(m), static_cast<int>(p) });
        assert(matches.size() == 1 && "expected exactly one node-glue relation per base edge");
        glue.push_back({ i, j, matches[0].first, matches[0].second });
    }
    return glue;
}

std::map<std::pair<int, int>, std::vector<std::array<int, 3>>> growing_edge_level_up_map(
    const std::map<std::pair<int, int>, std::array<int, 4>>& edge_glue_map) {
    std::map<std::pair<int, int>, std::vector<std::array<int, 3>>> m;
    for (const auto& [key, unused] : edge_glue_map) {
        (void)unused;
        int p = key.first, q = key.second;
        m[key] = { { p, p, q }, { q, p, q } };
    }
    return m;
}

std::map<int, std::pair<int, int>> identity_node_level_up_map(int num_leaf) {
    std::map<int, std::pair<int, int>> m;
    for (int vtx = 0; vtx < num_leaf; vtx++) m[vtx] = { vtx, vtx };
    return m;
}

SubFlakeResult node_edge_merge_flake_rec(
    int n, int num_leaf, int num_subs, const std::vector<std::pair<int, int>>& edges,
    const std::map<std::pair<int, int>, std::array<int, 4>>& edge_glue_map,
    const std::map<std::pair<int, int>, std::pair<int, int>>& node_glue_map,
    const std::map<std::pair<int, int>, std::vector<std::array<int, 3>>>& edge_level_up_map,
    const std::map<int, std::pair<int, int>>& node_level_up_map) {
    if (n == 1) {
        auto adj = zero_adj(num_leaf);
        for (const auto& e : edges) {
            adj[e.first][e.second] = 1;
            adj[e.second][e.first] = 1;
        }
        std::map<std::pair<int, int>, std::vector<int>> chains;
        for (const auto& [key, segs] : edge_level_up_map) { (void)segs; chains[key] = { key.first, key.second }; }
        std::vector<int> corners(num_leaf);
        std::iota(corners.begin(), corners.end(), 0);
        return { std::move(adj), std::move(corners), std::move(chains) };
    }

    std::vector<SubFlakeResult> subs;
    for (int s = 0; s < num_subs; s++)
        subs.push_back(node_edge_merge_flake_rec(
            n - 1, num_leaf, num_subs, edges, edge_glue_map, node_glue_map, edge_level_up_map,
            node_level_up_map));

    std::vector<std::pair<std::pair<int,int>, std::pair<int,int>>> merges;
    for (const auto& [key, eg] : edge_glue_map) {
        int p = key.first, q = key.second;
        auto [self_c, self_d, other_e, other_f] = eg;

        int self_lo = std::min(self_c, self_d), self_hi = std::max(self_c, self_d);
        std::vector<int> chain_self = subs[p].edge_chains.at({ self_lo, self_hi });
        if (self_c > self_d) std::reverse(chain_self.begin(), chain_self.end()); // start at self_c
        int other_lo = std::min(other_e, other_f), other_hi = std::max(other_e, other_f);
        std::vector<int> chain_other = subs[q].edge_chains.at({ other_lo, other_hi });
        if (other_e > other_f) std::reverse(chain_other.begin(), chain_other.end()); // start at other_e

        for (size_t idx = 0; idx < chain_self.size(); idx++)
            merges.push_back({ { p, chain_self[idx] }, { q, chain_other[idx] } });
    }
    for (const auto& [key, ng] : node_glue_map) {
        int p = key.first, q = key.second;
        auto [m, np] = ng;
        merges.push_back({ { p, subs[p].corners[m] }, { q, subs[q].corners[np] } });
    }

    // merge_boards is (pos, adj)-shaped - pass empty per-node positions (emb_dim=0 convention, see
    // zero_adj's own callers) since these boards track no position at all, and discard the result's
    // own (unused) pos.
    std::vector<RawBoard> sub_boards;
    for (auto& s : subs) sub_boards.push_back(RawBoard{ std::vector<std::vector<unsigned>>(s.adj.size()), s.adj });
    auto [combined, maps] = merge_boards(sub_boards, merges);

    // Only the first num_leaf sub-copies correspond to actual leaf-vertex attachment points (see
    // fractal.h's own doc comment) - any further entries (e.g. an auxiliary central copy) are purely
    // internal structure, not exposed as one of this call's own corners. node_level_up_map decouples
    // which sub-copy/corner each leaf vertex chases (see fractal.h's own doc comment) from leaf
    // vertex `vtx`'s own numbering - every shape here still chases sub-copy `vtx`'s own corner `vtx`,
    // but not because that's hardcoded here.
    std::vector<int> corners_out(num_leaf);
    for (int vtx = 0; vtx < num_leaf; vtx++) {
        auto [sub_idx, subflake_node] = node_level_up_map.at(vtx);
        corners_out[vtx] = maps[sub_idx][subs[sub_idx].corners[subflake_node]];
    }

    std::map<std::pair<int, int>, std::vector<int>> edge_chains;
    for (const auto& [key, segments] : edge_level_up_map) {
        std::vector<int> chain;
        for (const auto& seg : segments) {
            int sub_idx = seg[0], a = seg[1], b = seg[2];
            int lo = std::min(a, b), hi = std::max(a, b);
            std::vector<int> s = subs[sub_idx].edge_chains.at({ lo, hi });
            if (a > b) std::reverse(s.begin(), s.end()); // start at a
            for (auto idx : s) chain.push_back(maps[sub_idx][idx]);
        }
        edge_chains[key] = std::move(chain);
    }

    return { std::move(combined.adj), std::move(corners_out), std::move(edge_chains) };
}

const FlakeEdgeGlueData& dodecahedron_flake_data() {
    static FlakeEdgeGlueData data;
    static bool computed = false;
    if (computed) return data;

    const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
    const double scale = phi / 2.0; // normalizes edge length (2/phi at the raw scale above) to exactly 1
    auto s = [](int bit) { return bit == 0 ? 1.0 : -1.0; };
    auto x_idx = [](int sa, int sb, int sc) { return sa * 4 + sb * 2 + sc; };
    auto y_idx = [](int sb, int sc) { return 8 + sb * 2 + sc; };
    auto z_idx = [](int sa, int sb) { return 12 + sa * 2 + sb; };
    auto w_idx = [](int sa, int sc) { return 16 + sa * 2 + sc; };

    std::vector<std::vector<double>> verts(20);
    for (int sa = 0; sa < 2; sa++)
        for (int sb = 0; sb < 2; sb++)
            for (int sc = 0; sc < 2; sc++)
                verts[x_idx(sa, sb, sc)] = { s(sa) * scale, s(sb) * scale, s(sc) * scale };
    for (int sb = 0; sb < 2; sb++)
        for (int sc = 0; sc < 2; sc++)
            verts[y_idx(sb, sc)] = { 0.0, (s(sb) / phi) * scale, s(sc) * phi * scale };
    for (int sa = 0; sa < 2; sa++)
        for (int sb = 0; sb < 2; sb++)
            verts[z_idx(sa, sb)] = { (s(sa) / phi) * scale, s(sb) * phi * scale, 0.0 };
    for (int sa = 0; sa < 2; sa++)
        for (int sc = 0; sc < 2; sc++)
            verts[w_idx(sa, sc)] = { s(sa) * phi * scale, 0.0, (s(sc) / phi) * scale };

    std::vector<std::pair<int, int>> edges;
    for (int sa = 0; sa < 2; sa++)
        for (int sb = 0; sb < 2; sb++)
            for (int sc = 0; sc < 2; sc++) {
                int x = x_idx(sa, sb, sc);
                edges.push_back({ x, y_idx(sb, sc) });
                edges.push_back({ x, z_idx(sa, sb) });
                edges.push_back({ x, w_idx(sa, sc) });
            }
    for (int sc = 0; sc < 2; sc++) edges.push_back({ y_idx(0, sc), y_idx(1, sc) });
    for (int sb = 0; sb < 2; sb++) edges.push_back({ z_idx(0, sb), z_idx(1, sb) });
    for (int sa = 0; sa < 2; sa++) edges.push_back({ w_idx(sa, 0), w_idx(sa, 1) });

    const double r = 1.0 / (2.0 + phi);
    const double c = phi * phi * r;
    data.edges = std::move(edges);
    data.glue = compute_flake_glue(verts, data.edges, r, c);
    computed = true;
    return data;
}

const FlakeEdgeGlueData& icosahedron_flake_data() {
    static FlakeEdgeGlueData data;
    static bool computed = false;
    if (computed) return data;

    const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
    const double scale = 0.5; // normalizes edge length (2 at the raw scale above) to exactly 1
    auto s = [](int bit) { return bit == 0 ? 1.0 : -1.0; };
    auto a_idx = [](int sp, int sq) { return sp * 2 + sq; };
    auto b_idx = [](int sp, int sq) { return 4 + sp * 2 + sq; };
    auto c_idx = [](int sp, int sq) { return 8 + sp * 2 + sq; };

    std::vector<std::vector<double>> verts(12);
    for (int sp = 0; sp < 2; sp++)
        for (int sq = 0; sq < 2; sq++) {
            verts[a_idx(sp, sq)] = { 0.0, s(sp) * scale, s(sq) * phi * scale };
            verts[b_idx(sp, sq)] = { s(sp) * scale, s(sq) * phi * scale, 0.0 };
            verts[c_idx(sp, sq)] = { s(sq) * phi * scale, 0.0, s(sp) * scale };
        }

    auto adj = zero_adj(12);
    auto connect = [&](int i, int j) { adj[i][j] = 1; adj[j][i] = 1; };
    for (int sq = 0; sq < 2; sq++) {
        connect(a_idx(0, sq), a_idx(1, sq));
        connect(b_idx(0, sq), b_idx(1, sq));
        connect(c_idx(0, sq), c_idx(1, sq));
    }
    for (int sp = 0; sp < 2; sp++)
        for (int sq = 0; sq < 2; sq++)
            for (int free = 0; free < 2; free++) {
                connect(a_idx(sp, sq), b_idx(free, sp));
                connect(b_idx(sp, sq), c_idx(free, sp));
                connect(c_idx(sp, sq), a_idx(free, sp));
            }
    std::vector<std::pair<int, int>> edges;
    for (int i = 0; i < 12; i++)
        for (int j = i + 1; j < 12; j++)
            if (adj[i][j]) edges.push_back({ i, j });

    const double r = 1.0 / (1.0 + phi);
    const double c_coef = phi * r;
    data.edges = std::move(edges);
    data.glue = compute_flake_glue(verts, data.edges, r, c_coef);
    computed = true;
    return data;
}

// Vertex `2k`/`2k+1` are the `+-1` points on axis `k` (matching octahedron_board()'s own
// orthoplex_board(3) indexing) and are each other's antipode; every non-antipodal pair is an edge.
// Unlike dodeca/icosahedron, octahedron's own non-edges (the 3 antipodal pairs) are deliberately
// left out of compute_flake_glue()'s search entirely - not because they don't coincide (they do:
// every copy's own antipodal-attachment corner lands on the same shared center point, since
// `S_i(v_{antipode(i)}) = r*(-v_i) + c*v_i = (c-r)*v_i = 0` once `c = r`, for every `i`), but
// because that coincidence needs no glue entry of its own: it already follows transitively from the
// 12 real-edge glue relations, resolved automatically by node_edge_merge_flake_rec()'s own
// merge_boards() call (see that function's own doc comment).
const FlakeEdgeGlueData& octahedron_flake_data() {
    static FlakeEdgeGlueData data;
    static bool computed = false;
    if (computed) return data;

    const double edge_scale = 1.0 / std::sqrt(2.0); // matches orthoplex_board()'s own unit-edge scale
    std::vector<std::vector<double>> verts(6, std::vector<double>(3, 0.0));
    for (int k = 0; k < 3; k++) {
        verts[2 * k][k] = edge_scale;
        verts[2 * k + 1][k] = -edge_scale;
    }
    auto antipode = [](int i) { return i % 2 == 0 ? i + 1 : i - 1; };
    std::vector<std::pair<int, int>> edges;
    for (int i = 0; i < 6; i++)
        for (int j = i + 1; j < 6; j++)
            if (j != antipode(i)) edges.push_back({ i, j });

    const double r = 0.5, c = 0.5;
    data.edges = std::move(edges);
    data.glue = compute_flake_glue(verts, data.edges, r, c);
    computed = true;
    return data;
}

// Mirrors shared/fractal.ts's regularPolygonFlakeRC() - see its own doc comment for the full
// derivation: r+c=1 (a copy's own self vertex stays exactly where the outer polygon's vertex was)
// plus a closed-form coincidence ratio `c/r = 1 + 2*sum_{j=1}^{k} cos(2*pi*j/n_sides)`, for
// `k = n_sides/4 - 1` (merge by edge, `n_sides` a multiple of 4) or `k = n_sides/4` truncated (merge
// by node, otherwise - C++ integer division already floors for positive `n_sides`).
static std::pair<double, double> regular_polygon_flake_rc(int n_sides) {
    const double pi = std::acos(-1.0);
    bool is_edge_merge = n_sides % 4 == 0;
    int k = is_edge_merge ? n_sides / 4 - 1 : n_sides / 4;
    double cos_sum = 0.0;
    for (int j = 1; j <= k; j++) cos_sum += std::cos(2.0 * pi * j / n_sides);
    double ratio = 1.0 + 2.0 * cos_sum; // c/r
    double r = 1.0 / (1.0 + ratio), c = ratio / (1.0 + ratio);
    return { r, c };
}

const RegularPolygonFlakeData& regular_polygon_flake_data(int n_sides) {
    static std::map<int, RegularPolygonFlakeData> cache;
    auto it = cache.find(n_sides);
    if (it != cache.end()) return it->second;

    const double pi = std::acos(-1.0);
    double rad = 1.0 / (2.0 * std::sin(pi / n_sides)); // matches regular_polygon_board()'s own unit-edge scale
    std::vector<std::vector<double>> verts(n_sides);
    for (int k = 0; k < n_sides; k++) {
        double theta = 2.0 * pi * k / n_sides;
        verts[k] = { rad * std::cos(theta), rad * std::sin(theta) };
    }
    std::vector<std::pair<int, int>> edges;
    for (int k = 0; k < n_sides; k++) {
        int a = k, b = (k + 1) % n_sides;
        edges.push_back(a < b ? std::pair<int, int>{ a, b } : std::pair<int, int>{ b, a });
    }

    auto [r, c] = regular_polygon_flake_rc(n_sides);
    RegularPolygonFlakeData data;
    data.edges = edges;
    if (n_sides % 4 == 0) data.edge_glue = compute_flake_glue(verts, edges, r, c);
    else data.node_glue = compute_node_glue(verts, edges, r, c);

    auto result = cache.emplace(n_sides, std::move(data));
    return result.first->second;
}
