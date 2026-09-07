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
// search, this is a single-point search - regular_polygon_fractal_descr()'s own non-multiple-of-4
// case (see fractal.h's own doc comment): node-merge copies share exactly one point per base edge,
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

// Mirrors shared/fractal.ts's POINT_GLUE_OBJECT/EDGE_GLUE_OBJECT - see fractal.h's own doc comment
// for why these are plain shared constants (stateless, no captures), unlike
// menger_hyperface_glue_object() below.
const GlueObjectType POINT_GLUE_OBJECT{ [](const std::vector<int>& tags) -> std::vector<GlueStep> {
    int v = tags[0];
    return { GlueStep{ v, { v } } };
} };
const GlueObjectType EDGE_GLUE_OBJECT{ [](const std::vector<int>& tags) -> std::vector<GlueStep> {
    int c = tags[0], d = tags[1];
    return { GlueStep{ c, { c, d } }, GlueStep{ d, { c, d } } };
} };

std::vector<std::string> glue_object_addresses(const GlueObjectType& object, const std::vector<int>& tags, int depth) {
    if (depth == 1) {
        std::vector<std::string> addrs;
        for (int t : tags) addrs.push_back(std::to_string(t));
        return addrs;
    }
    std::vector<std::string> addrs;
    for (const auto& step : object.step(tags)) {
        auto child_addrs = glue_object_addresses(object, step.tags, depth - 1);
        for (auto& a : child_addrs) addrs.push_back(std::to_string(step.sub_slot) + "," + a);
    }
    return addrs;
}

SubFlakeResult node_edge_merge_flake_rec(int n, const FractalDescr& descr) {
    if (n == 1) {
        auto adj = zero_adj(descr.num_leaf);
        for (const auto& e : descr.leaf_conn) { adj[e.first][e.second] = 1; adj[e.second][e.first] = 1; }
        std::map<std::string, int> labels;
        for (int v = 0; v < descr.num_leaf; v++) labels[std::to_string(v)] = v;
        return { std::move(adj), std::move(labels) };
    }

    std::vector<SubFlakeResult> subs;
    for (int s = 0; s < descr.num_subs; s++) { (void)s; subs.push_back(node_edge_merge_flake_rec(n - 1, descr)); }

    std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> merges;
    for (const auto& [key, entry] : descr.glue_map) {
        int p = key.first, q = key.second;
        auto self_addrs = glue_object_addresses(entry.object, entry.self_vertices, n - 1);
        auto other_addrs = glue_object_addresses(entry.object, entry.other_vertices, n - 1);
        for (size_t i = 0; i < self_addrs.size(); i++)
            merges.push_back({ { p, subs[p].labels.at(self_addrs[i]) }, { q, subs[q].labels.at(other_addrs[i]) } });
    }

    // Each sub's own addresses are relative to ITS OWN numbering - prepend its slot index so they
    // become valid addresses at THIS level, before combining (mirrors shared/fractal.ts's own
    // nodeEdgeMergeFlakeRec() doc comment). merge_boards() is (pos, adj, labels)-shaped - pass empty
    // per-node positions (emb_dim=0 convention, see zero_adj's own callers) since these boards track
    // no position at all, and discard the result's own (unused) pos.
    std::vector<RawBoard> boards_for_merge;
    for (int slot = 0; slot < descr.num_subs; slot++) {
        std::map<std::string, int> relabeled;
        for (const auto& [addr, idx] : subs[slot].labels) relabeled[std::to_string(slot) + "," + addr] = idx;
        boards_for_merge.push_back(RawBoard{
            std::vector<std::vector<unsigned>>(subs[slot].adj.size()), subs[slot].adj, std::move(relabeled) });
    }
    auto [combined, maps] = merge_boards(boards_for_merge, merges);
    (void)maps;

    return { std::move(combined.adj), std::move(combined.labels) };
}

std::vector<std::vector<int>> build_fractal(int n, const FractalDescr& descr) {
    return node_edge_merge_flake_rec(n, descr).adj;
}

const FractalDescr& dodecahedron_fractal_descr() {
    static FractalDescr descr;
    static bool computed = false;
    if (computed) return descr;

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
    auto glue = compute_flake_glue(verts, edges, r, c);

    descr.num_leaf = 20;
    descr.num_subs = 20;
    descr.leaf_conn = std::move(edges);
    for (const auto& g : glue)
        descr.glue_map[{ g[0], g[1] }] = GlueEntry{ EDGE_GLUE_OBJECT, { g[2], g[3] }, { g[4], g[5] } };

    computed = true;
    return descr;
}

const FractalDescr& icosahedron_fractal_descr() {
    static FractalDescr descr;
    static bool computed = false;
    if (computed) return descr;

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
    auto glue = compute_flake_glue(verts, edges, r, c_coef);

    descr.num_leaf = 12;
    descr.num_subs = 12;
    descr.leaf_conn = std::move(edges);
    for (const auto& g : glue)
        descr.glue_map[{ g[0], g[1] }] = GlueEntry{ EDGE_GLUE_OBJECT, { g[2], g[3] }, { g[4], g[5] } };

    computed = true;
    return descr;
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
const FractalDescr& octahedron_fractal_descr() {
    static FractalDescr descr;
    static bool computed = false;
    if (computed) return descr;

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
    auto glue = compute_flake_glue(verts, edges, r, c);

    descr.num_leaf = 6;
    descr.num_subs = 6;
    descr.leaf_conn = std::move(edges);
    for (const auto& g : glue)
        descr.glue_map[{ g[0], g[1] }] = GlueEntry{ EDGE_GLUE_OBJECT, { g[2], g[3] }, { g[4], g[5] } };

    computed = true;
    return descr;
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

// Raw base edges + glue relations shared by regular_polygon_fractal_descr()/
// central_pentagon_fractal_descr() - a distinct caching layer with no direct TS counterpart:
// shared/fractal.ts's own regularPolygonFractalDescr()/centralPentagonFractalDescr() each
// independently recompute their own verts/edges/glue (a JS Map cache per function is cheap enough
// there), whereas here central_pentagon_fractal_descr() needs the exact same base pentagon
// edges/node_glue regular_polygon_fractal_descr(5, ...) does, so this avoids a second distance
// search for identical data - mirrors the pre-generalization version of this file's own
// regular_polygon_flake_data(), unchanged in spirit, just feeding the new GlueEntry-based descr
// builders below instead of the old edge_glue_map/node_glue_map split.
struct RegularPolygonRawData {
    std::vector<std::pair<int, int>> edges;
    std::vector<std::array<int, 6>> edge_glue; // populated iff n_sides % 4 == 0
    std::vector<std::array<int, 4>> node_glue; // populated iff n_sides % 4 != 0
};
static const RegularPolygonRawData& regular_polygon_raw_data(int n_sides) {
    static std::map<int, RegularPolygonRawData> cache;
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
    RegularPolygonRawData data;
    data.edges = edges;
    if (n_sides % 4 == 0) data.edge_glue = compute_flake_glue(verts, edges, r, c);
    else data.node_glue = compute_node_glue(verts, edges, r, c);

    auto result = cache.emplace(n_sides, std::move(data));
    return result.first->second;
}

const FractalDescr& regular_polygon_fractal_descr(int n_sides, bool center) {
    static std::map<std::pair<int, bool>, FractalDescr> cache;
    auto key = std::make_pair(n_sides, center);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    const auto& raw = regular_polygon_raw_data(n_sides);
    FractalDescr descr;
    descr.num_leaf = n_sides;
    descr.leaf_conn = raw.edges;
    int num_subs = n_sides;
    if (n_sides % 4 == 0) {
        for (const auto& g : raw.edge_glue)
            descr.glue_map[{ g[0], g[1] }] = GlueEntry{ EDGE_GLUE_OBJECT, { g[2], g[3] }, { g[4], g[5] } };
    } else {
        for (const auto& g : raw.node_glue)
            descr.glue_map[{ g[0], g[1] }] = GlueEntry{ POINT_GLUE_OBJECT, { g[2] }, { g[3] } };
    }

    if (center && n_sides % 2 == 0 && n_sides > 4) {
        int center_idx = num_subs;
        num_subs += 1;
        for (int i = 0; i < n_sides; i++)
            descr.glue_map[{ i, center_idx }] =
                GlueEntry{ POINT_GLUE_OBJECT, { (i + n_sides / 2) % n_sides }, { i } };
    }
    descr.num_subs = num_subs;

    auto result = cache.emplace(key, std::move(descr));
    return result.first->second;
}

const FractalDescr& central_pentagon_fractal_descr() {
    static FractalDescr descr;
    static bool computed = false;
    if (computed) return descr;

    const int n_sides = 5;
    const auto& raw = regular_polygon_raw_data(n_sides); // 5 % 4 != 0 - node_glue is the populated one

    descr.num_leaf = n_sides;
    descr.leaf_conn = raw.edges;
    for (const auto& g : raw.node_glue)
        descr.glue_map[{ g[0], g[1] }] = GlueEntry{ POINT_GLUE_OBJECT, { g[2] }, { g[3] } };

    int center_idx = n_sides;
    descr.num_subs = n_sides + 1;
    for (int i = 0; i < n_sides; i++) {
        int a = i, b = (i + 1) % n_sides;
        int j = (i + 3) % n_sides;
        descr.glue_map[{ j, center_idx }] = GlueEntry{ EDGE_GLUE_OBJECT, { a, b }, { b, a } };
    }

    computed = true;
    return descr;
}

// Mirrors shared/fractal.ts's radixTuples(): all `base^k` length-`k` tuples with entries in
// `0..base-1`, in mixed-radix counting order.
static std::vector<std::vector<int>> radix_tuples(int k, int base) {
    std::vector<std::vector<int>> tuples = { {} };
    for (int d = 0; d < k; d++) {
        std::vector<std::vector<int>> next;
        for (const auto& t : tuples)
            for (int v = 0; v < base; v++) {
                auto nt = t;
                nt.push_back(v);
                next.push_back(std::move(nt));
            }
        tuples = std::move(next);
    }
    return tuples;
}

// Mirrors shared/fractal.ts's encodeMengerVertex()/decodeMengerVertex().
static int encode_menger_vertex(const std::vector<int>& coords) {
    int acc = 0;
    for (int c : coords) acc = acc * 2 + c;
    return acc;
}
static std::vector<int> decode_menger_vertex(int dim, int v) {
    std::vector<int> coords(dim);
    for (int i = 0; i < dim; i++) coords[i] = (v >> (dim - 1 - i)) & 1;
    return coords;
}

// Mirrors shared/fractal.ts's mengerFaceTags() - the general form, taking a SET of fixed axes (see
// that TS function's own doc comment for why a single glue object needs to handle any fixed-axis-set
// size, from 1 up to `dim`).
static std::vector<int> menger_face_tags(int dim, const std::vector<int>& fixed_axes, const std::vector<int>& fixed_vals) {
    std::vector<int> free_axes;
    for (int a = 0; a < dim; a++)
        if (std::find(fixed_axes.begin(), fixed_axes.end(), a) == fixed_axes.end()) free_axes.push_back(a);

    std::vector<int> tags;
    int r = 1 << free_axes.size();
    for (int i = 0; i < r; i++) {
        std::vector<int> c(dim, 0);
        for (size_t j = 0; j < fixed_axes.size(); j++) c[fixed_axes[j]] = fixed_vals[j];
        for (size_t j = 0; j < free_axes.size(); j++) c[free_axes[j]] = (i >> j) & 1;
        tags.push_back(encode_menger_vertex(c));
    }
    return tags;
}

// Mirrors shared/fractal.ts's isMengerGridKept().
static bool is_menger_grid_kept(const std::vector<int>& grid, const std::vector<int>& indicator) {
    int off_center = 0;
    for (int v : grid) if (v != 1) off_center++;
    return indicator[off_center] == 1;
}

// Mirrors shared/fractal.ts's grid.join(',') string-keying of `slotOf` - `positions`' own grid
// position -> subDescr slot lookup, closed over by menger_hyperface_glue_object() below.
static std::string join_grid(const std::vector<int>& grid) {
    std::string s;
    for (size_t i = 0; i < grid.size(); i++) { if (i) s += ","; s += std::to_string(grid[i]); }
    return s;
}

// Mirrors shared/fractal.ts's mengerHyperfaceGlueObject() - see its own doc comment for the full
// derivation (finding the FULL SET of axes every input tag agrees on, rather than just the first,
// is what lets one implementation handle every possible shared sub-face dimension uniformly, down to
// a single shared corner point). Closes over `dim`/`indicator`/`slot_of` by value (mirroring the TS
// side's own JS closure) - see fractal.h's own doc comment for why this, unlike
// POINT_GLUE_OBJECT/EDGE_GLUE_OBJECT above, cannot be a stateless shared constant.
static GlueObjectType menger_hyperface_glue_object(int dim, std::vector<int> indicator, std::map<std::string, int> slot_of) {
    return GlueObjectType{
        [dim, indicator, slot_of](const std::vector<int>& tags) -> std::vector<GlueStep> {
            std::vector<std::vector<int>> coords;
            for (int t : tags) coords.push_back(decode_menger_vertex(dim, t));

            std::vector<int> fixed_axes;
            for (int axis = 0; axis < dim; axis++) {
                bool all_agree = true;
                for (const auto& c : coords) if (c[axis] != coords[0][axis]) { all_agree = false; break; }
                if (all_agree) fixed_axes.push_back(axis);
            }
            assert(!fixed_axes.empty() && "menger_hyperface_glue_object: tags do not lie on a common sub-face");
            std::vector<int> free_axes;
            for (int a = 0; a < dim; a++)
                if (std::find(fixed_axes.begin(), fixed_axes.end(), a) == fixed_axes.end()) free_axes.push_back(a);
            std::vector<int> fixed_vals;
            for (int ax : fixed_axes) fixed_vals.push_back(coords[0][ax]);
            std::vector<int> fixed_grids;
            for (int v : fixed_vals) fixed_grids.push_back(v == 0 ? 0 : 2);

            std::vector<GlueStep> steps;
            for (const auto& free : radix_tuples(static_cast<int>(free_axes.size()), 3)) {
                std::vector<int> grid(dim, 1);
                for (size_t j = 0; j < fixed_axes.size(); j++) grid[fixed_axes[j]] = fixed_grids[j];
                for (size_t j = 0; j < free_axes.size(); j++) grid[free_axes[j]] = free[j];
                if (!is_menger_grid_kept(grid, indicator)) continue;
                auto it = slot_of.find(join_grid(grid));
                assert(it != slot_of.end() && "menger_hyperface_glue_object: no sub-cube at grid position");
                steps.push_back(GlueStep{ it->second, menger_face_tags(dim, fixed_axes, fixed_vals) });
            }
            return steps;
        }
    };
}

const FractalDescr& menger_fractal_descr(int dim, const std::vector<int>& indicator) {
    static std::map<int, std::map<int, FractalDescr>> cache;
    assert(dim >= 1 && "dim must be a positive integer");
    assert(static_cast<int>(indicator.size()) == dim + 1 &&
        "indicator must be a length-(dim+1) list of 0/1 entries");

    int indicator_bits = 0;
    for (int b : indicator) indicator_bits = indicator_bits * 2 + b;
    auto& by_indicator = cache[dim];
    auto found = by_indicator.find(indicator_bits);
    if (found != by_indicator.end()) return found->second;

    int num_corners = 1 << dim;
    std::vector<std::pair<int, int>> leaf_conn;
    for (int v1 = 0; v1 < num_corners; v1++)
        for (int v2 = v1 + 1; v2 < num_corners; v2++) {
            auto c1 = decode_menger_vertex(dim, v1), c2 = decode_menger_vertex(dim, v2);
            int diff = 0;
            for (int i = 0; i < dim; i++) if (c1[i] != c2[i]) diff++;
            if (diff == 1) leaf_conn.push_back({ v1, v2 });
        }

    std::vector<std::vector<int>> positions;
    for (const auto& g : radix_tuples(dim, 3)) if (is_menger_grid_kept(g, indicator)) positions.push_back(g);
    std::map<std::string, int> slot_of;
    for (size_t i = 0; i < positions.size(); i++) slot_of[join_grid(positions[i])] = static_cast<int>(i);

    GlueObjectType object = menger_hyperface_glue_object(dim, indicator, slot_of);
    FractalDescr descr;
    descr.num_leaf = num_corners;
    descr.num_subs = static_cast<int>(positions.size());
    descr.leaf_conn = std::move(leaf_conn);

    // One glue_map entry per pair of surviving grid positions that TOUCH (every axis differs by at
    // most 1, at least one axis differs) - mirrors shared/fractal.ts's mengerFractalDescr() own
    // `glueMap`-building loop.
    for (size_t i = 0; i < positions.size(); i++)
        for (size_t j = i + 1; j < positions.size(); j++) {
            const auto& gi = positions[i];
            const auto& gj = positions[j];
            std::vector<int> diff_axes;
            bool touching = true;
            for (int axis = 0; axis < dim; axis++) {
                int d = gj[axis] - gi[axis];
                if (d == 0) continue;
                if (std::abs(d) != 1) { touching = false; break; }
                diff_axes.push_back(axis);
            }
            if (!touching || diff_axes.empty()) continue;

            std::vector<int> self_vals, other_vals;
            for (int ax : diff_axes) {
                int sv = gi[ax] < gj[ax] ? 1 : 0;
                self_vals.push_back(sv);
                other_vals.push_back(1 - sv);
            }
            descr.glue_map[{ static_cast<int>(i), static_cast<int>(j) }] = GlueEntry{
                object, menger_face_tags(dim, diff_axes, self_vals), menger_face_tags(dim, diff_axes, other_vals)
            };
        }

    auto result = by_indicator.emplace(indicator_bits, std::move(descr));
    return result.first->second;
}
