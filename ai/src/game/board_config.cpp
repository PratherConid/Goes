#include "game/board_config.h"
#include "game/geometry.h"
#include "game/topology.h"
#include <cassert>
#include <algorithm>
#include <array>
#include <functional>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <map>
#include <set>

static BoardConfig make_bc(std::vector<std::vector<int>> adj,
                            unsigned emb_dim,
                            std::vector<std::vector<unsigned>> emb) {
    int N = static_cast<int>(emb.size());
    assert(static_cast<int>(adj.size()) == N &&
           (N == 0 || static_cast<int>(adj[0].size()) == N) && "adj must be N×N");
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            assert(adj[i][j] == adj[j][i] && "adj must be symmetric");
    return {N, std::move(adj), emb_dim, std::move(emb)};
}

static std::vector<std::vector<int>> zero_adj(int N) {
    return std::vector<std::vector<int>>(N, std::vector<int>(N, 0));
}

BoardConfig quotient_board(const BoardConfig& bc,
                           const std::vector<std::pair<int,int>>& quot) {
    int N = bc.N;
    std::vector<int> parent(N);
    std::iota(parent.begin(), parent.end(), 0);

    std::function<int(int)> find = [&](int x) -> int {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };

    for (auto [a, b] : quot) {
        assert(0 <= a && a < N && 0 <= b && b < N && "quot indices out of bounds");
        int pa = find(a), pb = find(b);
        if (pa != pb) parent[pa] = pb;
    }

    std::vector<int> roots(N);
    for (int i = 0; i < N; i++) roots[i] = find(i);
    std::vector<int> unique_roots = roots;
    std::sort(unique_roots.begin(), unique_roots.end());
    unique_roots.erase(std::unique(unique_roots.begin(), unique_roots.end()),
                       unique_roots.end());
    int new_n = static_cast<int>(unique_roots.size());

    std::vector<int> root_to_new(N, -1);
    for (int i = 0; i < new_n; i++) root_to_new[unique_roots[i]] = i;
    std::vector<int> node_to_new(N);
    for (int i = 0; i < N; i++) node_to_new[i] = root_to_new[roots[i]];

    // New positions: average of class members (all emb_dim coordinates, not just the first two -
    // needed for merge_close to work on cubical/hypercube boards, not just 2D ones).
    std::vector<std::vector<unsigned>> new_embed(new_n, std::vector<unsigned>(bc.emb_dim, 0u));
    std::vector<int> cnt(new_n, 0);
    for (int i = 0; i < N; i++) {
        int ni = node_to_new[i];
        for (unsigned k = 0; k < bc.emb_dim; k++) new_embed[ni][k] += bc.embed[i][k];
        cnt[ni]++;
    }
    for (int ni = 0; ni < new_n; ni++)
        for (unsigned k = 0; k < bc.emb_dim; k++) new_embed[ni][k] /= cnt[ni];

    auto new_adj = zero_adj(new_n);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            if (!bc.adj[i][j]) continue;
            int ni = node_to_new[i], nj = node_to_new[j];
            if (ni != nj) new_adj[ni][nj] = 1;
        }

    return make_bc(std::move(new_adj), bc.emb_dim, std::move(new_embed));
}

BoardConfig edge_split(const BoardConfig& bc, int split_n) {
    assert(split_n >= 1 && "split_n must be at least 1");
    int N = bc.N;
    std::vector<std::vector<unsigned>> pos(N);
    for (int i = 0; i < N; i++) {
        pos[i].resize(bc.emb_dim);
        for (unsigned k = 0; k < bc.emb_dim; k++)
            pos[i][k] = bc.embed[i][k] * split_n;
    }
    std::vector<std::pair<int,int>> edges;
    for (int i = 0; i < N; i++)
        for (int j = i+1; j < N; j++) {
            if (!bc.adj[i][j]) continue;
            int prev = i;
            for (int k = 1; k < split_n; k++) {
                std::vector<unsigned> np(bc.emb_dim);
                for (unsigned d = 0; d < bc.emb_dim; d++)
                    np[d] = (unsigned)((int)pos[i][d] + k * ((int)bc.embed[j][d] - (int)bc.embed[i][d]));
                int idx = (int)pos.size();
                pos.push_back(std::move(np));
                edges.push_back({prev, idx});
                prev = idx;
            }
            edges.push_back({prev, j});
        }
    int new_n = (int)pos.size();
    auto adj = zero_adj(new_n);
    for (auto& [a, b] : edges) { adj[a][b] = 1; adj[b][a] = 1; }
    return make_bc(std::move(adj), bc.emb_dim, std::move(pos));
}

BoardConfig rectify(const BoardConfig& bc) {
    // A real (always-active) check, not assert() - rectify's connectivity is decided by the angular
    // ordering of real edge directions around each vertex (see convex_hull_edges below), which is
    // undefined for an emb_dim=0 board (e.g. dodeca/icosa/tetra/regpoly, or triform/sqform output -
    // see board_config.h). Left unchecked, the convex-hull LP degenerates on 0-dimensional points and
    // silently returns no edges at all, rather than failing loudly.
    if (bc.emb_dim == 0)
        throw std::runtime_error(
            "rectify: requires a real (non-zero) embedding, got emb_dim=0 - this board has no "
            "coordinates to compute edge directions from");
    int N = bc.N;
    unsigned emb_dim = bc.emb_dim;

    // One new node per original edge (i<j), at the midpoint - embed[i]+embed[j] is already 2x the
    // true midpoint (exact integer, no rounding).
    std::map<std::pair<int,int>, int> edge_idx;
    std::vector<std::vector<unsigned>> pos;
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) {
            if (!bc.adj[i][j]) continue;
            edge_idx[{i, j}] = (int)pos.size();
            std::vector<unsigned> mid(emb_dim);
            for (unsigned k = 0; k < emb_dim; k++) mid[k] = bc.embed[i][k] + bc.embed[j][k];
            pos.push_back(std::move(mid));
        }

    // Edges incident to each original node, as [midpoint node index] lists.
    std::vector<std::vector<int>> incident(N);
    for (auto& [ij, idx] : edge_idx) {
        incident[ij.first].push_back(idx);
        incident[ij.second].push_back(idx);
    }

    auto adj = zero_adj((int)pos.size());
    for (int v = 0; v < N; v++) {
        auto& mids = incident[v];
        if (mids.size() < 2) continue;
        // pos[midIdx] is at the doubled scale (see above), so subtract 2*embed[v] (not embed[v]) to
        // get a consistently-scaled direction vector before normalizing - mixing scales here would
        // give the wrong direction entirely (this is the exact bug the TS port hit and fixed).
        std::vector<std::vector<double>> dirs;
        for (int mid_idx : mids) {
            std::vector<double> d(emb_dim);
            double len_sq = 0;
            for (unsigned k = 0; k < emb_dim; k++) {
                d[k] = (double)pos[mid_idx][k] - 2.0 * (double)bc.embed[v][k];
                len_sq += d[k] * d[k];
            }
            double len = std::sqrt(len_sq);
            for (unsigned k = 0; k < emb_dim; k++) d[k] /= len;
            dirs.push_back(std::move(d));
        }
        for (auto& [a, b] : convex_hull_edges(dirs)) {
            adj[mids[a]][mids[b]] = 1;
            adj[mids[b]][mids[a]] = 1;
        }
    }

    return make_bc(std::move(adj), emb_dim, std::move(pos));
}

BoardConfig merge_close(const BoardConfig& bc, double dist) {
    assert(dist > 0 && "dist must be positive");
    // A real (always-active) check, not assert() - merge_close needs real coordinates to compute a
    // meaningful distance. Left unchecked, an emb_dim=0 board (e.g. dodeca/icosa/tetra/regpoly, or
    // triform/sqform output - see board_config.h) makes every pairwise distance compute to 0 (the
    // loop over emb_dim coordinates never runs), silently collapsing the entire board into one node.
    if (bc.emb_dim == 0)
        throw std::runtime_error(
            "merge_close: requires a real (non-zero) embedding, got emb_dim=0 - this board has no "
            "coordinates to compute a distance from");
    double dist2 = dist * dist;
    int N = bc.N;
    std::vector<std::pair<int,int>> quot;
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) {
            double d2 = 0.0;
            for (unsigned k = 0; k < bc.emb_dim; k++) {
                double diff = (double)bc.embed[i][k] - (double)bc.embed[j][k];
                d2 += diff * diff;
            }
            if (d2 < dist2) quot.push_back({i, j});
        }
    return quotient_board(bc, quot);
}

BoardConfig triangle_form(const BoardConfig& bc, int w) {
    assert(w >= 1 && "w must be at least 1");
    int N = bc.N;
    auto triangles = find_triangles(bc.adj); // each {A, B, C} with A < B < C

    int n_face = w * (w + 1) / 2;
    auto local_idx = [](int i, int j) { return i * (i + 1) / 2 + j; };
    auto global_idx = [&](int t, int i, int j) { return N + t * n_face + local_idx(i, j); };

    std::set<std::pair<int,int>> is_triangle_side;
    for (auto& tri : triangles) {
        is_triangle_side.insert({tri[0], tri[1]});
        is_triangle_side.insert({tri[0], tri[2]});
        is_triangle_side.insert({tri[1], tri[2]});
    }

    int total_n = N + (int)triangles.size() * n_face;
    auto adj = zero_adj(total_n);
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) {
            if (!bc.adj[i][j] || is_triangle_side.count({i, j})) continue;
            adj[i][j] = 1;
            adj[j][i] = 1;
        }

    const int dirs[6][2] = {{1,0},{1,1},{0,1},{-1,0},{-1,-1},{0,-1}};
    for (int t = 0; t < (int)triangles.size(); t++)
        for (int i = 0; i < w; i++)
            for (int j = 0; j <= i; j++)
                for (auto& d : dirs) {
                    int ni = i + d[0], nj = j + d[1];
                    if (ni >= 0 && ni < w && nj >= 0 && nj <= ni)
                        adj[global_idx(t, i, j)][global_idx(t, ni, nj)] = 1;
                }

    // Boundary node sequence (as (i,j) pairs) for triangle t's edge between vertex-pair (p, q) -
    // same left/right/bottom convention as triangular_board's own row/col indexing.
    auto boundary_seq = [&](int t, int p, int q) {
        auto& tri = triangles[t];
        int A = tri[0], B = tri[1], C = tri[2];
        std::vector<std::pair<int,int>> seq(w);
        if (p == A && q == B) { for (int i = 0; i < w; i++) seq[i] = {i, 0}; }
        else if (p == A && q == C) { for (int i = 0; i < w; i++) seq[i] = {i, i}; }
        else if (p == B && q == C) { for (int j = 0; j < w; j++) seq[j] = {w - 1, j}; }
        else throw std::runtime_error("triangle_form: triangle has no such edge");
        return seq;
    };

    std::vector<std::pair<int,int>> quot;
    for (int t = 0; t < (int)triangles.size(); t++) {
        auto& tri = triangles[t];
        quot.push_back({tri[0], global_idx(t, 0, 0)});
        quot.push_back({tri[1], global_idx(t, w - 1, 0)});
        quot.push_back({tri[2], global_idx(t, w - 1, w - 1)});
    }
    std::map<std::pair<int,int>, std::vector<int>> edge_to_triangles;
    for (int t = 0; t < (int)triangles.size(); t++) {
        auto& tri = triangles[t];
        for (auto& pq : {std::pair{tri[0], tri[1]}, std::pair{tri[0], tri[2]}, std::pair{tri[1], tri[2]}})
            edge_to_triangles[pq].push_back(t);
    }
    for (auto& [pq, ts] : edge_to_triangles) {
        if (ts.size() < 2) continue;
        auto canonical = boundary_seq(ts[0], pq.first, pq.second);
        for (size_t k = 1; k < ts.size(); k++) {
            auto seq = boundary_seq(ts[k], pq.first, pq.second);
            for (int idx = 0; idx < w; idx++)
                quot.push_back({
                    global_idx(ts[0], canonical[idx].first, canonical[idx].second),
                    global_idx(ts[k], seq[idx].first, seq[idx].second),
                });
        }
    }

    std::vector<std::vector<unsigned>> embed(total_n); // emb_dim=0 - see board_config.h's doc comment
    BoardConfig combined = make_bc(std::move(adj), 0u, std::move(embed));
    return quotient_board(combined, quot);
}

BoardConfig square_form(const BoardConfig& bc, int w) {
    assert(w >= 1 && "w must be at least 1");
    int N = bc.N;
    auto squares = find_squares(bc.adj); // each {A, B, C, D} in cycle order

    int n_face = w * w;
    auto local_idx = [&](int i, int j) { return i * w + j; };
    auto global_idx = [&](int t, int i, int j) { return N + t * n_face + local_idx(i, j); };

    std::set<std::pair<int,int>> is_square_side;
    for (auto& sq : squares) {
        int A = sq[0], B = sq[1], C = sq[2], D = sq[3];
        for (auto& pq : {std::pair{A, B}, std::pair{B, C}, std::pair{C, D}, std::pair{D, A}})
            is_square_side.insert({std::min(pq.first, pq.second), std::max(pq.first, pq.second)});
    }

    int total_n = N + (int)squares.size() * n_face;
    auto adj = zero_adj(total_n);
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) {
            if (!bc.adj[i][j] || is_square_side.count({i, j})) continue;
            adj[i][j] = 1;
            adj[j][i] = 1;
        }

    const int dirs[4][2] = {{0,1},{1,0},{0,-1},{-1,0}};
    for (int t = 0; t < (int)squares.size(); t++)
        for (int i = 0; i < w; i++)
            for (int j = 0; j < w; j++)
                for (auto& d : dirs) {
                    int ni = i + d[0], nj = j + d[1];
                    if (ni >= 0 && ni < w && nj >= 0 && nj < w)
                        adj[global_idx(t, i, j)][global_idx(t, ni, nj)] = 1;
                }

    // The "natural" boundary sequence (local (i,j) pairs, k=0..w-1) for square t's side `side`
    // (0=A-B top row, 1=B-C right col, 2=C-D bottom row, 3=D-A left col), running from that side's
    // first-listed corner (k=0) to its second (k=w-1) - matches shared/boardConfig.ts's naturalSeq().
    auto natural_seq = [&](int side) {
        std::vector<std::pair<int,int>> seq(w);
        if (side == 0) { for (int j = 0; j < w; j++) seq[j] = {0, j}; }
        else if (side == 1) { for (int i = 0; i < w; i++) seq[i] = {i, w - 1}; }
        else if (side == 2) { for (int k = 0; k < w; k++) seq[k] = {w - 1, w - 1 - k}; }
        else { for (int k = 0; k < w; k++) seq[k] = {w - 1 - k, 0}; }
        return seq;
    };

    std::vector<std::pair<int,int>> quot;
    for (int t = 0; t < (int)squares.size(); t++) {
        auto& sq = squares[t];
        quot.push_back({sq[0], global_idx(t, 0, 0)});
        quot.push_back({sq[1], global_idx(t, 0, w - 1)});
        quot.push_back({sq[2], global_idx(t, w - 1, w - 1)});
        quot.push_back({sq[3], global_idx(t, w - 1, 0)});
    }
    // Unlike triangle_form's A < B < C corner convention, a square's cycle order isn't globally
    // monotonic in vertex index, so each side's natural sequence is explicitly re-oriented here to
    // always run from min(endpoint) to max(endpoint) - the shared canonical direction every square
    // touching that original edge agrees on, regardless of its own cycle orientation.
    std::map<std::pair<int,int>, std::vector<std::pair<int, std::vector<std::pair<int,int>>>>> edge_to_seqs;
    for (int t = 0; t < (int)squares.size(); t++) {
        auto& sq = squares[t];
        int A = sq[0], B = sq[1], C = sq[2], D = sq[3];
        int sides[4][3] = {{A, B, 0}, {B, C, 1}, {C, D, 2}, {D, A, 3}};
        for (auto& s : sides) {
            int ep1 = s[0], ep2 = s[1], side = s[2];
            auto key = std::pair{std::min(ep1, ep2), std::max(ep1, ep2)};
            auto seq = natural_seq(side);
            if (ep1 > ep2) std::reverse(seq.begin(), seq.end());
            edge_to_seqs[key].push_back({t, seq});
        }
    }
    for (auto& [key, entries] : edge_to_seqs) {
        if (entries.size() < 2) continue;
        auto& [t0, canonical] = entries[0];
        for (size_t k = 1; k < entries.size(); k++) {
            auto& [tk, seq] = entries[k];
            for (int idx = 0; idx < w; idx++)
                quot.push_back({
                    global_idx(t0, canonical[idx].first, canonical[idx].second),
                    global_idx(tk, seq[idx].first, seq[idx].second),
                });
        }
    }

    std::vector<std::vector<unsigned>> embed(total_n); // emb_dim=0 - see board_config.h's doc comment
    BoardConfig combined = make_bc(std::move(adj), 0u, std::move(embed));
    return quotient_board(combined, quot);
}

BoardConfig global_centralize(const BoardConfig& bc) {
    int N = bc.N;
    int hub = N;
    auto adj = zero_adj(N + 1);
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) {
            if (!bc.adj[i][j]) continue;
            adj[i][j] = 1;
            adj[j][i] = 1;
        }
    for (int i = 0; i < N; i++) {
        adj[i][hub] = 1;
        adj[hub][i] = 1;
    }

    std::vector<std::vector<unsigned>> embed(N + 1); // emb_dim=0 - see board_config.h's doc comment
    return make_bc(std::move(adj), 0u, std::move(embed));
}

BoardConfig sq_octarize(const BoardConfig& bc) {
    int N = bc.N;
    auto squares = find_squares(bc.adj); // each {A, B, C, D} in cycle order

    int total_n = N + (int)squares.size() * 2;
    auto adj = zero_adj(total_n);
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) {
            if (!bc.adj[i][j]) continue;
            adj[i][j] = 1;
            adj[j][i] = 1;
        }

    for (int s = 0; s < (int)squares.size(); s++) {
        int top = N + s * 2, bottom = top + 1;
        for (int c : squares[s]) {
            adj[c][top] = 1;
            adj[top][c] = 1;
            adj[c][bottom] = 1;
            adj[bottom][c] = 1;
        }
    }

    std::vector<std::vector<unsigned>> embed(total_n); // emb_dim=0 - see board_config.h's doc comment
    return make_bc(std::move(adj), 0u, std::move(embed));
}

BoardConfig scale_board(const BoardConfig& bc, double /*factor*/) {
    return bc;
}

BoardConfig product(const BoardConfig& bc1, const BoardConfig& bc2) {
    int N1 = bc1.N, N2 = bc2.N;
    unsigned emb_dim = bc1.emb_dim + bc2.emb_dim;
    auto idx = [&](int i, int j) { return i * N2 + j; };

    std::vector<std::vector<unsigned>> embed(N1 * N2);
    for (int i = 0; i < N1; i++)
        for (int j = 0; j < N2; j++) {
            auto& e = embed[idx(i, j)];
            e.reserve(emb_dim);
            e.insert(e.end(), bc1.embed[i].begin(), bc1.embed[i].end());
            e.insert(e.end(), bc2.embed[j].begin(), bc2.embed[j].end());
        }

    auto adj = zero_adj(N1 * N2);
    for (int i = 0; i < N1; i++)
        for (int j = 0; j < N2; j++) {
            for (int i2 = 0; i2 < N1; i2++)
                if (bc1.adj[i][i2]) adj[idx(i, j)][idx(i2, j)] = 1;
            for (int j2 = 0; j2 < N2; j2++)
                if (bc2.adj[j][j2]) adj[idx(i, j)][idx(i, j2)] = 1;
        }

    return make_bc(std::move(adj), emb_dim, std::move(embed));
}

BoardConfig apply_modifier(const BoardConfig& bc, const BoardModifier& modifier) {
    switch (modifier.kind) {
        case ModifierKind::Rectify:    return rectify(bc);
        case ModifierKind::EdgeSplit:  return edge_split(bc, modifier.split_n);
        case ModifierKind::MergeClose: return merge_close(bc, modifier.dist);
        case ModifierKind::TriangleForm: return triangle_form(bc, modifier.split_n);
        case ModifierKind::SquareForm: return square_form(bc, modifier.split_n);
        case ModifierKind::Prod:
            return product(bc, build_board_config(modifier.board_type, modifier.board_args));
        case ModifierKind::GlobalCentralize: return global_centralize(bc);
        case ModifierKind::SqOctarize: return sq_octarize(bc);
        case ModifierKind::Scale: return scale_board(bc, modifier.dist);
        case ModifierKind::BeginProd:
        case ModifierKind::EndProd:
            throw std::runtime_error(
                "apply_modifier: BeginProd/EndProd must be applied via apply_modifiers, not directly");
    }
    throw std::runtime_error("apply_modifier: unknown ModifierKind");
}

BoardConfig apply_modifiers(const BoardConfig& bc, const std::vector<BoardModifier>& modifiers) {
    BoardConfig current = bc;
    std::vector<BoardConfig> stack;
    for (auto& m : modifiers) {
        if (m.kind == ModifierKind::BeginProd) {
            stack.push_back(std::move(current));
            current = build_board_config(m.board_type, m.board_args);
        } else if (m.kind == ModifierKind::EndProd) {
            if (stack.empty())
                throw std::runtime_error("apply_modifiers: endprod with no matching beginprod");
            BoardConfig outer = std::move(stack.back());
            stack.pop_back();
            current = product(outer, current);
        } else {
            current = apply_modifier(current, m);
        }
    }
    if (!stack.empty())
        throw std::runtime_error("apply_modifiers: unmatched beginprod(s)");
    return current;
}

BoardConfig linear_board(int w) {
    assert(w > 0 && "w must be positive");
    std::vector<std::vector<unsigned>> pos;
    for (unsigned i = 0; i < (unsigned)w; i++) pos.push_back({i});
    auto adj = zero_adj(w);
    for (int i = 0; i < w - 1; i++) {
        adj[i][i + 1] = 1;
        adj[i + 1][i] = 1;
    }
    return make_bc(std::move(adj), 1u, std::move(pos));
}

BoardConfig rectangular_board(int w, int h) {
    return hypercuboid_board(2, {w, h});
}

BoardConfig rectangular_diagonal_board(int w, int h, int m) {
    assert(w > 0 && h > 0 && m > 0 && "w, h, m must be positive");
    auto adj = zero_adj(w * h);
    std::vector<std::vector<unsigned>> pos;
    for (unsigned r = 0; r < h; r++)
        for (unsigned c = 0; c < w; c++)
            pos.push_back({c, r});
    const int dirs[6][2] = {{0,1},{1,0},{0,-1},{-1,0},{1,1},{-1,1}};
    for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++)
            for (int di = 0; di < 6; di++) {
                int dr = dirs[di][0], dc = dirs[di][1];
                int nr = r + dr, nc = c + dc;
                if (nr < 0 || nr >= h || nc < 0 || nc >= w) continue;
                if (std::abs(dr) == 1 && std::abs(dc) == 1 && m > 1 &&
                    ((r + (dr - 1) / 2) % m != 0 || c % m != 0)) continue;
                adj[r*w+c][nr*w+nc] = 1;
                adj[nr*w+nc][r*w+c] = 1;
            }
    return make_bc(std::move(adj), 2u, std::move(pos));
}

BoardConfig cube_lattice_board(int w, int h, int d) {
    return hypercuboid_board(3, {w, h, d});
}

// Mirrors shared/boardConfig.ts's hypercuboidBoard() exactly: a node survives (occurs on the
// board) iff at most `meshdim` of its coordinates are strictly interior to their own axis (not 0,
// not dims[i]-1); surviving nodes keep the plain grid adjacency (connected iff they differ by
// exactly 1 in exactly one coordinate). Unlike the TS side (which centers positions at the origin
// via `- (dims[i]-1)/2`), this keeps the plain uncentered [0, dims[i]) integer coordinates - same
// convention rectangular_board/cube_lattice_board already used before becoming thin wrappers
// around this function (centering would need exact-half-integer coordinates for an even-sized
// dimension, which this file's exact-integer embed[] can't represent - see merge_close's own doc
// comment). See the TS side's own doc comment for what `meshdim` means geometrically (the
// m-skeleton of the hypercuboid) and its worked examples.
BoardConfig hypercuboid_board(int meshdim, const std::vector<int>& dims) {
    assert(!dims.empty() && "dims must have at least 1 entry");
    for (int d : dims) assert(d > 0 && "every dimension must be positive");
    assert(meshdim >= 0 && "meshdim must be non-negative");
    int k = static_cast<int>(dims.size());
    int full_n = 1;
    for (int d : dims) full_n *= d;

    std::vector<int> strides(k);
    strides[0] = 1;
    for (int i = 1; i < k; i++) strides[i] = strides[i - 1] * dims[i - 1];
    auto coords_of = [&](int n) {
        std::vector<int> coords(k);
        for (int i = 0; i < k; i++) { coords[i] = n % dims[i]; n /= dims[i]; }
        return coords;
    };
    auto full_idx = [&](const std::vector<int>& coords) {
        int s = 0;
        for (int i = 0; i < k; i++) s += coords[i] * strides[i];
        return s;
    };
    auto is_interior = [](int c, int dim) { return c > 0 && c < dim - 1; };

    // Only surviving nodes get a board index (compacted, in ascending full-lattice-index order) -
    // board_idx_of maps a full-lattice index to that compacted index, absent for a culled node.
    std::map<int, int> board_idx_of;
    std::vector<std::vector<int>> surviving_coords;
    std::vector<std::vector<unsigned>> pos;
    for (int n = 0; n < full_n; n++) {
        auto coords = coords_of(n);
        int interior_count = 0;
        for (int i = 0; i < k; i++)
            if (is_interior(coords[i], dims[i])) interior_count++;
        if (interior_count > meshdim) continue;
        board_idx_of[n] = static_cast<int>(surviving_coords.size());
        surviving_coords.push_back(coords);
        std::vector<unsigned> p(k);
        for (int i = 0; i < k; i++) p[i] = static_cast<unsigned>(coords[i]);
        pos.push_back(std::move(p));
    }
    int N = static_cast<int>(surviving_coords.size());

    auto adj = zero_adj(N);
    for (int bi = 0; bi < N; bi++) {
        const auto& coords = surviving_coords[bi];
        for (int i = 0; i < k; i++)
            for (int delta : {1, -1}) {
                int nc = coords[i] + delta;
                if (nc < 0 || nc >= dims[i]) continue;
                auto ncoords = coords;
                ncoords[i] = nc;
                auto it = board_idx_of.find(full_idx(ncoords));
                if (it == board_idx_of.end()) continue; // that neighbor didn't survive the meshdim filter
                adj[bi][it->second] = 1;
            }
    }
    return make_bc(std::move(adj), static_cast<unsigned>(k), std::move(pos));
}

BoardConfig triangular_board(int w) {
    assert(w > 0 && "w must be positive");
    std::vector<std::vector<unsigned>> pos;
    for (unsigned i = 0; i < w; i++)
        for (unsigned j = 0; j <= i; j++)
            pos.push_back({j, i});
    int N = w * (w + 1) / 2;
    auto adj = zero_adj(N);
    auto idx = [&](int i, int j) { return i*(i+1)/2 + j; };
    const int dirs[6][2] = {{1,0},{1,1},{0,1},{-1,0},{-1,-1},{0,-1}};
    for (int i = 0; i < w; i++)
        for (int j = 0; j <= i; j++)
            for (auto& d : dirs) {
                int ni = i+d[0], nj = j+d[1];
                if (ni>=0 && ni<w && nj>=0 && nj<=ni)
                    adj[idx(i,j)][idx(ni,nj)] = 1;
            }
    return make_bc(std::move(adj), 2u, std::move(pos));
}

// A plain (pos, adj) pair - the recursive intermediate form merge_boards()/sierpinski_rec()/
// node_edge_merge_flake_rec() all build on before an outermost caller wraps the result into a full
// BoardConfig via make_bc().
struct RawBoard {
    std::vector<std::vector<unsigned>> pos;
    std::vector<std::vector<int>> adj;
};

// Mirrors shared/boardConfig.ts's unionFindClasses(): given N nodes and a list of pairs to merge,
// returns each node's equivalence-class index, compressed to a dense 0..M-1 range (M = number of
// distinct classes) in ascending order of each class's lowest original member. Internal to
// merge_boards() (below) - not shared with quotient_board()'s own separate, pre-existing union-find
// (a different enough job - single already-combined board, position-averaging - to leave alone).
static std::vector<int> union_find_classes(int n, const std::vector<std::pair<int,int>>& pairs) {
    std::vector<int> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    std::function<int(int)> find = [&](int x) -> int {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    for (auto [a, b] : pairs) {
        int pa = find(a), pb = find(b);
        if (pa != pb) parent[pa] = pb;
    }
    std::vector<int> roots(n);
    for (int i = 0; i < n; i++) roots[i] = find(i);
    std::vector<int> unique_roots = roots;
    std::sort(unique_roots.begin(), unique_roots.end());
    unique_roots.erase(std::unique(unique_roots.begin(), unique_roots.end()), unique_roots.end());
    std::vector<int> root_to_new(n, -1);
    for (size_t i = 0; i < unique_roots.size(); i++) root_to_new[unique_roots[i]] = static_cast<int>(i);
    std::vector<int> node_to_new(n);
    for (int i = 0; i < n; i++) node_to_new[i] = root_to_new[roots[i]];
    return node_to_new;
}

// Mirrors shared/boardConfig.ts's mergeBoards(), generalized to a list of boards - see that
// function's own doc comment: every `merges` instruction ((board index, that board's own local
// node index), (board index, local node index)) is resolved in one batch via
// union_find_classes(), not board-by-board, so a merge between two boards that haven't been
// introduced to each other by any other merge is handled exactly like any other - no caller needs
// to fold boards in one at a time just to keep every merge target "already placed". The merged
// node keeps whichever input position is encountered first. Returns the combined board plus, for
// each input board (same order as `boards`), a map from that board's own local indices to its
// final index in the combined board.
static std::pair<RawBoard, std::vector<std::vector<int>>> merge_boards(
    const std::vector<RawBoard>& boards,
    const std::vector<std::pair<std::pair<int,int>, std::pair<int,int>>>& merges) {
    std::vector<int> offset(boards.size(), 0);
    for (size_t i = 1; i < boards.size(); i++)
        offset[i] = offset[i - 1] + static_cast<int>(boards[i - 1].adj.size());
    int total = 0;
    for (const auto& b : boards) total += static_cast<int>(b.adj.size());
    auto g = [&](int b, int local) { return offset[b] + local; };

    std::vector<std::pair<int,int>> pairs;
    for (const auto& m : merges)
        pairs.push_back({ g(m.first.first, m.first.second), g(m.second.first, m.second.second) });
    std::vector<int> node_to_new = union_find_classes(total, pairs);
    int new_n = total == 0 ? 0 : *std::max_element(node_to_new.begin(), node_to_new.end()) + 1;

    std::vector<std::vector<unsigned>> pos(new_n);
    for (size_t b = 0; b < boards.size(); b++)
        for (size_t local = 0; local < boards[b].pos.size(); local++)
            pos[node_to_new[g(static_cast<int>(b), static_cast<int>(local))]] = boards[b].pos[local];

    auto adj = zero_adj(new_n);
    for (size_t b = 0; b < boards.size(); b++) {
        const auto& board_adj = boards[b].adj;
        for (size_t i = 0; i < board_adj.size(); i++)
            for (size_t j = 0; j < board_adj.size(); j++)
                if (board_adj[i][j])
                    adj[node_to_new[g(static_cast<int>(b), static_cast<int>(i))]]
                       [node_to_new[g(static_cast<int>(b), static_cast<int>(j))]] = 1;
    }

    std::vector<std::vector<int>> maps(boards.size());
    for (size_t b = 0; b < boards.size(); b++) {
        maps[b].resize(boards[b].adj.size());
        for (size_t local = 0; local < boards[b].adj.size(); local++)
            maps[b][local] = node_to_new[g(static_cast<int>(b), static_cast<int>(local))];
    }

    return { RawBoard{ std::move(pos), std::move(adj) }, std::move(maps) };
}

// Mirrors shared/boardConfig.ts's sierpinskiRec(), generalized from 3 fixed corners to dim+1 (same
// as the TS side's own generalization). Returns (pos, adj, corners), corners being the
// (possibly-merged) node index of each entry of `corners` (length dim+1), in the same order.
static std::tuple<std::vector<std::vector<unsigned>>, std::vector<std::vector<int>>, std::vector<int>>
sierpinski_rec(int n, const std::vector<std::vector<unsigned>>& corners) {
    int k = static_cast<int>(corners.size()); // dim + 1
    if (n == 1) {
        auto adj = zero_adj(k);
        for (int i = 0; i < k; i++)
            for (int j = i + 1; j < k; j++) { adj[i][j] = 1; adj[j][i] = 1; }
        std::vector<int> ids(k);
        std::iota(ids.begin(), ids.end(), 0);
        return { corners, adj, ids };
    }
    auto mid = [](const std::vector<unsigned>& a, const std::vector<unsigned>& b) {
        std::vector<unsigned> m(a.size());
        for (size_t d = 0; d < a.size(); d++) m[d] = (a[d] + b[d]) / 2;
        return m;
    };

    std::vector<RawBoard> subs;
    std::vector<std::vector<int>> sub_corners_list;
    for (int k_ = 0; k_ < k; k_++) {
        std::vector<std::vector<unsigned>> sub_corners_in(k);
        for (int p = 0; p < k; p++) sub_corners_in[p] = (p == k_) ? corners[k_] : mid(corners[k_], corners[p]);
        auto [pos, adj, sub_corners] = sierpinski_rec(n - 1, sub_corners_in);
        subs.push_back(RawBoard{ std::move(pos), std::move(adj) });
        sub_corners_list.push_back(std::move(sub_corners));
    }

    std::vector<std::pair<std::pair<int,int>, std::pair<int,int>>> merges;
    for (int a = 0; a < k; a++)
        for (int b = a + 1; b < k; b++)
            merges.push_back({ { a, sub_corners_list[a][b] }, { b, sub_corners_list[b][a] } });

    auto [combined, maps] = merge_boards(subs, merges);
    std::vector<int> outer_corners(k);
    for (int k_ = 0; k_ < k; k_++) outer_corners[k_] = maps[k_][sub_corners_list[k_][k_]];

    return { std::move(combined.pos), std::move(combined.adj), std::move(outer_corners) };
}

// Mirrors shared/boardConfig.ts's sierpinskiSimplex(), with one simplification the TS side doesn't
// have: BoardConfig::embed is exact-integer only, so rather than port sierpinskiSimplex's real-
// valued, centroid-at-origin regularSimplexCoords() (irrational for dim >= 2), this instead places
// corner 0 of the outer side-`side` simplex at the origin and corner k (1 <= k <= dim) at `side`
// times the k-th standard basis vector - dim+1 affinely-independent corners with minimal edge
// length `side` (corner 0 to any other corner). Recursive midpoints stay exact integers throughout
// since `side` only ever appears as a power of 2 above the n=1 base case.
BoardConfig sierpinski_simplex_board(int dim, int n) {
    assert(dim >= 1 && "dim must be at least 1");
    assert(n >= 1 && "n must be at least 1");

    unsigned side = 1u << (n - 1);
    std::vector<std::vector<unsigned>> corners(dim + 1, std::vector<unsigned>(dim, 0u));
    for (int k = 1; k <= dim; k++) corners[k][k - 1] = side;

    auto [pos, adj, outer_corners] = sierpinski_rec(n, corners);
    (void)outer_corners; // only needed by recursive callers, same as the TS top-level driver
    return make_bc(std::move(adj), static_cast<unsigned>(dim), std::move(pos));
}

BoardConfig regular_polygon_board(int n) {
    assert(n >= 3 && "n must be at least 3");
    auto adj = zero_adj(n);
    for (int k = 0; k < n; k++) {
        int next = (k + 1) % n;
        adj[k][next] = 1;
        adj[next][k] = 1;
    }
    std::vector<std::vector<unsigned>> embed(n); // emb_dim=0: every node's embed[] is empty
    return make_bc(std::move(adj), 0u, std::move(embed));
}

// Mirrors shared/boardConfig.ts's starBoard() connectivity (adjacency only - no
// position/embedding, see board_config.h's own doc comment on this function). Node 0 is the
// center, nodes 1..n are the outer nodes.
BoardConfig star_board(int n) {
    assert(n >= 1 && "n must be at least 1");
    auto adj = zero_adj(n + 1);
    for (int k = 1; k <= n; k++) {
        adj[0][k] = 1;
        adj[k][0] = 1;
    }
    std::vector<std::vector<unsigned>> embed(n + 1); // emb_dim=0
    return make_bc(std::move(adj), 0u, std::move(embed));
}

// Mirrors shared/boardConfig.ts's tetrahedronBoard() connectivity (adjacency only - no
// position/embedding, see board_config.h's own doc comment on this function). 4 faces, each
// triangular_board(w)-shaped (n_face = w*(w+1)/2 nodes, same left/right/bottom boundary convention
// as triangular_board's own row/col indexing), glued along shared tetrahedron edges via
// quotient_board - every pair of distinct faces shares exactly one edge (the 2 vertex indices not
// excluded by either face), and since each face lists its own 3 corners in ascending vertex-index
// order, both faces' boundary node sequences for a shared edge are always already aligned
// position-for-position (see the TS side's doc comment for the full argument), so no
// direction-flipping is ever needed here.
BoardConfig tetrahedron_board() {
    auto adj = zero_adj(4);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            if (i != j) adj[i][j] = 1;
    std::vector<std::vector<unsigned>> embed(4); // emb_dim=0
    return make_bc(std::move(adj), 0u, std::move(embed));
}

BoardConfig octahedron_board() {
    return orthoplex_board(3);
}

// Mirrors shared/boardConfig.ts's orthoplexBoard(), with one simplification: rather than the TS
// side's real-valued +-1/sqrt(2) coordinates, this uses only the integer values {0, 1, 2} - vertex
// 2k (the "+" pole on axis k) has coordinate k = 2, vertex 2k+1 (the "-" pole) has coordinate k =
// 1, every other coordinate 0; connectivity (every vertex adjacent to every other except its own
// antipode) is unaffected.
BoardConfig orthoplex_board(int n) {
    assert(n >= 1 && "n must be at least 1");
    int N = 2 * n;
    std::vector<std::vector<unsigned>> pos(N, std::vector<unsigned>(n, 0u));
    for (int k = 0; k < n; k++) {
        pos[2 * k][k] = 2u;
        pos[2 * k + 1][k] = 1u;
    }
    auto adj = zero_adj(N);
    auto antipode = [](int i) { return i % 2 == 0 ? i + 1 : i - 1; };
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (i != j && j != antipode(i)) adj[i][j] = 1;
    return make_bc(std::move(adj), static_cast<unsigned>(n), std::move(pos));
}

// Mirrors shared/boardConfig.ts's dodecahedronBoard() connectivity (adjacency only - no
// position/embedding, see board_config.h's own doc comment on this function).
BoardConfig dodecahedron_board() {
    auto x_idx = [](int sa, int sb, int sc) { return sa * 4 + sb * 2 + sc; };
    auto y_idx = [](int sb, int sc) { return 8 + sb * 2 + sc; };
    auto z_idx = [](int sa, int sb) { return 12 + sa * 2 + sb; };
    auto w_idx = [](int sa, int sc) { return 16 + sa * 2 + sc; };

    auto adj = zero_adj(20);
    auto connect = [&](int i, int j) { adj[i][j] = 1; adj[j][i] = 1; };
    for (int sa = 0; sa < 2; sa++)
        for (int sb = 0; sb < 2; sb++)
            for (int sc = 0; sc < 2; sc++) {
                int x = x_idx(sa, sb, sc);
                connect(x, y_idx(sb, sc));
                connect(x, z_idx(sa, sb));
                connect(x, w_idx(sa, sc));
            }
    for (int sc = 0; sc < 2; sc++) connect(y_idx(0, sc), y_idx(1, sc));
    for (int sb = 0; sb < 2; sb++) connect(z_idx(0, sb), z_idx(1, sb));
    for (int sa = 0; sa < 2; sa++) connect(w_idx(sa, 0), w_idx(sa, 1));

    std::vector<std::vector<unsigned>> embed(20); // emb_dim=0
    return make_bc(std::move(adj), 0u, std::move(embed));
}

// Mirrors shared/boardConfig.ts's icosahedronBoard() connectivity (adjacency only - no
// position/embedding, see board_config.h's own doc comment on this function).
BoardConfig icosahedron_board() {
    auto a_idx = [](int sp, int sq) { return sp * 2 + sq; };
    auto b_idx = [](int sp, int sq) { return 4 + sp * 2 + sq; };
    auto c_idx = [](int sp, int sq) { return 8 + sp * 2 + sq; };

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

    std::vector<std::vector<unsigned>> embed(12); // emb_dim=0
    return make_bc(std::move(adj), 0u, std::move(embed));
}

// --- dodecahedron_flake_board()/icosahedron_flake_board()/octahedron_flake_board()/
// regular_polygon_flake_board(): mirrors shared/boardConfig.ts's dodecahedronFlake()/
// icosahedronFlake()/octahedronFlake()/regularPolygonFlake() - see board_config.h's own doc comment
// on dodecahedron_flake_board() for why these never compute or store node positions at all. ---

// Mirrors shared/boardConfig.ts's computeFlakeGlue(): for each base edge (i, j), exhaustively finds
// the unique pair of other base edges - (m1, m2) within the copy at i, (n1, n2) within the copy at
// j - whose S_i/S_j-transformed endpoints coincide two-for-two (S_i(x) = r*x + c*verts[i]). `verts`
// here are plain doubles used ONLY as scratch data for this one-time search - unlike
// BoardConfig::embed elsewhere in this file, they are never required to be exact integers, and are
// never exposed as a board's own position data (see this section's own top comment).
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

// Mirrors shared/boardConfig.ts's computeNodeGlue(): for each base edge (i, j), exhaustively finds
// the unique pair of vertices (m, p) - m within the copy at i, p within the copy at j - whose
// S_i/S_j-transformed positions coincide. Unlike compute_flake_glue()'s own edge-to-edge (2-point)
// search, this is a single-point search - regular_polygon_flake_board()'s own non-multiple-of-4
// case (see that function's own doc comment): node-merge copies share exactly one point per base
// edge, not a whole growing edge, so there is no second point to match and no chain to track.
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

// Mirrors dodecaFlakeRec()/icosaFlakeRec()'s own object-shaped return value ({ pos, adj, corners,
// edgeChains }) - minus `pos`, since these two boards track no position at all (see this section's
// own top comment).
struct FlakeRecResult {
    std::vector<std::vector<int>> adj;
    std::vector<int> corners;
    std::map<std::pair<int, int>, std::vector<int>> edge_chains;
};

// Mirrors shared/boardConfig.ts's growingEdgeLevelUpMap(): derives the standard `edge_level_up_map`
// for a shape whose growing shared edges are exactly its `edge_glue_map` entries between ADJACENT
// sub-copies (dodeca/icosa/octahedron/regular_polygon_flake_board()'s own 4n-gon case) - see
// node_edge_merge_flake_rec()'s own doc comment for the `{{P,P,Q},{Q,P,Q}}` derivation.
static std::map<std::pair<int, int>, std::vector<std::array<int, 3>>> growing_edge_level_up_map(
    const std::map<std::pair<int, int>, std::array<int, 4>>& edge_glue_map) {
    std::map<std::pair<int, int>, std::vector<std::array<int, 3>>> m;
    for (const auto& [key, unused] : edge_glue_map) {
        (void)unused;
        int p = key.first, q = key.second;
        m[key] = { { p, p, q }, { q, p, q } };
    }
    return m;
}

// Mirrors shared/boardConfig.ts's nodeEdgeMergeFlakeRec() - see its own doc comment for the full
// recursive construction. `num_subs` sub-copies are built each recursion (mirroring
// shared/boardConfig.ts's `subDescr` list): the first `num_leaf` correspond to leaf vertices (0 ..
// num_leaf-1, `edges`' own endpoint range) and expose `corners`/`corners_out`; any further entries
// (num_leaf <= idx < num_subs) are purely auxiliary, with no leaf vertex of their own (e.g.
// central_regular_polygon_flake_board()'s/central_pentagon_flake_board()'s own central copy) - same
// split as shared/boardConfig.ts's FractalDescr's own doc comment. `edge_glue_map`/`node_glue_map`
// key every glued pair `(P, Q)` (sub-copy indices, not necessarily an `edges` member) to EITHER a
// whole, growing shared edge OR a single shared point, never both: an `edge_glue_map` entry pairs
// up `subs[P]`'s own chain for `(C, D)` with `subs[Q]`'s own chain for `(E, F)` position-by-
// position, while a `node_glue_map` entry contributes exactly one merge pair (`subs[P]`'s own
// corner `m`, `subs[Q]`'s own corner `p`) and no chain at all. The structural merge is built as one
// `merges` list across every entry of both maps and resolved by a single merge_boards() call (see
// that function's own doc comment for why this - rather than folding subs in one at a time - is
// what makes octahedron_flake_board()'s own transitive antipodal-corner coincidence come out
// correct).
//
// `edge_level_up_map` keys a leaf-vertex pair `(A, B)` (always two valid indices below `num_leaf` -
// NOT a sub-copy-index pair like `edge_glue_map`/`node_glue_map`) to an ordered list of `(sub_idx,
// a, b)` triples: this shape's own chain for edge `(A, B)`, at ANY recursion order, is the
// concatenation - in list order - of `subs[sub_idx]`'s own chain for its OWN edge `(a, b)`
// (oriented to start at `a`). Every shape with a growing shared edge between ADJACENT sub-copies
// (dodeca/icosa/octahedron/regular_polygon_flake_board()'s own 4n-gon case) sets this via
// growing_edge_level_up_map() - for exactly those shapes, an `edge_glue_map` key `(P, Q)` doubles as
// a valid leaf-vertex pair (sub-copy index == leaf-vertex index one-to-one), giving
// `{{P,P,Q},{Q,P,Q}}` (`subs[P]`'s own `(P,Q)`-chain then `subs[Q]`'s own `(P,Q)`-chain),
// reproducing this map's own pre-generalization behavior exactly (concatenate `subs[P]`'s and
// `subs[Q]`'s own same-keyed chain - see git history). central_pentagon_flake_board() is the one
// shape needing a genuinely different map: its own "corner i to corner i+1" chain exists (same
// two-segment shape, using the leaf-adjacent indices `i`, `i+1` in place of `P`, `Q`) even though it
// is never itself an `edge_glue_map` key - it goes entirely unused by the plain (non-central)
// pentagon flake, which merges its own adjacent copies by a single node instead - only becoming
// load-bearing once a central copy is added, which glues to it via `edge_glue_map`.
//
// One deliberate deviation from mirroring TS's own per-shape functions (no deviation needed - this
// is already shared TS-side by every caller): with positions dropped entirely (see this section's
// own top comment) the recursion is purely combinatorial, so this single function serves every flake
// board below, parameterized only by `num_leaf`, `num_subs`, `edges`, `edge_glue_map`,
// `node_glue_map`, and `edge_level_up_map`.
static FlakeRecResult node_edge_merge_flake_rec(
    int n, int num_leaf, int num_subs, const std::vector<std::pair<int, int>>& edges,
    const std::map<std::pair<int, int>, std::array<int, 4>>& edge_glue_map,
    const std::map<std::pair<int, int>, std::pair<int, int>>& node_glue_map,
    const std::map<std::pair<int, int>, std::vector<std::array<int, 3>>>& edge_level_up_map) {
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

    std::vector<FlakeRecResult> subs;
    for (int s = 0; s < num_subs; s++)
        subs.push_back(node_edge_merge_flake_rec(
            n - 1, num_leaf, num_subs, edges, edge_glue_map, node_glue_map, edge_level_up_map));

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
    // this function's own doc comment) - any further entries (e.g. an auxiliary central copy) are
    // purely internal structure, not exposed as one of this call's own corners.
    std::vector<int> corners_out(num_leaf);
    for (int vtx = 0; vtx < num_leaf; vtx++) corners_out[vtx] = maps[vtx][subs[vtx].corners[vtx]];

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

// Vertex/edge/glue data dodecahedron_flake_board() needs - mirrors shared/boardConfig.ts's
// dodecahedronFlakeData(), minus `verts` in the return (see this section's own top comment for
// why). Cached in function-local statics since `edges`/`glue` only ever depend on the
// dodecahedron's own fixed structure and never change.
struct FlakeEdgeGlueData {
    std::vector<std::pair<int, int>> edges;
    std::vector<std::array<int, 6>> glue;
};

static const FlakeEdgeGlueData& dodecahedron_flake_data() {
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

// Vertex/edge/glue data icosahedron_flake_board() needs - mirrors shared/boardConfig.ts's
// icosahedronFlakeData(), same caching/no-`verts`-in-the-return reasoning as
// dodecahedron_flake_data() above.
static const FlakeEdgeGlueData& icosahedron_flake_data() {
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

// Vertex/edge/glue data octahedron_flake_board() needs - mirrors shared/boardConfig.ts's
// octahedronFlakeData(), same caching/no-`verts`-in-the-return reasoning as
// dodecahedron_flake_data() above. Vertex `2k`/`2k+1` are the `+-1` points on axis `k` (matching
// octahedron_board()'s own orthoplex_board(3) indexing) and are each other's antipode; every
// non-antipodal pair is an edge. Unlike dodeca/icosahedron, octahedron's own non-edges (the 3
// antipodal pairs) are deliberately left out of compute_flake_glue()'s search entirely - not
// because they don't coincide (they do: every copy's own antipodal-attachment corner lands on the
// same shared center point, since `S_i(v_{antipode(i)}) = r*(-v_i) + c*v_i = (c-r)*v_i = 0` once
// `c = r`, for every `i`), but because that coincidence needs no glue entry of its own: it already
// follows transitively from the 12 real-edge glue relations, resolved automatically by
// node_edge_merge_flake_rec()'s own merge_boards() call (see that function's own doc comment).
static const FlakeEdgeGlueData& octahedron_flake_data() {
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

// Mirrors shared/boardConfig.ts's dodecahedronFlake() - see board_config.h's own doc comment for
// the high-level construction (why there's no embedding at all here).
BoardConfig dodecahedron_flake_board(int n) {
    assert(n >= 1 && "n must be at least 1");
    const auto& data = dodecahedron_flake_data();
    std::map<std::pair<int, int>, std::array<int, 4>> edge_glue_map;
    for (const auto& g : data.glue) edge_glue_map[{ g[0], g[1] }] = { g[2], g[3], g[4], g[5] };

    auto edge_level_up_map = growing_edge_level_up_map(edge_glue_map);
    auto result = node_edge_merge_flake_rec(n, 20, 20, data.edges, edge_glue_map, {}, edge_level_up_map);
    std::vector<std::vector<unsigned>> embed(result.adj.size()); // emb_dim=0
    return make_bc(std::move(result.adj), 0u, std::move(embed));
}

// Mirrors shared/boardConfig.ts's icosahedronFlake() - see board_config.h's own doc comment on
// dodecahedron_flake_board() for the high-level construction, which this shares in full.
BoardConfig icosahedron_flake_board(int n) {
    assert(n >= 1 && "n must be at least 1");
    const auto& data = icosahedron_flake_data();
    std::map<std::pair<int, int>, std::array<int, 4>> edge_glue_map;
    for (const auto& g : data.glue) edge_glue_map[{ g[0], g[1] }] = { g[2], g[3], g[4], g[5] };

    auto edge_level_up_map = growing_edge_level_up_map(edge_glue_map);
    auto result = node_edge_merge_flake_rec(n, 12, 12, data.edges, edge_glue_map, {}, edge_level_up_map);
    std::vector<std::vector<unsigned>> embed(result.adj.size()); // emb_dim=0
    return make_bc(std::move(result.adj), 0u, std::move(embed));
}

// Mirrors shared/boardConfig.ts's octahedronFlake() - see board_config.h's own doc comment on
// dodecahedron_flake_board() for the high-level construction, which this shares in full, just with
// 6 vertices/12 edges instead of 20 vertices/30 edges or 12 vertices/30 edges.
BoardConfig octahedron_flake_board(int n) {
    assert(n >= 1 && "n must be at least 1");
    const auto& data = octahedron_flake_data();
    std::map<std::pair<int, int>, std::array<int, 4>> edge_glue_map;
    for (const auto& g : data.glue) edge_glue_map[{ g[0], g[1] }] = { g[2], g[3], g[4], g[5] };

    auto edge_level_up_map = growing_edge_level_up_map(edge_glue_map);
    auto result = node_edge_merge_flake_rec(n, 6, 6, data.edges, edge_glue_map, {}, edge_level_up_map);
    std::vector<std::vector<unsigned>> embed(result.adj.size()); // emb_dim=0
    return make_bc(std::move(result.adj), 0u, std::move(embed));
}

// Mirrors shared/boardConfig.ts's regularPolygonFlakeRC() - see its own doc comment for the full
// derivation: r+c=1 (a copy's own self vertex stays exactly where the outer polygon's vertex was)
// plus a closed-form coincidence ratio `c/r = 1 + 2*sum_{j=1}^{k} cos(2*pi*j/n_sides)`, for
// `k = n_sides/4 - 1` (merge by edge, `n_sides` a multiple of 4) or `k = n_sides/4` truncated
// (merge by node, otherwise - C++ integer division already floors for positive `n_sides`).
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

// Vertex/edge/glue data regular_polygon_flake_board() needs - mirrors shared/boardConfig.ts's
// regularPolygonFlakeData(). Cached per `n_sides` (unlike dodeca/icosa/octahedron's own single-shape
// caches) since regular_polygon_flake_board() isn't a fixed shape. Exactly one of `edge_glue`/
// `node_glue` is populated (never both), per `n_sides % 4` - see board_config.h's own doc comment on
// regular_polygon_flake_board() for which point(s) and why.
struct RegularPolygonFlakeData {
    std::vector<std::pair<int, int>> edges;
    std::vector<std::array<int, 6>> edge_glue;
    std::vector<std::array<int, 4>> node_glue;
};

static const RegularPolygonFlakeData& regular_polygon_flake_data(int n_sides) {
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

// Mirrors shared/boardConfig.ts's regularPolygonFlake() - see board_config.h's own doc comment for
// the high-level construction (why there's no embedding at all here, same as regular_polygon_board()
// itself - both have inherently irrational coordinates for most `n_sides`).
BoardConfig regular_polygon_flake_board(int n_sides, int order) {
    assert(n_sides >= 3 && "n_sides must be at least 3");
    assert(order >= 1 && "order must be at least 1");
    const auto& data = regular_polygon_flake_data(n_sides);

    std::map<std::pair<int, int>, std::array<int, 4>> edge_glue_map;
    for (const auto& g : data.edge_glue) edge_glue_map[{ g[0], g[1] }] = { g[2], g[3], g[4], g[5] };
    std::map<std::pair<int, int>, std::pair<int, int>> node_glue_map;
    for (const auto& g : data.node_glue) node_glue_map[{ g[0], g[1] }] = { g[2], g[3] };

    auto edge_level_up_map = growing_edge_level_up_map(edge_glue_map);
    auto result = node_edge_merge_flake_rec(
        order, n_sides, n_sides, data.edges, edge_glue_map, node_glue_map, edge_level_up_map);
    std::vector<std::vector<unsigned>> embed(result.adj.size()); // emb_dim=0
    return make_bc(std::move(result.adj), 0u, std::move(embed));
}

// Mirrors shared/boardConfig.ts's centralRegularPolygonFlake() - see board_config.h's own doc
// comment for the high-level construction. Reuses regular_polygon_flake_data(n_sides)'s own base
// edge_glue/node_glue (the plain regular_polygon_flake_board() relation between ADJACENT copies,
// untouched), adding one further auxiliary sub-copy (index n_sides, num_leaf..num_subs-1 - see
// node_edge_merge_flake_rec()'s own doc comment) glued to every regular copy via node_glue_map only
// - a closed-form vertex correspondence (`(i + n_sides/2) % n_sides`, `i`), needing no distance
// search unlike compute_flake_glue()/compute_node_glue() above (mirrors
// shared/boardConfig.ts's regularPolygonFractalDescr() derivation exactly).
BoardConfig central_regular_polygon_flake_board(int n_sides, int order) {
    assert(n_sides >= 3 && "n_sides must be at least 3");
    assert(order >= 1 && "order must be at least 1");
    const auto& data = regular_polygon_flake_data(n_sides);

    std::map<std::pair<int, int>, std::array<int, 4>> edge_glue_map;
    for (const auto& g : data.edge_glue) edge_glue_map[{ g[0], g[1] }] = { g[2], g[3], g[4], g[5] };
    std::map<std::pair<int, int>, std::pair<int, int>> node_glue_map;
    for (const auto& g : data.node_glue) node_glue_map[{ g[0], g[1] }] = { g[2], g[3] };

    int num_subs = n_sides;
    if (n_sides % 2 == 0 && n_sides > 4) {
        int center_idx = n_sides;
        num_subs = n_sides + 1;
        for (int i = 0; i < n_sides; i++)
            node_glue_map[{ i, center_idx }] = { (i + n_sides / 2) % n_sides, i };
    }

    auto edge_level_up_map = growing_edge_level_up_map(edge_glue_map);
    auto result = node_edge_merge_flake_rec(
        order, n_sides, num_subs, data.edges, edge_glue_map, node_glue_map, edge_level_up_map);
    std::vector<std::vector<unsigned>> embed(result.adj.size()); // emb_dim=0
    return make_bc(std::move(result.adj), 0u, std::move(embed));
}

// Mirrors shared/boardConfig.ts's centralPentagonFlake() - see board_config.h's own doc comment for
// the high-level construction. Reuses regular_polygon_flake_data(5)'s own base node_glue (pentagon's
// plain adjacent-copy relation is always a node merge - 5 % 4 != 0); the central copy's own relation
// is a closed-form edge_glue_map/edge_level_up_map pair, mirroring
// shared/boardConfig.ts's centralPentagonFractalDescr() exactly, no distance search needed.
BoardConfig central_pentagon_flake_board(int order) {
    assert(order >= 1 && "order must be at least 1");
    const int n_sides = 5;
    const auto& data = regular_polygon_flake_data(n_sides);

    std::map<std::pair<int, int>, std::pair<int, int>> node_glue_map;
    for (const auto& g : data.node_glue) node_glue_map[{ g[0], g[1] }] = { g[2], g[3] };

    int center_idx = n_sides;
    int num_subs = n_sides + 1;
    std::map<std::pair<int, int>, std::array<int, 4>> edge_glue_map;
    std::map<std::pair<int, int>, std::vector<std::array<int, 3>>> edge_level_up_map;
    for (int i = 0; i < n_sides; i++) {
        int a = i, b = (i + 1) % n_sides;
        int j = (i + 3) % n_sides;
        edge_glue_map[{ j, center_idx }] = { a, b, b, a };
        int lo = std::min(a, b), hi = std::max(a, b);
        edge_level_up_map[{ lo, hi }] = { { a, a, b }, { b, a, b } };
    }

    auto result = node_edge_merge_flake_rec(
        order, n_sides, num_subs, data.edges, edge_glue_map, node_glue_map, edge_level_up_map);
    std::vector<std::vector<unsigned>> embed(result.adj.size()); // emb_dim=0
    return make_bc(std::move(result.adj), 0u, std::move(embed));
}

BoardConfig triangular_hex_board(int d) {
    assert(d >= 0 && "d must be non-negative");
    std::vector<std::pair<int,int>> coords;
    for (int q = -d; q <= d; q++) {
        int r_lo = std::max(-d, -d - q);
        int r_hi = std::min(d, d - q);
        for (int r = r_lo; r <= r_hi; r++)
            coords.push_back({q, r});
    }
    int N = static_cast<int>(coords.size());
    std::vector<std::vector<unsigned>> pos(N);
    std::map<std::pair<int,int>, int> idx;
    for (int i = 0; i < N; i++) {
        auto [q, r] = coords[i];
        pos[i] = {static_cast<unsigned>(q + d), static_cast<unsigned>(r + d)};
        idx[{q, r}] = i;
    }
    auto adj = zero_adj(N);
    const int dirs[6][2] = {{1,0},{1,-1},{0,-1},{-1,0},{-1,1},{0,1}};
    for (int i = 0; i < N; i++) {
        auto [q, r] = coords[i];
        for (auto& dv : dirs) {
            auto it = idx.find({q + dv[0], r + dv[1]});
            if (it != idx.end()) adj[i][it->second] = 1;
        }
    }
    return make_bc(std::move(adj), 2u, std::move(pos));
}

BoardConfig hex_board(int d) {
    assert(d >= 0 && "d must be non-negative");
    const int dirs[6][2] = {{1,0},{1,-1},{0,-1},{-1,0},{-1,1},{0,1}};

    // Hexagon centers: the (q, r) = (a+2b, a-b) sublattice (color (q-r) mod 3 == 0, since
    // q-r = (a+2b)-(a-b) = 3b), restricted to hex-distance <= d in its own (a, b) coordinates -
    // mirrors shared/boardConfig.ts's hexBoard().
    std::vector<std::pair<int,int>> centers;
    for (int a = -d; a <= d; a++) {
        int b_lo = std::max(-d, -d - a);
        int b_hi = std::min(d, d - a);
        for (int b = b_lo; b <= b_hi; b++)
            centers.push_back({a + 2*b, a - b});
    }

    std::map<std::pair<int,int>, int> idx;
    std::vector<std::pair<int,int>> coords;
    auto add_vertex = [&](int q, int r) {
        auto key = std::make_pair(q, r);
        if (idx.count(key)) return;
        idx[key] = static_cast<int>(coords.size());
        coords.push_back(key);
    };
    for (auto& [q, r] : centers)
        for (auto& dv : dirs)
            add_vertex(q + dv[0], r + dv[1]);

    int N = static_cast<int>(coords.size());
    int offset = 2*d + 1;
    std::vector<std::vector<unsigned>> pos(N);
    for (int i = 0; i < N; i++) {
        auto [q, r] = coords[i];
        pos[i] = {static_cast<unsigned>(q + offset), static_cast<unsigned>(r + offset)};
    }
    auto adj = zero_adj(N);
    for (int i = 0; i < N; i++) {
        auto [q, r] = coords[i];
        for (auto& dv : dirs) {
            auto it = idx.find({q + dv[0], r + dv[1]});
            if (it != idx.end()) adj[i][it->second] = 1;
        }
    }
    return make_bc(std::move(adj), 2u, std::move(pos));
}

BoardConfig trihex_board(int d) {
    assert(d >= 0 && "d must be non-negative");
    const int dirs[6][2] = {{1,0},{1,-1},{0,-1},{-1,0},{-1,1},{0,1}};

    // Hexagon centers: the (q, r) = (2a, 2b) sublattice (both axial coordinates even), restricted
    // to hex-distance <= d in its own halved (a, b) coordinates - mirrors
    // shared/boardConfig.ts's trihexBoard().
    std::vector<std::pair<int,int>> centers;
    for (int a = -d; a <= d; a++) {
        int b_lo = std::max(-d, -d - a);
        int b_hi = std::min(d, d - a);
        for (int b = b_lo; b <= b_hi; b++)
            centers.push_back({2*a, 2*b});
    }

    std::map<std::pair<int,int>, int> idx;
    std::vector<std::pair<int,int>> coords;
    auto add_vertex = [&](int q, int r) {
        auto key = std::make_pair(q, r);
        if (idx.count(key)) return;
        idx[key] = static_cast<int>(coords.size());
        coords.push_back(key);
    };
    for (auto& [q, r] : centers)
        for (auto& dv : dirs)
            add_vertex(q + dv[0], r + dv[1]);

    int N = static_cast<int>(coords.size());
    int offset = 2*d + 1;
    std::vector<std::vector<unsigned>> pos(N);
    for (int i = 0; i < N; i++) {
        auto [q, r] = coords[i];
        pos[i] = {static_cast<unsigned>(q + offset), static_cast<unsigned>(r + offset)};
    }
    auto adj = zero_adj(N);
    for (int i = 0; i < N; i++) {
        auto [q, r] = coords[i];
        for (auto& dv : dirs) {
            auto it = idx.find({q + dv[0], r + dv[1]});
            if (it != idx.end()) adj[i][it->second] = 1;
        }
    }
    return make_bc(std::move(adj), 2u, std::move(pos));
}

// gap=0.0 → glue_twisted_square_board, gap=1.0 → twisted_square_board
static std::tuple<std::vector<std::vector<unsigned>>,
                  std::vector<std::vector<int>>,
                  std::vector<std::pair<int,int>>>
tilted_disconnected_square_board(int w, int h, int g, int gap) {
    const unsigned sq_width = (g - 1) * 2 + gap;
    std::vector<std::vector<unsigned>> pos;
    for (int rb = 0; rb < h; rb++)
        for (int cb = 0; cb < w; cb++) {
            unsigned bx = cb * sq_width;
            unsigned by = rb * sq_width;
            for (unsigned r = 0; r < g; r++)
                for (unsigned c = 0; c < g; c++)
                    // use `g - 1 - r` to avoid unsigned underflow
                    pos.push_back({bx + c + (g - 1 - r), by + c + r});
        }
    int N = w * h * g * g;
    auto adj = zero_adj(N);
    auto b_idx = [&](int rb, int cb) { return (rb*w + cb)*g*g; };
    const int dirs[4][2] = {{0,1},{1,0},{0,-1},{-1,0}};

    for (int rb = 0; rb < h; rb++)
        for (int cb = 0; cb < w; cb++) {
            int b = b_idx(rb, cb);
            for (int r = 0; r < g; r++)
                for (int c = 0; c < g; c++)
                    for (auto& d : dirs) {
                        int nr = r+d[0], nc = c+d[1];
                        if (nr>=0 && nr<g && nc>=0 && nc<g)
                            adj[b+r*g+c][b+nr*g+nc] = 1;
                    }
        }

    std::vector<std::pair<int,int>> inter_conn;
    for (int rb = 0; rb < h; rb++)
        for (int cb = 0; cb < w; cb++) {
            int b = b_idx(rb, cb);
            for (auto& d : dirs) {
                int nrb = rb+d[0], ncb = cb+d[1];
                if (nrb<0||nrb>=h||ncb<0||ncb>=w) continue;
                int nb_b = b_idx(nrb, ncb);
                int dr = d[0], dc = d[1];
                int self_idx  = ((dr - dc + 1) >> 1) * g * (g-1) + ((dr + dc + 1) >> 1) * (g-1);
                int other_idx = g*g - 1 - self_idx;
                inter_conn.push_back({b + self_idx, nb_b + other_idx});
            }
        }

    return {std::move(pos), std::move(adj), std::move(inter_conn)};
}

BoardConfig snub_square_board(int w, int h, int g) {
    assert(w > 0 && h > 0 && g > 0 && "w, h, and g must be positive");
    // Same 45deg-integer-rotation embedding as tilted_disconnected_square_board (gap=0, i.e. the
    // glue_twisted_square_board case) - embed coordinates must be integers, unlike
    // shared/boardConfig.ts's own literal +-30-degree floating-point layout.
    const unsigned sq_width = (g - 1) * 2;
    std::vector<std::vector<unsigned>> pos;
    for (int rb = 0; rb < h; rb++)
        for (int cb = 0; cb < w; cb++) {
            unsigned bx = cb * sq_width;
            unsigned by = rb * sq_width;
            for (unsigned r = 0; r < g; r++)
                for (unsigned c = 0; c < g; c++)
                    pos.push_back({bx + c + (g - 1 - r), by + c + r});
        }
    int N = w * h * g * g;
    auto adj = zero_adj(N);
    auto b_idx = [&](int rb, int cb) { return (rb*w + cb)*g*g; };
    const int dirs[4][2] = {{0,1},{1,0},{0,-1},{-1,0}};

    // Edges within each square (ordinary rectangular grid)
    for (int rb = 0; rb < h; rb++)
        for (int cb = 0; cb < w; cb++) {
            int b = b_idx(rb, cb);
            for (int r = 0; r < g; r++)
                for (int c = 0; c < g; c++)
                    for (auto& d : dirs) {
                        int nr = r+d[0], nc = c+d[1];
                        if (nr>=0 && nr<g && nc>=0 && nc<g)
                            adj[b+r*g+c][b+nr*g+nc] = 1;
                    }
        }

    // Corner (r,c) offsets: 0=NW 1=NE 2=SW 3=SE - matches shared/boardConfig.ts's cornerRC.
    auto corner = [&](int which) -> std::pair<int,int> {
        switch (which) {
            case 0: return {0, 0};
            case 1: return {0, g-1};
            case 2: return {g-1, 0};
            default: return {g-1, g-1};
        }
    };
    // glue/tri corner indices per self-cell checkerboard parity and direction (H: dr=0,dc=1;
    // V: dr=1,dc=0) - mirrors shared/boardConfig.ts's snubSquareBoard CONN table: each
    // orthogonal neighbor shares one glued corner plus one new triangle-connecting edge between two
    // of their other corners.
    struct Conn { int glue_self, glue_other, tri_self, tri_other; };
    const Conn conn[2][2] = {
        // parity 0:   H (dc=1)        V (dr=1)
        {  {3,2, 1,0},   {2,0, 3,1}  },
        // parity 1:   H (dc=1)        V (dr=1)
        {  {1,0, 3,2},   {3,1, 2,0}  },
    };

    std::vector<std::pair<int,int>> inter_conn;
    for (int rb = 0; rb < h; rb++)
        for (int cb = 0; cb < w; cb++) {
            int b = b_idx(rb, cb);
            int parity = (rb + cb) % 2;
            for (int dir = 0; dir < 2; dir++) { // 0=H(dr=0,dc=1), 1=V(dr=1,dc=0)
                int dr = dir==1 ? 1 : 0, dc = dir==0 ? 1 : 0;
                int nrb = rb+dr, ncb = cb+dc;
                if (nrb<0||nrb>=h||ncb<0||ncb>=w) continue;
                int nb = b_idx(nrb, ncb);
                const Conn& c = conn[parity][dir];
                auto [sr,sc] = corner(c.glue_self);
                auto [or_,oc] = corner(c.glue_other);
                inter_conn.push_back({b + sr*g + sc, nb + or_*g + oc});
                auto [tsr,tsc] = corner(c.tri_self);
                auto [tor,toc] = corner(c.tri_other);
                int i = b + tsr*g + tsc, j = nb + tor*g + toc;
                adj[i][j] = 1;
                adj[j][i] = 1;
            }
        }

    return quotient_board(make_bc(std::move(adj), 2u, std::move(pos)), inter_conn);
}

BoardConfig snub_square_tri_board(int w, int h, int g) {
    assert(w > 0 && h > 0 && g > 0 && "w, h, and g must be positive");

    int n_tri = g * (g + 1) / 2;
    auto tri_idx = [&](int i, int j) { return i*(i+1)/2 + j; };
    auto sq_idx = [&](int x, int y) { return (y*w + x)*g*g; };
    int sq_n = w * h * g * g;
    int h_count = (w-1) * h;
    int v_count = w * (h-1);
    auto h_base = [&](int x, int y) { return sq_n + (y*(w-1) + x)*n_tri; };
    auto v_base = [&](int x, int y) { return sq_n + h_count*n_tri + (y*w + x)*n_tri; };
    int N = sq_n + (h_count + v_count) * n_tri;

    // Squares: same per-cell 45deg-integer-rotation shape {c+(g-1-r), c+r} as snub_square_board -
    // but NOT the same bx=x*sq_width, by=y*sq_width placement. That placement only needs to support
    // snub_square_board's own single-corner glue (which never derives a *new* position - it just
    // identifies two already-computed corners via inter_conn, so it works regardless of how the
    // corners' raw values relate). Here, a triangle's interior nodes are *derived* by interpolating
    // between the two square corners it glues to, and two vertically-stacked h-triangles (or
    // horizontally-adjacent v-triangles) must derive IDENTICAL values at their shared third side -
    // which requires the two squares each one interpolates from to already have matching corners
    // *before* interpolation, not just after quotient_board's merge. Solving that (square(x,y) and
    // square(x,y+1)'s relevant H-triangle corners equal; square(x,y) and square(x+1,y)'s relevant
    // V-triangle corners equal) for a per-cell offset bx(x,y), by(x,y) gives bx = (x-y)*(g-1) (shifted
    // by +(h-1)*(g-1) to stay non-negative) and by = (x+y)*(g-1) - unlike the simple bx=x*sq_width
    // used elsewhere, both offsets now depend on both x and y.
    std::vector<std::vector<unsigned>> pos(N);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            unsigned bx = (unsigned)(x - y + (h - 1)) * (g - 1);
            unsigned by = (unsigned)(x + y) * (g - 1);
            int b = sq_idx(x, y);
            for (unsigned r = 0; r < g; r++)
                for (unsigned c = 0; c < g; c++)
                    pos[b + r*g + c] = {bx + c + (g - 1 - r), by + c + r};
        }
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w - 1; x++) {
            int p = (x + y) % 2;
            int base = h_base(x, y);
            for (int i = 0; i < g; i++) {
                int self_r = p == 0 ? g-1-i : i;
                const auto& left = pos[sq_idx(x, y) + self_r*g + (g-1)];
                const auto& right = pos[sq_idx(x+1, y) + self_r*g + 0];
                for (int j = 0; j <= i; j++) {
                    double t = i == 0 ? 0.0 : (double)j / (double)i;
                    pos[base + tri_idx(i,j)] = {
                        (unsigned)std::lround(left[0] + ((double)right[0] - (double)left[0]) * t),
                        (unsigned)std::lround(left[1] + ((double)right[1] - (double)left[1]) * t)
                    };
                }
            }
        }
    for (int y = 0; y < h - 1; y++)
        for (int x = 0; x < w; x++) {
            int p = (x + y) % 2;
            int base = v_base(x, y);
            for (int i = 0; i < g; i++) {
                int self_c = p == 0 ? i : g-1-i;
                const auto& left = pos[sq_idx(x, y) + (g-1)*g + self_c];
                const auto& right = pos[sq_idx(x, y+1) + 0*g + self_c];
                for (int j = 0; j <= i; j++) {
                    double t = i == 0 ? 0.0 : (double)j / (double)i;
                    pos[base + tri_idx(i,j)] = {
                        (unsigned)std::lround(left[0] + ((double)right[0] - (double)left[0]) * t),
                        (unsigned)std::lround(left[1] + ((double)right[1] - (double)left[1]) * t)
                    };
                }
            }
        }

    auto adj = zero_adj(N);
    const int dirs4[4][2] = {{0,1},{1,0},{0,-1},{-1,0}};

    // Intra-square edges (ordinary rectangular grid; no direct square-to-square connections here).
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            int b = sq_idx(x, y);
            for (int r = 0; r < g; r++)
                for (int c = 0; c < g; c++)
                    for (auto& d : dirs4) {
                        int nr = r+d[0], nc = c+d[1];
                        if (nr>=0 && nr<g && nc>=0 && nc<g)
                            adj[b+r*g+c][b+nr*g+nc] = 1;
                    }
        }

    // Intra-triangle edges (mirrors triangular_board's own edge loop).
    const int dirs6[6][2] = {{1,0},{1,1},{0,1},{-1,0},{-1,-1},{0,-1}};
    auto add_tri_edges = [&](int base) {
        for (int i = 0; i < g; i++)
            for (int j = 0; j <= i; j++)
                for (auto& d : dirs6) {
                    int ni = i+d[0], nj = j+d[1];
                    if (ni>=0 && ni<g && nj>=0 && nj<=ni)
                        adj[base+tri_idx(i,j)][base+tri_idx(ni,nj)] = 1;
                }
    };
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w - 1; x++) add_tri_edges(h_base(x, y));
    for (int y = 0; y < h - 1; y++)
        for (int x = 0; x < w; x++) add_tri_edges(v_base(x, y));

    // Gluing: every square-triangle and triangle-triangle edge, merged via a single quotient_board
    // call - mirrors shared/boardConfig.ts's snubSquareTriBoard.
    std::vector<std::pair<int,int>> inter_conn;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w - 1; x++) {
            int p = (x + y) % 2;
            int base = h_base(x, y);
            for (int i = 0; i < g; i++) {
                int self_r = p == 0 ? g-1-i : i;
                inter_conn.push_back({base + tri_idx(i,0), sq_idx(x, y) + self_r*g + (g-1)});
                inter_conn.push_back({base + tri_idx(i,i), sq_idx(x+1, y) + self_r*g + 0});
            }
            if (p == 1 && y + 1 <= h - 1) {
                int nbase = h_base(x, y+1);
                for (int j = 0; j < g; j++)
                    inter_conn.push_back({base + tri_idx(g-1,j), nbase + tri_idx(g-1,j)});
            }
        }
    for (int y = 0; y < h - 1; y++)
        for (int x = 0; x < w; x++) {
            int p = (x + y) % 2;
            int base = v_base(x, y);
            for (int i = 0; i < g; i++) {
                int self_c = p == 0 ? i : g-1-i;
                inter_conn.push_back({base + tri_idx(i,0), sq_idx(x, y) + (g-1)*g + self_c});
                inter_conn.push_back({base + tri_idx(i,i), sq_idx(x, y+1) + 0*g + self_c});
            }
            if (p == 0 && x + 1 <= w - 1) {
                int nbase = v_base(x+1, y);
                for (int j = 0; j < g; j++)
                    inter_conn.push_back({base + tri_idx(g-1,j), nbase + tri_idx(g-1,j)});
            }
        }

    return quotient_board(make_bc(std::move(adj), 2u, std::move(pos)), inter_conn);
}

BoardConfig glue_twisted_square_board(int w, int h, int g) {
    assert(w > 0 && h > 0 && g > 0 && "w, h, g must be positive");
    auto [pos, adj, inter_conn] = tilted_disconnected_square_board(w, h, g, 0);
    return quotient_board(make_bc(std::move(adj), 2u, std::move(pos)), inter_conn);
}

BoardConfig twisted_square_board(int w, int h, int g) {
    assert(w > 0 && h > 0 && g > 0 && "w, h, g must be positive");
    auto [pos, adj, inter_conn] = tilted_disconnected_square_board(w, h, g, 1);
    for (auto [i, j] : inter_conn) {
        adj[i][j] = 1;
        adj[j][i] = 1;
    }
    return make_bc(std::move(adj), 2u, std::move(pos));
}

BoardConfig build_board_config(const std::string& kind, const std::vector<int>& args) {
    const auto& v = args;
    if (kind == "line")  return linear_board(v[0]);
    if (kind == "rect")  return rectangular_board(v[0], v[1]);
    if (kind == "rectd") return rectangular_diagonal_board(v[0], v[1], v[2]);
    if (kind == "cublat") return cube_lattice_board(v[0], v[1], v[2]);
    if (kind == "hcub")  return hypercuboid_board(v[0], std::vector<int>(v.begin() + 1, v.end()));
    if (kind == "tri")   return triangular_board(v[0]);
    if (kind == "sier")  return sierpinski_simplex_board(v[0], v[1]);
    if (kind == "regpoly") return regular_polygon_board(v[0]);
    if (kind == "star")  return star_board(v[0]);
    if (kind == "tetra") return tetrahedron_board();
    if (kind == "octa") return octahedron_board();
    if (kind == "ortho") return orthoplex_board(v[0]);
    if (kind == "dodeca") return dodecahedron_board();
    if (kind == "icosa") return icosahedron_board();
    if (kind == "dodflake") return dodecahedron_flake_board(v[0]);
    if (kind == "icoflake") return icosahedron_flake_board(v[0]);
    if (kind == "octaflake") return octahedron_flake_board(v[0]);
    if (kind == "polyflake") return regular_polygon_flake_board(v[0], v[1]);
    if (kind == "cpolyflake") return central_regular_polygon_flake_board(v[0], v[1]);
    if (kind == "cpentflake") return central_pentagon_flake_board(v[0]);
    if (kind == "trihex") return triangular_hex_board(v[0]);
    if (kind == "hex")   return hex_board(v[0]);
    if (kind == "hexdel") return trihex_board(v[0]);
    if (kind == "snubsq") return snub_square_board(v[0], v[1], v[2]);
    if (kind == "snubsqtri") return snub_square_tri_board(v[0], v[1], v[2]);
    if (kind == "twsq")  return twisted_square_board(v[0], v[1], v[2]);
    if (kind == "gtsq")  return glue_twisted_square_board(v[0], v[1], v[2]);
    throw std::runtime_error("Unknown board type: " + kind);
}
