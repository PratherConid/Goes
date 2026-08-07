#include "game/board_config.h"
#include "game/geometry.h"
#include "game/topology.h"
#include <cassert>
#include <algorithm>
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
    assert(w > 0 && h > 0 && "w and h must be positive");
    std::vector<std::vector<unsigned>> pos;
    for (unsigned r = 0; r < h; r++)
        for (unsigned c = 0; c < w; c++)
            pos.push_back({c, r});
    auto adj = zero_adj(w * h);
    const int dirs[4][2] = {{0,1},{1,0},{0,-1},{-1,0}};
    for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++)
            for (auto& d : dirs) {
                int nr = r + d[0], nc = c + d[1];
                if (nr >= 0 && nr < h && nc >= 0 && nc < w)
                    adj[r*w+c][nr*w+nc] = 1;
            }
    return make_bc(std::move(adj), 2u, std::move(pos));
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
    assert(w > 0 && h > 0 && d > 0 && "w, h, d must be positive");
    std::vector<std::vector<unsigned>> pos;
    for (unsigned s = 0; s < d; s++)
        for (unsigned r = 0; r < h; r++)
            for (unsigned c = 0; c < w; c++)
                pos.push_back({c, r, s});
    int N = w * h * d;
    auto adj = zero_adj(N);
    auto idx = [&](int r, int c, int s) { return s*h*w + r*w + c; };
    const int dirs[6][3] = {{0,1,0},{1,0,0},{0,-1,0},{-1,0,0},{0,0,1},{0,0,-1}};
    for (int s = 0; s < d; s++)
        for (int r = 0; r < h; r++)
            for (int c = 0; c < w; c++)
                for (auto& dv : dirs) {
                    int nr = r+dv[0], nc = c+dv[1], ns = s+dv[2];
                    if (nr>=0 && nr<h && nc>=0 && nc<w && ns>=0 && ns<d)
                        adj[idx(r,c,s)][idx(nr,nc,ns)] = 1;
                }
    return make_bc(std::move(adj), 3u, std::move(pos));
}

BoardConfig hypercube_board(int w, int h, int d, int t) {
    assert(w > 0 && h > 0 && d > 0 && t > 0 && "w, h, d, t must be positive");
    std::vector<std::vector<unsigned>> pos;
    for (unsigned s = 0; s < t; s++)
        for (unsigned u = 0; u < d; u++)
            for (unsigned r = 0; r < h; r++)
                for (unsigned c = 0; c < w; c++)
                    pos.push_back({c, r, u, s});
    int N = w * h * d * t;
    auto adj = zero_adj(N);
    auto idx = [&](int r, int c, int u, int s) {
        return ((s*d + u)*h + r)*w + c;
    };
    const int dirs[8][4] = {
        {0,1,0,0},{1,0,0,0},{0,-1,0,0},{-1,0,0,0},
        {0,0,1,0},{0,0,-1,0},{0,0,0,1},{0,0,0,-1}
    };
    for (int s = 0; s < t; s++)
        for (int u = 0; u < d; u++)
            for (int r = 0; r < h; r++)
                for (int c = 0; c < w; c++)
                    for (auto& dv : dirs) {
                        int nr=r+dv[0], nc=c+dv[1], nu=u+dv[2], ns=s+dv[3];
                        if (nr>=0&&nr<h&&nc>=0&&nc<w&&nu>=0&&nu<d&&ns>=0&&ns<t)
                            adj[idx(r,c,u,s)][idx(nr,nc,nu,ns)] = 1;
                    }
    return make_bc(std::move(adj), 4u, std::move(pos));
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

// Mirrors shared/boardConfig.ts's starBoard() connectivity exactly (adjacency only - no
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

// Mirrors shared/boardConfig.ts's tetrahedronBoard() connectivity exactly (adjacency only - no
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

// Mirrors shared/boardConfig.ts's octahedronBoard() connectivity exactly (adjacency only - no
// position/embedding, see board_config.h's own doc comment on this function). Vertex 2k and 2k+1
// are always antipodal pairs, by construction - each vertex connects to every other vertex except
// its own antipode.
BoardConfig octahedron_board() {
    auto adj = zero_adj(6);
    auto antipode = [](int i) { return i % 2 == 0 ? i + 1 : i - 1; };
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
            if (i != j && j != antipode(i)) adj[i][j] = 1;
    std::vector<std::vector<unsigned>> embed(6); // emb_dim=0
    return make_bc(std::move(adj), 0u, std::move(embed));
}

// Mirrors shared/boardConfig.ts's dodecahedronBoard() connectivity exactly (adjacency only - no
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

// Mirrors shared/boardConfig.ts's icosahedronBoard() connectivity exactly (adjacency only - no
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
    // V: dr=1,dc=0) - mirrors shared/boardConfig.ts's snubSquareBoard CONN table exactly: each
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
    // call - mirrors shared/boardConfig.ts's snubSquareTriBoard exactly.
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
    if (kind == "hcub")  return hypercube_board(v[0], v[1], v[2], v[3]);
    if (kind == "tri")   return triangular_board(v[0]);
    if (kind == "regpoly") return regular_polygon_board(v[0]);
    if (kind == "star")  return star_board(v[0]);
    if (kind == "tetra") return tetrahedron_board();
    if (kind == "octa") return octahedron_board();
    if (kind == "dodeca") return dodecahedron_board();
    if (kind == "icosa") return icosahedron_board();
    if (kind == "trihex") return triangular_hex_board(v[0]);
    if (kind == "hex")   return hex_board(v[0]);
    if (kind == "hexdel") return trihex_board(v[0]);
    if (kind == "snubsq") return snub_square_board(v[0], v[1], v[2]);
    if (kind == "snubsqtri") return snub_square_tri_board(v[0], v[1], v[2]);
    if (kind == "twsq")  return twisted_square_board(v[0], v[1], v[2]);
    if (kind == "gtsq")  return glue_twisted_square_board(v[0], v[1], v[2]);
    throw std::runtime_error("Unknown board type: " + kind);
}
