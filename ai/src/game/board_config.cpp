#include "game/board_config.h"
#include "game/geometry.h"
#include "game/topology.h"
#include "game/fractal.h"
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

    // New positions: average of class members, over all emb_dim coordinates.
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
    // A real (always-active) check, not assert(): an emb_dim=0 board (e.g. dodeca/icosa/regpoly, or
    // triform/sqform output - see board_config.h) has no edge directions, and the convex-hull LP
    // would silently return no edges on 0-dimensional points rather than failing loudly.
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
        // pos[mid_idx] is at the doubled scale (see above), so subtract 2*embed[v] (TS's pos2[v]).
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

// Mirrors shared/boardConfig.ts's truncate()'s connectivity but not its position formula: TS's
// per-vertex fraction t_v is irrational in general, so here every near-point instead sits a fixed
// 1/3 of the way along its edge from its own endpoint, exact-integer after scaling the whole board
// by 3: the near-i point of edge (i,j) is 3*embed[i] + (embed[j]-embed[i]) = 2*embed[i]+embed[j],
// and symmetrically the near-j point is embed[i]+2*embed[j].
BoardConfig truncate(const BoardConfig& bc) {
    // Same rationale as rectify's own real-embedding check above - the vertex-ring connectivity
    // needs real edge directions.
    if (bc.emb_dim == 0)
        throw std::runtime_error(
            "truncate: requires a real (non-zero) embedding, got emb_dim=0 - this board has no "
            "coordinates to compute edge directions from");
    int N = bc.N;
    unsigned emb_dim = bc.emb_dim;

    // Two new nodes per original edge (i<j): near_i (2*embed[i]+embed[j]) and near_j
    // (embed[i]+2*embed[j]) - see this function's own doc comment above.
    std::map<std::pair<int,int>, std::pair<int,int>> near_idx; // (i,j) -> (near_i idx, near_j idx)
    std::vector<std::vector<unsigned>> pos;
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) {
            if (!bc.adj[i][j]) continue;
            std::vector<unsigned> near_i(emb_dim), near_j(emb_dim);
            for (unsigned k = 0; k < emb_dim; k++) {
                near_i[k] = 2 * bc.embed[i][k] + bc.embed[j][k];
                near_j[k] = bc.embed[i][k] + 2 * bc.embed[j][k];
            }
            int idx_i = static_cast<int>(pos.size()); pos.push_back(std::move(near_i));
            int idx_j = static_cast<int>(pos.size()); pos.push_back(std::move(near_j));
            near_idx[{i, j}] = { idx_i, idx_j };
        }

    // Edges incident to each original node, as [other endpoint] lists.
    std::vector<std::vector<int>> incident(N);
    for (auto& [ij, idx] : near_idx) {
        (void)idx;
        incident[ij.first].push_back(ij.second);
        incident[ij.second].push_back(ij.first);
    }

    auto adj = zero_adj(static_cast<int>(pos.size()));
    // Shrunk original edges: connect the two near-points of each original edge.
    for (auto& [ij, near] : near_idx) {
        (void)ij;
        adj[near.first][near.second] = 1;
        adj[near.second][near.first] = 1;
    }
    // Vertex rings: connect hull-adjacent near-v points around each original vertex, same
    // direction/convex-hull technique as rectify() above.
    for (int v = 0; v < N; v++) {
        auto& us = incident[v];
        if (us.size() < 2) continue;
        std::vector<std::vector<double>> dirs;
        std::vector<int> near_at(us.size());
        for (size_t idx = 0; idx < us.size(); idx++) {
            int u = us[idx];
            auto key = v < u ? std::make_pair(v, u) : std::make_pair(u, v);
            auto& near = near_idx[key];
            near_at[idx] = (v < u) ? near.first : near.second;

            std::vector<double> d(emb_dim);
            double len_sq = 0;
            for (unsigned k = 0; k < emb_dim; k++) {
                d[k] = static_cast<double>(bc.embed[u][k]) - static_cast<double>(bc.embed[v][k]);
                len_sq += d[k] * d[k];
            }
            double len = std::sqrt(len_sq);
            for (unsigned k = 0; k < emb_dim; k++) d[k] /= len;
            dirs.push_back(std::move(d));
        }
        for (auto& [a, b] : convex_hull_edges(dirs)) {
            adj[near_at[a]][near_at[b]] = 1;
            adj[near_at[b]][near_at[a]] = 1;
        }
    }

    return make_bc(std::move(adj), emb_dim, std::move(pos));
}

BoardConfig merge_close(const BoardConfig& bc, double dist) {
    assert(dist > 0 && "dist must be positive");
    // A real (always-active) check, not assert() - merge_close needs real coordinates to compute a
    // meaningful distance. Left unchecked, an emb_dim=0 board (e.g. dodeca/icosa/regpoly, or
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

BoardConfig generic_form(const BoardConfig& bc, int w, const std::vector<FormSelector>& sels, ClegRandom& random) {
    assert(w >= 1 && "w must be at least 1");
    int N = bc.N;

    std::set<std::pair<int,int>> is_face_side; // (p,q), p<q - an original edge consumed by some face
    // canonical (p,q), p<q -> per-face boundary getters, oriented p (k=0) to q (k=w-1) by add_side.
    std::map<std::pair<int,int>, std::vector<std::function<int(int)>>> edge_to_seqs;
    std::vector<std::pair<int,int>> quot; // corner glues, plus cross-edge glues appended below

    auto add_side = [&](int p, int q, std::function<int(int)> at_k) {
        auto key = std::pair{std::min(p, q), std::max(p, q)};
        is_face_side.insert(key);
        std::function<int(int)> oriented = p < q ? at_k : std::function<int(int)>(
            [at_k, w](int k) { return at_k(w - 1 - k); });
        edge_to_seqs[key].push_back(std::move(oriented));
    };

    // TS's extraEdges; no extraPos counterpart, since the result is emb_dim=0 (see board_config.h).
    std::vector<std::pair<int,int>> extra_edges;
    int next_idx = N;

    // Mirrors genericForm()'s buildQuadGridLattice().
    auto build_quad_grid_lattice = [&](const std::vector<BoardQuad>& quads, const std::vector<std::pair<int,int>>& dirs) {
        int n_face = w * w;
        auto local_idx = [&](int i, int j) { return i * w + j; };
        for (auto& [A, B, C, D] : quads) {
            int offset = next_idx;
            next_idx += n_face;
            auto global_idx = [=](int i, int j) { return offset + local_idx(i, j); };
            for (int i = 0; i < w; i++)
                for (int j = 0; j < w; j++)
                    for (auto& [di, dj] : dirs) {
                        int ni = i + di, nj = j + dj;
                        if (ni >= 0 && ni < w && nj >= 0 && nj < w)
                            extra_edges.push_back({global_idx(i, j), global_idx(ni, nj)});
                    }
            quot.push_back({A, global_idx(0, 0)});
            quot.push_back({B, global_idx(0, w - 1)});
            quot.push_back({C, global_idx(w - 1, w - 1)});
            quot.push_back({D, global_idx(w - 1, 0)});
            add_side(A, B, [=](int k) { return global_idx(0, k); });
            add_side(B, C, [=](int k) { return global_idx(k, w - 1); });
            add_side(C, D, [=](int k) { return global_idx(w - 1, w - 1 - k); });
            add_side(D, A, [=](int k) { return global_idx(w - 1 - k, 0); });
        }
    };

    for (auto& fsel : sels) {
        if (fsel.kind == FormSelKind::TriForm) {
            auto sel = fsel.sel.value_or(Selector{SelectorOp::All, simp_type(2)});
            auto triangles = select_triangle(bc.adj, bc.embed, sel, random);
            int n_face = w * (w + 1) / 2;
            auto local_idx = [](int i, int j) { return i * (i + 1) / 2 + j; };
            const int dirs[6][2] = {{1,0},{1,1},{0,1},{-1,0},{-1,-1},{0,-1}};
            for (auto& simplex : triangles) {
                int A = simplex.nodes[0], B = simplex.nodes[1], C = simplex.nodes[2];
                int offset = next_idx;
                next_idx += n_face;
                auto global_idx = [=](int i, int j) { return offset + local_idx(i, j); };
                for (int i = 0; i < w; i++)
                    for (int j = 0; j <= i; j++)
                        for (auto& d : dirs) {
                            int ni = i + d[0], nj = j + d[1];
                            if (ni >= 0 && ni < w && nj >= 0 && nj <= ni)
                                extra_edges.push_back({global_idx(i, j), global_idx(ni, nj)});
                        }
                quot.push_back({A, global_idx(0, 0)});
                quot.push_back({B, global_idx(w - 1, 0)});
                quot.push_back({C, global_idx(w - 1, w - 1)});
                add_side(A, B, [=](int k) { return global_idx(k, 0); });
                add_side(A, C, [=](int k) { return global_idx(k, k); });
                add_side(B, C, [=](int k) { return global_idx(w - 1, k); });
            }
        } else if (fsel.kind == FormSelKind::QuadForm) {
            auto sel = fsel.sel.value_or(Selector{SelectorOp::All, SelectorType{SelectorKind::Quad}});
            build_quad_grid_lattice(select_quad(bc.adj, bc.embed, sel, random), {{0,1},{1,0},{0,-1},{-1,0}});
        } else if (fsel.kind == FormSelKind::QuadKnightForm) {
            auto sel = fsel.sel.value_or(Selector{SelectorOp::All, SelectorType{SelectorKind::Quad}});
            build_quad_grid_lattice(select_quad(bc.adj, bc.embed, sel, random),
                {{1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}});
        } else if (fsel.kind == FormSelKind::QuadBishopForm) {
            auto sel = fsel.sel.value_or(Selector{SelectorOp::All, SelectorType{SelectorKind::Quad}});
            build_quad_grid_lattice(select_quad(bc.adj, bc.embed, sel, random), {{1,1},{1,-1},{-1,1},{-1,-1}});
        } else { // FormSelKind::QuadDiagForm
            auto sel = fsel.sel.value_or(Selector{SelectorOp::All, SelectorType{SelectorKind::Quad}});
            auto quads = select_quad(bc.adj, bc.embed, sel, random);
            int n_primary = w * w;
            auto primary_idx = [&](int i, int j) { return i * w + j; };
            auto center_idx = [&](int i, int j) { return n_primary + i * (w - 1) + j; };
            int n_face = n_primary + (w - 1) * (w - 1);
            for (auto& [A, B, C, D] : quads) {
                int offset = next_idx;
                next_idx += n_face;
                auto global_primary = [=](int i, int j) { return offset + primary_idx(i, j); };
                auto global_center = [=](int i, int j) { return offset + center_idx(i, j); };
                for (int i = 0; i < w - 1; i++)
                    for (int j = 0; j < w - 1; j++) {
                        extra_edges.push_back({global_center(i, j), global_primary(i, j)});
                        extra_edges.push_back({global_center(i, j), global_primary(i + 1, j)});
                        extra_edges.push_back({global_center(i, j), global_primary(i, j + 1)});
                        extra_edges.push_back({global_center(i, j), global_primary(i + 1, j + 1)});
                    }
                quot.push_back({A, global_primary(0, 0)});
                quot.push_back({B, global_primary(0, w - 1)});
                quot.push_back({C, global_primary(w - 1, w - 1)});
                quot.push_back({D, global_primary(w - 1, 0)});
                add_side(A, B, [=](int k) { return global_primary(0, k); });
                add_side(B, C, [=](int k) { return global_primary(k, w - 1); });
                add_side(C, D, [=](int k) { return global_primary(w - 1, w - 1 - k); });
                add_side(D, A, [=](int k) { return global_primary(w - 1 - k, 0); });
            }
        }
    }

    int total_n = next_idx;
    auto adj = zero_adj(total_n);
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) {
            if (!bc.adj[i][j] || is_face_side.count({i, j})) continue;
            adj[i][j] = 1;
            adj[j][i] = 1;
        }
    for (auto& [a, b] : extra_edges) {
        adj[a][b] = 1;
        adj[b][a] = 1;
    }

    for (auto& [key, seqs] : edge_to_seqs) {
        if (seqs.size() < 2) continue;
        for (size_t s = 1; s < seqs.size(); s++)
            for (int k = 0; k < w; k++) quot.push_back({seqs[0](k), seqs[s](k)});
    }

    std::vector<std::vector<unsigned>> embed(total_n); // emb_dim=0 - see board_config.h's doc comment
    BoardConfig combined = make_bc(std::move(adj), 0u, std::move(embed));
    return quotient_board(combined, quot);
}

BoardConfig triangle_form(const BoardConfig& bc, int w, ClegRandom& random, std::optional<Selector> sel) {
    return generic_form(bc, w, { FormSelector{FormSelKind::TriForm, sel} }, random);
}

BoardConfig quad_form(const BoardConfig& bc, int w, ClegRandom& random, std::optional<Selector> sel) {
    return generic_form(bc, w, { FormSelector{FormSelKind::QuadForm, sel} }, random);
}

BoardConfig quad_diag_form(const BoardConfig& bc, int w, ClegRandom& random, std::optional<Selector> sel) {
    return generic_form(bc, w, { FormSelector{FormSelKind::QuadDiagForm, sel} }, random);
}

BoardConfig quad_knight_form(const BoardConfig& bc, int w, ClegRandom& random, std::optional<Selector> sel) {
    return generic_form(bc, w, { FormSelector{FormSelKind::QuadKnightForm, sel} }, random);
}

BoardConfig quad_bishop_form(const BoardConfig& bc, int w, ClegRandom& random, std::optional<Selector> sel) {
    return generic_form(bc, w, { FormSelector{FormSelKind::QuadBishopForm, sel} }, random);
}

// Mirrors shared/boardConfig.ts's genericLocalReplace(), minus its position logic (barycenters,
// apexes, QuadOctarize's extra-dimension padding): the result is always emb_dim=0 (see
// board_config.h).
BoardConfig generic_local_replace(const BoardConfig& bc, const std::vector<LocalReplaceSelector>& selectors, ClegRandom& random) {
    int N = bc.N;
    int next_idx = N;
    // (i,j) i<j - a selected face's own original edge, excluded from bc.adj's straight copy below.
    std::set<std::pair<int,int>> consumed;
    auto mark_consumed = [&](int a, int b) { consumed.insert({std::min(a, b), std::max(a, b)}); };
    std::vector<std::pair<int,int>> extra_edges;

    for (auto& s : selectors) {
        if (s.kind == LocalReplaceKind::QuadCentralize || s.kind == LocalReplaceKind::QuadOctarize ||
            s.kind == LocalReplaceKind::QuadCentering) {
            Selector sel = s.sel.value_or(Selector{SelectorOp::All, SelectorType{SelectorKind::Quad}});
            auto quads = select_quad(bc.adj, bc.embed, sel, random);
            for (auto& [A, B, C, D] : quads) {
                mark_consumed(A, B); mark_consumed(B, C); mark_consumed(C, D); mark_consumed(D, A);
                if (s.kind != LocalReplaceKind::QuadCentering) {
                    extra_edges.push_back({A, B});
                    extra_edges.push_back({B, C});
                    extra_edges.push_back({C, D});
                    extra_edges.push_back({D, A});
                }
                if (s.kind == LocalReplaceKind::QuadOctarize) {
                    int top = next_idx++, bottom = next_idx++;
                    for (int c : {A, B, C, D}) {
                        extra_edges.push_back({top, c});
                        extra_edges.push_back({bottom, c});
                    }
                } else {
                    // QuadCentralize or QuadCentering - a single hub either way, differing only in
                    // whether the quad's own 4-cycle edges survive (see above).
                    int hub = next_idx++;
                    extra_edges.push_back({hub, A});
                    extra_edges.push_back({hub, B});
                    extra_edges.push_back({hub, C});
                    extra_edges.push_back({hub, D});
                }
            }
        } else {
            int n = s.n;
            assert(n >= 2 && "generic_local_replace: n must be at least 2");
            Selector sel = s.sel.value_or(Selector{SelectorOp::All, simp_type(n)});
            auto simplices = select_simp(bc.adj, bc.embed, sel, random);
            for (auto& simplex : simplices) {
                auto& nodes = simplex.nodes;
                for (size_t i = 0; i < nodes.size(); i++)
                    for (size_t j = i + 1; j < nodes.size(); j++) {
                        mark_consumed(nodes[i], nodes[j]);
                        // SimpCentering is the one branch that does NOT re-add the simplex's own
                        // clique edges - see this function's own header doc comment.
                        if (s.kind != LocalReplaceKind::SimpCentering) extra_edges.push_back({nodes[i], nodes[j]});
                    }
                int hub = next_idx++;
                for (int v : nodes) extra_edges.push_back({hub, v});
            }
        }
    }

    int total_n = next_idx;
    auto adj = zero_adj(total_n);
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) {
            if (!bc.adj[i][j] || consumed.count({i, j})) continue;
            adj[i][j] = 1;
            adj[j][i] = 1;
        }
    for (auto& [a, b] : extra_edges) {
        adj[a][b] = 1;
        adj[b][a] = 1;
    }

    std::vector<std::vector<unsigned>> embed(total_n); // emb_dim=0 - see board_config.h's doc comment
    return make_bc(std::move(adj), 0u, std::move(embed));
}

BoardConfig simp_centralize(const BoardConfig& bc, int n, ClegRandom& random, std::optional<Selector> sel) {
    assert(n >= 2 && "simp_centralize: n must be at least 2");
    return generic_local_replace(bc, { LocalReplaceSelector{LocalReplaceKind::SimpCentralize, n, sel} }, random);
}

BoardConfig simp_centering(const BoardConfig& bc, int n, ClegRandom& random, std::optional<Selector> sel) {
    assert(n >= 2 && "simp_centering: n must be at least 2");
    return generic_local_replace(bc, { LocalReplaceSelector{LocalReplaceKind::SimpCentering, n, sel} }, random);
}

BoardConfig tri_centralize(const BoardConfig& bc, ClegRandom& random, std::optional<Selector> sel) {
    return simp_centralize(bc, 2, random, sel);
}

BoardConfig tri_centering(const BoardConfig& bc, ClegRandom& random, std::optional<Selector> sel) {
    return simp_centering(bc, 2, random, sel);
}

BoardConfig quad_centralize(const BoardConfig& bc, ClegRandom& random, std::optional<Selector> sel) {
    return generic_local_replace(bc, { LocalReplaceSelector{LocalReplaceKind::QuadCentralize, 0, sel} }, random);
}

BoardConfig quad_centering(const BoardConfig& bc, ClegRandom& random, std::optional<Selector> sel) {
    return generic_local_replace(bc, { LocalReplaceSelector{LocalReplaceKind::QuadCentering, 0, sel} }, random);
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

BoardConfig quad_octarize(const BoardConfig& bc, ClegRandom& random, std::optional<Selector> sel) {
    return generic_local_replace(bc, { LocalReplaceSelector{LocalReplaceKind::QuadOctarize, 0, sel} }, random);
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

BoardConfig node_induced_subgraph(const BoardConfig& bc, const std::set<int>& nodes) {
    std::vector<int> kept;
    for (int i = 0; i < bc.N; i++) if (nodes.count(i)) kept.push_back(i);

    std::vector<std::vector<unsigned>> embed;
    embed.reserve(kept.size());
    for (int i : kept) embed.push_back(bc.embed[i]);

    auto adj = zero_adj(static_cast<int>(kept.size()));
    for (size_t a = 0; a < kept.size(); a++)
        for (size_t b = a + 1; b < kept.size(); b++)
            if (bc.adj[kept[a]][kept[b]]) { adj[a][b] = 1; adj[b][a] = 1; }

    return make_bc(std::move(adj), bc.emb_dim, std::move(embed));
}

// Mirrors shared/boardConfig.ts's inducedSubgraphFromEdges().
static BoardConfig induced_subgraph_from_edges(
    const BoardConfig& bc, std::set<int> initially_touched, const std::vector<BoardEdge>& edges)
{
    std::set<int> touched = std::move(initially_touched);
    for (auto& e : edges) { touched.insert(e.n1); touched.insert(e.n2); }
    std::vector<int> kept;
    for (int i = 0; i < bc.N; i++) if (touched.count(i)) kept.push_back(i);
    std::vector<int> new_idx(bc.N, -1);
    for (size_t i = 0; i < kept.size(); i++) new_idx[kept[i]] = static_cast<int>(i);

    std::vector<std::vector<unsigned>> embed;
    embed.reserve(kept.size());
    for (int i : kept) embed.push_back(bc.embed[i]);

    auto adj = zero_adj(static_cast<int>(kept.size()));
    for (auto& e : edges) {
        int a = new_idx[e.n1], b = new_idx[e.n2];
        adj[a][b] = 1;
        adj[b][a] = 1;
    }

    return make_bc(std::move(adj), bc.emb_dim, std::move(embed));
}

BoardConfig edge_induced_subgraph(const BoardConfig& bc, const std::vector<BoardEdge>& edges) {
    return induced_subgraph_from_edges(bc, {}, edges);
}

BoardConfig node_edge_induced_subgraph(
    const BoardConfig& bc, const std::set<int>& nodes, const std::vector<BoardEdge>& edges)
{
    return induced_subgraph_from_edges(bc, nodes, edges);
}

BoardConfig apply_modifier(const BoardConfig& bc, const BoardModifier& modifier, ClegRandom& random) {
    switch (modifier.kind) {
        case ModifierKind::Rectify:    return rectify(bc);
        case ModifierKind::Truncate:   return truncate(bc);
        case ModifierKind::EdgeSplit:  return edge_split(bc, modifier.split_n);
        case ModifierKind::MergeClose: return merge_close(bc, modifier.dist);
        case ModifierKind::TriangleForm: return triangle_form(bc, modifier.split_n, random, modifier.form_sel);
        case ModifierKind::QuadForm: return quad_form(bc, modifier.split_n, random, modifier.form_sel);
        case ModifierKind::QuadDiagForm: return quad_diag_form(bc, modifier.split_n, random, modifier.form_sel);
        case ModifierKind::QuadKnightForm: return quad_knight_form(bc, modifier.split_n, random, modifier.form_sel);
        case ModifierKind::QuadBishopForm: return quad_bishop_form(bc, modifier.split_n, random, modifier.form_sel);
        case ModifierKind::Form: return generic_form(bc, modifier.split_n, modifier.form_sels, random);
        case ModifierKind::LocalReplace: return generic_local_replace(bc, modifier.selectors, random);
        case ModifierKind::GlobalCentralize: return global_centralize(bc);
        case ModifierKind::Scale: return scale_board(bc, modifier.dist);
        case ModifierKind::NodeInducedSubgraph: return node_induced_subgraph(bc, select_node(bc.adj, bc.embed, modifier.sel, random));
        case ModifierKind::EdgeInducedSubgraph: return edge_induced_subgraph(bc, select_edge(bc.adj, bc.embed, modifier.sel, random));
    }
    throw std::runtime_error("apply_modifier: unknown ModifierKind");
}

BoardConfig apply_modifiers(const BoardConfig& bc, const std::vector<BoardModifier>& modifiers, ClegRandom& random) {
    BoardConfig current = bc;
    for (auto& m : modifiers) current = apply_modifier(current, m, random);
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

// Mirrors shared/boardConfig.ts's hypercuboidBoard(), except positions are the uncentered
// [0, dims[i]) integer coordinates rather than TS's `- (dims[i]-1)/2`-centered ones: centering
// would need half-integers for an even-sized dimension, which exact-integer embed[] can't represent.
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

// Mirrors shared/boardConfig.ts's sierpinskiRec(), returning (pos, adj, corners) as a tuple.
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

// Mirrors shared/boardConfig.ts's sierpinskiSimplex(), except for the corner embedding: TS's
// regularSimplexCoords() is irrational for dim >= 2, so here corner 0 of the outer simplex sits at
// the origin and corner k (1 <= k <= dim) at `side` times the k-th standard basis vector.
// Recursive midpoints stay exact integers since `side` = 2^(n-1).
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

// Mirrors shared/boardConfig.ts's simplexBoard(), with sierpinski_simplex_board()'s exact-integer
// corners (corner 0 at the origin, corner k at the k-th standard basis vector) in place of
// regularSimplexCoords(dim), so barycentric (c_0, ..., c_dim) embeds as simply (c_1, ..., c_dim).
BoardConfig simplex_board(int meshdim, int dim, int w) {
    assert(dim >= 1 && "dim must be at least 1");
    assert(w >= 1 && "w must be at least 1");
    assert(meshdim >= 0 && "meshdim must be non-negative");
    int m = dim + 1, n = w - 1;

    std::vector<std::vector<int>> all_coords;
    std::function<void(std::vector<int>&, int)> build = [&](std::vector<int>& prefix, int remaining) {
        if (static_cast<int>(prefix.size()) == m - 1) {
            prefix.push_back(remaining);
            all_coords.push_back(prefix);
            prefix.pop_back();
            return;
        }
        for (int c = 0; c <= remaining; c++) {
            prefix.push_back(c);
            build(prefix, remaining - c);
            prefix.pop_back();
        }
    };
    std::vector<int> prefix;
    build(prefix, n);

    std::vector<std::vector<unsigned>> corners(m, std::vector<unsigned>(dim, 0u));
    for (int k = 1; k <= dim; k++) corners[k][k - 1] = 1u;

    std::map<std::vector<int>, int> board_idx_of;
    std::vector<std::vector<int>> surviving_coords;
    std::vector<int> nonzero_counts;
    std::vector<std::vector<unsigned>> pos;
    for (auto& c : all_coords) {
        int nonzero_count = 0;
        for (int x : c) if (x > 0) nonzero_count++;
        if (nonzero_count > meshdim + 1) continue;
        board_idx_of[c] = static_cast<int>(surviving_coords.size());
        nonzero_counts.push_back(nonzero_count);
        surviving_coords.push_back(c);
        std::vector<unsigned> p(dim, 0u);
        for (int i = 0; i < m; i++)
            for (int d = 0; d < dim; d++) p[d] += static_cast<unsigned>(c[i]) * corners[i][d];
        pos.push_back(std::move(p));
    }
    int N = static_cast<int>(surviving_coords.size());

    auto adj = zero_adj(N);
    for (int bi = 0; bi < N; bi++) {
        auto& c = surviving_coords[bi];
        for (int i = 0; i < m; i++) {
            if (c[i] == 0) continue;
            for (int j = 0; j < m; j++) {
                if (j == i) continue;
                int extra = c[j] == 0 ? 1 : 0;
                if (nonzero_counts[bi] + extra > meshdim + 1) continue;
                std::vector<int> nc = c;
                nc[i]--; nc[j]++;
                auto it = board_idx_of.find(nc);
                if (it == board_idx_of.end()) continue;
                adj[bi][it->second] = 1;
            }
        }
    }
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

// Mirrors shared/boardConfig.ts's starBoard(), adjacency only (emb_dim=0 - see board_config.h).
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

// Mirrors shared/boardConfig.ts's tetrahedronBoard().
BoardConfig tetrahedron_board() {
    return simplex_board(3, 3, 2);
}

// Mirrors shared/boardConfig.ts's diamondCubicBoard(), adjacency only (emb_dim=0 - see
// board_config.h).
BoardConfig diamond_cubic_board(int w) {
    assert(w >= 1 && "w must be at least 1");
    int n = w - 1;

    // flat_key/node_idx: a flat-array stand-in for TS's Map-based nodeIdx.
    int stride = n + 1;
    auto flat_key = [&](int c0, int c1, int c2) { return (c0 * stride + c1) * stride + c2; };
    std::vector<int> node_idx(static_cast<size_t>(stride) * stride * stride, -1);
    int next_idx = 0;
    for (int c0 = 0; c0 <= n; c0++)
        for (int c1 = 0; c1 <= n - c0; c1++)
            for (int c2 = 0; c2 <= n - c0 - c1; c2++)
                node_idx[flat_key(c0, c1, c2)] = next_idx++;
    auto node = [&](int c0, int c1, int c2) { return node_idx[flat_key(c0, c1, c2)]; };

    std::vector<std::pair<int,int>> edges;

    // One hub per up-tetrahedron - see this function's own doc comment. No down-tetrahedron loop:
    // their own edges never survive regardless, so there's nothing left for one to add.
    for (int c0 = 0; c0 <= n - 1; c0++)
        for (int c1 = 0; c1 <= n - 1 - c0; c1++)
            for (int c2 = 0; c2 <= n - 1 - c0 - c1; c2++) {
                int corner[4] = {
                    node(c0 + 1, c1, c2), node(c0, c1 + 1, c2), node(c0, c1, c2 + 1), node(c0, c1, c2),
                };
                int hub = next_idx++;
                for (int c : corner) edges.push_back({hub, c});
            }

    int total_n = next_idx;
    auto adj = zero_adj(total_n);
    for (auto& [a, b] : edges) {
        adj[a][b] = 1;
        adj[b][a] = 1;
    }
    std::vector<std::vector<unsigned>> embed(total_n); // emb_dim=0 - see board_config.h's doc comment
    return make_bc(std::move(adj), 0u, std::move(embed));
}

BoardConfig octahedron_board() {
    return orthoplex_board(3);
}

// Mirrors shared/boardConfig.ts's orthoplexBoard(), but with integer coordinates in place of TS's
// +-1/sqrt(2): vertex 2k (the "+" pole on axis k) has coordinate k = 2, vertex 2k+1 (the "-" pole)
// has coordinate k = 1, every other coordinate 0; connectivity is unaffected.
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

// Mirrors shared/boardConfig.ts's reg24CellBoard(), except embed[] must be non-negative: stored
// positions are the raw {-1,0,1} coordinates shifted by +1 to {0,1,2} (instead of TS's 1/sqrt(2)
// scaling), while adjacency still uses the un-shifted raw dot product.
BoardConfig reg_24_cell_board() {
    std::vector<std::vector<int>> raw;
    for (int i = 0; i < 4; i++)
        for (int j = i + 1; j < 4; j++)
            for (int si : {1, -1})
                for (int sj : {1, -1}) {
                    std::vector<int> v(4, 0);
                    v[i] = si;
                    v[j] = sj;
                    raw.push_back(v);
                }

    int N = static_cast<int>(raw.size());
    std::vector<std::vector<unsigned>> pos(N, std::vector<unsigned>(4));
    for (int a = 0; a < N; a++)
        for (int k = 0; k < 4; k++) pos[a][k] = static_cast<unsigned>(raw[a][k] + 1);

    auto adj = zero_adj(N);
    for (int a = 0; a < N; a++)
        for (int b = a + 1; b < N; b++) {
            int dot = 0;
            for (int k = 0; k < 4; k++) dot += raw[a][k] * raw[b][k];
            if (dot == 1) { adj[a][b] = 1; adj[b][a] = 1; }
        }

    return make_bc(std::move(adj), 4u, std::move(pos));
}

// -- Helpers for reg_120_cell_board()/reg_600_cell_board() below, mirroring shared/boardConfig.ts's
// permsOf/evenPermsOf/signVariants (add_perms: one group's worth of signedPermutationVertices). --

static std::vector<std::vector<double>> perms_of(const std::vector<double>& arr) {
    if (arr.size() <= 1) return { arr };
    std::vector<std::vector<double>> out;
    for (size_t i = 0; i < arr.size(); i++) {
        std::vector<double> rest;
        for (size_t k = 0; k < arr.size(); k++) if (k != i) rest.push_back(arr[k]);
        for (auto& p : perms_of(rest)) {
            std::vector<double> full{ arr[i] };
            full.insert(full.end(), p.begin(), p.end());
            out.push_back(std::move(full));
        }
    }
    return out;
}

static std::vector<std::vector<double>> even_perms_of(const std::vector<double>& arr) {
    std::vector<double> idx(arr.size());
    std::iota(idx.begin(), idx.end(), 0.0);
    std::vector<std::vector<double>> out;
    for (auto& p : perms_of(idx)) {
        int inversions = 0;
        for (size_t i = 0; i < p.size(); i++)
            for (size_t j = i + 1; j < p.size(); j++)
                if (p[i] > p[j]) inversions++;
        if (inversions % 2 != 0) continue;
        std::vector<double> v(p.size());
        for (size_t i = 0; i < p.size(); i++) v[i] = arr[static_cast<size_t>(p[i])];
        out.push_back(std::move(v));
    }
    return out;
}

static std::vector<std::vector<double>> sign_variants(const std::vector<double>& v) {
    std::vector<size_t> nonzero;
    for (size_t i = 0; i < v.size(); i++) if (v[i] != 0.0) nonzero.push_back(i);
    size_t k = nonzero.size();
    std::vector<std::vector<double>> out;
    for (unsigned mask = 0; mask < (1u << k); mask++) {
        std::vector<double> w = v;
        for (size_t b = 0; b < k; b++) if (mask & (1u << b)) w[nonzero[b]] = -w[nonzero[b]];
        out.push_back(std::move(w));
    }
    return out;
}

// Mirrors shared/boardConfig.ts's signedPermutationVertices(), keying by llround(x * 100) in place of
// TS's toFixed(2) - the same 2-decimal rounding, as integers rather than strings.
static std::vector<std::vector<double>> signed_permutation_vertices(
    const std::vector<std::vector<std::vector<double>>>& perm_groups)
{
    std::vector<std::vector<double>> raw;
    std::set<std::string> seen;
    for (auto& perms : perm_groups)
        for (auto& p : perms)
            for (auto& v : sign_variants(p)) {
                std::string key;
                for (double x : v) key += std::to_string(std::llround(x * 1e2)) + ",";
                if (seen.insert(key).second) raw.push_back(v);
            }
    return raw;
}

// Mirrors shared/boardConfig.ts's adjacencyAtDistance().
static std::vector<std::vector<int>> adjacency_at_distance(const std::vector<std::vector<double>>& raw,
                                                           double edge_dist2) {
    int N = static_cast<int>(raw.size());
    auto adj = zero_adj(N);
    const double EPS = 1e-6;
    for (int a = 0; a < N; a++)
        for (int b = a + 1; b < N; b++) {
            double d2 = 0;
            for (size_t k = 0; k < raw[a].size(); k++) { double diff = raw[a][k] - raw[b][k]; d2 += diff * diff; }
            if (std::abs(d2 - edge_dist2) < EPS) { adj[a][b] = 1; adj[b][a] = 1; }
        }
    return adj;
}

// Mirrors shared/boardConfig.ts's reg120CellBoard(). Coordinates are a `double`-only intermediate
// (golden-ratio values have no exact-integer analog) and are discarded before returning: the result
// is emb_dim=0 (see board_config.h).
BoardConfig reg_120_cell_board() {
    const double PHI = (1.0 + std::sqrt(5.0)) / 2.0;
    const double PHI2 = PHI * PHI;
    const double IPHI = 1.0 / PHI;
    const double IPHI2 = 1.0 / (PHI * PHI);
    const double SQRT5 = std::sqrt(5.0);

    auto raw = signed_permutation_vertices({
        perms_of({0, 0, 2, 2}),
        perms_of({PHI, PHI, PHI, IPHI2}),
        perms_of({1, 1, 1, SQRT5}),
        perms_of({IPHI, IPHI, IPHI, PHI2}),
        even_perms_of({0, IPHI, PHI, SQRT5}),
        even_perms_of({0, IPHI2, 1, PHI2}),
        even_perms_of({IPHI, 1, PHI, 2}),
    });
    auto adj = adjacency_at_distance(raw, (3.0 - std::sqrt(5.0)) * (3.0 - std::sqrt(5.0)));
    std::vector<std::vector<unsigned>> embed(raw.size()); // emb_dim=0
    return make_bc(std::move(adj), 0u, std::move(embed));
}

// Mirrors shared/boardConfig.ts's reg600CellBoard() - same double-only-intermediate, emb_dim=0
// approach as reg_120_cell_board() above.
BoardConfig reg_600_cell_board() {
    const double PHI = (1.0 + std::sqrt(5.0)) / 2.0;
    const double IPHI = 1.0 / PHI;

    auto raw = signed_permutation_vertices({
        perms_of({0, 0, 0, 2}),
        perms_of({1, 1, 1, 1}),
        even_perms_of({PHI, 1, IPHI, 0}),
    });
    auto adj = adjacency_at_distance(raw, (2.0 / PHI) * (2.0 / PHI));
    std::vector<std::vector<unsigned>> embed(raw.size()); // emb_dim=0
    return make_bc(std::move(adj), 0u, std::move(embed));
}

// Mirrors shared/boardConfig.ts's antiprismBoard(), adjacency only (emb_dim=0 - see
// board_config.h).
BoardConfig antiprism_board(int n) {
    assert(n >= 3 && "n must be at least 3");
    int N = 2 * n;
    auto top = [](int k) { return k; };
    auto bot = [n](int k) { return n + k; };
    auto adj = zero_adj(N);
    auto connect = [&](int i, int j) { adj[i][j] = 1; adj[j][i] = 1; };
    for (int k = 0; k < n; k++) {
        connect(top(k), top((k + 1) % n));
        connect(bot(k), bot((k + 1) % n));
        connect(top(k), bot(k));
        connect(top(k), bot((k - 1 + n) % n));
    }
    std::vector<std::vector<unsigned>> embed(N); // emb_dim=0
    return make_bc(std::move(adj), 0u, std::move(embed));
}

// Mirrors shared/boardConfig.ts's dodecahedronBoard(), adjacency only (emb_dim=0 - see
// board_config.h).
BoardConfig dodecahedron_board() {
    const FractalDescr& descr = dodecahedron_fractal_descr();
    auto adj = zero_adj(descr.num_leaf);
    for (auto [a, b] : descr.leaf_conn) { adj[a][b] = 1; adj[b][a] = 1; }
    std::vector<std::vector<unsigned>> embed(descr.num_leaf); // emb_dim=0
    return make_bc(std::move(adj), 0u, std::move(embed));
}

// Mirrors shared/boardConfig.ts's icosahedronBoard(), adjacency only (emb_dim=0 - see
// board_config.h).
BoardConfig icosahedron_board() {
    const FractalDescr& descr = icosahedron_fractal_descr();
    auto adj = zero_adj(descr.num_leaf);
    for (auto [a, b] : descr.leaf_conn) { adj[a][b] = 1; adj[b][a] = 1; }
    std::vector<std::vector<unsigned>> embed(descr.num_leaf); // emb_dim=0
    return make_bc(std::move(adj), 0u, std::move(embed));
}

// --- Flake boards: thin build_fractal() wrappers (fractal.h), adjacency only (emb_dim=0 - see
// board_config.h's doc comment on dodecahedron_flake_board()). ---

// Mirrors shared/boardConfig.ts's dodecahedronFlake().
BoardConfig dodecahedron_flake_board(int n) {
    assert(n >= 1 && "n must be at least 1");
    auto adj = build_fractal(n, dodecahedron_fractal_descr());
    std::vector<std::vector<unsigned>> embed(adj.size()); // emb_dim=0
    return make_bc(std::move(adj), 0u, std::move(embed));
}

// Mirrors shared/boardConfig.ts's icosahedronFlake().
BoardConfig icosahedron_flake_board(int n) {
    assert(n >= 1 && "n must be at least 1");
    auto adj = build_fractal(n, icosahedron_fractal_descr());
    std::vector<std::vector<unsigned>> embed(adj.size()); // emb_dim=0
    return make_bc(std::move(adj), 0u, std::move(embed));
}

// Mirrors shared/boardConfig.ts's octahedronFlake().
BoardConfig octahedron_flake_board(int n) {
    assert(n >= 1 && "n must be at least 1");
    auto adj = build_fractal(n, octahedron_fractal_descr());
    std::vector<std::vector<unsigned>> embed(adj.size()); // emb_dim=0
    return make_bc(std::move(adj), 0u, std::move(embed));
}

// Mirrors shared/boardConfig.ts's regularPolygonFlake().
BoardConfig regular_polygon_flake_board(int n_sides, int order) {
    assert(n_sides >= 3 && "n_sides must be at least 3");
    assert(order >= 1 && "order must be at least 1");
    auto adj = build_fractal(order, regular_polygon_fractal_descr(n_sides, false));
    std::vector<std::vector<unsigned>> embed(adj.size()); // emb_dim=0
    return make_bc(std::move(adj), 0u, std::move(embed));
}

// Mirrors shared/boardConfig.ts's centralRegularPolygonFlake().
BoardConfig central_regular_polygon_flake_board(int n_sides, int order) {
    assert(n_sides >= 3 && "n_sides must be at least 3");
    assert(order >= 1 && "order must be at least 1");
    auto adj = build_fractal(order, regular_polygon_fractal_descr(n_sides, true));
    std::vector<std::vector<unsigned>> embed(adj.size()); // emb_dim=0
    return make_bc(std::move(adj), 0u, std::move(embed));
}

// Mirrors shared/boardConfig.ts's centralPentagonFlake().
BoardConfig central_pentagon_flake_board(int order) {
    assert(order >= 1 && "order must be at least 1");
    auto adj = build_fractal(order, central_pentagon_fractal_descr());
    std::vector<std::vector<unsigned>> embed(adj.size()); // emb_dim=0
    return make_bc(std::move(adj), 0u, std::move(embed));
}

// Mirrors shared/boardConfig.ts's mengerSpongeFlake().
BoardConfig menger_sponge_flake_board(int order, int dim, const std::vector<int>& indicator) {
    assert(order >= 1 && "order must be at least 1");
    assert(dim >= 1 && "dim must be at least 1");
    assert(static_cast<int>(indicator.size()) == dim + 1 &&
        "indicator must be a length-(dim+1) list of 0/1 entries");
    auto adj = build_fractal(order, menger_fractal_descr(dim, indicator));
    std::vector<std::vector<unsigned>> embed(adj.size()); // emb_dim=0
    return make_bc(std::move(adj), 0u, std::move(embed));
}

// Mirrors shared/boardConfig.ts's AXIAL_DIRS.
static constexpr int AXIAL_DIRS[6][2] = {{1,0},{1,-1},{0,-1},{-1,0},{-1,1},{0,1}};

// Mirrors shared/boardConfig.ts's triangularLatticeBoard(), embedding each axial (q, r) as the exact
// integers (q + offset, r + offset) in place of TS's real-valued (q + r/2, rowDist*r) - each caller
// passes the offset that makes its own coordinates non-negative (see board_config.h).
static BoardConfig triangular_lattice_board(const std::vector<std::pair<int,int>>& coords, int offset) {
    int N = static_cast<int>(coords.size());
    std::map<std::pair<int,int>, int> idx;
    for (int i = 0; i < N; i++) idx[coords[i]] = i;
    std::vector<std::vector<unsigned>> pos(N);
    for (int i = 0; i < N; i++) {
        auto [q, r] = coords[i];
        pos[i] = {static_cast<unsigned>(q + offset), static_cast<unsigned>(r + offset)};
    }
    auto adj = zero_adj(N);
    for (int i = 0; i < N; i++) {
        auto [q, r] = coords[i];
        for (auto& dv : AXIAL_DIRS) {
            auto it = idx.find({q + dv[0], r + dv[1]});
            if (it != idx.end()) adj[i][it->second] = 1;
        }
    }
    return make_bc(std::move(adj), 2u, std::move(pos));
}

BoardConfig triangular_hex_board(int d) {
    assert(d >= 0 && "d must be non-negative");
    std::vector<std::pair<int,int>> coords;
    for (int q = -d; q <= d; q++)
        for (int r = std::max(-d, -d - q); r <= std::min(d, d - q); r++)
            coords.push_back({q, r});
    return triangular_lattice_board(coords, d);
}

BoardConfig hex_board(int d) {
    assert(d >= 0 && "d must be non-negative");
    std::vector<std::pair<int,int>> centers;
    for (int a = -d; a <= d; a++)
        for (int b = std::max(-d, -d - a); b <= std::min(d, d - a); b++)
            centers.push_back({a + 2 * b, a - b});
    std::set<std::pair<int,int>> seen;
    std::vector<std::pair<int,int>> vertices;
    for (auto& [q, r] : centers)
        for (auto& dv : AXIAL_DIRS)
            if (seen.insert({q + dv[0], r + dv[1]}).second) vertices.push_back({q + dv[0], r + dv[1]});
    return triangular_lattice_board(vertices, 2 * d + 1);
}

BoardConfig trihex_board(int d) {
    assert(d >= 0 && "d must be non-negative");
    std::vector<std::pair<int,int>> centers;
    for (int a = -d; a <= d; a++)
        for (int b = std::max(-d, -d - a); b <= std::min(d, d - a); b++)
            centers.push_back({2 * a, 2 * b});
    std::set<std::pair<int,int>> seen;
    std::vector<std::pair<int,int>> vertices;
    for (auto& [q, r] : centers)
        for (auto& dv : AXIAL_DIRS)
            if (seen.insert({q + dv[0], r + dv[1]}).second) vertices.push_back({q + dv[0], r + dv[1]});
    return triangular_lattice_board(vertices, 2 * d + 1);
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

BoardConfig snub_square_board(int w, int h) {
    assert(w > 0 && h > 0 && "w and h must be positive");
    // Same 45deg-integer-rotation embedding as tilted_disconnected_square_board (gap=0, i.e. the
    // glue_twisted_square_board case) - embed coordinates must be integers, unlike
    // shared/boardConfig.ts's own literal +-30-degree floating-point layout.
    std::vector<std::vector<unsigned>> pos;
    for (int rb = 0; rb < h; rb++)
        for (int cb = 0; cb < w; cb++) {
            unsigned bx = cb * 2, by = rb * 2;
            pos.push_back({bx + 1, by});     // NW
            pos.push_back({bx + 2, by + 1}); // NE
            pos.push_back({bx,     by + 1}); // SW
            pos.push_back({bx + 1, by + 2}); // SE
        }
    int N = w * h * 4;
    auto adj = zero_adj(N);
    auto b_idx = [&](int rb, int cb) { return (rb*w + cb)*4; };

    // Each cell's own 4-cycle (a genuine quad face - no diagonal edges). 0=NW 1=NE 2=SW 3=SE,
    // matching the position order pushed above and shared/boardConfig.ts's cornerIdx.
    const int sides[4][2] = {{0,1},{0,2},{1,3},{2,3}};
    for (int rb = 0; rb < h; rb++)
        for (int cb = 0; cb < w; cb++) {
            int b = b_idx(rb, cb);
            for (auto& s : sides) {
                adj[b+s[0]][b+s[1]] = 1;
                adj[b+s[1]][b+s[0]] = 1;
            }
        }

    // glue/tri corner indices per self-cell checkerboard parity and direction (H: dr=0,dc=1;
    // V: dr=1,dc=0) - mirrors shared/boardConfig.ts's snubSquareBoard() CONN table.
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
                inter_conn.push_back({b + c.glue_self, nb + c.glue_other});
                int i = b + c.tri_self, j = nb + c.tri_other;
                adj[i][j] = 1;
                adj[j][i] = 1;
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

int board_arg_number(const BoardArgEntry& e) {
    assert(e.kind == BoardArgKind::Number && "expected a Number board arg");
    return e.value;
}
const std::vector<int>& board_arg_list(const BoardArgEntry& e) {
    assert(e.kind != BoardArgKind::Number && "expected a list board arg");
    return e.values;
}

std::string format_board_arg_entry(const BoardArgEntry& e) {
    if (e.kind == BoardArgKind::Number) return std::to_string(e.value);
    std::string sep = e.kind == BoardArgKind::CommaSeparatedNumbers ? "," : "";
    std::string s;
    for (size_t i = 0; i < e.values.size(); i++) { if (i) s += sep; s += std::to_string(e.values[i]); }
    return s;
}

BoardConfig build_board_config(const std::string& kind, const std::vector<BoardArgEntry>& args) {
    const auto& v = args;
    auto num = [](const BoardArgEntry& e) { return board_arg_number(e); };
    auto list = [](const BoardArgEntry& e) { return board_arg_list(e); };
    if (kind == "line")  return linear_board(num(v[0]));
    if (kind == "rect")  return rectangular_board(num(v[0]), num(v[1]));
    if (kind == "rectd") return rectangular_diagonal_board(num(v[0]), num(v[1]), num(v[2]));
    if (kind == "cublat") return cube_lattice_board(num(v[0]), num(v[1]), num(v[2]));
    if (kind == "hcub")  return hypercuboid_board(num(v[0]), list(v[1]));
    if (kind == "tri")   return triangular_board(num(v[0]));
    if (kind == "sier")  return sierpinski_simplex_board(num(v[0]), num(v[1]));
    if (kind == "simplex") return simplex_board(num(v[0]), num(v[1]), num(v[2]));
    if (kind == "regpoly") return regular_polygon_board(num(v[0]));
    if (kind == "star")  return star_board(num(v[0]));
    if (kind == "tetra") return tetrahedron_board();
    if (kind == "diamondCubic") return diamond_cubic_board(num(v[0]));
    if (kind == "octa") return octahedron_board();
    if (kind == "ortho") return orthoplex_board(num(v[0]));
    if (kind == "reg24Cell") return reg_24_cell_board();
    if (kind == "reg120Cell") return reg_120_cell_board();
    if (kind == "reg600Cell") return reg_600_cell_board();
    if (kind == "ap")    return antiprism_board(num(v[0]));
    if (kind == "dodeca") return dodecahedron_board();
    if (kind == "icosa") return icosahedron_board();
    if (kind == "dodflake") return dodecahedron_flake_board(num(v[0]));
    if (kind == "icoflake") return icosahedron_flake_board(num(v[0]));
    if (kind == "octaflake") return octahedron_flake_board(num(v[0]));
    if (kind == "polyflake") return regular_polygon_flake_board(num(v[0]), num(v[1]));
    if (kind == "cpolyflake") return central_regular_polygon_flake_board(num(v[0]), num(v[1]));
    if (kind == "cpentflake") return central_pentagon_flake_board(num(v[0]));
    if (kind == "menger") return menger_sponge_flake_board(num(v[0]), num(v[1]), list(v[2]));
    if (kind == "trihex") return triangular_hex_board(num(v[0]));
    if (kind == "hex")   return hex_board(num(v[0]));
    if (kind == "hexdel") return trihex_board(num(v[0]));
    if (kind == "snubsq") return snub_square_board(num(v[0]), num(v[1]));
    if (kind == "twsq")  return twisted_square_board(num(v[0]), num(v[1]), num(v[2]));
    if (kind == "gtsq")  return glue_twisted_square_board(num(v[0]), num(v[1]), num(v[2]));
    throw std::runtime_error("Unknown board type: " + kind);
}
