#include "game/board_config.h"
#include "game/geometry.h"
#include <cassert>
#include <algorithm>
#include <functional>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <map>

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

    // New positions: average of class members
    std::vector<std::vector<unsigned>> new_embed(new_n, std::vector<unsigned>(bc.emb_dim, 0u));
    std::vector<int> cnt(new_n, 0);
    for (int i = 0; i < N; i++) {
        int ni = node_to_new[i];
        new_embed[ni][0] += bc.embed[i][0];
        new_embed[ni][1] += bc.embed[i][1];
        cnt[ni]++;
    }
    for (int ni = 0; ni < new_n; ni++) {
        new_embed[ni][0] /= cnt[ni];
        new_embed[ni][1] /= cnt[ni];
    }

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

BoardConfig apply_modifier(const BoardConfig& bc, const BoardModifier& modifier) {
    switch (modifier.kind) {
        case ModifierKind::Rectify:   return rectify(bc);
        case ModifierKind::EdgeSplit: return edge_split(bc, modifier.split_n);
    }
    throw std::runtime_error("apply_modifier: unknown ModifierKind");
}

BoardConfig apply_modifiers(const BoardConfig& bc, const std::vector<BoardModifier>& modifiers) {
    BoardConfig result = bc;
    for (auto& m : modifiers) result = apply_modifier(result, m);
    return result;
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

BoardConfig cubical_board(int w, int h, int d) {
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
    if (kind == "rect")  return rectangular_board(v[0], v[1]);
    if (kind == "rectd") return rectangular_diagonal_board(v[0], v[1], v[2]);
    if (kind == "cub")   return cubical_board(v[0], v[1], v[2]);
    if (kind == "hcub")  return hypercube_board(v[0], v[1], v[2], v[3]);
    if (kind == "tri")   return triangular_board(v[0]);
    if (kind == "trihex") return triangular_hex_board(v[0]);
    if (kind == "hex")   return hex_board(v[0]);
    if (kind == "hexdel") return trihex_board(v[0]);
    if (kind == "snubsq") return snub_square_board(v[0], v[1], v[2]);
    if (kind == "snubsqtri") return snub_square_tri_board(v[0], v[1], v[2]);
    if (kind == "twsq")  return twisted_square_board(v[0], v[1], v[2]);
    if (kind == "gtsq")  return glue_twisted_square_board(v[0], v[1], v[2]);
    throw std::runtime_error("Unknown board type: " + kind);
}
