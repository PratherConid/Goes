#include "game/board_config.h"
#include <cassert>
#include <algorithm>
#include <functional>
#include <numeric>
#include <cmath>

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
                if (dr == 1 && dc == 1 && m > 1 && (r % m != 0 || c % m != 0)) continue;
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
