#include "game/topology.h"
#include <algorithm>
#include <numeric>
#include <functional>

std::vector<std::vector<int>> zero_adj(int N) {
    return std::vector<std::vector<int>>(N, std::vector<int>(N, 0));
}

// Mirrors shared/topology.ts's own module-private unionFindClasses(): given N nodes and a list of
// pairs to merge, returns each node's equivalence-class index, compressed to a dense 0..M-1 range
// (M = number of distinct classes) in ascending order of each class's lowest original member.
// Internal to merge_boards() (below) - resolves every merge instruction at once, so a chain like
// (0,3)~(1,5) and (1,5)~(2,7) correctly collapses (0,3) and (2,7) into the same node too, even
// though no single instruction names both directly.
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

std::pair<RawBoard, std::vector<std::vector<int>>> merge_boards(
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

    std::map<std::string, int> labels;
    for (size_t b = 0; b < boards.size(); b++)
        for (const auto& [addr, local] : boards[b].labels) labels[addr] = maps[b][local];

    return { RawBoard{ std::move(pos), std::move(adj), std::move(labels) }, std::move(maps) };
}

AdjacencyList to_adjacency_list(const std::vector<std::vector<int>>& adj) {
    int N = (int)adj.size();
    AdjacencyList list(N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (adj[i][j]) list[i].insert(j);
    return list;
}

std::vector<BoardTriangle> find_triangles(const std::vector<std::vector<int>>& adj) {
    int N = (int)adj.size();
    auto adj_list = to_adjacency_list(adj);
    std::vector<BoardTriangle> triangles;
    for (int u = 0; u < N; u++)
        for (int v : adj_list[u]) {
            if (v <= u) continue;
            for (int w : adj_list[v]) {
                if (w <= v) continue;
                if (adj_list[u].count(w)) triangles.push_back(make_board_triangle(u, v, w));
            }
        }
    return triangles;
}

std::vector<BoardSquare> find_squares(const std::vector<std::vector<int>>& adj) {
    int N = (int)adj.size();
    auto adj_list = to_adjacency_list(adj);
    std::vector<BoardSquare> squares;
    for (int p = 0; p < N; p++)
        for (int q = p + 1; q < N; q++) {
            if (adj_list[p].count(q)) continue; // p-q would be an edge, not a diagonal
            std::vector<int> common;
            for (int x : adj_list[p]) if (adj_list[q].count(x)) common.push_back(x);
            for (size_t i = 0; i < common.size(); i++)
                for (size_t j = i + 1; j < common.size(); j++) {
                    int r = std::min(common[i], common[j]);
                    int s = std::max(common[i], common[j]);
                    if (adj_list[r].count(s)) continue; // r-s would be an edge, not a diagonal
                    if (p < r) squares.push_back(make_board_square(p, r, q, s));
                }
        }
    return squares;
}
