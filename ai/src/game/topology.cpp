#include "game/topology.h"
#include <algorithm>

AdjacencyList to_adjacency_list(const std::vector<std::vector<int>>& adj) {
    int N = (int)adj.size();
    AdjacencyList list(N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (adj[i][j]) list[i].insert(j);
    return list;
}

std::vector<std::array<int, 3>> find_triangles(const std::vector<std::vector<int>>& adj) {
    int N = (int)adj.size();
    auto adj_list = to_adjacency_list(adj);
    std::vector<std::array<int, 3>> triangles;
    for (int u = 0; u < N; u++)
        for (int v : adj_list[u]) {
            if (v <= u) continue;
            for (int w : adj_list[v]) {
                if (w <= v) continue;
                if (adj_list[u].count(w)) triangles.push_back({u, v, w});
            }
        }
    return triangles;
}

std::vector<std::array<int, 4>> find_squares(const std::vector<std::vector<int>>& adj) {
    int N = (int)adj.size();
    auto adj_list = to_adjacency_list(adj);
    std::vector<std::array<int, 4>> squares;
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
                    if (p < r) squares.push_back({p, r, q, s});
                }
        }
    return squares;
}
