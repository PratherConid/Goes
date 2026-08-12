#pragma once
#include <vector>
#include <set>
#include <array>
#include <utility>

// Graph-topology utilities operating on plain N×N adjacency matrices (the same representation as
// BoardConfig::adj), independent of any board-specific geometry. Mirrors shared/topology.ts.

// An all-zero N×N adjacency matrix - the usual starting point before filling in edges. Mirrors
// shared/topology.ts's zeroAdj().
std::vector<std::vector<int>> zero_adj(int N);

// A plain (pos, adj) pair - the recursive intermediate form merge_boards() and its callers
// (board_config.cpp's sierpinski_rec(), fractal.cpp's node_edge_merge_flake_rec()) build on before
// an outermost caller wraps the result into a full BoardConfig.
struct RawBoard {
    std::vector<std::vector<unsigned>> pos;
    std::vector<std::vector<int>> adj;
};

// Combines a list of boards into one, additionally identifying every `((b1, i1), (b2, i2))` pair in
// `merges` (board index, that board's own local node index) as the same node - every merge is
// resolved in one batch via an internal union-find, not board-by-board, so a merge between two
// boards that haven't been introduced to each other by any other merge is handled exactly like any
// other. The merged node keeps whichever input position is encountered first. Returns the combined
// board plus, for each input board (same order as `boards`), a map from that board's own local
// indices to its final index in the combined board. Mirrors shared/topology.ts's mergeBoards().
std::pair<RawBoard, std::vector<std::vector<int>>> merge_boards(
    const std::vector<RawBoard>& boards,
    const std::vector<std::pair<std::pair<int,int>, std::pair<int,int>>>& merges);

// An adjacency-list view of a graph: list[i] is the set of i's neighbors.
using AdjacencyList = std::vector<std::set<int>>;

// Converts an N×N adjacency matrix into an adjacency list, each node's neighbors stored as a
// std::set (not a vector) so membership checks - the hot path for both find_triangles/find_squares
// below - are O(log degree) instead of O(degree).
AdjacencyList to_adjacency_list(const std::vector<std::vector<int>>& adj);

// Finds every triangle (3 distinct, pairwise-adjacent vertices) in adj, each reported exactly once
// as {u, v, w} with u < v < w. Mirrors shared/topology.ts's findTriangles() - see its own doc
// comment for why the increasing-order search both dedupes and stays efficient on sparse graphs.
std::vector<std::array<int, 3>> find_triangles(const std::vector<std::vector<int>>& adj);

// Finds every "square" - 4 distinct vertices a, b, c, d forming a cycle a-b-c-d-a whose two
// diagonals a-c and b-d are BOTH absent - each reported exactly once as {a, b, c, d} in that cycle
// order. Mirrors shared/topology.ts's findSquares() - see its own doc comment for the
// common-neighbor-pair search and its deduplication rule.
std::vector<std::array<int, 4>> find_squares(const std::vector<std::vector<int>>& adj);
