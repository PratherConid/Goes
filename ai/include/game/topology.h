#pragma once
#include "game/selector.h"
#include <vector>
#include <set>
#include <map>
#include <string>
#include <utility>

// Graph-topology utilities operating on plain N×N adjacency matrices (the same representation as
// BoardConfig::adj), independent of any board-specific geometry. Mirrors shared/topology.ts.

// An all-zero N×N adjacency matrix - the usual starting point before filling in edges. Mirrors
// shared/topology.ts's zeroAdj().
std::vector<std::vector<int>> zero_adj(int N);

// A plain (pos, adj) pair - the recursive intermediate form merge_boards() and its callers
// (board_config.cpp's sierpinski_rec(), fractal.cpp's node_edge_merge_flake_rec()) build on before
// an outermost caller wraps the result into a full BoardConfig. `labels` mirrors
// shared/topology.ts's mergeBoards() own optional per-board `labels` (address string -> that
// board's own local node index - see shared/fractal.ts's own doc comment on addresses/
// `SubFlakeResult` for what an address is): carried through the SAME remapping as `pos`/`adj` and
// combined into one map keyed the same way. Left empty (as sierpinski_rec()'s own boards do) simply
// contributes nothing to the combined map - there is no separate "has labels" flag needed, since an
// empty map already behaves that way.
struct RawBoard {
    std::vector<std::vector<unsigned>> pos;
    std::vector<std::vector<int>> adj;
    std::map<std::string, int> labels;
};

// Combines a list of boards into one, additionally identifying every `((b1, i1), (b2, i2))` pair in
// `merges` (board index, that board's own local node index) as the same node - every merge is
// resolved in one batch via an internal union-find, not board-by-board, so a merge between two
// boards that haven't been introduced to each other by any other merge is handled exactly like any
// other. The merged node keeps whichever input position is encountered first. Returns the combined
// board (its own `labels` combined the same way, see `RawBoard`'s own doc comment) plus, for each
// input board (same order as `boards`), a map from that board's own local indices to its final index
// in the combined board. Mirrors shared/topology.ts's mergeBoards().
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
// as a BoardTriangle (already n1 < n2 < n3 by construction - see make_board_triangle - since this
// always discovers a triangle's own 3 vertices in increasing order to begin with, so canonicalizing
// it costs nothing extra). Mirrors shared/topology.ts's findTriangles() - see its own doc comment for
// why the increasing-order search both dedupes and stays efficient on sparse graphs.
std::vector<BoardTriangle> find_triangles(const std::vector<std::vector<int>>& adj);

// Finds every "square" - 4 distinct vertices a, b, c, d forming a cycle a-b-c-d-a whose two
// diagonals a-c and b-d are BOTH absent - each reported exactly once as a BoardSquare, canonicalized
// via make_board_square (see its own doc comment, game/selector.h) from the a-b-c-d cycle order this
// function itself discovers it in - that canonicalization is a genuine relabeling (not necessarily
// a, b, c, d verbatim), unlike find_triangles' own free ride, since a square's own discovery order
// isn't already the lexicographically-least one in general. Mirrors shared/topology.ts's
// findSquares() - see its own doc comment for the common-neighbor-pair search and its deduplication
// rule.
std::vector<BoardSquare> find_squares(const std::vector<std::vector<int>>& adj);
