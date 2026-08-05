#pragma once
#include <vector>
#include <set>
#include <array>

// Graph-topology utilities operating on plain N×N adjacency matrices (the same representation as
// BoardConfig::adj), independent of any board-specific geometry. Mirrors shared/topology.ts.

// An adjacency-list view of a graph: list[i] is the set of i's neighbors.
using AdjacencyList = std::vector<std::set<int>>;

// Converts an N×N adjacency matrix into an adjacency list, each node's neighbors stored as a
// std::set (not a vector) so membership checks - the hot path for both find_triangles/find_squares
// below - are O(log degree) instead of O(degree).
AdjacencyList to_adjacency_list(const std::vector<std::vector<int>>& adj);

// Finds every triangle (3 distinct, pairwise-adjacent vertices) in adj, each reported exactly once
// as {u, v, w} with u < v < w. Mirrors shared/topology.ts's findTriangles() exactly - see its own
// doc comment for why the increasing-order search both dedupes and stays efficient on sparse graphs.
std::vector<std::array<int, 3>> find_triangles(const std::vector<std::vector<int>>& adj);

// Finds every "square" - 4 distinct vertices a, b, c, d forming a cycle a-b-c-d-a whose two
// diagonals a-c and b-d are BOTH absent - each reported exactly once as {a, b, c, d} in that cycle
// order. Mirrors shared/topology.ts's findSquares() exactly - see its own doc comment for the
// common-neighbor-pair search and its deduplication rule.
std::vector<std::array<int, 4>> find_squares(const std::vector<std::vector<int>>& adj);
