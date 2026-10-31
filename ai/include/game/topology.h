#pragma once
#include "game/selector.h"
#include <vector>
#include <set>
#include <map>
#include <string>
#include <utility>

// Mirrors shared/topology.ts.

// Mirrors shared/topology.ts's zeroAdj().
std::vector<std::vector<int>> zero_adj(int N);

// The per-board element type of shared/topology.ts's mergeBoards() - the intermediate form
// merge_boards()'s recursive callers (sierpinski_rec(), node_edge_merge_flake_rec()) build on
// before wrapping the result into a BoardConfig. An empty `labels` stands in for TS's omitted one.
struct RawBoard {
    std::vector<std::vector<unsigned>> pos;
    std::vector<std::vector<int>> adj;
    std::map<std::string, int> labels;
};

// Mirrors shared/topology.ts's mergeBoards(), returning (combined board, maps) as a pair.
std::pair<RawBoard, std::vector<std::vector<int>>> merge_boards(
    const std::vector<RawBoard>& boards,
    const std::vector<std::pair<std::pair<int,int>, std::pair<int,int>>>& merges);

// Mirrors shared/topology.ts's AdjacencyList.
using AdjacencyList = std::vector<std::set<int>>;

// Mirrors shared/topology.ts's toAdjacencyList(); std::set membership checks are O(log degree)
// rather than TS Set's O(1).
AdjacencyList to_adjacency_list(const std::vector<std::vector<int>>& adj);

// Mirrors shared/topology.ts's findSimplices().
std::vector<BoardSimplex> find_simplices(const std::vector<std::vector<int>>& adj, int n);

// Mirrors shared/topology.ts's findQuads().
std::vector<BoardQuad> find_quads(const std::vector<std::vector<int>>& adj);
