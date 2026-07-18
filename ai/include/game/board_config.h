#pragma once
#include <vector>
#include <utility>
#include <string>

struct BoardConfig {
    int N;
    std::vector<std::vector<int>> adj;       // N×N symmetric adjacency matrix (0/1)
    unsigned emb_dim;                        // embedding dimension of embed coordinates
    std::vector<std::vector<unsigned>> embed; // N×emb_dim node positions
};

// Glue pairs of nodes in quot together. The position of the merged node is the
// average of its predecessors' positions.
BoardConfig quotient_board(const BoardConfig& bc,
                           const std::vector<std::pair<int,int>>& quot);

// A rectangular board with width w and height h. Each node is identified by
// (col, row) where 0 <= col < w, 0 <= row < h.
BoardConfig rectangular_board(int w, int h);

// A rectangular board with width w and height h where diagonally adjacent nodes
// are also connected, but only at every m-th square.
BoardConfig rectangular_diagonal_board(int w, int h, int m);

// A cubical board with width w, height h and depth d. Each node is identified
// by (col, row, slice) where 0 <= col < w, 0 <= row < h, 0 <= slice < d.
BoardConfig cubical_board(int w, int h, int d);

// A hypercubical board with width w, height h, depth d and hyperdepth t. Each
// node is identified by (col, row, slice, hyperslice) where 0 <= col < w,
// 0 <= row < h, 0 <= slice < d, 0 <= hyperslice < t.
BoardConfig hypercube_board(int w, int h, int d, int t);

// A triangular board with side length w.
BoardConfig triangular_board(int w);

// A board of w x h squares each rotated 45 degrees, arranged in a rectangle.
// The squares have the usual square topology. The closest nodes of two adjacent
// squares are glued together (merged into one node).
BoardConfig glue_twisted_square_board(int w, int h, int g);

// A board of w x h squares each rotated 45 degrees, arranged in a rectangle.
// The squares have the usual square topology. The closest nodes of two adjacent
// squares are connected by an edge.
BoardConfig twisted_square_board(int w, int h, int g);

// Dispatches to the board builder above matching `kind` ("rect" | "rectd" |
// "cub" | "hcub" | "tri" | "twsq" | "gtsq" - matches shared/types.ts's
// GameConfig.boardType strings), passing `args` as that builder's positional
// parameters. Throws std::runtime_error for an unknown kind. Shared by
// train.cpp (via GameConfig::board_type/board_args, loaded from
// --game-config) and server.cpp (via the /move request's boardType/boardArgs)
// so there's one board-kind switch instead of two near-identical copies.
BoardConfig build_board_config(const std::string& kind, const std::vector<int>& args);
