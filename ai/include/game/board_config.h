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

// A triangular-lattice board arranged in a hexagon shape, with d layers of triangles surrounding
// the central point. Not tiled by hexagons - see hex_board below for that. Cells are
// identified by axial coordinates (q, r) with max(|q|, |r|, |q+r|) <= d, embedded (shifted to
// non-negative) as (q+d, r+d); each cell connects to its up to six axial neighbors.
BoardConfig triangular_hex_board(int d);

// A board actually tiled by regular hexagons: a central hexagonal cell surrounded by d further
// layers of hexagonal cells (honeycomb topology - degree 3 in the interior, degree 2 on the
// boundary). Mirrors shared/boardConfig.ts's hexBoard() - see that function's doc comment for the
// full construction (carving the honeycomb out of triangular_hex_board's own triangular lattice by
// erasing one 3-coloring class, whose points mark the hexagonal faces' centers). Embedded (shifted
// to non-negative) as (q+2d+1, r+2d+1) - a vertex's axial (q, r) never strays more than 2d+1 from
// the origin, one hex-lattice step past the d-ringed centers' own [-2d, 2d] range.
BoardConfig hex_board(int d);

// A trihexagonal ("hexdel") tiling: hexagons and triangles alternate, 2 of each around every
// vertex (degree 4 in the interior) - d layers of hexagons, connected by triangles, surrounding a
// central hexagon. Mirrors shared/boardConfig.ts's trihexBoard() - see that function's doc comment
// for the full construction (same erase-a-sublattice technique as hex_board, but erasing only the
// coarser 1-of-4 "both axial coordinates even" sublattice, which leaves triangular faces intact
// alongside the hexagonal ones). Embedded (shifted to non-negative) as (q+2d+1, r+2d+1), same bound
// as hex_board's own embedding.
BoardConfig trihex_board(int d);

// A w x h grid of g x g squares, each rotated +-30 degrees in a checkerboard pattern, arranged as
// a snub square tiling. Mirrors shared/boardConfig.ts's snubSquareBoard() for the connectivity
// (each pair of horizontally/vertically adjacent squares shares one glued corner plus one
// triangle-connecting edge between two of their other corners - see that function's CONN table),
// but embeds nodes the same integer way as tilted_disconnected_square_board/
// glue_twisted_square_board (each g x g square rotated 45 degrees via integer arithmetic) rather
// than the TypeScript side's literal +-30-degree floating-point layout, since embed coordinates
// must be integers here.
BoardConfig snub_square_board(int w, int h, int g);

// A board of w x h squares each rotated 45 degrees, arranged in a rectangle.
// The squares have the usual square topology. The closest nodes of two adjacent
// squares are glued together (merged into one node).
BoardConfig glue_twisted_square_board(int w, int h, int g);

// A board of w x h squares each rotated 45 degrees, arranged in a rectangle.
// The squares have the usual square topology. The closest nodes of two adjacent
// squares are connected by an edge.
BoardConfig twisted_square_board(int w, int h, int g);

// Dispatches to the board builder above matching `kind` ("rect" | "rectd" |
// "cub" | "hcub" | "tri" | "trihex" | "hex" | "hexdel" | "snubsq" | "twsq" | "gtsq" - matches shared/types.ts's
// GameConfig.boardType strings), passing `args` as that builder's positional
// parameters. Throws std::runtime_error for an unknown kind. Shared by
// train.cpp (via GameConfig::board_type/board_args, loaded from
// --game-config) and server.cpp (via the /move request's boardType/boardArgs)
// so there's one board-kind switch instead of two near-identical copies.
BoardConfig build_board_config(const std::string& kind, const std::vector<int>& args);
