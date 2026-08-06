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

// Splits every edge of bc into split_n sub-edges, inserting split_n-1 evenly-spaced new nodes
// along each original edge. Mirrors shared/boardConfig.ts's edgeSplit() exactly, but computed with
// plain integer arithmetic instead of floats: original embed coordinates are scaled by split_n
// first, so new node k (1 <= k < split_n) along edge (i, j) is embed[i]*split_n + k*(embed[j] -
// embed[i]) - always an exact integer, no rounding, since k*(embed[j]-embed[i]) is a multiple of 1
// by construction (unlike a fractional split_n-th of an arbitrary distance).
BoardConfig edge_split(const BoardConfig& bc, int split_n);

// Rectifies bc: one new node per original edge, at that edge's midpoint (embed[i]+embed[j], the
// exact-integer "scaled-by-2" convention - same trick as edge_split, but no division needed here
// since summing two already-doubled positions and halving the sum is the same as summing the two
// undoubled positions directly). Two new nodes are connected iff their edges are angularly adjacent
// around a shared original vertex v: normalize each incident edge's direction from v to a unit
// vector (the one non-integer-exact step in this function - see geometry.h's doc comment for why
// there's no exact-integer alternative), then connect the pairs joined by an edge on the convex hull
// of v's directions (see geometry.h's convex_hull_edges). Mirrors shared/boardConfig.ts's rectify()
// exactly, including its own fix for a scale-mismatch bug (direction vectors are computed against
// 2*embed[v], not embed[v] directly, to match the doubled scale of the midpoint positions). Throws if
// bc.emb_dim == 0 - there are no real coordinates to compute edge directions from.
BoardConfig rectify(const BoardConfig& bc);

// Merges every pair of nodes whose Euclidean distance (in the natural embedding dimension) is
// strictly less than dist into a single node, via quotient_board. Closeness is transitive under
// quotient_board's union-find, so a chain of nodes each within dist of the next all collapse into
// one node, not just each individual close pair. Mirrors shared/boardConfig.ts's mergeClose()
// exactly. Unlike edge_split/rectify's own node *positions*, dist and the distance test are
// floating point - embed coordinates stay exact integers throughout, but there's no exact-integer
// way to compare an arbitrary real-valued threshold against a Euclidean distance (same reasoning as
// rectify()'s direction normalization - see geometry.h's doc comment). Throws if bc.emb_dim == 0 -
// there are no real coordinates to compute a distance from.
BoardConfig merge_close(const BoardConfig& bc, double dist);

// Replaces every triangle (3 mutually-adjacent, distinct vertices - see topology.h's
// find_triangles) in bc with a triangular_board(w)-shaped lattice, gluing new corners back to the
// original vertices and gluing any triangles that share an edge along that shared boundary too.
// Mirrors shared/boardConfig.ts's triangleForm() exactly, with one difference: the TS version
// computes real (generally irrational, since it divides by w-1) node positions, but every C++
// board with a genuine (non-zero) emb_dim is an exact-integer invariant throughout this file, so
// this always produces an emb_dim = 0 board instead (same reasoning as regular_polygon_board /
// tetrahedron_board / dodecahedron_board / icosahedron_board) - adjacency only, regardless of
// whether bc itself had a real embedding.
BoardConfig triangle_form(const BoardConfig& bc, int w);

// Replaces every square (4 distinct vertices forming a cycle with no diagonal edges - see
// topology.h's find_squares) in bc with a w-by-w grid, the same way triangle_form replaces
// triangles with triangular_board(w)-shaped lattices - see that function's own doc comment for why
// this always produces an emb_dim = 0 board. Mirrors shared/boardConfig.ts's squareForm() exactly.
BoardConfig square_form(const BoardConfig& bc, int w);

// The Cartesian (box) product of two board configs: N = bc1.N * bc2.N, one new node per pair (i, j)
// (i from bc1, j from bc2), at the concatenated position embed[i] followed by embed[j] (emb_dim =
// bc1.emb_dim + bc2.emb_dim - both stay exact integers here, unlike merge_close/rectify, since
// concatenation needs no arithmetic at all). (i, j) is adjacent to (i2, j2) iff exactly one of:
// i == i2 and j is adjacent to j2 in bc2, or j == j2 and i is adjacent to i2 in bc1 (the standard
// graph Cartesian product - e.g. cube_lattice_board(w, h, d) is, up to embedding, the product of three
// path graphs). Mirrors shared/boardConfig.ts's product() for N/adj/embed exactly; unlike the TS
// side, there's no projMat to construct here - see BoardConfig's own fields above, C++ never renders.
BoardConfig product(const BoardConfig& bc1, const BoardConfig& bc2);

// A BoardConfig-transforming operation - see apply_modifier/apply_modifiers. Mirrors
// shared/boardConfig.ts's BoardModifier, minus a C++ port of parseModifier(name, args): that parses
// interactive command text, which has no analog here - train.cpp/server.cpp get their whole
// GameConfig (including board_modifiers) from a JSON file/HTTP body via parse_game_cfg
// (training/self_play.cpp) instead.
enum class ModifierKind {
    Rectify, EdgeSplit, MergeClose, TriangleForm, SquareForm, Prod, BeginProd, EndProd
};
struct BoardModifier {
    ModifierKind kind;
    // split_n is reused for TriangleForm/SquareForm's own single int parameter (its w) - all three
    // are "one plain int argument" modifiers, same as Prod/BeginProd already share board_type/
    // board_args below.
    int split_n = 0;         // meaningful when kind == ModifierKind::EdgeSplit / TriangleForm/SquareForm
    double dist = 0.0;             // only meaningful when kind == ModifierKind::MergeClose
    std::string board_type;        // only meaningful when kind == ModifierKind::Prod / BeginProd
    std::vector<int> board_args;   // only meaningful when kind == ModifierKind::Prod / BeginProd

    // Needed for std::vector<BoardModifier>::operator== (used by weak_equal, training/self_play.cpp)
    // - C++17 has no defaulted struct equality (that's a C++20 feature), so this is spelled out.
    bool operator==(const BoardModifier& other) const {
        return kind == other.kind && split_n == other.split_n && dist == other.dist &&
               board_type == other.board_type && board_args == other.board_args;
    }
};

// Applies modifier to bc, dispatching to rectify / edge_split / merge_close / triangle_form /
// square_form / product (Prod builds a fresh board from its own board_type/board_args via
// build_board_config, then multiplies it into bc - a one-shot immediate product, unlike
// BeginProd/EndProd below). Does NOT accept
// BeginProd/EndProd - those have no meaning applied to a single board in isolation (BeginProd starts
// a whole new board for apply_modifiers to build up separately - potentially with further modifiers
// of its own before the product happens - and EndProd's product() needs that suspended outer board
// back too) - see apply_modifiers, the only valid way to apply a modifier list containing them.
// Mirrors shared/boardConfig.ts's applyModifier() exactly, including this same rejection.
BoardConfig apply_modifier(const BoardConfig& bc, const BoardModifier& modifier);

// Applies every modifier in modifiers, in order, to bc. Most modifiers just transform the "current"
// board via apply_modifier, but BeginProd/EndProd (rejected by apply_modifier itself - see its own
// comment) are handled specially here, via a stack of boards suspended to be multiplied back in
// later: BeginProd pushes the current board onto the stack and starts a fresh "current" board (via
// build_board_config, from its board_type/board_args), so modifiers up to the matching EndProd
// transform this new board instead of the outer one; EndProd pops the suspended outer board and
// replaces "current" with product(outer, current). BeginProd/EndProd pairs may nest. Throws on an
// EndProd with no matching BeginProd, or if modifiers ends with an unmatched BeginProd. Mirrors
// shared/boardConfig.ts's applyModifiers() exactly.
BoardConfig apply_modifiers(const BoardConfig& bc, const std::vector<BoardModifier>& modifiers);

// A board with w nodes forming a simple line: node i is connected to node i + 1, at position [i]
// (emb_dim = 1 - unlike rectangular_board(w, 1), the useless constant second dimension is dropped).
BoardConfig linear_board(int w);

// A rectangular board with width w and height h. Each node is identified by
// (col, row) where 0 <= col < w, 0 <= row < h.
BoardConfig rectangular_board(int w, int h);

// A rectangular board with width w and height h where diagonally adjacent nodes
// are also connected, but only at every m-th square.
BoardConfig rectangular_diagonal_board(int w, int h, int m);

// A cubical board with width w, height h and depth d. Each node is identified
// by (col, row, slice) where 0 <= col < w, 0 <= row < h, 0 <= slice < d.
BoardConfig cube_lattice_board(int w, int h, int d);

// A hypercubical board with width w, height h, depth d and hyperdepth t. Each
// node is identified by (col, row, slice, hyperslice) where 0 <= col < w,
// 0 <= row < h, 0 <= slice < d, 0 <= hyperslice < t.
BoardConfig hypercube_board(int w, int h, int d, int t);

// A triangular board with side length w.
BoardConfig triangular_board(int w);

// A regular polygon with n edges (a simple n-cycle graph), n >= 3. Unlike every other board type
// here, a unit-edge-length regular n-gon has no exact-integer Cartesian embedding for general n
// (see shared/boardConfig.ts's regularPolygonBoard() and ai/Readme.md's now-resolved TODO note).
// Rather than force an approximate/scaled embedding that would behave inconsistently with every
// other board's "1 embed unit = 1 real unit" convention (see merge_close's own doc comment), this
// uses emb_dim = 0 and an empty embed[] per node - adjacency (a plain n-cycle) is exact and
// complete on its own, and only CNN/UNet actually need real 2D coordinates (see their own guards
// against emb_dim != 2 in cnn.cpp/unet.cpp) - neither is grid-shaped anyway, so nothing is lost.
BoardConfig regular_polygon_board(int n);

// A regular tetrahedron: 4 vertices, all mutually adjacent (K4), 6 edges. Same emb_dim = 0 / empty
// embed[] approach as regular_polygon_board, for the same reason - see shared/boardConfig.ts's
// tetrahedronBoard() for the coordinates this mirrors, adjacency only. A side-length-w subdivision
// of its 4 triangular faces is built via triangle_form(w), not in here directly - see its own doc
// comment; find_triangles finds exactly this board's 4 faces, since every 3-subset of K4's
// vertices is a triangle.
BoardConfig tetrahedron_board();

// A regular dodecahedron: 20 vertices, 12 pentagonal faces, 30 edges (every vertex degree 3).
// Same emb_dim = 0 / empty embed[] approach as regular_polygon_board, for the same reason
// (dodecahedron vertices are inherently irrational - see shared/boardConfig.ts's
// dodecahedronBoard() for the coordinates/connectivity derivation this mirrors, adjacency only).
BoardConfig dodecahedron_board();

// A regular icosahedron: 12 vertices, 20 triangular faces, 30 edges (every vertex degree 5).
// Same emb_dim = 0 / empty embed[] approach, for the same reason - see shared/boardConfig.ts's
// icosahedronBoard() for the coordinates/connectivity derivation this mirrors, adjacency only.
BoardConfig icosahedron_board();

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

// Triangle-inflated variant of snub_square_board: same w x h grid of g x g squares (same per-cell
// 45-degree-integer shape), but every square-to-square gap is filled by an actual side-length-g
// triangular sub-board (same construction as triangular_board(g)) instead of a single glued corner
// plus a bare edge - mirrors shared/boardConfig.ts's snubSquareTriBoard() for the connectivity. A
// triangle's boundary nodes copy their glued square corner's embed value exactly; interior nodes are
// placed by rounded linear interpolation between them. Unlike snub_square_board, each square's
// per-cell placement offset depends on both x and y (not just its own axis) - required so that two
// triangles gluing to the same pair of squares from opposite sides derive identical interpolated
// values before quotient_board ever runs, not just after merging (see the .cpp for the derivation).
BoardConfig snub_square_tri_board(int w, int h, int g);

// A board of w x h squares each rotated 45 degrees, arranged in a rectangle.
// The squares have the usual square topology. The closest nodes of two adjacent
// squares are glued together (merged into one node).
BoardConfig glue_twisted_square_board(int w, int h, int g);

// A board of w x h squares each rotated 45 degrees, arranged in a rectangle.
// The squares have the usual square topology. The closest nodes of two adjacent
// squares are connected by an edge.
BoardConfig twisted_square_board(int w, int h, int g);

// Dispatches to the board builder above matching `kind` ("line" | "rect" | "rectd" |
// "cublat" | "hcub" | "tri" | "regpoly" | "tetra" | "dodeca" | "icosa" | "trihex" |
// "hex" | "hexdel" | "snubsq" | "snubsqtri" | "twsq" | "gtsq" - matches
// shared/types.ts's GameConfig.boardType strings), passing `args` as that
// builder's positional parameters. Throws std::runtime_error for an unknown
// kind. Shared by
// train.cpp (via GameConfig::board_type/board_args, loaded from
// --game-config) and server.cpp (via the /move request's boardType/boardArgs)
// so there's one board-kind switch instead of two near-identical copies.
BoardConfig build_board_config(const std::string& kind, const std::vector<int>& args);
