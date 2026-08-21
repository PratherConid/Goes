#pragma once
#include "game/selector.h"
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
// along each original edge. Mirrors shared/boardConfig.ts's edgeSplit(), but computed with
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
// of v's directions (see geometry.h's convex_hull_edges). Mirrors shared/boardConfig.ts's rectify(),
// including its own fix for a scale-mismatch bug (direction vectors are computed against
// 2*embed[v], not embed[v] directly, to match the doubled scale of the midpoint positions). Throws if
// bc.emb_dim == 0 - there are no real coordinates to compute edge directions from.
BoardConfig rectify(const BoardConfig& bc);

// Merges every pair of nodes whose Euclidean distance (in the natural embedding dimension) is
// strictly less than dist into a single node, via quotient_board. Closeness is transitive under
// quotient_board's union-find, so a chain of nodes each within dist of the next all collapse into
// one node, not just each individual close pair. Mirrors shared/boardConfig.ts's mergeClose().
// Unlike edge_split/rectify's own node *positions*, dist and the distance test are
// floating point - embed coordinates stay exact integers throughout, but there's no exact-integer
// way to compare an arbitrary real-valued threshold against a Euclidean distance (same reasoning as
// rectify()'s direction normalization - see geometry.h's doc comment). Throws if bc.emb_dim == 0 -
// there are no real coordinates to compute a distance from.
BoardConfig merge_close(const BoardConfig& bc, double dist);

// Replaces every triangle (3 mutually-adjacent, distinct vertices - see topology.h's
// find_triangles) in bc with a triangular_board(w)-shaped lattice, gluing new corners back to the
// original vertices and gluing any triangles that share an edge along that shared boundary too.
// Mirrors shared/boardConfig.ts's triangleForm(), with one difference: the TS version
// computes real (generally irrational, since it divides by w-1) node positions, but every C++
// board with a genuine (non-zero) emb_dim is an exact-integer invariant throughout this file, so
// this always produces an emb_dim = 0 board instead (same reasoning as regular_polygon_board /
// tetrahedron_board / dodecahedron_board / icosahedron_board) - adjacency only, regardless of
// whether bc itself had a real embedding.
BoardConfig triangle_form(const BoardConfig& bc, int w);

// Replaces every square (4 distinct vertices forming a cycle with no diagonal edges - see
// topology.h's find_squares) in bc with a w-by-w grid, the same way triangle_form replaces
// triangles with triangular_board(w)-shaped lattices - see that function's own doc comment for why
// this always produces an emb_dim = 0 board. Mirrors shared/boardConfig.ts's squareForm().
BoardConfig square_form(const BoardConfig& bc, int w);

// Adds one new node connected to every existing node of bc, at the barycenter of bc's existing
// node positions - mirrors shared/boardConfig.ts's globalCentralize() connectivity, but
// (like triangle_form/square_form above) always produces an emb_dim = 0 board regardless of bc's
// own embedding: the barycenter is generally not an exact integer (it divides by N), and even
// where it happens to be, a hub node adjacent to the *entire* board doesn't fit the local-grid
// shape CNN/UNet expect anyway (see cnn.cpp/unet.cpp) - so nothing real is lost by dropping to
// adjacency-only here, same reasoning as those two functions' own doc comments.
BoardConfig global_centralize(const BoardConfig& bc);

// Replaces every square (4 distinct vertices forming a cycle with no diagonal edges - see
// topology.h's find_squares, same squares square_form finds) with an octahedron: two new "apex"
// nodes, one on each side of the square, each connected to all 4 of that square's corners - the
// square's own 4-cycle edges (already present, untouched) become the octahedron's equatorial ring,
// and the two apexes are NOT connected to each other (antipodal, like octahedron_board's own apex
// pairs - see that function's doc comment for why a plain square graph plus two such apex nodes is
// exactly an octahedron's edge set). Mirrors shared/boardConfig.ts's sqOctarize() connectivity,
// but (like triangle_form/square_form/global_centralize above) always produces an
// emb_dim = 0 board regardless of bc's own embedding: the TS side gives each apex a real position
// on a genuinely new dimension, offset by the (generally irrational, since it's a Euclidean
// distance) average corner-to-barycenter distance - there is no exact-integer equivalent here, and
// C++ never renders anyway (see BoardConfig's own fields, board_config.h's top comment), so nothing
// is lost by dropping to adjacency-only.
BoardConfig sq_octarize(const BoardConfig& bc);

// A no-op: mirrors shared/boardConfig.ts's scaleBoard() in name only. embed[] coordinates here must
// stay exact integers (see merge_close's own doc comment) and are otherwise unrelated to gameplay -
// C++ never renders (see BoardConfig's own fields, board_config.h's top comment) - so multiplying
// them by an arbitrary factor would only lose precision for no benefit. Returns bc unchanged.
BoardConfig scale_board(const BoardConfig& bc, double factor);

// The Cartesian (box) product of two board configs: N = bc1.N * bc2.N, one new node per pair (i, j)
// (i from bc1, j from bc2), at the concatenated position embed[i] followed by embed[j] (emb_dim =
// bc1.emb_dim + bc2.emb_dim - both stay exact integers here, unlike merge_close/rectify, since
// concatenation needs no arithmetic at all). (i, j) is adjacent to (i2, j2) iff exactly one of:
// i == i2 and j is adjacent to j2 in bc2, or j == j2 and i is adjacent to i2 in bc1 (the standard
// graph Cartesian product - e.g. cube_lattice_board(w, h, d) is, up to embedding, the product of three
// path graphs). Mirrors shared/boardConfig.ts's product() for N/adj/embed; unlike the TS
// side, there's no projMat to construct here - see BoardConfig's own fields above, C++ never renders.
BoardConfig product(const BoardConfig& bc1, const BoardConfig& bc2);

// The subgraph induced by sel (evaluated via game/selector.h's select_node): keeps only the nodes
// sel selects - compacted to a fresh 0..k-1 index range, in ascending original-index order,
// embed/emb_dim otherwise untouched - with two surviving nodes adjacent iff they were already
// adjacent in bc. Unlike quotient_board/merge_close, nothing is merged or repositioned; a
// non-selected node's own incident edges are simply dropped along with it. Mirrors
// shared/boardConfig.ts's nodeInducedSubgraph().
BoardConfig node_induced_subgraph(const BoardConfig& bc, const Selector& sel);

// The subgraph induced by sel (evaluated via game/selector.h's select_edge): keeps only the edges
// sel selects, and only the nodes touched by at least one of them - compacted to a fresh 0..k-1
// index range, in ascending original-index order, embed/emb_dim otherwise untouched. Unlike
// node_induced_subgraph (which keeps every original edge between two surviving nodes, since it
// starts from a node selection), this keeps exactly the selected edges themselves - the standard
// graph-theory distinction between a node-induced and an edge-induced subgraph - so a node with no
// selected incident edge doesn't survive at all, even if it's adjacent to other surviving nodes via
// a non-selected edge. Mirrors shared/boardConfig.ts's edgeInducedSubgraph().
BoardConfig edge_induced_subgraph(const BoardConfig& bc, const Selector& sel);

// One positional board-construction arg, tagged with its own kind - mirrors shared/boardConfig.ts's
// BoardArgType/BoardArgEntry (see that TS type's own doc comment for the full rationale: exactly
// one BoardArgEntry per positional arg, never a flattened/anonymous number list, since a variable-
// length list arg's own length can depend on ANOTHER arg's value - menger_sponge_flake_board()'s
// own indicator, whose length is dim+1 - making it impossible to unambiguously re-derive the
// grouping from a flat array alone). `value` is meaningful iff `kind == Number`; `values` iff
// `kind == CommaSeparatedNumbers` or `ZeroOneList` (both are plain int lists once parsed - the
// distinction only matters for the ORIGINAL command-line token's own syntax, comma-joined vs a bare
// 0/1 string, which C++ never re-parses from text - see board_arg_number()/board_arg_list()).
enum class BoardArgKind { Number, CommaSeparatedNumbers, ZeroOneList };
struct BoardArgEntry {
    BoardArgKind kind = BoardArgKind::Number;
    int value = 0;
    std::vector<int> values;

    bool operator==(const BoardArgEntry& other) const {
        return kind == other.kind && value == other.value && values == other.values;
    }
};

// Mirrors shared/boardConfig.ts's numArg()/csvArg()/zolArg() - shorthand for building a BoardArgEntry
// by hand (as opposed to parsing one from JSON - see training/self_play.cpp's own to_json/from_json).
inline BoardArgEntry num_arg(int value) { return BoardArgEntry{ BoardArgKind::Number, value, {} }; }
inline BoardArgEntry csv_arg(std::vector<int> values) {
    return BoardArgEntry{ BoardArgKind::CommaSeparatedNumbers, 0, std::move(values) };
}
inline BoardArgEntry zol_arg(std::vector<int> values) {
    return BoardArgEntry{ BoardArgKind::ZeroOneList, 0, std::move(values) };
}

// Mirrors shared/boardConfig.ts's boardArgNumber()/boardArgList() - read a BoardArgEntry back to the
// single number/number list it must hold, asserting (not throwing - see their own .cpp definitions)
// if it's actually the other kind. Shared by build_board_config's own dispatch below.
int board_arg_number(const BoardArgEntry& e);
const std::vector<int>& board_arg_list(const BoardArgEntry& e);

// Mirrors shared/boardConfig.ts's formatBoardArgEntry() - the command-line token e was parsed from
// (Number -> the plain value; CommaSeparatedNumbers -> comma-joined; ZeroOneList -> concatenated
// digits, no separator). Used for diagnostic printouts (train.cpp).
std::string format_board_arg_entry(const BoardArgEntry& e);

// A BoardConfig-transforming operation - see apply_modifier/apply_modifiers. Mirrors
// shared/boardConfig.ts's BoardModifier, minus a C++ port of parseModifier(name, args)/
// parseModifiers(text) (which parse interactive command text - no analog here): train.cpp/
// server.cpp get their whole GameConfig (including board_modifiers) from a JSON file/HTTP body via
// parse_game_cfg (training/self_play.cpp) instead, which reads the already-tree-shaped JSON
// directly (see parse_game_cfg's own parse_board_modifiers()).
enum class ModifierKind {
    Rectify, EdgeSplit, MergeClose, TriangleForm, SquareForm, Prod, Repeat,
    GlobalCentralize, SqOctarize, Scale, NodeInducedSubgraph, EdgeInducedSubgraph
};
struct BoardModifier {
    ModifierKind kind;
    // split_n is reused for TriangleForm/SquareForm's own single int parameter (its w), and for
    // Repeat's own single int parameter (its count) - all four are "one plain int argument"
    // modifiers, same as Prod already shares board_type/board_args below.
    int split_n = 0;   // meaningful when kind == ModifierKind::EdgeSplit/TriangleForm/SquareForm/Repeat
    // dist is reused for Scale's own single double parameter (its factor) - both are "one plain
    // double argument" modifiers.
    double dist = 0.0;             // meaningful when kind == ModifierKind::MergeClose / Scale
    std::string board_type;        // only meaningful when kind == ModifierKind::Prod
    std::vector<BoardArgEntry> board_args; // only meaningful when kind == ModifierKind::Prod
    // Prod/Repeat's own nested modifier list, applied (via apply_modifiers) to the fresh board Prod
    // builds from board_type/board_args, or repeatedly to the current board for Repeat - this is
    // what makes BoardModifier tree-shaped, mirroring shared/boardConfig.ts's own Prod/Repeat
    // (which replaced a separate, non-tree-shaped Prod/BeginProd/EndProd trio - see that file's own
    // history/doc comments). A plain std::vector<BoardModifier> member of BoardModifier itself is
    // legal (no forward-declaration/pointer indirection needed) since C++17 relaxed std::vector's
    // completeness requirement for its own element type.
    std::vector<BoardModifier> modifiers; // meaningful when kind == ModifierKind::Prod / Repeat
    // Only meaningful when kind == ModifierKind::NodeInducedSubgraph / EdgeInducedSubgraph - see
    // game/selector.h.
    Selector sel;

    // Needed for std::vector<BoardModifier>::operator== (used by weak_equal, training/self_play.cpp)
    // - C++17 has no defaulted struct equality (that's a C++20 feature), so this is spelled out.
    bool operator==(const BoardModifier& other) const {
        return kind == other.kind && split_n == other.split_n && dist == other.dist &&
               board_type == other.board_type && board_args == other.board_args &&
               modifiers == other.modifiers && sel == other.sel;
    }
};

// Applies modifier to bc, dispatching to rectify / edge_split / merge_close / triangle_form /
// square_form / global_centralize / sq_octarize / scale_board / node_induced_subgraph /
// edge_induced_subgraph (Prod builds a fresh board from its own board_type/board_args via
// build_board_config, applies its own nested modifiers to that fresh board via apply_modifiers, then
// multiplies the result into bc; Repeat applies its own nested modifiers to bc, via apply_modifiers,
// split_n times in a row). Mirrors shared/boardConfig.ts's applyModifier().
BoardConfig apply_modifier(const BoardConfig& bc, const BoardModifier& modifier);

// Applies every modifier in modifiers, in order, to bc. Mirrors shared/boardConfig.ts's
// applyModifiers() - now a plain fold, since BoardModifier's own Prod/Repeat (unlike the old
// Prod/BeginProd/EndProd trio) are already self-contained tree nodes needing no cross-modifier stack
// here.
BoardConfig apply_modifiers(const BoardConfig& bc, const std::vector<BoardModifier>& modifiers);

// A board with w nodes forming a simple line: node i is connected to node i + 1, at position [i]
// (emb_dim = 1 - unlike rectangular_board(w, 1), the useless constant second dimension is dropped).
BoardConfig linear_board(int w);

// A rectangular board with width w and height h - the meshdim=2 case of hypercuboid_board below
// (so nothing is ever excluded). Each node is identified by (col, row) where 0 <= col < w,
// 0 <= row < h.
BoardConfig rectangular_board(int w, int h);

// A rectangular board with width w and height h where diagonally adjacent nodes
// are also connected, but only at every m-th square.
BoardConfig rectangular_diagonal_board(int w, int h, int m);

// A cubical board with width w, height h and depth d - the meshdim=3 case of hypercuboid_board
// below (so nothing is ever excluded). Each node is identified by (col, row, slice) where
// 0 <= col < w, 0 <= row < h, 0 <= slice < d.
BoardConfig cube_lattice_board(int w, int h, int d);

// Mirrors shared/boardConfig.ts's hypercuboidBoard() - the meshdim-skeleton of a
// dims.size()-dimensional hypercuboid with dims[i] points along axis i: a node survives iff at
// most `meshdim` of its coordinates are strictly interior to their own axis (see the .cpp file's
// own comment for the full construction and what meshdim means geometrically). rectangular_board
// (above)/cube_lattice_board (above) are now just this function called with meshdim equal to
// their own full dimension count and 2/3 dims.
BoardConfig hypercuboid_board(int meshdim, const std::vector<int>& dims);

// A triangular board with side length w.
BoardConfig triangular_board(int w);

// Mirrors shared/boardConfig.ts's sierpinskiSimplex()/sierpinskiRec()/mergeBoards() (n >= 1; n=1:
// unit dim-simplex; n>1: dim+1 copies of n-1 glued at touching corners via the same dim+1-way
// merge - see the .cpp file's merge_boards/sierpinski_rec), with one simplification: the
// TS side's regularSimplexCoords() is real-valued and centroid-at-origin (irrational for dim >= 2 -
// no exact-integer analog, since BoardConfig::embed here is exact-integer only, see merge_close's
// own doc comment), so this instead places corner 0 at the origin and corner k (1 <= k <= dim) at
// a standard basis vector scaled by its own side length (see the .cpp file's own comment).
BoardConfig sierpinski_simplex_board(int dim, int n);

// A regular polygon with n edges (a simple n-cycle graph), n >= 3. Unlike every other board type
// here, a unit-edge-length regular n-gon has no exact-integer Cartesian embedding for general n
// (see shared/boardConfig.ts's regularPolygonBoard() and ai/Readme.md's now-resolved TODO note).
// Rather than force an approximate/scaled embedding that would behave inconsistently with every
// other board's "1 embed unit = 1 real unit" convention (see merge_close's own doc comment), this
// uses emb_dim = 0 and an empty embed[] per node - adjacency (a plain n-cycle) is exact and
// complete on its own, and only CNN/UNet actually need real 2D coordinates (see their own guards
// against emb_dim != 2 in cnn.cpp/unet.cpp) - neither is grid-shaped anyway, so nothing is lost.
BoardConfig regular_polygon_board(int n);

// A star graph: 1 center node connected to n outer nodes (outer nodes are not connected to each
// other). Same emb_dim = 0 / empty embed[] approach as regular_polygon_board, for the same reason
// (the outer nodes sit at angle 2*pi*k/n around the center, irrational for general n - see
// shared/boardConfig.ts's starBoard() for the coordinates this mirrors, adjacency only).
BoardConfig star_board(int n);

// A regular tetrahedron: 4 vertices, all mutually adjacent (K4), 6 edges. Same emb_dim = 0 / empty
// embed[] approach as regular_polygon_board, for the same reason - see shared/boardConfig.ts's
// tetrahedronBoard() for the coordinates this mirrors, adjacency only. A side-length-w subdivision
// of its 4 triangular faces is built via triangle_form(w), not in here directly - see its own doc
// comment; find_triangles finds exactly this board's 4 faces, since every 3-subset of K4's
// vertices is a triangle.
BoardConfig tetrahedron_board();

// A regular octahedron: the n=3 case of orthoplex_board() below - 6 vertices, 12 edges, 8
// triangular faces (every vertex degree 4). A side-length-w subdivision of its 8 triangular faces
// can be applied via triangle_form(w).
BoardConfig octahedron_board();

// Mirrors shared/boardConfig.ts's orthoplexBoard(), with one simplification: rather than the TS
// side's real-valued +-1/sqrt(2) coordinates, this uses only the integer values {0, 1, 2} - vertex
// 2k (the "+" pole on axis k) has coordinate k = 2, vertex 2k+1 (the "-" pole) has coordinate k =
// 1, every other coordinate 0; connectivity (every vertex adjacent to every other except its own
// antipode) is unaffected. n=3 is the regular octahedron (see octahedron_board() above).
BoardConfig orthoplex_board(int n);

// A uniform n-gonal antiprism: 2n vertices - a "top" n-gon (indices 0..n-1) and a "bottom" n-gon
// (indices n..2n-1, rotated by half a step), joined by 2n "slant" edges (top k to bottom k and
// bottom (k-1+n)%n, its two nearest bottom neighbors), forming 2n triangles around the belt in
// addition to the two n-gon rings. Same emb_dim = 0 / empty embed[] approach as regular_polygon_board,
// for the same reason (the circumradius/height are inherently irrational for essentially every n -
// see shared/boardConfig.ts's antiprismBoard() for the coordinates/connectivity derivation this
// mirrors, adjacency only). n=3 is graph-isomorphic to the regular octahedron (see octahedron_board()
// above), though this keeps its own top/bottom-triangle + belt vertex numbering rather than
// orthoplex_board()'s antipodal-pair one.
BoardConfig antiprism_board(int n);

// A regular dodecahedron: 20 vertices, 12 pentagonal faces, 30 edges (every vertex degree 3).
// Same emb_dim = 0 / empty embed[] approach as regular_polygon_board, for the same reason
// (dodecahedron vertices are inherently irrational - see shared/boardConfig.ts's
// dodecahedronBoard() for the coordinates/connectivity derivation this mirrors, adjacency only).
BoardConfig dodecahedron_board();

// A regular icosahedron: 12 vertices, 20 triangular faces, 30 edges (every vertex degree 5).
// Same emb_dim = 0 / empty embed[] approach, for the same reason - see shared/boardConfig.ts's
// icosahedronBoard() for the coordinates/connectivity derivation this mirrors, adjacency only.
BoardConfig icosahedron_board();

// Mirrors shared/boardConfig.ts's dodecahedronFlake() (n >= 1: n=1 is the plain dodecahedron_board()
// itself; n>1 recurses into 20 order-(n-1) copies, one per vertex, each sharing a full, growing
// edge - not just a point - with every adjacent copy; see the .cpp file's own comment for the full
// construction, including the C++ port of computeFlakeGlue()). Same emb_dim = 0 / empty embed[]
// approach as dodecahedron_board() itself (see that function's own doc comment) - but unlike every
// other board here, this never computes or stores node positions at all, not even internally as
// scratch data beyond the one-time glue-table search: the TS side's entire construction is built
// around a real-valued position transform (S_i(x) = r*x + c*verts[i]), which this drops entirely
// once the glue table itself is known, since the recursive merge only ever needs *which* nodes
// coincide (integer indices), never *where* they are.
BoardConfig dodecahedron_flake_board(int n);

// Mirrors shared/boardConfig.ts's icosahedronFlake() - same construction as
// dodecahedron_flake_board() above (see its own doc comment), just with 12 vertices/30 edges
// instead of 20 vertices/30 edges.
BoardConfig icosahedron_flake_board(int n);

// Mirrors shared/boardConfig.ts's octahedronFlake() - same construction as
// dodecahedron_flake_board() above (see its own doc comment), just with 6 vertices/12 edges instead
// of 20 vertices/30 edges. No embedding here either, same as dodecahedron_flake_board() (unlike
// octahedron_board() itself, which does have an integer embedding via orthoplex_board()) - this may
// change if a future need for octahedron flake's own positions comes up, but for now the C++ side
// mirrors dodeca/icosahedron flake's own no-embedding convention rather than octahedron_board()'s.
BoardConfig octahedron_flake_board(int n);

// Mirrors shared/boardConfig.ts's regularPolygonFlake() - same overall construction as
// dodecahedron_flake_board() above (see its own doc comment), just with n_sides vertices/edges
// instead of a fixed shape, so cached per n_sides rather than once (see fractal.cpp's own comment
// on regular_polygon_fractal_descr()). Unlike dodeca/icosa/octahedron (always merges by whole edge, or
// - octahedron - transitively equivalent to one), a regular polygon's base edges merge by a whole
// growing edge when n_sides is a multiple of 4, and by a single non-growing node otherwise (see the
// .cpp file's own comment on regular_polygon_flake_rc() for the closed-form r/c this forces). No
// embedding here either, same as regular_polygon_board() itself (also emb_dim=0, for the same
// inherently-irrational-coordinates reason).
BoardConfig regular_polygon_flake_board(int n_sides, int order);

// Mirrors shared/boardConfig.ts's centralRegularPolygonFlake() - same construction as
// regular_polygon_flake_board() above, with one further auxiliary sub-copy added at every recursion
// level (mirroring shared/boardConfig.ts's FractalDescr's own "beyond leafPos.length" subDescr
// entries - see the .cpp file's own comment on node_edge_merge_flake_rec()): a central copy glued to
// all n_sides regular copies at once, via a plain node merge (central copy's own vertex i coincides
// with regular copy i's own vertex i+n_sides/2 - closed-form, mirroring
// shared/boardConfig.ts's regularPolygonFractalDescr() derivation exactly; no distance search
// needed, unlike compute_flake_glue()/compute_node_glue() above). Only actually adds the central
// copy when n_sides is even and greater than 4 (same condition as the TS side, for the same
// reason); for any other n_sides, silently falls back to regular_polygon_flake_board()'s own
// construction rather than rejecting the input.
BoardConfig central_regular_polygon_flake_board(int n_sides, int order);

// Mirrors shared/boardConfig.ts's centralPentagonFlake() - the pentagon-specific special case
// centralRegularPolygonFlake()'s own even-n_sides construction can't cover (pentagon is odd, so a
// same-orientation central copy has no non-degenerate fixed point - see
// shared/boardConfig.ts's centralPentagonFractalDescr() for the full derivation this mirrors): a
// central copy at the SAME scale magnitude but OPPOSITE orientation, glued to every regular copy by
// a whole shared EDGE rather than a single node (closed-form vertex correspondence, same as
// central_regular_polygon_flake_board() above - no distance search needed). This is the one shape
// needing an EDGE_GLUE_OBJECT entry (fractal.h) whose two named corners exist but go unused by the
// plain (non-central) pentagon flake, which merges its own adjacent copies by a single node instead
// - only becoming load-bearing once the central copy needs it to glue against.
BoardConfig central_pentagon_flake_board(int order);

// Mirrors shared/boardConfig.ts's mengerSpongeFlake() - order >= 1, dim >= 1, `indicator` a
// length-(dim+1) list of 0/1 entries (see menger_fractal_descr()'s own doc comment in fractal.h for
// the full derivation - dim=1 is the Cantor set, dim=2 the Sierpinski carpet, dim=3 the classical
// Menger sponge, and so on; {0, 0, 1, 1} at dim=3 reproduces the classical Menger sponge exactly).
// order=1 is the plain unit dim-cube itself; order>1 recurses into one order-(order-1) copy per
// surviving sub-cube of `indicator`'s own subdivision, each sharing a whole growing sub-face - not
// just a point - with every touching copy. Same emb_dim = 0 / empty embed[] approach and no-position
// construction as dodecahedron_flake_board() above (see its own doc comment).
BoardConfig menger_sponge_flake_board(int order, int dim, const std::vector<int>& indicator);

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
// "cublat" | "hcub" | "tri" | "sier" | "regpoly" | "tetra" | "octa" | "ortho" | "ap" | "dodeca" |
// "icosa" | "dodflake" | "icoflake" | "octaflake" | "polyflake" | "cpolyflake" | "cpentflake" |
// "menger" | "trihex" | "hex" | "hexdel" | "snubsq" | "snubsqtri" | "twsq" | "gtsq" | "star" - matches
// shared/types.ts's GameConfig.boardType strings), reading each of `args` back via
// board_arg_number()/board_arg_list() as that builder's own positional parameters expect. Throws
// std::runtime_error for an unknown kind. Shared by
// train.cpp (via GameConfig::board_type/board_args, loaded from
// --game-config) and server.cpp (via the /move request's boardType/boardArgs)
// so there's one board-kind switch instead of two near-identical copies.
BoardConfig build_board_config(const std::string& kind, const std::vector<BoardArgEntry>& args);
