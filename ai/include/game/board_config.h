#pragma once
#include "game/selector.h"
#include <vector>
#include <utility>
#include <string>
#include <optional>
#include <set>

struct BoardConfig {
    int N;
    std::vector<std::vector<int>> adj;       // N×N symmetric adjacency matrix (0/1)
    unsigned emb_dim;                        // embedding dimension of embed coordinates
    std::vector<std::vector<unsigned>> embed; // N×emb_dim node positions
};

// Mirrors shared/boardConfig.ts's quotientBoard().
BoardConfig quotient_board(const BoardConfig& bc,
                           const std::vector<std::pair<int,int>>& quot);

// Mirrors shared/boardConfig.ts's edgeSplit(), in integer arithmetic: new node k along edge (i, j)
// is embed[i]*split_n + k*(embed[j] - embed[i]) - exact, no division.
BoardConfig edge_split(const BoardConfig& bc, int split_n);

// Mirrors shared/boardConfig.ts's rectify(). Midpoints (embed[i]+embed[j], the same doubled scale)
// stay exact integers; only the direction normalization is floating point (see geometry.h's doc
// comment). Throws if bc.emb_dim == 0 - there are no real coordinates to compute edge directions
// from.
BoardConfig rectify(const BoardConfig& bc);

// Mirrors shared/boardConfig.ts's truncate()'s connectivity but not its per-vertex-fraction
// positions (irrational in general - no exact-integer analog): every near-point instead sits at a
// fixed 1/3 or 2/3 of the way along its edge, on a 3x-scaled board (see the .cpp file's comment).
// Throws if bc.emb_dim == 0, same as rectify().
BoardConfig truncate(const BoardConfig& bc);

// Mirrors shared/boardConfig.ts's mergeClose(). embed coordinates stay exact integers throughout;
// only dist and the distance test are floating point. Throws if bc.emb_dim == 0 - there are no real
// coordinates to compute a distance from.
BoardConfig merge_close(const BoardConfig& bc, double dist);

// Mirrors shared/types.ts's FormSelector; `sel` nullopt = TS's omitted `sel`.
enum class FormSelKind { TriForm, QuadForm, QuadDiagForm, QuadKnightForm, QuadBishopForm };
struct FormSelector {
    FormSelKind kind = FormSelKind::TriForm;
    std::optional<Selector> sel;
    bool operator==(const FormSelector& other) const { return kind == other.kind && sel == other.sel; }
};

// Mirrors shared/boardConfig.ts's genericForm(), but always produces an emb_dim = 0 board
// (adjacency only, regardless of bc's own embedding): the TS side's interpolated lattice positions
// are generally non-integer, and every C++ board with a non-zero emb_dim is exact-integer.
BoardConfig generic_form(const BoardConfig& bc, int w, const std::vector<FormSelector>& sels, ClegRandom& random);

// Mirrors shared/boardConfig.ts's triangleForm(); emb_dim = 0 output, same as generic_form().
BoardConfig triangle_form(const BoardConfig& bc, int w, ClegRandom& random, std::optional<Selector> sel = std::nullopt);

// Mirrors shared/boardConfig.ts's quadForm(); emb_dim = 0 output, same as generic_form().
BoardConfig quad_form(const BoardConfig& bc, int w, ClegRandom& random, std::optional<Selector> sel = std::nullopt);

// Mirrors shared/boardConfig.ts's quadDiagForm(); emb_dim = 0 output, same as generic_form().
BoardConfig quad_diag_form(const BoardConfig& bc, int w, ClegRandom& random, std::optional<Selector> sel = std::nullopt);

// Mirrors shared/boardConfig.ts's quadKnightForm(); emb_dim = 0 output, same as generic_form().
BoardConfig quad_knight_form(const BoardConfig& bc, int w, ClegRandom& random, std::optional<Selector> sel = std::nullopt);

// Mirrors shared/boardConfig.ts's quadBishopForm(); emb_dim = 0 output, same as generic_form().
BoardConfig quad_bishop_form(const BoardConfig& bc, int w, ClegRandom& random, std::optional<Selector> sel = std::nullopt);

// Mirrors shared/types.ts's LocalReplaceSelector; `sel` nullopt = TS's omitted `sel`.
enum class LocalReplaceKind { QuadCentralize, SimpCentralize, QuadOctarize, QuadCentering, SimpCentering };
struct LocalReplaceSelector {
    LocalReplaceKind kind = LocalReplaceKind::QuadCentralize;
    int n = 0; // meaningful iff kind == SimpCentralize/SimpCentering
    std::optional<Selector> sel;
    bool operator==(const LocalReplaceSelector& other) const {
        return kind == other.kind && n == other.n && sel == other.sel;
    }
};

// Mirrors shared/boardConfig.ts's genericLocalReplace(), but always produces an emb_dim = 0 board
// (adjacency only, regardless of bc's own embedding): the TS side's barycenter/apex positions are
// generally non-integer - same reasoning as generic_form().
BoardConfig generic_local_replace(const BoardConfig& bc, const std::vector<LocalReplaceSelector>& selectors, ClegRandom& random);

// Mirrors shared/boardConfig.ts's simpCentralize().
BoardConfig simp_centralize(const BoardConfig& bc, int n, ClegRandom& random, std::optional<Selector> sel = std::nullopt);

// Mirrors shared/boardConfig.ts's simpCentering().
BoardConfig simp_centering(const BoardConfig& bc, int n, ClegRandom& random, std::optional<Selector> sel = std::nullopt);

// Mirrors shared/boardConfig.ts's triCentralize().
BoardConfig tri_centralize(const BoardConfig& bc, ClegRandom& random, std::optional<Selector> sel = std::nullopt);

// Mirrors shared/boardConfig.ts's triCentering().
BoardConfig tri_centering(const BoardConfig& bc, ClegRandom& random, std::optional<Selector> sel = std::nullopt);

// Mirrors shared/boardConfig.ts's quadCentralize().
BoardConfig quad_centralize(const BoardConfig& bc, ClegRandom& random, std::optional<Selector> sel = std::nullopt);

// Mirrors shared/boardConfig.ts's quadCentering().
BoardConfig quad_centering(const BoardConfig& bc, ClegRandom& random, std::optional<Selector> sel = std::nullopt);

// Mirrors shared/boardConfig.ts's globalCentralize() connectivity, but always produces an
// emb_dim = 0 board: the barycenter is generally not an exact integer (it divides by N), and a hub
// adjacent to the entire board doesn't fit the local-grid shape CNN/UNet expect anyway (see
// cnn.cpp/unet.cpp).
BoardConfig global_centralize(const BoardConfig& bc);

// Mirrors shared/boardConfig.ts's quadOctarize() connectivity, but always produces an emb_dim = 0
// board: the TS apex offset along the new dimension is a (generally irrational) Euclidean distance,
// with no exact-integer equivalent.
BoardConfig quad_octarize(const BoardConfig& bc, ClegRandom& random, std::optional<Selector> sel = std::nullopt);

// A no-op: mirrors shared/boardConfig.ts's scaleBoard() in name only. embed[] coordinates must stay
// exact integers and are otherwise unrelated to gameplay, so multiplying them by an arbitrary factor
// would only lose precision for no benefit. Returns bc unchanged.
BoardConfig scale_board(const BoardConfig& bc, double factor);

// Mirrors shared/boardConfig.ts's product().
BoardConfig product(const BoardConfig& bc1, const BoardConfig& bc2);

// Mirrors shared/boardConfig.ts's nodeInducedSubgraph().
BoardConfig node_induced_subgraph(const BoardConfig& bc, const std::set<int>& nodes);

// Mirrors shared/boardConfig.ts's edgeInducedSubgraph().
BoardConfig edge_induced_subgraph(const BoardConfig& bc, const std::vector<BoardEdge>& edges);

// Mirrors shared/boardConfig.ts's nodeEdgeInducedSubgraph().
BoardConfig node_edge_induced_subgraph(
    const BoardConfig& bc, const std::set<int>& nodes, const std::vector<BoardEdge>& edges);

// Mirrors shared/types.ts's BoardArgType/BoardArgEntry, as one struct rather than a union: `value`
// is meaningful iff `kind == Number`, `values` otherwise. C++ never parses board-arg tokens from
// text, so CommaSeparatedNumbers/ZeroOneList only differ as tags.
enum class BoardArgKind { Number, CommaSeparatedNumbers, ZeroOneList };
struct BoardArgEntry {
    BoardArgKind kind = BoardArgKind::Number;
    int value = 0;
    std::vector<int> values;

    bool operator==(const BoardArgEntry& other) const {
        return kind == other.kind && value == other.value && values == other.values;
    }
};

// Mirrors shared/types.ts's numArg()/csvArg()/zolArg(). The only caller is game/cleg_eval.cpp.
inline BoardArgEntry num_arg(int value) { return BoardArgEntry{ BoardArgKind::Number, value, {} }; }
inline BoardArgEntry csv_arg(std::vector<int> values) {
    return BoardArgEntry{ BoardArgKind::CommaSeparatedNumbers, 0, std::move(values) };
}
inline BoardArgEntry zol_arg(std::vector<int> values) {
    return BoardArgEntry{ BoardArgKind::ZeroOneList, 0, std::move(values) };
}

// Mirrors shared/types.ts's boardArgNumber()/boardArgList(), but asserts (not throws) on the wrong
// kind.
int board_arg_number(const BoardArgEntry& e);
const std::vector<int>& board_arg_list(const BoardArgEntry& e);

// Mirrors shared/types.ts's formatBoardArgEntry(). Kept for parity; no C++ caller needs it (board
// descriptions are printed as their own cleg source text instead - see train.cpp).
std::string format_board_arg_entry(const BoardArgEntry& e);

// Mirrors shared/types.ts's BoardModifier, as one struct whose fields are shared across kinds.
enum class ModifierKind {
    Rectify, Truncate, EdgeSplit, MergeClose, TriangleForm, QuadForm, QuadDiagForm, QuadKnightForm,
    QuadBishopForm, Form, LocalReplace, GlobalCentralize, Scale, NodeInducedSubgraph, EdgeInducedSubgraph
};
struct BoardModifier {
    ModifierKind kind;
    // split_n doubles as TS's `w` for TriangleForm/QuadForm/QuadDiagForm/QuadKnightForm/
    // QuadBishopForm/Form.
    int split_n = 0;   // meaningful when kind == ModifierKind::EdgeSplit/TriangleForm/QuadForm/QuadDiagForm/QuadKnightForm/QuadBishopForm/Form
    // dist doubles as TS's `factor` for Scale.
    double dist = 0.0;             // meaningful when kind == ModifierKind::MergeClose / Scale
    // Only meaningful when kind == ModifierKind::NodeInducedSubgraph / EdgeInducedSubgraph
    // (non-optional, as in TS).
    Selector sel;
    // TS's optional `sel` for the single-kind form modifiers - a separate field from `sel` above,
    // since that one is mandatory.
    std::optional<Selector> form_sel; // meaningful when kind == TriangleForm/QuadForm/QuadDiagForm/QuadKnightForm/QuadBishopForm
    // TS's Form `sels`.
    std::vector<FormSelector> form_sels; // meaningful when kind == ModifierKind::Form
    std::vector<LocalReplaceSelector> selectors; // meaningful when kind == ModifierKind::LocalReplace

    // Needed for std::vector<BoardModifier>::operator==; spelled out since C++17 has no defaulted
    // struct equality.
    bool operator==(const BoardModifier& other) const {
        return kind == other.kind && split_n == other.split_n && dist == other.dist &&
               sel == other.sel && form_sel == other.form_sel && form_sels == other.form_sels &&
               selectors == other.selectors;
    }
};

// Mirrors shared/boardConfig.ts's applyModifier().
BoardConfig apply_modifier(const BoardConfig& bc, const BoardModifier& modifier, ClegRandom& random);

// Mirrors shared/boardConfig.ts's applyModifiers().
BoardConfig apply_modifiers(const BoardConfig& bc, const std::vector<BoardModifier>& modifiers, ClegRandom& random);

// Mirrors shared/boardConfig.ts's linearBoard(), with node i at uncentered position [i].
BoardConfig linear_board(int w);

// Mirrors shared/boardConfig.ts's rectangularBoard().
BoardConfig rectangular_board(int w, int h);

// Mirrors shared/boardConfig.ts's rectangularDiagonalBoard().
BoardConfig rectangular_diagonal_board(int w, int h, int m);

// Mirrors shared/boardConfig.ts's cubeLatticeBoard().
BoardConfig cube_lattice_board(int w, int h, int d);

// Mirrors shared/boardConfig.ts's hypercuboidBoard(); see the .cpp file's comment for the
// embedding difference.
BoardConfig hypercuboid_board(int meshdim, const std::vector<int>& dims);

// Mirrors shared/boardConfig.ts's triangularBoard().
BoardConfig triangular_board(int w);

// Mirrors shared/boardConfig.ts's sierpinskiSimplex(), but instead of the real-valued
// regularSimplexCoords() (irrational for dim >= 2), corner 0 sits at the origin and corner k at a
// scaled standard basis vector (see the .cpp file's comment).
BoardConfig sierpinski_simplex_board(int dim, int n);

// Mirrors shared/boardConfig.ts's simplexBoard(), with the same exact-integer embedding deviation as
// sierpinski_simplex_board() (see the .cpp file's comment).
BoardConfig simplex_board(int meshdim, int dim, int w);

// Mirrors shared/boardConfig.ts's regularPolygonBoard() connectivity only: a unit-edge regular
// n-gon has no exact-integer embedding for general n, so this uses emb_dim = 0 and an empty embed[]
// per node. Only CNN/UNet need real 2D coordinates (see cnn.cpp/unet.cpp), and this board isn't
// grid-shaped anyway, so nothing is lost.
BoardConfig regular_polygon_board(int n);

// Mirrors shared/boardConfig.ts's starBoard() connectivity only - same emb_dim = 0 approach as
// regular_polygon_board, for the same reason.
BoardConfig star_board(int n);

// Mirrors shared/boardConfig.ts's tetrahedronBoard(), so it inherits simplex_board()'s
// exact-integer embedding.
BoardConfig tetrahedron_board();

// Mirrors shared/boardConfig.ts's diamondCubicBoard(), but always produces an emb_dim = 0 board: a
// hub's position would need real-number barycenter averaging (same reasoning as
// generic_local_replace()).
BoardConfig diamond_cubic_board(int w);

// Mirrors shared/boardConfig.ts's octahedronBoard().
BoardConfig octahedron_board();

// Mirrors shared/boardConfig.ts's orthoplexBoard(), but with integer coordinates in place of
// +-1/sqrt(2): vertex 2k has coordinate k = 2, vertex 2k+1 has coordinate k = 1, every other
// coordinate 0.
BoardConfig orthoplex_board(int n);

// Mirrors shared/boardConfig.ts's reg24CellBoard(), exact-integer - see the .cpp file's comment for
// the +1 shift its stored embed[] coordinates need (BoardConfig::embed is unsigned-only).
BoardConfig reg_24_cell_board();

// Mirrors shared/boardConfig.ts's reg120CellBoard(), but always produces an emb_dim = 0 board -
// golden-ratio coordinates have no exact-integer analog. See the .cpp file's comment.
BoardConfig reg_120_cell_board();

// Mirrors shared/boardConfig.ts's reg600CellBoard() - same emb_dim = 0 approach as
// reg_120_cell_board().
BoardConfig reg_600_cell_board();

// Mirrors shared/boardConfig.ts's antiprismBoard() connectivity only - same emb_dim = 0 approach as
// regular_polygon_board (the circumradius/height are irrational for essentially every n).
BoardConfig antiprism_board(int n);

// Mirrors shared/boardConfig.ts's dodecahedronBoard() connectivity only - same emb_dim = 0 approach
// as regular_polygon_board (dodecahedron vertices are inherently irrational).
BoardConfig dodecahedron_board();

// Mirrors shared/boardConfig.ts's icosahedronBoard() connectivity only - same emb_dim = 0 approach,
// for the same reason.
BoardConfig icosahedron_board();

// Mirrors shared/boardConfig.ts's dodecahedronFlake(), with the same emb_dim = 0 approach as
// dodecahedron_board() - but this never computes or stores node positions at all beyond
// fractal.cpp's one-time glue-table search: the TS construction is built around a real-valued
// position transform (S_i(x) = r*x + c*verts[i]), which this drops once the glue table is known,
// since the recursive merge only needs *which* nodes coincide, never *where* they are.
BoardConfig dodecahedron_flake_board(int n);

// Mirrors shared/boardConfig.ts's icosahedronFlake() - same no-position approach as
// dodecahedron_flake_board().
BoardConfig icosahedron_flake_board(int n);

// Mirrors shared/boardConfig.ts's octahedronFlake() - same no-position approach as
// dodecahedron_flake_board(), unlike octahedron_board() itself (which has an integer embedding via
// orthoplex_board()).
BoardConfig octahedron_flake_board(int n);

// Mirrors shared/boardConfig.ts's regularPolygonFlake() - same no-position approach as
// dodecahedron_flake_board().
BoardConfig regular_polygon_flake_board(int n_sides, int order);

// Mirrors shared/boardConfig.ts's centralRegularPolygonFlake() - same no-position approach as
// dodecahedron_flake_board().
BoardConfig central_regular_polygon_flake_board(int n_sides, int order);

// Mirrors shared/boardConfig.ts's centralPentagonFlake() - same no-position approach as
// dodecahedron_flake_board().
BoardConfig central_pentagon_flake_board(int order);

// Mirrors shared/boardConfig.ts's mengerSpongeFlake() - same no-position approach as
// dodecahedron_flake_board().
BoardConfig menger_sponge_flake_board(int order, int dim, const std::vector<int>& indicator);

// Mirrors shared/boardConfig.ts's triangularHexBoard(), but embedded as axial (q+d, r+d) (shifted to
// non-negative).
BoardConfig triangular_hex_board(int d);

// Mirrors shared/boardConfig.ts's hexBoard(), but embedded as axial (q+2d+1, r+2d+1) (shifted to
// non-negative) - a vertex's axial (q, r) never strays more than 2d+1 from the origin, one
// hex-lattice step past the d-ringed centers' own [-2d, 2d] range.
BoardConfig hex_board(int d);

// Mirrors shared/boardConfig.ts's trihexBoard(), but embedded as axial (q+2d+1, r+2d+1), same bound
// as hex_board's own embedding.
BoardConfig trihex_board(int d);

// Mirrors shared/boardConfig.ts's snubSquareBoard() for the connectivity, but embeds nodes the same
// integer way as tilted_disconnected_square_board/glue_twisted_square_board (each unit square
// rotated 45 degrees) rather than the TS side's +-30-degree floating-point layout, since embed
// coordinates must be integers here.
BoardConfig snub_square_board(int w, int h);

// Mirrors shared/boardConfig.ts's glueTwistedSquareBoard().
BoardConfig glue_twisted_square_board(int w, int h, int g);

// Mirrors shared/boardConfig.ts's twistedSquareBoard().
BoardConfig twisted_square_board(int w, int h, int g);

// Mirrors shared/boardConfig.ts's PrescribedBoardFns, keyed by `kind` ("line" | "rect" | "rectd" |
// "cublat" | "hcub" | "tri" | "sier" | "simplex" | "regpoly" | "tetra" | "diamondCubic" | "octa" |
// "ortho" | "reg24Cell" | "reg120Cell" | "reg600Cell" | "ap" | "dodeca" |
// "icosa" | "dodflake" | "icoflake" | "octaflake" | "polyflake" | "cpolyflake" | "cpentflake" |
// "menger" | "trihex" | "hex" | "hexdel" | "snubsq" | "twsq" | "gtsq" | "star"). Throws
// std::runtime_error for an unknown kind. Only called by game/cleg_eval.cpp's board-constructor
// builtins (cleg name = `kind` + "B", e.g. "rectB" -> "rect"); train.cpp/server.cpp go through
// build_board_from_cleg() instead.
BoardConfig build_board_config(const std::string& kind, const std::vector<BoardArgEntry>& args);
