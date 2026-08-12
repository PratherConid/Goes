#pragma once
#include <vector>
#include <map>
#include <array>
#include <utility>

// The recursive "flake" fractal core, and each shape's own static edge/glue data builder - mirrors
// shared/fractal.ts (see that file's own top comment for why this is split out from board_config.h:
// a self-contained unit distinct from that file's many one-off, non-recursive board constructors).
// The actual BoardConfig-returning functions built on these (dodecahedron_flake_board()/etc.) stay
// in board_config.h/.cpp, which calls node_edge_merge_flake_rec()/the data builders declared below -
// keeping this file free of BoardConfig/embedding concerns entirely: every value here is either a
// leaf-space position (used only for the one-time glue search, in fractal.cpp, never exposed) or a
// plain index/adjacency structure. See board_config.h's own doc comment on dodecahedron_flake_board()
// for why these boards never compute or store real node positions at all.

// Mirrors dodecaFlakeRec()/icosaFlakeRec()'s own object-shaped return value ({ pos, adj, corners,
// edgeChains }) - minus `pos`, since these boards track no position at all (see this file's own top
// comment).
struct FlakeRecResult {
    std::vector<std::vector<int>> adj;
    std::vector<int> corners;
    std::map<std::pair<int, int>, std::vector<int>> edge_chains;
};

// Mirrors shared/fractal.ts's growingEdgeLevelUpMap(): derives the standard `edge_level_up_map` for
// a shape whose growing shared edges are exactly its `edge_glue_map` entries between ADJACENT
// sub-copies (dodeca/icosa/octahedron/regular_polygon_flake_board()'s own 4n-gon case) - see
// node_edge_merge_flake_rec()'s own doc comment for the `{{P,P,Q},{Q,P,Q}}` derivation.
std::map<std::pair<int, int>, std::vector<std::array<int, 3>>> growing_edge_level_up_map(
    const std::map<std::pair<int, int>, std::array<int, 4>>& edge_glue_map);

// Mirrors shared/fractal.ts's identityNodeLevelUpMap(): the standard `node_level_up_map` every shape
// here uses - leaf vertex `vtx` maps to `(vtx, vtx)` (sub-copy `vtx`'s own corner `vtx`) - see
// node_edge_merge_flake_rec()'s own doc comment for what this map means/is consumed for.
std::map<int, std::pair<int, int>> identity_node_level_up_map(int num_leaf);

// Mirrors shared/fractal.ts's nodeEdgeMergeFlakeRec() - see its own doc comment for the full
// recursive construction. `num_subs` sub-copies are built each recursion (mirroring
// shared/fractal.ts's `subDescr` list): the first `num_leaf` correspond to leaf vertices (0 ..
// num_leaf-1, `edges`' own endpoint range) and expose `corners`/`corners_out`; any further entries
// (num_leaf <= idx < num_subs) are purely auxiliary, with no leaf vertex of their own (e.g.
// central_regular_polygon_flake_board()'s/central_pentagon_flake_board()'s own central copy) - same
// split as shared/fractal.ts's FractalDescr's own doc comment. `edge_glue_map`/`node_glue_map` key
// every glued pair `(P, Q)` (sub-copy indices, not necessarily an `edges` member) to EITHER a whole,
// growing shared edge OR a single shared point, never both: an `edge_glue_map` entry pairs up
// `subs[P]`'s own chain for `(C, D)` with `subs[Q]`'s own chain for `(E, F)` position-by-position,
// while a `node_glue_map` entry contributes exactly one merge pair (`subs[P]`'s own corner `m`,
// `subs[Q]`'s own corner `p`) and no chain at all. The structural merge is built as one `merges`
// list across every entry of both maps and resolved by a single topology.h merge_boards() call (see
// that function's own doc comment for why this - rather than folding subs in one at a time - is
// what makes octahedron_flake_board()'s own transitive antipodal-corner coincidence come out
// correct).
//
// `edge_level_up_map` keys a leaf-vertex pair `(A, B)` (always two valid indices below `num_leaf` -
// NOT a sub-copy-index pair like `edge_glue_map`/`node_glue_map`) to an ordered list of `(sub_idx,
// a, b)` triples: this shape's own chain for edge `(A, B)`, at ANY recursion order, is the
// concatenation - in list order - of `subs[sub_idx]`'s own chain for its OWN edge `(a, b)`
// (oriented to start at `a`). Every shape with a growing shared edge between ADJACENT sub-copies
// (dodeca/icosa/octahedron/regular_polygon_flake_board()'s own 4n-gon case) sets this via
// growing_edge_level_up_map() - for exactly those shapes, an `edge_glue_map` key `(P, Q)` doubles as
// a valid leaf-vertex pair (sub-copy index == leaf-vertex index one-to-one), giving
// `{{P,P,Q},{Q,P,Q}}` (`subs[P]`'s own `(P,Q)`-chain then `subs[Q]`'s own `(P,Q)`-chain),
// reproducing this map's own pre-generalization behavior exactly (concatenate `subs[P]`'s and
// `subs[Q]`'s own same-keyed chain - see git history). central_pentagon_flake_board() is the one
// shape needing a genuinely different map: its own "corner i to corner i+1" chain exists (same
// two-segment shape, using the leaf-adjacent indices `i`, `i+1` in place of `P`, `Q`) even though it
// is never itself an `edge_glue_map` key - it goes entirely unused by the plain (non-central)
// pentagon flake, which merges its own adjacent copies by a single node instead - only becoming
// load-bearing once a central copy is added, which glues to it via `edge_glue_map`.
//
// `node_level_up_map` keys a leaf-vertex index `vtx` (NOT a sub-copy index like `edge_glue_map`/
// `node_glue_map`'s own keys) to a `(sub_idx, subflake_node)` pair: this shape's own corner `vtx`,
// at recursion order `n > 1`, is `subs[sub_idx]`'s own corner `subflake_node` (recursively). This is
// what lets a shape's RECURSION-STEP topology (which sub-copy attaches where, and via which of that
// sub-copy's own corners) be picked independently of its LEAF board's own topology (`edges`) - every
// shape here still sets it via identity_node_level_up_map() (leaf vertex `vtx` maps to `(vtx, vtx)`:
// sub-copy `vtx`'s own corner `vtx`), the "chase the same index every level" convention this map
// replaced (see git history), but a future shape's own recursion step need not follow leaf vertex
// `vtx`'s own numbering at all.
//
// One deliberate deviation from mirroring TS's own per-shape functions (no deviation needed - this
// is already shared TS-side by every caller): with positions dropped entirely (see this file's own
// top comment) the recursion is purely combinatorial, so this single function serves every flake
// board in board_config.h/.cpp, parameterized only by `num_leaf`, `num_subs`, `edges`,
// `edge_glue_map`, `node_glue_map`, `edge_level_up_map`, and `node_level_up_map`.
FlakeRecResult node_edge_merge_flake_rec(
    int n, int num_leaf, int num_subs, const std::vector<std::pair<int, int>>& edges,
    const std::map<std::pair<int, int>, std::array<int, 4>>& edge_glue_map,
    const std::map<std::pair<int, int>, std::pair<int, int>>& node_glue_map,
    const std::map<std::pair<int, int>, std::vector<std::array<int, 3>>>& edge_level_up_map,
    const std::map<int, std::pair<int, int>>& node_level_up_map);

// Vertex/edge/glue data dodecahedron_flake_board()/icosahedron_flake_board()/octahedron_flake_board()
// each need - mirrors shared/fractal.ts's dodecahedronFractalDescr()/icosahedronFractalDescr()/
// octahedronFractalDescr(), minus `verts` in the return (see this file's own top comment for why -
// they're scratch data internal to fractal.cpp's one-time glue search, never exposed). Cached in
// fractal.cpp's own function-local statics since `edges`/`glue` only ever depend on each shape's own
// fixed structure and never change.
struct FlakeEdgeGlueData {
    std::vector<std::pair<int, int>> edges;
    std::vector<std::array<int, 6>> glue;
};
const FlakeEdgeGlueData& dodecahedron_flake_data();
const FlakeEdgeGlueData& icosahedron_flake_data();
const FlakeEdgeGlueData& octahedron_flake_data();

// Vertex/edge/glue data regular_polygon_flake_board()/central_regular_polygon_flake_board()/
// central_pentagon_flake_board() each need - mirrors shared/fractal.ts's
// regularPolygonFractalDescr(), minus the `center` argument (the central-copy relations it derives
// are closed-form - see board_config.h's own doc comment on central_regular_polygon_flake_board()/
// central_pentagon_flake_board() - so board_config.cpp builds them directly, reusing this same base
// `edges`/`edge_glue`/`node_glue` unchanged rather than bundling a second cached variant per shape).
// Cached per `n_sides` (unlike dodeca/icosa/octahedron's own single-shape caches) since
// regular_polygon_flake_board() isn't a fixed shape. Exactly one of `edge_glue`/`node_glue` is
// populated (never both), per `n_sides % 4` - see board_config.h's own doc comment on
// regular_polygon_flake_board() for which point(s) and why.
struct RegularPolygonFlakeData {
    std::vector<std::pair<int, int>> edges;
    std::vector<std::array<int, 6>> edge_glue;
    std::vector<std::array<int, 4>> node_glue;
};
const RegularPolygonFlakeData& regular_polygon_flake_data(int n_sides);
