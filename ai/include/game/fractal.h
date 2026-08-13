#pragma once
#include <vector>
#include <map>
#include <string>
#include <array>
#include <utility>
#include <functional>

// The recursive "flake" fractal core, and each shape's own static glue-data builder - mirrors
// shared/fractal.ts (see that file's own top comment for why this is split out from board_config.h:
// a self-contained unit distinct from that file's many one-off, non-recursive board constructors).
// The actual BoardConfig-returning functions built on these (dodecahedron_flake_board()/etc.) stay
// in board_config.h/.cpp, which calls build_fractal()/the descr builders declared below - keeping
// this file free of BoardConfig/embedding concerns entirely: every value here is either a leaf-space
// position (used only for the one-time glue search, in fractal.cpp, never exposed) or a plain
// index/adjacency structure. See board_config.h's own doc comment on dodecahedron_flake_board() for
// why these boards never compute or store real node positions at all.
//
// One further, C++-specific deviation beyond dropping positions: shared/fractal.ts's `FractalDescr`
// carries a `SubDescr { scale, shift }` per sub-copy, since the TS side's own recursion threads a
// real-valued affine transform through every level to build `pos`. With `pos` dropped entirely here,
// `scale`/`shift` are unused by anything the C++ side actually computes (the STRUCTURAL recursion -
// how many sub-copies, which pairs glue to which) - so `FractalDescr` below carries plain `num_subs`
// (the count `subDescr.length` would have been) in its place, same simplification the
// pre-generalization version of this file already made for the same reason (see git history).

// Mirrors shared/fractal.ts's GlueObjectType's own `step()` return shape: for one of a glue object's
// own sub-pieces, which outer sub-slot to chase into (`sub_slot`) and that sub-piece's own tags one
// level deeper (`tags`).
struct GlueStep {
    int sub_slot;
    std::vector<int> tags;
};

// Mirrors shared/fractal.ts's GlueObjectType - see its own doc comment for the full contract `step`
// must satisfy (purely combinatorial, single-recursion-step, ad hoc per object). A `std::function`
// here (rather than a bare function pointer) is what lets menger_hyperface_glue_object() below close
// over its own `dim`/`indicator`/`slot_of`, mirroring the TS side's own JS closure.
struct GlueObjectType {
    std::function<std::vector<GlueStep>(const std::vector<int>&)> step;
};

// Mirrors shared/fractal.ts's POINT_GLUE_OBJECT/EDGE_GLUE_OBJECT - see their own doc comments there
// for the full derivation; both are stateless (no captures), so - unlike menger_hyperface_glue_object()
// below - these are plain shared constants, one instance reused by every `GlueEntry` that needs them.
extern const GlueObjectType POINT_GLUE_OBJECT;
extern const GlueObjectType EDGE_GLUE_OBJECT;

// Mirrors shared/fractal.ts's glueObjectAddresses() - GENERIC, shared code, common to every glue
// object, applying its own single-step `step()` repeatedly to enumerate, in a fixed, self-consistent
// order, every address realizing `object`'s own structure at recursion depth `depth`. See the TS
// function's own doc comment for the full base-case/recursive-case derivation, which this mirrors
// exactly.
std::vector<std::string> glue_object_addresses(const GlueObjectType& object, const std::vector<int>& tags, int depth);

// Mirrors shared/fractal.ts's GlueEntry - see its own doc comment: sub-copy `P`'s own realization of
// `object` (instantiated with `self_vertices`) coincides with sub-copy `Q`'s own realization
// (instantiated with `other_vertices`), point-for-point in array order.
struct GlueEntry {
    GlueObjectType object;
    std::vector<int> self_vertices;
    std::vector<int> other_vertices;
};

// Mirrors shared/fractal.ts's FractalDescr - minus `leafPos`/`subDescr`'s own `scale`/`shift` and
// `globalScale` (all position-only - see this file's own top comment), plus `num_leaf`/`num_subs`
// (`leafPos.length`/`subDescr.length`, needed since the arrays themselves are dropped). `leaf_conn`
// mirrors `leafPos`'s own sibling `leafConn` unchanged (already position-free). `glue_map` mirrors
// `glueMap` exactly, just keyed by a real `std::pair<int,int>` rather than a `"P,Q"` string (a
// C++-specific difference with no TS counterpart: TS strings its keys only because a JS `Map` needs a
// primitive key and has no compound-key support, which does not apply here).
struct FractalDescr {
    int num_leaf;
    int num_subs;
    std::vector<std::pair<int, int>> leaf_conn;
    std::map<std::pair<int, int>, GlueEntry> glue_map;
};

// Mirrors shared/fractal.ts's SubFlakeResult - minus `pos` (see this file's own top comment).
struct SubFlakeResult {
    std::vector<std::vector<int>> adj;
    std::map<std::string, int> labels;
};

// Mirrors shared/fractal.ts's nodeEdgeMergeFlakeRec() - minus `scale`/`offset` (both position-only,
// see this file's own top comment); `descr.num_subs` takes the role of iterating `descr.subDescr`
// (only its own COUNT is needed, per `FractalDescr`'s own doc comment). See the TS function's own
// doc comment for the full recursive construction (base case, addresses/merges, mergeBoards()) this
// mirrors line-for-line, using merge_boards()'s own `labels` support (topology.h) in place of TS's
// mergeBoards() own `labels`.
SubFlakeResult node_edge_merge_flake_rec(int n, const FractalDescr& descr);

// Mirrors shared/fractal.ts's buildFractal() - minus `pos` (see this file's own top comment), and so
// minus the `offset`/`globalScale` machinery only `pos` ever needed.
std::vector<std::vector<int>> build_fractal(int n, const FractalDescr& descr);

// Mirrors shared/fractal.ts's dodecahedronFractalDescr()/icosahedronFractalDescr()/
// octahedronFractalDescr()/regularPolygonFractalDescr()/centralPentagonFractalDescr() - see each TS
// function's own doc comment for the full derivation (vertex/edge construction, `r`/`c`, the glue
// search) each of these mirrors exactly, minus positions (this file's own top comment) and glue
// entries built via `POINT_GLUE_OBJECT`/`EDGE_GLUE_OBJECT` in place of the TS side's own. Cached the
// same way as their TS counterparts (dodeca/icosa/octahedron once each; regular-polygon per
// `(n_sides, center)`).
const FractalDescr& dodecahedron_fractal_descr();
const FractalDescr& icosahedron_fractal_descr();
const FractalDescr& octahedron_fractal_descr();
const FractalDescr& regular_polygon_fractal_descr(int n_sides, bool center);
const FractalDescr& central_pentagon_fractal_descr();

// Mirrors shared/fractal.ts's mengerFractalDescr(dim, indicator) - see its own doc comment for the
// full derivation (the `3^dim` grid removal rule, the general "hyperface" glue object handling any
// number of simultaneously-differing axes down to a single shared corner point). `indicator` must
// have exactly `dim + 1` entries (0/1), one per `offCenter = 0..dim` class. Cached per `(dim,
// indicator)`, `indicator`'s own bits (MSB-first) reduced to a single integer key nested under a
// first-level cache keyed by `dim` itself - same two-level scheme as the TS side's own cache.
const FractalDescr& menger_fractal_descr(int dim, const std::vector<int>& indicator);
