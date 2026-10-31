#pragma once
#include <vector>
#include <map>
#include <string>
#include <array>
#include <utility>
#include <functional>

// Mirrors shared/fractal.ts, minus node positions: these boards never compute or store real
// positions (see board_config.h's dodecahedron_flake_board()), so every TS `pos` output and every
// position-only input (`leafPos`, `SubDescr`'s `scale`/`shift`, `globalScale`) is dropped.
// Leaf-space positions survive only as scratch data for fractal.cpp's one-time glue search.

// Mirrors one element of shared/fractal.ts's GlueObjectType `step()` return value.
struct GlueStep {
    int sub_slot;
    std::vector<int> tags;
};

// Mirrors shared/fractal.ts's GlueObjectType. `step` is a `std::function` (not a bare function
// pointer) so menger_hyperface_glue_object() can capture state, as its TS closure does.
struct GlueObjectType {
    std::function<std::vector<GlueStep>(const std::vector<int>&)> step;
};

// Mirrors shared/fractal.ts's POINT_GLUE_OBJECT/EDGE_GLUE_OBJECT.
extern const GlueObjectType POINT_GLUE_OBJECT;
extern const GlueObjectType EDGE_GLUE_OBJECT;

// Mirrors shared/fractal.ts's glueObjectAddresses().
std::vector<std::string> glue_object_addresses(const GlueObjectType& object, const std::vector<int>& tags, int depth);

// Mirrors shared/fractal.ts's GlueEntry.
struct GlueEntry {
    GlueObjectType object;
    std::vector<int> self_vertices;
    std::vector<int> other_vertices;
};

// Mirrors shared/fractal.ts's FractalDescr, minus its position-only fields (see top comment);
// `num_leaf`/`num_subs` stand in for `leafPos.length`/`subDescr.length`. `glue_map` is keyed by
// `std::pair<int,int>` rather than a `"P,Q"` string (TS stringifies only because a JS `Map` has no
// compound keys).
struct FractalDescr {
    int num_leaf;
    int num_subs;
    std::vector<std::pair<int, int>> leaf_conn;
    std::map<std::pair<int, int>, GlueEntry> glue_map;
};

// Mirrors shared/fractal.ts's SubFlakeResult, minus `pos`.
struct SubFlakeResult {
    std::vector<std::vector<int>> adj;
    std::map<std::string, int> labels;
};

// Mirrors shared/fractal.ts's nodeEdgeMergeFlakeRec(), minus the position-only `scale`/`offset`;
// recurses `descr.num_subs` times in place of iterating `descr.subDescr`.
SubFlakeResult node_edge_merge_flake_rec(int n, const FractalDescr& descr);

// Mirrors shared/fractal.ts's buildFractal(), minus `pos` (and so the `offset`/`globalScale` setup).
std::vector<std::vector<int>> build_fractal(int n, const FractalDescr& descr);

// Mirror shared/fractal.ts's dodecahedronFractalDescr()/icosahedronFractalDescr()/
// octahedronFractalDescr()/regularPolygonFractalDescr()/centralPentagonFractalDescr(), minus
// positions.
const FractalDescr& dodecahedron_fractal_descr();
const FractalDescr& icosahedron_fractal_descr();
const FractalDescr& octahedron_fractal_descr();
const FractalDescr& regular_polygon_fractal_descr(int n_sides, bool center);
const FractalDescr& central_pentagon_fractal_descr();

// Mirrors shared/fractal.ts's mengerFractalDescr(), minus positions.
const FractalDescr& menger_fractal_descr(int dim, const std::vector<int>& indicator);
