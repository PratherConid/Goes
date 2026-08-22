#pragma once
#include <algorithm>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

// Mirrors shared/selector.ts - see that file's own top comment for the full grammar (SEL):
//   (union SEL...) / (inter SEL...) / (diff SEL SEL) / (compl SEL) / (more [<num>] SEL) / (all) /
//   (none) / (deg <eq|gt|lt> <num>) / (conva <node|edge|tri|quad> SEL) /
//   (conve <node|edge|tri|quad> SEL) / (rrmn <num> SEL) / (rrmp <num> SEL)
// selecting a subset of a board's nodes, edges, triangles, or quads (a "triangle"/"quad" here is
// exactly what game/topology.h's find_triangles()/find_quads() finds). conva/conve convert
// between any two kinds (naming the source kind via their own leading node/edge/tri/quad token): a
// "to" object (whichever kind this Selector itself is) is selected iff ALL (conva) or AT LEAST ONE
// (conve) of its associated "from" objects are selected - two objects are associated iff one's own
// node set is completely contained in the other's (always well-defined for two differing kinds,
// since node/edge/triangle/quad have strictly increasing arity 1/2/3/4). Converting a kind to
// itself is a no-op; triangle <-> quad has no meaningful association and is rejected.

// Mirrors shared/types.ts's BoardEdge: n1 <= n2 always (see make_board_edge below).
struct BoardEdge {
    int n1 = 0, n2 = 0;
    bool operator==(const BoardEdge& other) const { return n1 == other.n1 && n2 == other.n2; }
};
inline BoardEdge make_board_edge(int a, int b) { return a <= b ? BoardEdge{a, b} : BoardEdge{b, a}; }

// Mirrors shared/types.ts's BoardTriangle: n1 < n2 < n3 always (see make_board_triangle below) -
// matches game/topology.h's find_triangles() own [u, v, w] convention.
struct BoardTriangle {
    int n1 = 0, n2 = 0, n3 = 0;
    bool operator==(const BoardTriangle& other) const {
        return n1 == other.n1 && n2 == other.n2 && n3 == other.n3;
    }
};
inline BoardTriangle make_board_triangle(int a, int b, int c) {
    int arr[3] = { a, b, c };
    std::sort(arr, arr + 3);
    return BoardTriangle{ arr[0], arr[1], arr[2] };
}

// Mirrors shared/types.ts's BoardQuad: n1-n2-n3-n4-n1 always a genuine cycle (n1-n2, n2-n3, n3-n4,
// n4-n1 the 4 real graph edges; n1-n3/n2-n4 the 2 absent diagonals) - see make_board_quad below.
// Unlike BoardTriangle (whose 3 members can always be sorted ascending with no loss, since a
// triangle's 3-cycle has full S3 symmetry - every permutation of 3 distinct elements is some
// rotation/reflection of it), a quad's cycle structure is real information a plain ascending sort
// would destroy (turning a diagonal into an apparent edge). So a BoardQuad is instead normalized to
// whichever of its own cycle's 8 rotation/reflection-equivalent relabelings (see make_board_quad)
// is lexicographically smallest - still giving every quad a unique representation regardless of
// which vertex/direction it was found from (the same guarantee BoardTriangle/BoardEdge have), while
// keeping n1-n2-n3-n4-n1 a genuine cycle (unlike game/topology.h's find_quads(), whose own
// {a, b, c, d} is *a* valid cycle order, but not necessarily this canonical one).
struct BoardQuad {
    int n1 = 0, n2 = 0, n3 = 0, n4 = 0;
    bool operator==(const BoardQuad& other) const {
        return n1 == other.n1 && n2 == other.n2 && n3 == other.n3 && n4 == other.n4;
    }
};
// Builds a BoardQuad from four node indices already in cycle order (a-b-c-d-a, as
// game/topology.h's find_quads() reports them) - normalizes to the lexicographically smallest of
// the cycle's own 8 rotation/reflection-equivalent relabelings (see BoardQuad's own doc comment),
// NOT a plain sort of all 4 values. Since every rotation starts with a different one of the 4
// (distinct) node indices, the smallest-starting rotation is unique - so only the 2 candidates
// starting at the minimum (one per direction) ever need comparing, and since all 4 inputs are
// distinct, those two candidates' second element can never tie either.
inline BoardQuad make_board_quad(int a, int b, int c, int d) {
    int seq[4] = { a, b, c, d };
    int i = static_cast<int>(std::min_element(seq, seq + 4) - seq);
    int fwd[4], bwd[4];
    for (int k = 0; k < 4; k++) {
        fwd[k] = seq[(i + k) % 4];
        bwd[k] = seq[(i - k + 4) % 4];
    }
    int* best = fwd[1] < bwd[1] ? fwd : bwd;
    return BoardQuad{ best[0], best[1], best[2], best[3] };
}

// Mirrors shared/selector.ts's SelectorType.
enum class SelectorType { Node, Edge, Tri, Quad };

// The operator tag of a Selector node - mirrors the `op` field of shared/selector.ts's own
// discriminated-union Selector type.
enum class SelectorOp { Union, Inter, Diff, Compl, More, All, None, Deg, Conva, Conve, Rrmn, Rrmp };

// Mirrors the comparator argument of a Deg selector ('eq'/'gt'/'lt' in the TS grammar).
enum class DegCmp { Eq, Gt, Lt };

// Mirrors shared/selector.ts's Selector - one monolithic recursive type (see that file's own top
// comment for why `type` is a plain field filled in by the parser, not a separate TS type per kind).
// Unlike BoardModifier's own self-recursive `modifiers` (a plain std::vector<BoardModifier> - always
// legal for a complete type since C++17, see board_config.h), most of a Selector's own children are a
// fixed 0/1/2-per-op arity, not a list - `a`/`b` are shared_ptr rather than Selector-by-value or a
// vector, since a Selector is only ever built once (by parse_node_selector/parse_edge_selector/etc. or
// from JSON) and never mutated afterward, so sharing sub-trees is exactly as safe as the TS side's own
// object sharing and avoids writing a deep-copy constructor by hand. Union/Inter are the one exception
// - a genuine variadic operand list (`items`, below), since `(union SEL...)`/`(inter SEL...)` take
// zero or more operands rather than a fixed arity.
struct Selector {
    SelectorOp op = SelectorOp::All;
    SelectorType type = SelectorType::Node;
    std::shared_ptr<Selector> a, b;  // meaningful per op - see each op's own case in select_node/select_edge/etc.
    // meaningful iff op == Union/Inter - the variadic operand list from `(union SEL...)`/
    // `(inter SEL...)` (a/b above are unused for these two ops). Zero items is the empty set for
    // Union, the universal set (every object of `type` - same as `(all)`) for Inter. A plain
    // std::vector<Selector> (not a vector of shared_ptr) is fine here for the same reason `a`/`b`
    // being shared_ptr is safe: a Selector is built once and never mutated afterward.
    std::vector<Selector> items;
    DegCmp cmp = DegCmp::Eq;         // meaningful iff op == Deg
    int n = 0;                       // meaningful iff op == Deg
    int count = 0;                   // meaningful iff op == Rrmn
    double frac = 0.0;               // meaningful iff op == Rrmp
    SelectorType from = SelectorType::Node; // meaningful iff op == Conva/Conve (the leading source-kind token)
    // meaningful iff op == More - the optional step count from `(more [<num>] SEL)`; nullopt means it
    // was omitted (defaults to 1 at evaluation, see select_node/select_edge), kept as nullopt rather
    // than eagerly filled in to 1 so format_selector can round-trip the exact text a Selector was
    // parsed from.
    std::optional<int> steps;

    bool operator==(const Selector& other) const;
};

// Parses `s` as a node selector (see this file's own top comment for the grammar) - throws
// std::runtime_error if `s` doesn't follow the grammar (an operator not valid for nodes is simply
// not recognized inside a node-selector context). Mirrors shared/selector.ts's parseNodeSelector(),
// including its mutual recursion with parse_edge_selector/parse_triangle_selector/
// parse_quad_selector below (via conva/conve's own operand) - see the .cpp file's own
// parse_node_sel_expr/parse_edge_sel_expr/parse_triangle_sel_expr/parse_quad_sel_expr.
Selector parse_node_selector(const std::string& s);

// Parses `s` as an edge selector - the edge-selector counterpart of parse_node_selector above.
// Mirrors shared/selector.ts's parseEdgeSelector().
Selector parse_edge_selector(const std::string& s);

// Parses `s` as a triangle selector - the triangle-selector counterpart of parse_node_selector
// above. Mirrors shared/selector.ts's parseTriangleSelector().
Selector parse_triangle_selector(const std::string& s);

// Parses `s` as a quad selector - the quad-selector counterpart of parse_node_selector above.
// Mirrors shared/selector.ts's parseQuadSelector().
Selector parse_quad_selector(const std::string& s);

// Formats `sel` back into the S-expression syntax the four parse_*_selector functions above accept -
// the inverse of parsing. Mirrors shared/selector.ts's formatSelector().
std::string format_selector(const Selector& sel);

// Evaluates a node Selector against a board's adjacency matrix, returning the set of selected node
// indices. Mutually recursive with select_edge()/select_triangle()/select_quad() via the conva/conve
// operators. `pos` isn't used by any selector in the current grammar, but is threaded through
// (matching the other three evaluators' own signatures) for future position-based selectors -
// mirrors shared/selector.ts's selectNode() exactly, including this same unused-for-now parameter.
std::set<int> select_node(const std::vector<std::vector<int>>& adj,
                           const std::vector<std::vector<unsigned>>& pos,
                           const Selector& sel);

// Evaluates an edge Selector against a board's adjacency matrix, returning the list of selected
// edges as BoardEdge values (deduplicated). Mutually recursive with select_node()/select_triangle()/
// select_quad() via the conva/conve operators. Mirrors shared/selector.ts's selectEdge().
std::vector<BoardEdge> select_edge(const std::vector<std::vector<int>>& adj,
                                    const std::vector<std::vector<unsigned>>& pos,
                                    const Selector& sel);

// Evaluates a triangle Selector against a board's adjacency matrix, returning the list of selected
// triangles as BoardTriangle values (deduplicated) - the triangle counterpart of select_edge above.
// `(all)` is every triangle game/topology.h's find_triangles() finds. Mutually recursive with
// select_node() via the conva/conve operators. Mirrors shared/selector.ts's selectTriangle().
std::vector<BoardTriangle> select_triangle(const std::vector<std::vector<int>>& adj,
                                            const std::vector<std::vector<unsigned>>& pos,
                                            const Selector& sel);

// Evaluates a quad Selector against a board's adjacency matrix, returning the list of selected
// quads as BoardQuad values (deduplicated) - the quad counterpart of select_triangle above.
// `(all)` is every quad game/topology.h's find_quads() finds. Mutually recursive with
// select_node() via the conva/conve operators. Mirrors shared/selector.ts's selectQuad().
std::vector<BoardQuad> select_quad(const std::vector<std::vector<int>>& adj,
                                    const std::vector<std::vector<unsigned>>& pos,
                                    const Selector& sel);
