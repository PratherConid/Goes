#pragma once
#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>

// Mirrors shared/selector.ts - see that file's own top comment for the full grammar (SEL):
//   (union SEL SEL) / (inter SEL SEL) / (diff SEL SEL) / (compl SEL) / (more SEL) / (all) / (none) /
//   (deg <eq|gt|lt> <num>) / (fromna SEL) / (fromne SEL) / (tona <edge|tri|sq> SEL) /
//   (tone <edge|tri|sq> SEL) / (rrmn <num> SEL) / (rrmp <num> SEL)
// selecting a subset of a board's nodes, edges, triangles, or squares (a "triangle"/"square" here is
// exactly what game/topology.h's find_triangles()/find_squares() finds).

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

// Mirrors shared/types.ts's BoardSquare: n1 < n2 < n3 < n4 always (see make_board_square below) -
// unlike game/topology.h's find_squares(), which reports its own [a, b, c, d] in cycle order, not
// sorted.
struct BoardSquare {
    int n1 = 0, n2 = 0, n3 = 0, n4 = 0;
    bool operator==(const BoardSquare& other) const {
        return n1 == other.n1 && n2 == other.n2 && n3 == other.n3 && n4 == other.n4;
    }
};
inline BoardSquare make_board_square(int a, int b, int c, int d) {
    int arr[4] = { a, b, c, d };
    std::sort(arr, arr + 4);
    return BoardSquare{ arr[0], arr[1], arr[2], arr[3] };
}

// Mirrors shared/selector.ts's SelectorType.
enum class SelectorType { Node, Edge, Tri, Sq };

// The non-node kinds fromna/fromne convert a node selector into, and tona/tone convert back from -
// mirrors shared/selector.ts's ObjectType.
enum class ObjectType { Edge, Tri, Sq };

// The operator tag of a Selector node - mirrors the `op` field of shared/selector.ts's own
// discriminated-union Selector type.
enum class SelectorOp { Union, Inter, Diff, Compl, More, All, None, Deg, Fromna, Fromne, Tona, Tone, Rrmn, Rrmp };

// Mirrors the comparator argument of a Deg selector ('eq'/'gt'/'lt' in the TS grammar).
enum class DegCmp { Eq, Gt, Lt };

// Mirrors shared/selector.ts's Selector - one monolithic recursive type (see that file's own top
// comment for why `type` is a plain field filled in by the parser, not a separate TS type per kind).
// Unlike BoardModifier's own self-recursive `modifiers` (a plain std::vector<BoardModifier> - always
// legal for a complete type since C++17, see board_config.h), a Selector's own children are a fixed
// 0/1/2-per-op arity, not a list - `a`/`b` are shared_ptr rather than Selector-by-value or a vector,
// since a Selector is only ever built once (by parse_node_selector/parse_edge_selector/etc. or from
// JSON) and never mutated afterward, so sharing sub-trees is exactly as safe as the TS side's own
// object sharing and avoids writing a deep-copy constructor by hand.
struct Selector {
    SelectorOp op = SelectorOp::All;
    SelectorType type = SelectorType::Node;
    std::shared_ptr<Selector> a, b;  // meaningful per op - see each op's own case in select_node/select_edge/etc.
    DegCmp cmp = DegCmp::Eq;         // meaningful iff op == Deg
    int n = 0;                       // meaningful iff op == Deg
    int count = 0;                   // meaningful iff op == Rrmn
    double frac = 0.0;               // meaningful iff op == Rrmp
    ObjectType from = ObjectType::Edge; // meaningful iff op == Tona/Tone (the leading edge/tri/sq token)

    bool operator==(const Selector& other) const;
};

// Parses `s` as a node selector (see this file's own top comment for the grammar) - throws
// std::runtime_error if `s` doesn't follow the grammar (an operator not valid for nodes is simply
// not recognized inside a node-selector context). Mirrors shared/selector.ts's parseNodeSelector(),
// including its mutual recursion with parse_edge_selector/parse_triangle_selector/
// parse_square_selector below (via tona/tone's own operand) - see the .cpp file's own
// parse_node_sel_expr/parse_edge_sel_expr/parse_triangle_sel_expr/parse_square_sel_expr.
Selector parse_node_selector(const std::string& s);

// Parses `s` as an edge selector - the edge-selector counterpart of parse_node_selector above.
// Mirrors shared/selector.ts's parseEdgeSelector().
Selector parse_edge_selector(const std::string& s);

// Parses `s` as a triangle selector - the triangle-selector counterpart of parse_node_selector
// above. Mirrors shared/selector.ts's parseTriangleSelector().
Selector parse_triangle_selector(const std::string& s);

// Parses `s` as a square selector - the square-selector counterpart of parse_node_selector above.
// Mirrors shared/selector.ts's parseSquareSelector().
Selector parse_square_selector(const std::string& s);

// Formats `sel` back into the S-expression syntax the four parse_*_selector functions above accept -
// the inverse of parsing. Mirrors shared/selector.ts's formatSelector().
std::string format_selector(const Selector& sel);

// Evaluates a node Selector against a board's adjacency matrix, returning the set of selected node
// indices. Mutually recursive with select_edge()/select_triangle()/select_square() via the tona/tone
// operators. `pos` isn't used by any selector in the current grammar, but is threaded through
// (matching the other three evaluators' own signatures) for future position-based selectors -
// mirrors shared/selector.ts's selectNode() exactly, including this same unused-for-now parameter.
std::set<int> select_node(const std::vector<std::vector<int>>& adj,
                           const std::vector<std::vector<unsigned>>& pos,
                           const Selector& sel);

// Evaluates an edge Selector against a board's adjacency matrix, returning the list of selected
// edges as BoardEdge values (deduplicated). Mutually recursive with select_node() via the
// fromna/fromne operators. Mirrors shared/selector.ts's selectEdge().
std::vector<BoardEdge> select_edge(const std::vector<std::vector<int>>& adj,
                                    const std::vector<std::vector<unsigned>>& pos,
                                    const Selector& sel);

// Evaluates a triangle Selector against a board's adjacency matrix, returning the list of selected
// triangles as BoardTriangle values (deduplicated) - the triangle counterpart of select_edge above.
// `(all)` is every triangle game/topology.h's find_triangles() finds. Mutually recursive with
// select_node() via the fromna/fromne operators. Mirrors shared/selector.ts's selectTriangle().
std::vector<BoardTriangle> select_triangle(const std::vector<std::vector<int>>& adj,
                                            const std::vector<std::vector<unsigned>>& pos,
                                            const Selector& sel);

// Evaluates a square Selector against a board's adjacency matrix, returning the list of selected
// squares as BoardSquare values (deduplicated) - the square counterpart of select_triangle above.
// `(all)` is every square game/topology.h's find_squares() finds. Mutually recursive with
// select_node() via the fromna/fromne operators. Mirrors shared/selector.ts's selectSquare().
std::vector<BoardSquare> select_square(const std::vector<std::vector<int>>& adj,
                                        const std::vector<std::vector<unsigned>>& pos,
                                        const Selector& sel);
