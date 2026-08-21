#pragma once
#include <memory>
#include <set>
#include <string>
#include <vector>

// Mirrors shared/selector.ts - see that file's own top comment for the full grammar (SEL):
//   (union SEL SEL) / (inter SEL SEL) / (diff SEL SEL) / (compl SEL) / (all) / (none) /
//   (deg <eq|gt|lt> <num>) / (e2n SEL) / (n2e SEL) / (rrmn <num> SEL) / (rrmp <num> SEL)
// selecting a subset of a board's nodes or edges.

// Mirrors shared/types.ts's BoardEdge: n1 <= n2 always (see make_board_edge below).
struct BoardEdge {
    int n1 = 0, n2 = 0;
    bool operator==(const BoardEdge& other) const { return n1 == other.n1 && n2 == other.n2; }
};
inline BoardEdge make_board_edge(int a, int b) { return a <= b ? BoardEdge{a, b} : BoardEdge{b, a}; }

// Mirrors shared/selector.ts's SelectorType.
enum class SelectorType { Node, Edge };

// The operator tag of a Selector node - mirrors the `op` field of shared/selector.ts's own
// discriminated-union Selector type.
enum class SelectorOp { Union, Inter, Diff, Compl, All, None, Deg, E2N, N2E, Rrmn, Rrmp };

// Mirrors the comparator argument of a Deg selector ('eq'/'gt'/'lt' in the TS grammar).
enum class DegCmp { Eq, Gt, Lt };

// Mirrors shared/selector.ts's Selector - one monolithic recursive type (see that file's own top
// comment for why `type` is a plain field filled in by the parser, not two separate TS types).
// Unlike BoardModifier's own self-recursive `modifiers` (a plain std::vector<BoardModifier> - always
// legal for a complete type since C++17, see board_config.h), a Selector's own children are a fixed
// 0/1/2-per-op arity, not a list - `a`/`b` are shared_ptr rather than Selector-by-value or a vector,
// since a Selector is only ever built once (by parse_node_selector/parse_edge_selector or from JSON)
// and never mutated afterward, so sharing sub-trees is exactly as safe as the TS side's own object
// sharing and avoids writing a deep-copy constructor by hand.
struct Selector {
    SelectorOp op = SelectorOp::All;
    SelectorType type = SelectorType::Node;
    std::shared_ptr<Selector> a, b;  // meaningful per op - see each op's own case in select_node/select_edge
    DegCmp cmp = DegCmp::Eq;         // meaningful iff op == Deg
    int n = 0;                       // meaningful iff op == Deg
    int count = 0;                   // meaningful iff op == Rrmn
    double frac = 0.0;               // meaningful iff op == Rrmp

    bool operator==(const Selector& other) const;
};

// Parses `s` as a node selector (see this file's own top comment for the grammar) - throws
// std::runtime_error if `s` doesn't follow the grammar (an edge-only operator like e2n is simply
// not a recognized operator inside a node-selector context). Mirrors shared/selector.ts's
// parseNodeSelector(), including its mutual recursion with parse_edge_selector below (via n2e's own
// operand) - see the .cpp file's own parse_node_sel_expr/parse_edge_sel_expr.
Selector parse_node_selector(const std::string& s);

// Parses `s` as an edge selector - the edge-selector counterpart of parse_node_selector above.
// Mirrors shared/selector.ts's parseEdgeSelector().
Selector parse_edge_selector(const std::string& s);

// Formats `sel` back into the S-expression syntax parse_node_selector()/parse_edge_selector() accept
// - the inverse of parsing. Mirrors shared/selector.ts's formatSelector().
std::string format_selector(const Selector& sel);

// Evaluates a node Selector against a board's adjacency matrix, returning the set of selected node
// indices. Mutually recursive with select_edge() via the n2e operator. `pos` isn't used by any
// selector in the current grammar, but is threaded through (matching select_edge()'s own signature)
// for future position-based selectors - mirrors shared/selector.ts's selectNode() exactly, including
// this same unused-for-now parameter.
std::set<int> select_node(const std::vector<std::vector<int>>& adj,
                           const std::vector<std::vector<unsigned>>& pos,
                           const Selector& sel);

// Evaluates an edge Selector against a board's adjacency matrix, returning the list of selected
// edges as BoardEdge values (deduplicated). Mutually recursive with select_node() via the e2n
// operator. Mirrors shared/selector.ts's selectEdge().
std::vector<BoardEdge> select_edge(const std::vector<std::vector<int>>& adj,
                                    const std::vector<std::vector<unsigned>>& pos,
                                    const Selector& sel);
