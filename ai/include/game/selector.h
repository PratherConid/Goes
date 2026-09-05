#pragma once
#include <algorithm>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <vector>

// Mirrors shared/selector.ts - see that file's own top comment for the full grammar and semantics
// (conva/conve's association rule, bottom-up type inference, etc.); repeated here only as a compact
// grammar reference (SEL):
//   (union SEL...) / (inter SEL...) / (diff SEL SEL) / (compl SEL) / (more [<num>] SEL) /
//   (all <node|edge|simp N|tri|quad>) / (none <node|edge|simp N|tri|quad>) /
//   (deg <eq|gt|lt> <num>) / (conva <node|edge|simp N|tri|quad> SEL) /
//   (conve <node|edge|simp N|tri|quad> SEL) /
//   (convlt|conveq|convgt|convclt|convceq|convcgt <node|edge|simp N|tri|quad> <num> SEL) /
//   (rrmn <num> SEL) / (rrmp <num> SEL) / (rpkn <num> SEL) / (rpkp <num> SEL)
// "tri" is sugar for "simp 2" everywhere (both parse to the identical Selector). rpkn/rpkp are the
// pick-instead-of-remove counterparts of rrmn/rrmp - same count/portion argument, but keep those
// elements instead of dropping them. convlt/conveq/convgt/convclt/convceq/convcgt are the
// threshold-counting generalization of conva/conve: a "to" object is selected iff the COUNT of its
// associated "from" objects that are selected (convlt/conveq/convgt) - or NOT selected, the "c"
// variants (convclt/convceq/convcgt) - is </=/> <num>. conva is exactly convceq with <num> 0; conve
// is exactly convgt with <num> 0 - see conv_cmp_params() in the .cpp file.

// Mirrors shared/types.ts's BoardEdge: n1 <= n2 always (see make_board_edge below).
struct BoardEdge {
    int n1 = 0, n2 = 0;
    bool operator==(const BoardEdge& other) const { return n1 == other.n1 && n2 == other.n2; }
};
inline BoardEdge make_board_edge(int a, int b) { return a <= b ? BoardEdge{a, b} : BoardEdge{b, a}; }

// Mirrors shared/types.ts's BoardSimplex: an (n+1)-node clique, `nodes` always sorted ascending (see
// make_board_simplex below) - matches game/topology.h's find_simplices() own increasing-order
// convention. Generalizes what used to be a fixed 3-field BoardTriangle.
struct BoardSimplex {
    std::vector<int> nodes;
    bool operator==(const BoardSimplex& other) const { return nodes == other.nodes; }
};
inline BoardSimplex make_board_simplex(std::vector<int> nodes) {
    std::sort(nodes.begin(), nodes.end());
    return BoardSimplex{ std::move(nodes) };
}

// Mirrors shared/types.ts's BoardQuad: n1-n2-n3-n4-n1 always a genuine cycle - see that type's own
// doc comment for why (unlike BoardSimplex) this can't just be sorted ascending, and make_board_quad
// below for the canonicalization. No delta from the TS version beyond int fields instead of number.
struct BoardQuad {
    int n1 = 0, n2 = 0, n3 = 0, n4 = 0;
    bool operator==(const BoardQuad& other) const {
        return n1 == other.n1 && n2 == other.n2 && n3 == other.n3 && n4 == other.n4;
    }
};
// Mirrors shared/types.ts's makeBoardQuad() - same algorithm (see its own doc comment for the
// derivation), just plain arrays/a loop in place of .map().
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

// The four kinds a SelectorType can denote - mirrors shared/types.ts's own SelectorType string
// union, minus its `simp${number}` template-literal case (see SelectorType below for how that's
// represented here instead - C++ has no equivalent template-literal-type mechanism).
enum class SelectorKind { Node, Edge, Simp, Quad };

// Mirrors shared/types.ts's SelectorType ('node' | 'edge' | 'quad' | `simp${number}`) - `n` is
// meaningful only when `kind == SelectorKind::Simp` (the simplex arity: n+1 nodes), the closest C++
// analogue of the TS side's own template-literal-string trick. `simp_type(n)` below builds the Simp
// case; a bare `SelectorType{SelectorKind::Node}` (etc.) builds the other three, in place of the
// old plain-enum-value `SelectorType::Node` literals this replaces throughout the codebase.
struct SelectorType {
    SelectorKind kind = SelectorKind::Node;
    int n = 0; // meaningful iff kind == SelectorKind::Simp
    bool operator==(const SelectorType& other) const {
        return kind == other.kind && (kind != SelectorKind::Simp || n == other.n);
    }
    bool operator!=(const SelectorType& other) const { return !(*this == other); }
};
inline SelectorType simp_type(int n) { return SelectorType{ SelectorKind::Simp, n }; }
// Extracts n from a Simp SelectorType (simp_type's own inverse), or -1 for node/edge/quad - mirrors
// shared/types.ts's simpN().
inline int simp_n(const SelectorType& t) { return t.kind == SelectorKind::Simp ? t.n : -1; }

// The operator tag of a Selector node - mirrors the `op` field of shared/selector.ts's own
// discriminated-union Selector type. `Raw` has no textual grammar (format_selector rejects it) -
// it's built only by game/cleg.cpp, wrapping a `set`-typed cleg selector argument directly (mirrors
// shared/types.ts's Selector 'raw' variant).
enum class SelectorOp {
    Union, Inter, Diff, Compl, More, All, None, Deg, Conva, Conve,
    ConvLt, ConvEq, ConvGt, ConvClt, ConvCeq, ConvCgt,
    Rrmn, Rrmp, Rpkn, Rpkp, Raw,
};

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
    SelectorType type;
    std::shared_ptr<Selector> a, b;  // meaningful per op - see each op's own case in select_node/select_edge/etc.
    // meaningful iff op == Union/Inter - the variadic operand list from `(union SEL...)`/
    // `(inter SEL...)`, all sharing this same `type` (a/b above are unused for these two ops). A
    // plain std::vector<Selector> (not a vector of shared_ptr) is fine here for the same reason
    // `a`/`b` being shared_ptr is safe: a Selector is built once and never mutated afterward. Text
    // parsing never produces zero items (its `type` couldn't be inferred bottom-up from none - see
    // parse_sel_expr), though this field-level type doesn't itself forbid a hand-built Selector with
    // an empty `items`.
    std::vector<Selector> items;
    DegCmp cmp = DegCmp::Eq;         // meaningful iff op == Deg
    // meaningful iff op == Deg (the degree to compare against) or ConvLt/ConvEq/ConvGt/ConvClt/
    // ConvCeq/ConvCgt (the associated-object-count threshold from the grammar's own `<num>`) - Conva/
    // Conve don't store this themselves (their own threshold is always 0, folded in by
    // conv_cmp_params() in the .cpp file, mirroring shared/selector.ts's convCmpParams()).
    int n = 0;
    int count = 0;                   // meaningful iff op == Rrmn/Rpkn
    double frac = 0.0;               // meaningful iff op == Rrmp/Rpkp
    // meaningful iff op == Conva/Conve/ConvLt/ConvEq/ConvGt/ConvClt/ConvCeq/ConvCgt - the "from"
    // kind, read off sel.a's own bottom-up-inferred `type` at parse time (NOT a literal token -
    // `type` above is now what the leading node/edge/simp N/quad token in the grammar names, the
    // "to"/result kind; see parse_conversion/parse_conv_cmp).
    SelectorType from;
    // meaningful iff op == More - the optional step count from `(more [<num>] SEL)`; nullopt means it
    // was omitted (defaults to 1 at evaluation, see select_node/select_edge), kept as nullopt rather
    // than eagerly filled in to 1 so format_selector can round-trip the exact text a Selector was
    // parsed from.
    std::optional<int> steps;
    // meaningful iff op == Raw - the literal contents of a `set`-typed cleg selector argument (see
    // game/cleg.cpp's resolve_selector_arg/resolve_any_kind_selector_arg), one populated per `type`
    // (node -> raw_nodes, edge -> raw_edges, simp -> raw_simps, quad -> raw_quads) - mirrors
    // shared/types.ts's Selector 'raw' variant/SelectedVals. raw_nodes is a std::set (like
    // select_node's own return type) since node membership has genuine equality; the other three
    // stay plain vectors, matching SelectedVals' own doc comment on why.
    std::set<int> raw_nodes;
    std::vector<BoardEdge> raw_edges;
    std::vector<BoardSimplex> raw_simps;
    std::vector<BoardQuad> raw_quads;

    bool operator==(const Selector& other) const;
};

// Returns a NEW vector with exactly remove_count (clamped to [0, items.size()], since removing more
// than exist isn't meaningful) uniformly-randomly-chosen elements dropped, via a partial
// Fisher-Yates shuffle (only the first remove_count positions need to be randomized to pick which
// elements to drop) - mirrors shared/selector.ts's randomlyRemove(). A public header-only template
// (rather than selector.cpp-private) so game/cleg.cpp's randRmN/randRmP builtins can reuse it over
// a generic ClegValue too, not just Selector's own BoardEdge/BoardSimplex/BoardQuad - matches that
// TS builtin's own doc comment ("reuses that file's own randomlyRemove() rather than
// reimplementing"). Shares one process-wide RNG with select_node/select_edge/etc.'s own Rrmn/Rrmp
// cases (below) via cleg_selector_rng() - an inline function's local static is merged across
// translation units under the One Definition Rule, so this is the same generator either way.
inline std::mt19937& cleg_selector_rng() {
    static std::mt19937 rng(std::random_device{}());
    return rng;
}
template <typename T>
std::vector<T> randomly_remove(std::vector<T> items, int remove_count) {
    int n = static_cast<int>(items.size());
    int to_remove = std::min(std::max(remove_count, 0), n);
    for (int i = 0; i < to_remove; i++) {
        std::uniform_int_distribution<int> dist(i, n - 1);
        int j = dist(cleg_selector_rng());
        std::swap(items[i], items[j]);
    }
    items.erase(items.begin(), items.begin() + to_remove);
    return items;
}

// The take-instead-of-remove counterpart of randomly_remove just above - same partial Fisher-Yates
// shuffle (the first take_count positions become the randomized ones), but keeps THOSE positions
// (the taken elements) rather than erasing them - mirrors shared/selector.ts's randomlyTake().
template <typename T>
std::vector<T> randomly_take(std::vector<T> items, int take_count) {
    int n = static_cast<int>(items.size());
    int to_take = std::min(std::max(take_count, 0), n);
    for (int i = 0; i < to_take; i++) {
        std::uniform_int_distribution<int> dist(i, n - 1);
        int j = dist(cleg_selector_rng());
        std::swap(items[i], items[j]);
    }
    items.erase(items.begin() + to_take, items.end());
    return items;
}

// Parses `s` as a selector of whichever kind it turns out to be, inferred bottom-up from `s` itself
// (see this file's own top comment) - throws std::runtime_error if `s` doesn't follow the grammar.
// Unlike the four parse_*_selector functions below, doesn't check the result's own kind against
// anything; used wherever a selector's own kind isn't fixed ahead of the call (e.g. cleg's own mkSel
// builtin, game/cleg_eval.cpp). Mirrors shared/selector.ts's parseSelector().
Selector parse_selector(const std::string& s);

// Parses `s` as a node selector (see this file's own top comment for the grammar) - throws
// std::runtime_error if `s` doesn't follow the grammar, or parses to a selector of a different kind.
// Mirrors shared/selector.ts's parseNodeSelector() - built on parse_selector() above (see the .cpp
// file's own parse_top_level), the single context-free function all four parse_*_selector entry
// points below share, each just checking the result's own bottom-up-inferred `type` against what it
// promises to return.
Selector parse_node_selector(const std::string& s);

// Parses `s` as an edge selector - the edge-selector counterpart of parse_node_selector above.
// Mirrors shared/selector.ts's parseEdgeSelector().
Selector parse_edge_selector(const std::string& s);

// Parses `s` as a triangle (simp 2) selector - the triangle-selector counterpart of
// parse_node_selector above. Mirrors shared/selector.ts's parseTriangleSelector().
Selector parse_triangle_selector(const std::string& s);

// Parses `s` as a simp `n` selector - the general, n-parameterized counterpart of
// parse_triangle_selector above (its own n=2 special case). Mirrors shared/selector.ts's
// parseSimpSelector().
Selector parse_simp_selector(int n, const std::string& s);

// Parses `s` as a quad selector - the quad-selector counterpart of parse_node_selector above.
// Mirrors shared/selector.ts's parseQuadSelector().
Selector parse_quad_selector(const std::string& s);

// Formats `sel` back into the S-expression syntax the four parse_*_selector functions above accept -
// the inverse of parsing. Mirrors shared/selector.ts's formatSelector().
std::string format_selector(const Selector& sel);

// Evaluates a node Selector against a board's adjacency matrix, returning the set of selected node
// indices. Mutually recursive with select_edge()/select_simp()/select_quad() via the conva/conve
// operators. `pos` isn't used by any selector in the current grammar, but is threaded through
// (matching the other three evaluators' own signatures) for future position-based selectors -
// mirrors shared/selector.ts's selectNode() exactly, including this same unused-for-now parameter.
std::set<int> select_node(const std::vector<std::vector<int>>& adj,
                           const std::vector<std::vector<unsigned>>& pos,
                           const Selector& sel);

// Evaluates an edge Selector against a board's adjacency matrix, returning the list of selected
// edges as BoardEdge values (deduplicated). Mutually recursive with select_node()/select_simp()/
// select_quad() via the conva/conve operators. Mirrors shared/selector.ts's selectEdge().
std::vector<BoardEdge> select_edge(const std::vector<std::vector<int>>& adj,
                                    const std::vector<std::vector<unsigned>>& pos,
                                    const Selector& sel);

// Evaluates a simp Selector of any arity N (sel.type.kind == Simp) against a board's adjacency
// matrix, returning the list of selected N-simplices as BoardSimplex values (deduplicated) - the
// simplex counterpart of select_edge/select_quad. `(all)` is every N-simplex game/topology.h's
// find_simplices() finds. Mutually recursive with select_node()/select_edge()/select_quad() (and
// itself, for a simp M <-> simp N conversion) via the conva/conve operators. Mirrors
// shared/selector.ts's selectSimp().
std::vector<BoardSimplex> select_simp(const std::vector<std::vector<int>>& adj,
                                       const std::vector<std::vector<unsigned>>& pos,
                                       const Selector& sel);

// Parses `s` as a triangle (simp 2) selector and evaluates it - select_simp()'s own N=2 special
// case, kept as a thin sugar wrapper for callers that only ever deal in triangles
// (game/board_config.cpp's triangle_form/tri_centralize). Throws if sel isn't specifically simp 2
// (not just any simp N). Mirrors shared/selector.ts's selectTriangle().
std::vector<BoardSimplex> select_triangle(const std::vector<std::vector<int>>& adj,
                                           const std::vector<std::vector<unsigned>>& pos,
                                           const Selector& sel);

// Evaluates a quad Selector against a board's adjacency matrix, returning the list of selected
// quads as BoardQuad values (deduplicated) - the quad counterpart of select_simp above.
// `(all)` is every quad game/topology.h's find_quads() finds. Mutually recursive with
// select_node() via the conva/conve operators. Mirrors shared/selector.ts's selectQuad().
std::vector<BoardQuad> select_quad(const std::vector<std::vector<int>>& adj,
                                    const std::vector<std::vector<unsigned>>& pos,
                                    const Selector& sel);
