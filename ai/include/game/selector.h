#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

// Mirrors shared/selector.ts (see its top comment for the grammar) and the Selector-related types
// in shared/types.ts.

// Mirrors shared/types.ts's BoardEdge/makeBoardEdge().
struct BoardEdge {
    int n1 = 0, n2 = 0;
    bool operator==(const BoardEdge& other) const { return n1 == other.n1 && n2 == other.n2; }
};
inline BoardEdge make_board_edge(int a, int b) { return a <= b ? BoardEdge{a, b} : BoardEdge{b, a}; }

// Mirrors shared/types.ts's BoardSimplex/makeBoardSimplex().
struct BoardSimplex {
    std::vector<int> nodes;
    bool operator==(const BoardSimplex& other) const { return nodes == other.nodes; }
};
inline BoardSimplex make_board_simplex(std::vector<int> nodes) {
    std::sort(nodes.begin(), nodes.end());
    return BoardSimplex{ std::move(nodes) };
}

// Mirrors shared/types.ts's BoardQuad.
struct BoardQuad {
    int n1 = 0, n2 = 0, n3 = 0, n4 = 0;
    bool operator==(const BoardQuad& other) const {
        return n1 == other.n1 && n2 == other.n2 && n3 == other.n3 && n4 == other.n4;
    }
};
// Mirrors shared/types.ts's makeBoardQuad().
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

// The tag of SelectorType below (TS's SelectorType is a plain string union, with no separate tag).
enum class SelectorKind { Node, Edge, Simp, Quad };

// Mirrors shared/types.ts's SelectorType - a (kind, n) pair in place of its `simp${number}`
// template-literal string; simp_type() mirrors simpType().
struct SelectorType {
    SelectorKind kind = SelectorKind::Node;
    int n = 0; // meaningful iff kind == SelectorKind::Simp
    bool operator==(const SelectorType& other) const {
        return kind == other.kind && (kind != SelectorKind::Simp || n == other.n);
    }
    bool operator!=(const SelectorType& other) const { return !(*this == other); }
};
inline SelectorType simp_type(int n) { return SelectorType{ SelectorKind::Simp, n }; }
// Mirrors shared/types.ts's simpN(), with -1 in place of null.
inline int simp_n(const SelectorType& t) { return t.kind == SelectorKind::Simp ? t.n : -1; }

// Mirrors the `op` field of shared/types.ts's Selector union.
enum class SelectorOp {
    Union, Inter, Diff, Compl, More, All, None, Deg, Conva, Conve,
    ConvLt, ConvEq, ConvGt, ConvClt, ConvCeq, ConvCgt,
    Rrmn, Rrmp, Rpkn, Rpkp, Raw,
};

// Mirrors the comparator argument of a Deg selector ('eq'/'gt'/'lt' in the TS grammar).
enum class DegCmp { Eq, Gt, Lt };

// Mirrors shared/types.ts's Selector union, flattened into one struct whose per-op fields are
// meaningful only for the ops noted. `a`/`b` are shared_ptr since a Selector is never mutated after
// construction, so sharing sub-trees is safe and avoids a hand-written deep copy.
struct Selector {
    SelectorOp op = SelectorOp::All;
    SelectorType type;
    std::shared_ptr<Selector> a, b;  // meaningful per op - see each op's own case in select_node/select_edge/etc.
    // meaningful iff op == Union/Inter
    std::vector<Selector> items;
    DegCmp cmp = DegCmp::Eq;         // meaningful iff op == Deg
    // meaningful iff op == Deg or ConvLt/ConvEq/ConvGt/ConvClt/ConvCeq/ConvCgt
    int n = 0;
    int count = 0;                   // meaningful iff op == Rrmn/Rpkn
    double frac = 0.0;               // meaningful iff op == Rrmp/Rpkp
    // meaningful iff op == Conva/Conve/ConvLt/ConvEq/ConvGt/ConvClt/ConvCeq/ConvCgt
    SelectorType from;
    // meaningful iff op == More; nullopt mirrors TS's omitted `steps`
    std::optional<int> steps;
    // meaningful iff op == Raw - TS's tagged `items` (SelectedVals) as four fields, only the one
    // matching `type` populated.
    std::set<int> raw_nodes;
    std::vector<BoardEdge> raw_edges;
    std::vector<BoardSimplex> raw_simps;
    std::vector<BoardQuad> raw_quads;

    bool operator==(const Selector& other) const;
};

// Mirrors shared/selector.ts's ClegRandom - uint32_t arithmetic in place of Math.imul/`>>> 0`.
class ClegRandom {
public:
    explicit ClegRandom(uint32_t seed) : state_(seed) {}

    double next() {
        state_ += 0x6D2B79F5u;
        uint32_t t = (state_ ^ (state_ >> 15)) * (state_ | 1u);
        t = (t + (t ^ (t >> 7)) * (t | 61u)) ^ t;
        return static_cast<double>(t ^ (t >> 14)) / 4294967296.0;
    }

private:
    uint32_t state_;
};

// Mirror shared/selector.ts's randomlyRemove()/randomlyTake() - `less` is a std::sort-style
// predicate in place of TS's three-way `compare`. Header-only templates so game/cleg_eval.cpp's
// randRmN/randRmP/randTakeN/randTakeP builtins can reuse them over ClegValue too.
template <typename T, typename Less>
std::vector<T> randomly_remove(std::vector<T> items, int remove_count, Less less, ClegRandom& random) {
    int n = static_cast<int>(items.size());
    int to_remove = std::min(std::max(remove_count, 0), n);
    std::sort(items.begin(), items.end(), less);
    for (int i = 0; i < to_remove; i++) {
        int j = i + static_cast<int>(std::floor(random.next() * (n - i)));
        std::swap(items[i], items[j]);
    }
    items.erase(items.begin(), items.begin() + to_remove);
    return items;
}

template <typename T, typename Less>
std::vector<T> randomly_take(std::vector<T> items, int take_count, Less less, ClegRandom& random) {
    int n = static_cast<int>(items.size());
    int to_take = std::min(std::max(take_count, 0), n);
    std::sort(items.begin(), items.end(), less);
    for (int i = 0; i < to_take; i++) {
        int j = i + static_cast<int>(std::floor(random.next() * (n - i)));
        std::swap(items[i], items[j]);
    }
    items.erase(items.begin() + to_take, items.end());
    return items;
}

// Mirrors shared/selector.ts's parseSelector(). This and every other parse_*/select_* function
// below throws std::runtime_error where TS throws Error.
Selector parse_selector(const std::string& s);

// Mirrors shared/selector.ts's parseNodeSelector().
Selector parse_node_selector(const std::string& s);

// Mirrors shared/selector.ts's parseEdgeSelector().
Selector parse_edge_selector(const std::string& s);

// Mirrors shared/selector.ts's parseTriangleSelector().
Selector parse_triangle_selector(const std::string& s);

// Mirrors shared/selector.ts's parseSimpSelector().
Selector parse_simp_selector(int n, const std::string& s);

// Mirrors shared/selector.ts's parseQuadSelector().
Selector parse_quad_selector(const std::string& s);

// Mirrors shared/selector.ts's formatSelector().
std::string format_selector(const Selector& sel);

// Mirrors shared/selector.ts's selectNode() - returns a std::set (ascending order) where TS's Set
// iterates in insertion order.
std::set<int> select_node(const std::vector<std::vector<int>>& adj,
                           const std::vector<std::vector<unsigned>>& pos,
                           const Selector& sel, ClegRandom& random);

// Mirrors shared/selector.ts's selectEdge().
std::vector<BoardEdge> select_edge(const std::vector<std::vector<int>>& adj,
                                    const std::vector<std::vector<unsigned>>& pos,
                                    const Selector& sel, ClegRandom& random);

// Mirrors shared/selector.ts's selectSimp().
std::vector<BoardSimplex> select_simp(const std::vector<std::vector<int>>& adj,
                                       const std::vector<std::vector<unsigned>>& pos,
                                       const Selector& sel, ClegRandom& random);

// Mirrors shared/selector.ts's selectTriangle().
std::vector<BoardSimplex> select_triangle(const std::vector<std::vector<int>>& adj,
                                           const std::vector<std::vector<unsigned>>& pos,
                                           const Selector& sel, ClegRandom& random);

// Mirrors shared/selector.ts's selectQuad().
std::vector<BoardQuad> select_quad(const std::vector<std::vector<int>>& adj,
                                    const std::vector<std::vector<unsigned>>& pos,
                                    const Selector& sel, ClegRandom& random);
