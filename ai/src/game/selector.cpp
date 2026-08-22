#include "game/selector.h"
#include "game/topology.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

static std::mt19937 g_rng(std::random_device{}());

// ── parsing ──────────────────────────────────────────────────────────────────

// '(' and ')' are always their own token, even with no surrounding whitespace (e.g. "(deg eq 5)");
// every other maximal run of non-whitespace, non-paren characters is one token. Mirrors
// shared/selector.ts's tokenize().
static std::vector<std::string> tokenize(const std::string& s) {
    std::vector<std::string> tokens;
    size_t i = 0;
    while (i < s.size()) {
        char c = s[i];
        if (std::isspace(static_cast<unsigned char>(c))) { i++; continue; }
        if (c == '(' || c == ')') { tokens.push_back(std::string(1, c)); i++; continue; }
        size_t j = i + 1;
        while (j < s.size() && !std::isspace(static_cast<unsigned char>(s[j])) && s[j] != '(' && s[j] != ')') j++;
        tokens.push_back(s.substr(i, j - i));
        i = j;
    }
    return tokens;
}

// Mirrors shared/selector.ts's ParseCursor.
class ParseCursor {
public:
    explicit ParseCursor(std::vector<std::string> tokens) : tokens_(std::move(tokens)) {}

    bool at_end() const { return pos_ >= tokens_.size(); }
    std::string peek() const { return at_end() ? std::string() : tokens_[pos_]; }

    std::string next() {
        if (at_end()) throw std::runtime_error("selector: unexpected end of input");
        return tokens_[pos_++];
    }

    void expect(const std::string& tok) {
        std::string t = next();
        if (t != tok) throw std::runtime_error("selector: expected '" + tok + "', got '" + t + "'");
    }

private:
    std::vector<std::string> tokens_;
    size_t pos_ = 0;
};

// Shared numeric-argument validation for (deg .../rrmn ...)'s count and (rrmp ...)'s portion -
// `context` names the argument in the thrown error (e.g. "(deg ...) argument"). Mirrors
// shared/selector.ts's nextNonnegInt()/nextNonnegNumber().
static int next_nonneg_int(ParseCursor& c, const std::string& context) {
    std::string tok = c.next();
    size_t end = 0;
    double n = 0;
    bool ok = false;
    try { n = std::stod(tok, &end); ok = end == tok.size(); } catch (...) { ok = false; }
    if (!ok || !std::isfinite(n) || n != std::floor(n) || n < 0)
        throw std::runtime_error("selector: " + context + " must be a nonnegative integer, got '" + tok + "'");
    return static_cast<int>(n);
}

static double next_nonneg_number(ParseCursor& c, const std::string& context) {
    std::string tok = c.next();
    size_t end = 0;
    double n = 0;
    bool ok = false;
    try { n = std::stod(tok, &end); ok = end == tok.size(); } catch (...) { ok = false; }
    if (!ok || !std::isfinite(n) || n < 0)
        throw std::runtime_error("selector: " + context + " must be a nonnegative number, got '" + tok + "'");
    return n;
}

static Selector parse_node_sel_expr(ParseCursor& c);
static Selector parse_edge_sel_expr(ParseCursor& c);
static Selector parse_triangle_sel_expr(ParseCursor& c);
static Selector parse_square_sel_expr(ParseCursor& c);

// Reads tona/tone's own leading edge/tri/sq token and parses its operand via the matching one of
// parse_edge_sel_expr/parse_triangle_sel_expr/parse_square_sel_expr - shared by
// parse_node_sel_expr's own tona/tone case below (the only place either op appears, since both
// always produce type == SelectorType::Node). Mirrors shared/selector.ts's
// parseObjectSelExprFor().
static void parse_object_sel_expr_for(ParseCursor& c, const std::string& op_name, ObjectType& from, Selector& a) {
    std::string from_tok = c.next();
    if (from_tok == "edge") { from = ObjectType::Edge; a = parse_edge_sel_expr(c); return; }
    if (from_tok == "tri")  { from = ObjectType::Tri;  a = parse_triangle_sel_expr(c); return; }
    if (from_tok == "sq")   { from = ObjectType::Sq;   a = parse_square_sel_expr(c); return; }
    throw std::runtime_error(
        "selector: (" + op_name + " ...) source kind must be 'edge', 'tri', or 'sq', got '" + from_tok + "'");
}

// Parses a node SEL - mutually recursive with parse_edge_sel_expr/parse_triangle_sel_expr/
// parse_square_sel_expr via tona/tone's own operand. Every Selector this returns has
// type == SelectorType::Node. Mirrors shared/selector.ts's parseNodeSelExpr().
static Selector parse_node_sel_expr(ParseCursor& c) {
    c.expect("(");
    std::string op = c.next();
    if (op == "union" || op == "inter" || op == "diff") {
        Selector sel;
        sel.op = op == "union" ? SelectorOp::Union : op == "inter" ? SelectorOp::Inter : SelectorOp::Diff;
        sel.type = SelectorType::Node;
        sel.a = std::make_shared<Selector>(parse_node_sel_expr(c));
        sel.b = std::make_shared<Selector>(parse_node_sel_expr(c));
        c.expect(")");
        return sel;
    }
    if (op == "compl") {
        Selector sel;
        sel.op = SelectorOp::Compl;
        sel.type = SelectorType::Node;
        sel.a = std::make_shared<Selector>(parse_node_sel_expr(c));
        c.expect(")");
        return sel;
    }
    if (op == "more") {
        Selector sel;
        sel.op = SelectorOp::More;
        sel.type = SelectorType::Node;
        sel.a = std::make_shared<Selector>(parse_node_sel_expr(c));
        c.expect(")");
        return sel;
    }
    if (op == "all") { c.expect(")"); return Selector{SelectorOp::All, SelectorType::Node}; }
    if (op == "none") { c.expect(")"); return Selector{SelectorOp::None, SelectorType::Node}; }
    if (op == "deg") {
        std::string cmp_tok = c.next();
        DegCmp cmp;
        if (cmp_tok == "eq") cmp = DegCmp::Eq;
        else if (cmp_tok == "gt") cmp = DegCmp::Gt;
        else if (cmp_tok == "lt") cmp = DegCmp::Lt;
        else throw std::runtime_error(
            "selector: (deg ...) comparator must be 'eq', 'gt', or 'lt', got '" + cmp_tok + "'");
        int n = next_nonneg_int(c, "(deg ...) argument");
        c.expect(")");
        Selector sel;
        sel.op = SelectorOp::Deg; sel.type = SelectorType::Node; sel.cmp = cmp; sel.n = n;
        return sel;
    }
    if (op == "tona" || op == "tone") {
        Selector sel;
        sel.op = op == "tona" ? SelectorOp::Tona : SelectorOp::Tone;
        sel.type = SelectorType::Node;
        Selector a;
        parse_object_sel_expr_for(c, op, sel.from, a);
        sel.a = std::make_shared<Selector>(std::move(a));
        c.expect(")");
        return sel;
    }
    if (op == "rrmn") {
        int count = next_nonneg_int(c, "(rrmn ...) count");
        Selector sel;
        sel.op = SelectorOp::Rrmn; sel.type = SelectorType::Node; sel.count = count;
        sel.a = std::make_shared<Selector>(parse_node_sel_expr(c));
        c.expect(")");
        return sel;
    }
    if (op == "rrmp") {
        double frac = next_nonneg_number(c, "(rrmp ...) portion");
        Selector sel;
        sel.op = SelectorOp::Rrmp; sel.type = SelectorType::Node; sel.frac = frac;
        sel.a = std::make_shared<Selector>(parse_node_sel_expr(c));
        c.expect(")");
        return sel;
    }
    throw std::runtime_error("selector: unknown node-selector operator '" + op + "'");
}

// Parses an edge SEL - mutually recursive with parse_node_sel_expr via fromna/fromne's own operand.
// Every Selector this returns has type == SelectorType::Edge. Mirrors shared/selector.ts's
// parseEdgeSelExpr().
static Selector parse_edge_sel_expr(ParseCursor& c) {
    c.expect("(");
    std::string op = c.next();
    if (op == "union" || op == "inter" || op == "diff") {
        Selector sel;
        sel.op = op == "union" ? SelectorOp::Union : op == "inter" ? SelectorOp::Inter : SelectorOp::Diff;
        sel.type = SelectorType::Edge;
        sel.a = std::make_shared<Selector>(parse_edge_sel_expr(c));
        sel.b = std::make_shared<Selector>(parse_edge_sel_expr(c));
        c.expect(")");
        return sel;
    }
    if (op == "compl") {
        Selector sel;
        sel.op = SelectorOp::Compl;
        sel.type = SelectorType::Edge;
        sel.a = std::make_shared<Selector>(parse_edge_sel_expr(c));
        c.expect(")");
        return sel;
    }
    if (op == "more") {
        Selector sel;
        sel.op = SelectorOp::More;
        sel.type = SelectorType::Edge;
        sel.a = std::make_shared<Selector>(parse_edge_sel_expr(c));
        c.expect(")");
        return sel;
    }
    if (op == "all") { c.expect(")"); return Selector{SelectorOp::All, SelectorType::Edge}; }
    if (op == "none") { c.expect(")"); return Selector{SelectorOp::None, SelectorType::Edge}; }
    if (op == "fromna" || op == "fromne") {
        Selector sel;
        sel.op = op == "fromna" ? SelectorOp::Fromna : SelectorOp::Fromne;
        sel.type = SelectorType::Edge;
        sel.a = std::make_shared<Selector>(parse_node_sel_expr(c));
        c.expect(")");
        return sel;
    }
    if (op == "rrmn") {
        int count = next_nonneg_int(c, "(rrmn ...) count");
        Selector sel;
        sel.op = SelectorOp::Rrmn; sel.type = SelectorType::Edge; sel.count = count;
        sel.a = std::make_shared<Selector>(parse_edge_sel_expr(c));
        c.expect(")");
        return sel;
    }
    if (op == "rrmp") {
        double frac = next_nonneg_number(c, "(rrmp ...) portion");
        Selector sel;
        sel.op = SelectorOp::Rrmp; sel.type = SelectorType::Edge; sel.frac = frac;
        sel.a = std::make_shared<Selector>(parse_edge_sel_expr(c));
        c.expect(")");
        return sel;
    }
    throw std::runtime_error("selector: unknown edge-selector operator '" + op + "'");
}

// Parses a triangle SEL - mutually recursive with parse_node_sel_expr via fromna/fromne's own
// operand. Every Selector this returns has type == SelectorType::Tri. No deg/more/tona/tone here -
// see selector.h's own top comment. Mirrors shared/selector.ts's parseTriangleSelExpr().
static Selector parse_triangle_sel_expr(ParseCursor& c) {
    c.expect("(");
    std::string op = c.next();
    if (op == "union" || op == "inter" || op == "diff") {
        Selector sel;
        sel.op = op == "union" ? SelectorOp::Union : op == "inter" ? SelectorOp::Inter : SelectorOp::Diff;
        sel.type = SelectorType::Tri;
        sel.a = std::make_shared<Selector>(parse_triangle_sel_expr(c));
        sel.b = std::make_shared<Selector>(parse_triangle_sel_expr(c));
        c.expect(")");
        return sel;
    }
    if (op == "compl") {
        Selector sel;
        sel.op = SelectorOp::Compl;
        sel.type = SelectorType::Tri;
        sel.a = std::make_shared<Selector>(parse_triangle_sel_expr(c));
        c.expect(")");
        return sel;
    }
    if (op == "all") { c.expect(")"); return Selector{SelectorOp::All, SelectorType::Tri}; }
    if (op == "none") { c.expect(")"); return Selector{SelectorOp::None, SelectorType::Tri}; }
    if (op == "fromna" || op == "fromne") {
        Selector sel;
        sel.op = op == "fromna" ? SelectorOp::Fromna : SelectorOp::Fromne;
        sel.type = SelectorType::Tri;
        sel.a = std::make_shared<Selector>(parse_node_sel_expr(c));
        c.expect(")");
        return sel;
    }
    if (op == "rrmn") {
        int count = next_nonneg_int(c, "(rrmn ...) count");
        Selector sel;
        sel.op = SelectorOp::Rrmn; sel.type = SelectorType::Tri; sel.count = count;
        sel.a = std::make_shared<Selector>(parse_triangle_sel_expr(c));
        c.expect(")");
        return sel;
    }
    if (op == "rrmp") {
        double frac = next_nonneg_number(c, "(rrmp ...) portion");
        Selector sel;
        sel.op = SelectorOp::Rrmp; sel.type = SelectorType::Tri; sel.frac = frac;
        sel.a = std::make_shared<Selector>(parse_triangle_sel_expr(c));
        c.expect(")");
        return sel;
    }
    throw std::runtime_error("selector: unknown triangle-selector operator '" + op + "'");
}

// Parses a square SEL - the square counterpart of parse_triangle_sel_expr above (see its own doc
// comment). Every Selector this returns has type == SelectorType::Sq. Mirrors shared/selector.ts's
// parseSquareSelExpr().
static Selector parse_square_sel_expr(ParseCursor& c) {
    c.expect("(");
    std::string op = c.next();
    if (op == "union" || op == "inter" || op == "diff") {
        Selector sel;
        sel.op = op == "union" ? SelectorOp::Union : op == "inter" ? SelectorOp::Inter : SelectorOp::Diff;
        sel.type = SelectorType::Sq;
        sel.a = std::make_shared<Selector>(parse_square_sel_expr(c));
        sel.b = std::make_shared<Selector>(parse_square_sel_expr(c));
        c.expect(")");
        return sel;
    }
    if (op == "compl") {
        Selector sel;
        sel.op = SelectorOp::Compl;
        sel.type = SelectorType::Sq;
        sel.a = std::make_shared<Selector>(parse_square_sel_expr(c));
        c.expect(")");
        return sel;
    }
    if (op == "all") { c.expect(")"); return Selector{SelectorOp::All, SelectorType::Sq}; }
    if (op == "none") { c.expect(")"); return Selector{SelectorOp::None, SelectorType::Sq}; }
    if (op == "fromna" || op == "fromne") {
        Selector sel;
        sel.op = op == "fromna" ? SelectorOp::Fromna : SelectorOp::Fromne;
        sel.type = SelectorType::Sq;
        sel.a = std::make_shared<Selector>(parse_node_sel_expr(c));
        c.expect(")");
        return sel;
    }
    if (op == "rrmn") {
        int count = next_nonneg_int(c, "(rrmn ...) count");
        Selector sel;
        sel.op = SelectorOp::Rrmn; sel.type = SelectorType::Sq; sel.count = count;
        sel.a = std::make_shared<Selector>(parse_square_sel_expr(c));
        c.expect(")");
        return sel;
    }
    if (op == "rrmp") {
        double frac = next_nonneg_number(c, "(rrmp ...) portion");
        Selector sel;
        sel.op = SelectorOp::Rrmp; sel.type = SelectorType::Sq; sel.frac = frac;
        sel.a = std::make_shared<Selector>(parse_square_sel_expr(c));
        c.expect(")");
        return sel;
    }
    throw std::runtime_error("selector: unknown square-selector operator '" + op + "'");
}

// Shared by parse_node_selector/parse_edge_selector/parse_triangle_selector/parse_square_selector:
// tokenizes s, runs parse_expr over the whole thing, and rejects any leftover trailing input.
// Mirrors shared/selector.ts's parseTopLevel().
static Selector parse_top_level(const std::string& s, Selector (*parse_expr)(ParseCursor&)) {
    auto tokens = tokenize(s);
    if (tokens.empty()) throw std::runtime_error("selector: empty input");
    ParseCursor c(std::move(tokens));
    Selector sel = parse_expr(c);
    if (!c.at_end())
        throw std::runtime_error("selector: unexpected trailing input starting at '" + c.peek() + "'");
    return sel;
}

Selector parse_node_selector(const std::string& s) { return parse_top_level(s, parse_node_sel_expr); }
Selector parse_edge_selector(const std::string& s) { return parse_top_level(s, parse_edge_sel_expr); }
Selector parse_triangle_selector(const std::string& s) { return parse_top_level(s, parse_triangle_sel_expr); }
Selector parse_square_selector(const std::string& s) { return parse_top_level(s, parse_square_sel_expr); }

// Renders a double the way JS's default Number->string conversion would for the plain decimal
// literals (rrmp) actually seen in board presets ("0.1", "0.6666", "1", ...) - an integral value
// prints with no decimal point/trailing zeros; otherwise std::ostringstream's own default
// (non-fixed) formatting at 15 significant digits already drops trailing zeros the same way.
static std::string format_double(double d) {
    if (d == std::floor(d) && std::abs(d) < 1e15) return std::to_string(static_cast<long long>(d));
    std::ostringstream oss;
    oss.precision(15);
    oss << d;
    return oss.str();
}

// Mirrors shared/selector.ts's formatSelector() - the inverse of parsing.
std::string format_selector(const Selector& sel) {
    switch (sel.op) {
        case SelectorOp::Union: return "(union " + format_selector(*sel.a) + " " + format_selector(*sel.b) + ")";
        case SelectorOp::Inter: return "(inter " + format_selector(*sel.a) + " " + format_selector(*sel.b) + ")";
        case SelectorOp::Diff:  return "(diff " + format_selector(*sel.a) + " " + format_selector(*sel.b) + ")";
        case SelectorOp::Compl: return "(compl " + format_selector(*sel.a) + ")";
        case SelectorOp::More:  return "(more " + format_selector(*sel.a) + ")";
        case SelectorOp::All:   return "(all)";
        case SelectorOp::None:  return "(none)";
        case SelectorOp::Deg: {
            std::string cmp = sel.cmp == DegCmp::Eq ? "eq" : sel.cmp == DegCmp::Gt ? "gt" : "lt";
            return "(deg " + cmp + " " + std::to_string(sel.n) + ")";
        }
        case SelectorOp::Fromna: return "(fromna " + format_selector(*sel.a) + ")";
        case SelectorOp::Fromne: return "(fromne " + format_selector(*sel.a) + ")";
        case SelectorOp::Tona: case SelectorOp::Tone: {
            std::string name = sel.op == SelectorOp::Tona ? "tona" : "tone";
            std::string from = sel.from == ObjectType::Edge ? "edge" : sel.from == ObjectType::Tri ? "tri" : "sq";
            return "(" + name + " " + from + " " + format_selector(*sel.a) + ")";
        }
        case SelectorOp::Rrmn: return "(rrmn " + std::to_string(sel.count) + " " + format_selector(*sel.a) + ")";
        case SelectorOp::Rrmp: return "(rrmp " + format_double(sel.frac) + " " + format_selector(*sel.a) + ")";
    }
    throw std::runtime_error("format_selector: unknown op");
}

bool Selector::operator==(const Selector& other) const {
    if (op != other.op || type != other.type) return false;
    auto eq_ptr = [](const std::shared_ptr<Selector>& x, const std::shared_ptr<Selector>& y) {
        if (!x || !y) return x == y; // both null -> equal; exactly one null -> not equal
        return *x == *y;
    };
    if (!eq_ptr(a, other.a) || !eq_ptr(b, other.b)) return false;
    if (op == SelectorOp::Deg) return cmp == other.cmp && n == other.n;
    if (op == SelectorOp::Rrmn) return count == other.count;
    if (op == SelectorOp::Rrmp) return frac == other.frac;
    if (op == SelectorOp::Tona || op == SelectorOp::Tone) return from == other.from;
    return true;
}

// ── evaluation ───────────────────────────────────────────────────────────────

// "a node"/"an edge"/"a tri"/"a sq" - shared by each evaluator's own wrong-kind error message below.
static std::string describe_selector_type(SelectorType type) {
    switch (type) {
        case SelectorType::Node: return "a node";
        case SelectorType::Edge: return "an edge";
        case SelectorType::Tri:  return "a tri";
        case SelectorType::Sq:   return "a sq";
    }
    throw std::runtime_error("describe_selector_type: unknown type");
}

static int degree(const std::vector<std::vector<int>>& adj, int i) {
    int d = 0;
    for (int v : adj[i]) d += v ? 1 : 0;
    return d;
}

// BoardEdge/BoardTriangle/BoardSquare aren't valid std::set/std::unordered_map keys themselves (two
// structurally-equal values are different objects, and none has an operator< or std::hash), so
// union/inter/diff/compl below key them by these canonical string ids (n1 < n2 < ... already makes
// each one unique per object) whenever they need set-like membership tests. Mirrors
// shared/selector.ts's edgeKey()/triKey()/sqKey().
static std::string edge_key(const BoardEdge& e) {
    return std::to_string(e.n1) + "," + std::to_string(e.n2);
}
static std::string tri_key(const BoardTriangle& t) {
    return std::to_string(t.n1) + "," + std::to_string(t.n2) + "," + std::to_string(t.n3);
}
static std::string sq_key(const BoardSquare& s) {
    return std::to_string(s.n1) + "," + std::to_string(s.n2) + "," + std::to_string(s.n3) + "," + std::to_string(s.n4);
}

// Generic Map-based dedupe, keyed by key(item) - the last item seen for a given key overwrites the
// value, but the FIRST occurrence's position in iteration order is kept (matches a JS Map's own
// set-on-existing-key semantics, which shared/selector.ts's own dedupeByKey() relies on) - reproduced
// here via an index-into-`out` map alongside `out` itself, rather than plain std::map/unordered_map
// keyed directly by K (which would drop that first-seen ordering). Shared by every object kind's own
// `union` case below.
template <typename T, typename K>
static std::vector<T> dedupe_by_key(const std::vector<T>& items, const std::function<K(const T&)>& key) {
    std::unordered_map<K, size_t> first_seen;
    std::vector<T> out;
    for (auto& item : items) {
        K k = key(item);
        auto it = first_seen.find(k);
        if (it == first_seen.end()) { first_seen.emplace(k, out.size()); out.push_back(item); }
        else out[it->second] = item;
    }
    return out;
}

// Returns a NEW vector with exactly remove_count (clamped to [0, items.size()], since removing more
// than exist isn't meaningful) uniformly-randomly-chosen elements dropped, via a partial
// Fisher-Yates shuffle (only the first remove_count positions need to be randomized to pick which
// elements to drop) - mirrors shared/selector.ts's randomlyRemove().
template <typename T>
static std::vector<T> randomly_remove(std::vector<T> items, int remove_count) {
    int n = static_cast<int>(items.size());
    int to_remove = std::min(std::max(remove_count, 0), n);
    for (int i = 0; i < to_remove; i++) {
        std::uniform_int_distribution<int> dist(i, n - 1);
        int j = dist(g_rng);
        std::swap(items[i], items[j]);
    }
    items.erase(items.begin(), items.begin() + to_remove);
    return items;
}

// Shared by select_edge/select_triangle/select_square's own fromna/fromne case: filters `all`
// (every object of that kind in the whole graph) down to those whose own `members(obj)` nodes are
// either ALL in `node_set` (fromna) or at least ONE is (fromne). Mirrors shared/selector.ts's
// objectsFromNodes().
template <typename T>
static std::vector<T> objects_from_nodes(
    const std::vector<T>& all, const std::set<int>& node_set,
    std::vector<int> (*members)(const T&), bool require_all)
{
    std::vector<T> out;
    for (auto& obj : all) {
        auto m = members(obj);
        bool matches = require_all
            ? std::all_of(m.begin(), m.end(), [&](int n) { return node_set.count(n) > 0; })
            : std::any_of(m.begin(), m.end(), [&](int n) { return node_set.count(n) > 0; });
        if (matches) out.push_back(obj);
    }
    return out;
}

// Shared by select_node's own tona/tone case: for every node 0..N-1, looks at which of `all` (every
// object of the given kind in the whole graph) contain it (via `members`), and selects it iff ALL of
// those containing objects are in `selected_keys` (tona) or at least ONE is (tone) - vacuously
// true/false (respectively) for a node contained in no such object at all, per ordinary
// all_of/any_of semantics on an empty range. Mirrors shared/selector.ts's nodesFromObjects().
template <typename T, typename K>
static std::set<int> nodes_from_objects(
    int N, const std::vector<T>& all, std::vector<int> (*members)(const T&),
    const std::function<K(const T&)>& key, const std::set<K>& selected_keys, bool require_all)
{
    std::vector<std::vector<const T*>> containing_by_node(N);
    for (auto& obj : all) for (int n : members(obj)) containing_by_node[n].push_back(&obj);
    std::set<int> out;
    for (int n = 0; n < N; n++) {
        auto is_selected = [&](const T* obj) { return selected_keys.count(key(*obj)) > 0; };
        auto& containing = containing_by_node[n];
        bool matches = require_all
            ? std::all_of(containing.begin(), containing.end(), is_selected)
            : std::any_of(containing.begin(), containing.end(), is_selected);
        if (matches) out.insert(n);
    }
    return out;
}

std::set<int> select_node(const std::vector<std::vector<int>>& adj,
                           const std::vector<std::vector<unsigned>>& pos,
                           const Selector& sel) {
    if (sel.type != SelectorType::Node)
        throw std::runtime_error("select_node: expected a node selector, got " + describe_selector_type(sel.type));
    int N = static_cast<int>(adj.size());
    switch (sel.op) {
        case SelectorOp::Union: {
            auto a = select_node(adj, pos, *sel.a), b = select_node(adj, pos, *sel.b);
            a.insert(b.begin(), b.end());
            return a;
        }
        case SelectorOp::Inter: {
            auto a = select_node(adj, pos, *sel.a), b = select_node(adj, pos, *sel.b);
            std::set<int> out;
            for (int x : a) if (b.count(x)) out.insert(x);
            return out;
        }
        case SelectorOp::Diff: {
            auto a = select_node(adj, pos, *sel.a), b = select_node(adj, pos, *sel.b);
            std::set<int> out;
            for (int x : a) if (!b.count(x)) out.insert(x);
            return out;
        }
        case SelectorOp::Compl: {
            auto a = select_node(adj, pos, *sel.a);
            std::set<int> out;
            for (int i = 0; i < N; i++) if (!a.count(i)) out.insert(i);
            return out;
        }
        case SelectorOp::More: {
            auto a = select_node(adj, pos, *sel.a);
            std::set<int> out = a;
            for (int i : a)
                for (int j = 0; j < N; j++)
                    if (adj[i][j]) out.insert(j);
            return out;
        }
        case SelectorOp::All: {
            std::set<int> out;
            for (int i = 0; i < N; i++) out.insert(i);
            return out;
        }
        case SelectorOp::None:
            return {};
        case SelectorOp::Deg: {
            std::set<int> out;
            for (int i = 0; i < N; i++) {
                int d = degree(adj, i);
                bool match = (sel.cmp == DegCmp::Eq && d == sel.n) ||
                             (sel.cmp == DegCmp::Gt && d > sel.n) ||
                             (sel.cmp == DegCmp::Lt && d < sel.n);
                if (match) out.insert(i);
            }
            return out;
        }
        case SelectorOp::Tona: case SelectorOp::Tone: {
            bool require_all = sel.op == SelectorOp::Tona;
            if (sel.from == ObjectType::Edge) {
                auto all = select_edge(adj, pos, Selector{SelectorOp::All, SelectorType::Edge});
                auto selected = select_edge(adj, pos, *sel.a);
                std::set<std::string> selected_keys;
                for (auto& e : selected) selected_keys.insert(edge_key(e));
                return nodes_from_objects<BoardEdge, std::string>(
                    N, all,
                    +[](const BoardEdge& e) -> std::vector<int> { return { e.n1, e.n2 }; },
                    edge_key, selected_keys, require_all);
            }
            if (sel.from == ObjectType::Tri) {
                auto all = select_triangle(adj, pos, Selector{SelectorOp::All, SelectorType::Tri});
                auto selected = select_triangle(adj, pos, *sel.a);
                std::set<std::string> selected_keys;
                for (auto& t : selected) selected_keys.insert(tri_key(t));
                return nodes_from_objects<BoardTriangle, std::string>(
                    N, all,
                    +[](const BoardTriangle& t) -> std::vector<int> { return { t.n1, t.n2, t.n3 }; },
                    tri_key, selected_keys, require_all);
            }
            auto all = select_square(adj, pos, Selector{SelectorOp::All, SelectorType::Sq});
            auto selected = select_square(adj, pos, *sel.a);
            std::set<std::string> selected_keys;
            for (auto& s : selected) selected_keys.insert(sq_key(s));
            return nodes_from_objects<BoardSquare, std::string>(
                N, all,
                +[](const BoardSquare& s) -> std::vector<int> { return { s.n1, s.n2, s.n3, s.n4 }; },
                sq_key, selected_keys, require_all);
        }
        case SelectorOp::Rrmn: {
            auto base_set = select_node(adj, pos, *sel.a);
            std::vector<int> base(base_set.begin(), base_set.end());
            auto kept = randomly_remove(std::move(base), sel.count);
            return std::set<int>(kept.begin(), kept.end());
        }
        case SelectorOp::Rrmp: {
            auto base_set = select_node(adj, pos, *sel.a);
            std::vector<int> base(base_set.begin(), base_set.end());
            int remove_count = static_cast<int>(std::floor(sel.frac * static_cast<double>(base.size())));
            auto kept = randomly_remove(std::move(base), remove_count);
            return std::set<int>(kept.begin(), kept.end());
        }
        default:
            throw std::runtime_error("select_node: unexpected node-selector op");
    }
}

std::vector<BoardEdge> select_edge(const std::vector<std::vector<int>>& adj,
                                    const std::vector<std::vector<unsigned>>& pos,
                                    const Selector& sel) {
    if (sel.type != SelectorType::Edge)
        throw std::runtime_error("select_edge: expected an edge selector, got " + describe_selector_type(sel.type));
    int N = static_cast<int>(adj.size());
    switch (sel.op) {
        case SelectorOp::Union: {
            auto a = select_edge(adj, pos, *sel.a);
            auto b = select_edge(adj, pos, *sel.b);
            a.insert(a.end(), b.begin(), b.end());
            return dedupe_by_key<BoardEdge, std::string>(a, edge_key);
        }
        case SelectorOp::Inter: {
            auto a = select_edge(adj, pos, *sel.a);
            auto b = select_edge(adj, pos, *sel.b);
            std::set<std::string> b_keys;
            for (auto& e : b) b_keys.insert(edge_key(e));
            std::vector<BoardEdge> out;
            for (auto& e : a) if (b_keys.count(edge_key(e))) out.push_back(e);
            return out;
        }
        case SelectorOp::Diff: {
            auto a = select_edge(adj, pos, *sel.a);
            auto b = select_edge(adj, pos, *sel.b);
            std::set<std::string> b_keys;
            for (auto& e : b) b_keys.insert(edge_key(e));
            std::vector<BoardEdge> out;
            for (auto& e : a) if (!b_keys.count(edge_key(e))) out.push_back(e);
            return out;
        }
        case SelectorOp::Compl: {
            auto a = select_edge(adj, pos, *sel.a);
            std::set<std::string> a_keys;
            for (auto& e : a) a_keys.insert(edge_key(e));
            std::vector<BoardEdge> out;
            for (int i = 0; i < N; i++)
                for (int j = i + 1; j < N; j++) {
                    if (!adj[i][j]) continue;
                    BoardEdge e = make_board_edge(i, j);
                    if (!a_keys.count(edge_key(e))) out.push_back(e);
                }
            return out;
        }
        case SelectorOp::More: {
            auto a = select_edge(adj, pos, *sel.a);
            std::set<int> a_nodes;
            for (auto& e : a) { a_nodes.insert(e.n1); a_nodes.insert(e.n2); }
            std::vector<BoardEdge> out = a;
            for (int i = 0; i < N; i++)
                for (int j = i + 1; j < N; j++)
                    if (adj[i][j] && (a_nodes.count(i) || a_nodes.count(j))) out.push_back(make_board_edge(i, j));
            std::set<std::string> seen;
            std::vector<BoardEdge> deduped;
            for (auto& e : out) { std::string k = edge_key(e); if (seen.insert(k).second) deduped.push_back(e); }
            return deduped;
        }
        case SelectorOp::All: {
            std::vector<BoardEdge> out;
            for (int i = 0; i < N; i++)
                for (int j = i + 1; j < N; j++)
                    if (adj[i][j]) out.push_back(make_board_edge(i, j));
            return out;
        }
        case SelectorOp::None:
            return {};
        case SelectorOp::Fromna: case SelectorOp::Fromne: {
            auto nodes = select_node(adj, pos, *sel.a);
            auto all = select_edge(adj, pos, Selector{SelectorOp::All, SelectorType::Edge});
            return objects_from_nodes<BoardEdge>(
                all, nodes, +[](const BoardEdge& e) -> std::vector<int> { return { e.n1, e.n2 }; },
                sel.op == SelectorOp::Fromna);
        }
        case SelectorOp::Rrmn: {
            auto base = select_edge(adj, pos, *sel.a);
            return randomly_remove(base, sel.count);
        }
        case SelectorOp::Rrmp: {
            auto base = select_edge(adj, pos, *sel.a);
            int remove_count = static_cast<int>(std::floor(sel.frac * static_cast<double>(base.size())));
            return randomly_remove(base, remove_count);
        }
        default:
            throw std::runtime_error("select_edge: unexpected edge-selector op");
    }
}

std::vector<BoardTriangle> select_triangle(const std::vector<std::vector<int>>& adj,
                                            const std::vector<std::vector<unsigned>>& pos,
                                            const Selector& sel) {
    if (sel.type != SelectorType::Tri)
        throw std::runtime_error(
            "select_triangle: expected a triangle selector, got " + describe_selector_type(sel.type));
    switch (sel.op) {
        case SelectorOp::Union: {
            auto a = select_triangle(adj, pos, *sel.a);
            auto b = select_triangle(adj, pos, *sel.b);
            a.insert(a.end(), b.begin(), b.end());
            std::set<std::string> seen;
            std::vector<BoardTriangle> deduped;
            for (auto& t : a) if (seen.insert(tri_key(t)).second) deduped.push_back(t);
            return deduped;
        }
        case SelectorOp::Inter: {
            auto a = select_triangle(adj, pos, *sel.a);
            std::set<std::string> b_keys;
            for (auto& t : select_triangle(adj, pos, *sel.b)) b_keys.insert(tri_key(t));
            std::vector<BoardTriangle> out;
            for (auto& t : a) if (b_keys.count(tri_key(t))) out.push_back(t);
            return out;
        }
        case SelectorOp::Diff: {
            auto a = select_triangle(adj, pos, *sel.a);
            std::set<std::string> b_keys;
            for (auto& t : select_triangle(adj, pos, *sel.b)) b_keys.insert(tri_key(t));
            std::vector<BoardTriangle> out;
            for (auto& t : a) if (!b_keys.count(tri_key(t))) out.push_back(t);
            return out;
        }
        case SelectorOp::Compl: {
            std::set<std::string> a_keys;
            for (auto& t : select_triangle(adj, pos, *sel.a)) a_keys.insert(tri_key(t));
            std::vector<BoardTriangle> out;
            for (auto& uvw : find_triangles(adj)) {
                BoardTriangle t = make_board_triangle(uvw[0], uvw[1], uvw[2]);
                if (!a_keys.count(tri_key(t))) out.push_back(t);
            }
            return out;
        }
        case SelectorOp::All: {
            std::vector<BoardTriangle> out;
            for (auto& uvw : find_triangles(adj)) out.push_back(make_board_triangle(uvw[0], uvw[1], uvw[2]));
            return out;
        }
        case SelectorOp::None:
            return {};
        case SelectorOp::Fromna: case SelectorOp::Fromne: {
            auto nodes = select_node(adj, pos, *sel.a);
            auto all = select_triangle(adj, pos, Selector{SelectorOp::All, SelectorType::Tri});
            return objects_from_nodes<BoardTriangle>(
                all, nodes,
                +[](const BoardTriangle& t) -> std::vector<int> { return { t.n1, t.n2, t.n3 }; },
                sel.op == SelectorOp::Fromna);
        }
        case SelectorOp::Rrmn: {
            auto base = select_triangle(adj, pos, *sel.a);
            return randomly_remove(base, sel.count);
        }
        case SelectorOp::Rrmp: {
            auto base = select_triangle(adj, pos, *sel.a);
            int remove_count = static_cast<int>(std::floor(sel.frac * static_cast<double>(base.size())));
            return randomly_remove(base, remove_count);
        }
        default:
            throw std::runtime_error("select_triangle: unexpected triangle-selector op");
    }
}

std::vector<BoardSquare> select_square(const std::vector<std::vector<int>>& adj,
                                        const std::vector<std::vector<unsigned>>& pos,
                                        const Selector& sel) {
    if (sel.type != SelectorType::Sq)
        throw std::runtime_error(
            "select_square: expected a square selector, got " + describe_selector_type(sel.type));
    switch (sel.op) {
        case SelectorOp::Union: {
            auto a = select_square(adj, pos, *sel.a);
            auto b = select_square(adj, pos, *sel.b);
            a.insert(a.end(), b.begin(), b.end());
            std::set<std::string> seen;
            std::vector<BoardSquare> deduped;
            for (auto& s : a) if (seen.insert(sq_key(s)).second) deduped.push_back(s);
            return deduped;
        }
        case SelectorOp::Inter: {
            auto a = select_square(adj, pos, *sel.a);
            std::set<std::string> b_keys;
            for (auto& s : select_square(adj, pos, *sel.b)) b_keys.insert(sq_key(s));
            std::vector<BoardSquare> out;
            for (auto& s : a) if (b_keys.count(sq_key(s))) out.push_back(s);
            return out;
        }
        case SelectorOp::Diff: {
            auto a = select_square(adj, pos, *sel.a);
            std::set<std::string> b_keys;
            for (auto& s : select_square(adj, pos, *sel.b)) b_keys.insert(sq_key(s));
            std::vector<BoardSquare> out;
            for (auto& s : a) if (!b_keys.count(sq_key(s))) out.push_back(s);
            return out;
        }
        case SelectorOp::Compl: {
            std::set<std::string> a_keys;
            for (auto& s : select_square(adj, pos, *sel.a)) a_keys.insert(sq_key(s));
            std::vector<BoardSquare> out;
            for (auto& abcd : find_squares(adj)) {
                BoardSquare s = make_board_square(abcd[0], abcd[1], abcd[2], abcd[3]);
                if (!a_keys.count(sq_key(s))) out.push_back(s);
            }
            return out;
        }
        case SelectorOp::All: {
            std::vector<BoardSquare> out;
            for (auto& abcd : find_squares(adj)) out.push_back(make_board_square(abcd[0], abcd[1], abcd[2], abcd[3]));
            return out;
        }
        case SelectorOp::None:
            return {};
        case SelectorOp::Fromna: case SelectorOp::Fromne: {
            auto nodes = select_node(adj, pos, *sel.a);
            auto all = select_square(adj, pos, Selector{SelectorOp::All, SelectorType::Sq});
            return objects_from_nodes<BoardSquare>(
                all, nodes,
                +[](const BoardSquare& s) -> std::vector<int> { return { s.n1, s.n2, s.n3, s.n4 }; },
                sel.op == SelectorOp::Fromna);
        }
        case SelectorOp::Rrmn: {
            auto base = select_square(adj, pos, *sel.a);
            return randomly_remove(base, sel.count);
        }
        case SelectorOp::Rrmp: {
            auto base = select_square(adj, pos, *sel.a);
            int remove_count = static_cast<int>(std::floor(sel.frac * static_cast<double>(base.size())));
            return randomly_remove(base, remove_count);
        }
        default:
            throw std::runtime_error("select_square: unexpected square-selector op");
    }
}
