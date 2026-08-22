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
static Selector parse_quad_sel_expr(ParseCursor& c);

// Reads conva/conve's own leading node/edge/tri/quad token (the "from" kind) and parses its operand
// via the matching one of parse_node_sel_expr/parse_edge_sel_expr/parse_triangle_sel_expr/
// parse_quad_sel_expr - shared by all four of those functions' own conva/conve case below,
// `to_type` being whichever of them called this (the "to" kind, from parsing context, same as
// all/none). Throws if the (from, to_type) pair is the one with no defined association (triangle
// <-> quad - see selector.h's own top comment); returns the parsed operand directly, unwrapped,
// for a same-kind conversion (a no-op). Mirrors shared/selector.ts's parseConversion().
static Selector parse_conversion(ParseCursor& c, SelectorOp op, SelectorType to_type) {
    std::string from_tok = c.next();
    SelectorType from;
    Selector (*parse_from)(ParseCursor&);
    if (from_tok == "node") { from = SelectorType::Node; parse_from = parse_node_sel_expr; }
    else if (from_tok == "edge") { from = SelectorType::Edge; parse_from = parse_edge_sel_expr; }
    else if (from_tok == "tri") { from = SelectorType::Tri; parse_from = parse_triangle_sel_expr; }
    else if (from_tok == "quad") { from = SelectorType::Quad; parse_from = parse_quad_sel_expr; }
    else throw std::runtime_error(
        "selector: (" + std::string(op == SelectorOp::Conva ? "conva" : "conve") +
        " ...) source kind must be 'node', 'edge', 'tri', or 'quad', got '" + from_tok + "'");
    if ((from == SelectorType::Tri && to_type == SelectorType::Quad) ||
        (from == SelectorType::Quad && to_type == SelectorType::Tri))
        throw std::runtime_error("selector: (" + std::string(op == SelectorOp::Conva ? "conva" : "conve") +
            " ...) has no association defined between 'tri' and 'quad'");
    Selector a = parse_from(c);
    c.expect(")");
    if (from == to_type) return a; // same-kind conversion is a no-op
    Selector sel;
    sel.op = op; sel.type = to_type; sel.from = from;
    sel.a = std::make_shared<Selector>(std::move(a));
    return sel;
}

// Parses a node SEL - mutually recursive with parse_edge_sel_expr/parse_triangle_sel_expr/
// parse_quad_sel_expr via conva/conve's own operand. Every Selector this returns has
// type == SelectorType::Node. Mirrors shared/selector.ts's parseNodeSelExpr().
static Selector parse_node_sel_expr(ParseCursor& c) {
    c.expect("(");
    std::string op = c.next();
    if (op == "union" || op == "inter") {
        Selector sel;
        sel.op = op == "union" ? SelectorOp::Union : SelectorOp::Inter;
        sel.type = SelectorType::Node;
        while (c.peek() != ")") sel.items.push_back(parse_node_sel_expr(c));
        c.expect(")");
        return sel;
    }
    if (op == "diff") {
        Selector sel;
        sel.op = SelectorOp::Diff;
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
        if (c.peek() != "(") sel.steps = next_nonneg_int(c, "(more ...) step count");
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
    if (op == "conva" || op == "conve")
        return parse_conversion(c, op == "conva" ? SelectorOp::Conva : SelectorOp::Conve, SelectorType::Node);
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

// Parses an edge SEL - mutually recursive with parse_node_sel_expr/parse_triangle_sel_expr/
// parse_quad_sel_expr via conva/conve's own operand. Every Selector this returns has
// type == SelectorType::Edge. Mirrors shared/selector.ts's parseEdgeSelExpr().
static Selector parse_edge_sel_expr(ParseCursor& c) {
    c.expect("(");
    std::string op = c.next();
    if (op == "union" || op == "inter") {
        Selector sel;
        sel.op = op == "union" ? SelectorOp::Union : SelectorOp::Inter;
        sel.type = SelectorType::Edge;
        while (c.peek() != ")") sel.items.push_back(parse_edge_sel_expr(c));
        c.expect(")");
        return sel;
    }
    if (op == "diff") {
        Selector sel;
        sel.op = SelectorOp::Diff;
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
        if (c.peek() != "(") sel.steps = next_nonneg_int(c, "(more ...) step count");
        sel.a = std::make_shared<Selector>(parse_edge_sel_expr(c));
        c.expect(")");
        return sel;
    }
    if (op == "all") { c.expect(")"); return Selector{SelectorOp::All, SelectorType::Edge}; }
    if (op == "none") { c.expect(")"); return Selector{SelectorOp::None, SelectorType::Edge}; }
    if (op == "conva" || op == "conve")
        return parse_conversion(c, op == "conva" ? SelectorOp::Conva : SelectorOp::Conve, SelectorType::Edge);
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

// Parses a triangle SEL - mutually recursive with parse_node_sel_expr/parse_edge_sel_expr via
// conva/conve's own operand (never parse_quad_sel_expr - see selector.h's own top comment on why
// triangle <-> quad is rejected). Every Selector this returns has type == SelectorType::Tri. No
// deg/more here. Mirrors shared/selector.ts's parseTriangleSelExpr().
static Selector parse_triangle_sel_expr(ParseCursor& c) {
    c.expect("(");
    std::string op = c.next();
    if (op == "union" || op == "inter") {
        Selector sel;
        sel.op = op == "union" ? SelectorOp::Union : SelectorOp::Inter;
        sel.type = SelectorType::Tri;
        while (c.peek() != ")") sel.items.push_back(parse_triangle_sel_expr(c));
        c.expect(")");
        return sel;
    }
    if (op == "diff") {
        Selector sel;
        sel.op = SelectorOp::Diff;
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
    if (op == "conva" || op == "conve")
        return parse_conversion(c, op == "conva" ? SelectorOp::Conva : SelectorOp::Conve, SelectorType::Tri);
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

// Parses a quad SEL - the quad counterpart of parse_triangle_sel_expr above (see its own doc
// comment). Every Selector this returns has type == SelectorType::Quad. Mirrors shared/selector.ts's
// parseQuadSelExpr().
static Selector parse_quad_sel_expr(ParseCursor& c) {
    c.expect("(");
    std::string op = c.next();
    if (op == "union" || op == "inter") {
        Selector sel;
        sel.op = op == "union" ? SelectorOp::Union : SelectorOp::Inter;
        sel.type = SelectorType::Quad;
        while (c.peek() != ")") sel.items.push_back(parse_quad_sel_expr(c));
        c.expect(")");
        return sel;
    }
    if (op == "diff") {
        Selector sel;
        sel.op = SelectorOp::Diff;
        sel.type = SelectorType::Quad;
        sel.a = std::make_shared<Selector>(parse_quad_sel_expr(c));
        sel.b = std::make_shared<Selector>(parse_quad_sel_expr(c));
        c.expect(")");
        return sel;
    }
    if (op == "compl") {
        Selector sel;
        sel.op = SelectorOp::Compl;
        sel.type = SelectorType::Quad;
        sel.a = std::make_shared<Selector>(parse_quad_sel_expr(c));
        c.expect(")");
        return sel;
    }
    if (op == "all") { c.expect(")"); return Selector{SelectorOp::All, SelectorType::Quad}; }
    if (op == "none") { c.expect(")"); return Selector{SelectorOp::None, SelectorType::Quad}; }
    if (op == "conva" || op == "conve")
        return parse_conversion(c, op == "conva" ? SelectorOp::Conva : SelectorOp::Conve, SelectorType::Quad);
    if (op == "rrmn") {
        int count = next_nonneg_int(c, "(rrmn ...) count");
        Selector sel;
        sel.op = SelectorOp::Rrmn; sel.type = SelectorType::Quad; sel.count = count;
        sel.a = std::make_shared<Selector>(parse_quad_sel_expr(c));
        c.expect(")");
        return sel;
    }
    if (op == "rrmp") {
        double frac = next_nonneg_number(c, "(rrmp ...) portion");
        Selector sel;
        sel.op = SelectorOp::Rrmp; sel.type = SelectorType::Quad; sel.frac = frac;
        sel.a = std::make_shared<Selector>(parse_quad_sel_expr(c));
        c.expect(")");
        return sel;
    }
    throw std::runtime_error("selector: unknown quad-selector operator '" + op + "'");
}

// Shared by parse_node_selector/parse_edge_selector/parse_triangle_selector/parse_quad_selector:
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
Selector parse_quad_selector(const std::string& s) { return parse_top_level(s, parse_quad_sel_expr); }

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
        case SelectorOp::Union: case SelectorOp::Inter: {
            std::string name = sel.op == SelectorOp::Union ? "union" : "inter";
            std::string inner;
            for (size_t i = 0; i < sel.items.size(); i++) {
                if (i) inner += " ";
                inner += format_selector(sel.items[i]);
            }
            return inner.empty() ? "(" + name + ")" : "(" + name + " " + inner + ")";
        }
        case SelectorOp::Diff:  return "(diff " + format_selector(*sel.a) + " " + format_selector(*sel.b) + ")";
        case SelectorOp::Compl: return "(compl " + format_selector(*sel.a) + ")";
        case SelectorOp::More:
            return sel.steps.has_value()
                ? "(more " + std::to_string(*sel.steps) + " " + format_selector(*sel.a) + ")"
                : "(more " + format_selector(*sel.a) + ")";
        case SelectorOp::All:   return "(all)";
        case SelectorOp::None:  return "(none)";
        case SelectorOp::Deg: {
            std::string cmp = sel.cmp == DegCmp::Eq ? "eq" : sel.cmp == DegCmp::Gt ? "gt" : "lt";
            return "(deg " + cmp + " " + std::to_string(sel.n) + ")";
        }
        case SelectorOp::Conva: case SelectorOp::Conve: {
            std::string name = sel.op == SelectorOp::Conva ? "conva" : "conve";
            std::string from = sel.from == SelectorType::Node ? "node" : sel.from == SelectorType::Edge ? "edge"
                : sel.from == SelectorType::Tri ? "tri" : "quad";
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
    if (op == SelectorOp::Conva || op == SelectorOp::Conve) return from == other.from;
    if (op == SelectorOp::More) return steps == other.steps;
    if (op == SelectorOp::Union || op == SelectorOp::Inter) return items == other.items;
    return true;
}

// ── evaluation ───────────────────────────────────────────────────────────────

// "a node"/"an edge"/"a tri"/"a quad" - shared by each evaluator's own wrong-kind error message below.
static std::string describe_selector_type(SelectorType type) {
    switch (type) {
        case SelectorType::Node: return "a node";
        case SelectorType::Edge: return "an edge";
        case SelectorType::Tri:  return "a tri";
        case SelectorType::Quad: return "a quad";
    }
    throw std::runtime_error("describe_selector_type: unknown type");
}

static int degree(const std::vector<std::vector<int>>& adj, int i) {
    int d = 0;
    for (int v : adj[i]) d += v ? 1 : 0;
    return d;
}

// BoardEdge/BoardTriangle/BoardQuad aren't valid std::set/std::unordered_map keys themselves (two
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
static std::string quad_key(const BoardQuad& s) {
    return std::to_string(s.n1) + "," + std::to_string(s.n2) + "," + std::to_string(s.n3) + "," + std::to_string(s.n4);
}

// Generic Map-based dedupe, keyed by key(item) - the last item seen for a given key overwrites the
// value, but the FIRST occurrence's position in iteration order is kept (matches a JS Map's own
// set-on-existing-key semantics, which shared/selector.ts's own dedupeByKey() relies on) - reproduced
// here via an index-into-`out` map alongside `out` itself, rather than plain std::map/unordered_map
// keyed directly by K (which would drop that first-seen ordering). Shared by every object kind's own
// `union` case below.
template <typename T, typename K>
static std::vector<T> dedupe_by_key(const std::vector<T>& items, K (*key)(const T&)) {
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

// Mirrors shared/selector.ts's node member/key convention for edge/triangle/quad (a plain node
// index is its own key; a single-element member list) - used wherever "node" is conva/conve's own
// "from" or "to" kind in convert_objects below.
static std::vector<int> node_members(const int& n) { return { n }; }
static int node_key(const int& n) { return n; }

// True iff `a`'s own members are completely contained in `b`'s, or vice versa - the general
// "association" test conva/conve rely on (see selector.h's own top comment). Every object kind here
// has a fixed arity (node 1, edge 2, triangle 3, quad 4) and every object's own members are
// distinct node indices, so containment can only ever run from the smaller-arity list into the
// larger one; this checks whichever direction applies rather than assuming a fixed order.
static bool is_associated(const std::vector<int>& a, const std::vector<int>& b) {
    const std::vector<int>& small = a.size() <= b.size() ? a : b;
    const std::vector<int>& large = a.size() <= b.size() ? b : a;
    std::set<int> large_set(large.begin(), large.end());
    return std::all_of(small.begin(), small.end(), [&](int x) { return large_set.count(x) > 0; });
}

// Shared by every evaluator's own conva/conve case: `all_to`/`to_members` enumerate every object of
// THIS evaluator's own kind (the "to" kind) in the whole graph; `all_from`/`from_members`/`from_key`
// do the same for SEL's own declared source kind, and `selected_from_keys` is which of those SEL's
// own operand selects. A "to" object is kept iff ALL (require_all, conva) or AT LEAST ONE (conve) of
// its associated "from" objects (per is_associated above) are selected - vacuously true/false
// (respectively) for a "to" object with no associated "from" objects at all, per ordinary
// all_of/any_of semantics on an empty range. Mirrors shared/selector.ts's convertObjects().
template <typename F, typename T, typename K>
static std::vector<T> convert_objects(
    const std::vector<T>& all_to, std::vector<int> (*to_members)(const T&),
    const std::vector<F>& all_from, std::vector<int> (*from_members)(const F&),
    K (*from_key)(const F&), const std::set<K>& selected_from_keys, bool require_all)
{
    std::vector<T> out;
    for (auto& to : all_to) {
        auto to_m = to_members(to);
        std::vector<const F*> associated;
        for (auto& from : all_from) if (is_associated(to_m, from_members(from))) associated.push_back(&from);
        auto is_selected = [&](const F* from) { return selected_from_keys.count(from_key(*from)) > 0; };
        bool matches = require_all
            ? std::all_of(associated.begin(), associated.end(), is_selected)
            : std::any_of(associated.begin(), associated.end(), is_selected);
        if (matches) out.push_back(to);
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
            std::set<int> out;
            for (auto& item : sel.items) {
                auto s = select_node(adj, pos, item);
                out.insert(s.begin(), s.end());
            }
            return out;
        }
        case SelectorOp::Inter: {
            if (sel.items.empty()) return select_node(adj, pos, Selector{SelectorOp::All, SelectorType::Node});
            auto acc = select_node(adj, pos, sel.items[0]);
            for (size_t i = 1; i < sel.items.size(); i++) {
                auto next = select_node(adj, pos, sel.items[i]);
                std::set<int> out;
                for (int x : acc) if (next.count(x)) out.insert(x);
                acc = std::move(out);
            }
            return acc;
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
            // Repeats the one-step expansion sel.steps times (default 1, see selector.h's own doc
            // comment on Selector::steps) - each step only walks `frontier` (the nodes newly added by
            // the PREVIOUS step, not the whole accumulated `out` again), since a node's own neighbors
            // were already fully explored the one time it itself became part of the frontier.
            std::vector<int> frontier(a.begin(), a.end());
            int steps = sel.steps.value_or(1);
            for (int s = 0; s < steps && !frontier.empty(); s++) {
                std::vector<int> next_frontier;
                for (int i : frontier)
                    for (int j = 0; j < N; j++)
                        if (adj[i][j] && !out.count(j)) { out.insert(j); next_frontier.push_back(j); }
                frontier = std::move(next_frontier);
            }
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
        case SelectorOp::Conva: case SelectorOp::Conve: {
            bool require_all = sel.op == SelectorOp::Conva;
            if (sel.from == SelectorType::Node) return select_node(adj, pos, *sel.a); // same-kind: no-op (defensive)
            std::vector<int> to_nodes(N);
            for (int i = 0; i < N; i++) to_nodes[i] = i;
            if (sel.from == SelectorType::Edge) {
                auto all_from = select_edge(adj, pos, Selector{SelectorOp::All, SelectorType::Edge});
                std::set<std::string> selected_keys;
                for (auto& e : select_edge(adj, pos, *sel.a)) selected_keys.insert(edge_key(e));
                auto result = convert_objects<BoardEdge, int, std::string>(
                    to_nodes, node_members, all_from,
                    +[](const BoardEdge& e) -> std::vector<int> { return { e.n1, e.n2 }; },
                    edge_key, selected_keys, require_all);
                return std::set<int>(result.begin(), result.end());
            }
            if (sel.from == SelectorType::Tri) {
                auto all_from = select_triangle(adj, pos, Selector{SelectorOp::All, SelectorType::Tri});
                std::set<std::string> selected_keys;
                for (auto& t : select_triangle(adj, pos, *sel.a)) selected_keys.insert(tri_key(t));
                auto result = convert_objects<BoardTriangle, int, std::string>(
                    to_nodes, node_members, all_from,
                    +[](const BoardTriangle& t) -> std::vector<int> { return { t.n1, t.n2, t.n3 }; },
                    tri_key, selected_keys, require_all);
                return std::set<int>(result.begin(), result.end());
            }
            auto all_from = select_quad(adj, pos, Selector{SelectorOp::All, SelectorType::Quad});
            std::set<std::string> selected_keys;
            for (auto& s : select_quad(adj, pos, *sel.a)) selected_keys.insert(quad_key(s));
            auto result = convert_objects<BoardQuad, int, std::string>(
                to_nodes, node_members, all_from,
                +[](const BoardQuad& s) -> std::vector<int> { return { s.n1, s.n2, s.n3, s.n4 }; },
                quad_key, selected_keys, require_all);
            return std::set<int>(result.begin(), result.end());
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
            std::vector<BoardEdge> all;
            for (auto& item : sel.items) {
                auto s = select_edge(adj, pos, item);
                all.insert(all.end(), s.begin(), s.end());
            }
            return dedupe_by_key<BoardEdge, std::string>(all, edge_key);
        }
        case SelectorOp::Inter: {
            if (sel.items.empty()) return select_edge(adj, pos, Selector{SelectorOp::All, SelectorType::Edge});
            auto acc = select_edge(adj, pos, sel.items[0]);
            for (size_t i = 1; i < sel.items.size(); i++) {
                auto next = select_edge(adj, pos, sel.items[i]);
                std::set<std::string> next_keys;
                for (auto& e : next) next_keys.insert(edge_key(e));
                std::vector<BoardEdge> out;
                for (auto& e : acc) if (next_keys.count(edge_key(e))) out.push_back(e);
                acc = std::move(out);
            }
            return acc;
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
            std::vector<BoardEdge> out = a;
            std::set<std::string> seen;
            for (auto& e : a) seen.insert(edge_key(e));
            std::set<int> touched_nodes;
            for (auto& e : a) { touched_nodes.insert(e.n1); touched_nodes.insert(e.n2); }
            // Repeats the one-step expansion sel.steps times (default 1). Mirrors select_node's own
            // frontier trick: `frontier` is only the nodes newly touched by the PREVIOUS step, since a
            // node's own incident edges are all added to `out`/`seen` the one time it itself becomes
            // part of the frontier - a later step never needs to re-scan an already-touched node.
            std::vector<int> frontier(touched_nodes.begin(), touched_nodes.end());
            int steps = sel.steps.value_or(1);
            for (int s = 0; s < steps && !frontier.empty(); s++) {
                std::vector<int> next_frontier;
                for (int i : frontier)
                    for (int j = 0; j < N; j++) {
                        if (!adj[i][j]) continue;
                        BoardEdge e = make_board_edge(i, j);
                        if (seen.insert(edge_key(e)).second) out.push_back(e);
                        if (!touched_nodes.count(j)) { touched_nodes.insert(j); next_frontier.push_back(j); }
                    }
                frontier = std::move(next_frontier);
            }
            return out;
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
        case SelectorOp::Conva: case SelectorOp::Conve: {
            bool require_all = sel.op == SelectorOp::Conva;
            if (sel.from == SelectorType::Edge) return select_edge(adj, pos, *sel.a); // same-kind: no-op (defensive)
            auto all_to = select_edge(adj, pos, Selector{SelectorOp::All, SelectorType::Edge});
            auto edge_members = +[](const BoardEdge& e) -> std::vector<int> { return { e.n1, e.n2 }; };
            if (sel.from == SelectorType::Node) {
                std::set<int> selected_keys;
                for (int n : select_node(adj, pos, *sel.a)) selected_keys.insert(n);
                std::vector<int> all_from(N);
                for (int i = 0; i < N; i++) all_from[i] = i;
                return convert_objects<int, BoardEdge, int>(
                    all_to, edge_members, all_from, node_members, node_key, selected_keys, require_all);
            }
            if (sel.from == SelectorType::Tri) {
                auto all_from = select_triangle(adj, pos, Selector{SelectorOp::All, SelectorType::Tri});
                std::set<std::string> selected_keys;
                for (auto& t : select_triangle(adj, pos, *sel.a)) selected_keys.insert(tri_key(t));
                return convert_objects<BoardTriangle, BoardEdge, std::string>(
                    all_to, edge_members, all_from,
                    +[](const BoardTriangle& t) -> std::vector<int> { return { t.n1, t.n2, t.n3 }; },
                    tri_key, selected_keys, require_all);
            }
            auto all_from = select_quad(adj, pos, Selector{SelectorOp::All, SelectorType::Quad});
            std::set<std::string> selected_keys;
            for (auto& s : select_quad(adj, pos, *sel.a)) selected_keys.insert(quad_key(s));
            return convert_objects<BoardQuad, BoardEdge, std::string>(
                all_to, edge_members, all_from,
                +[](const BoardQuad& s) -> std::vector<int> { return { s.n1, s.n2, s.n3, s.n4 }; },
                quad_key, selected_keys, require_all);
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
            std::vector<BoardTriangle> all;
            for (auto& item : sel.items) {
                auto s = select_triangle(adj, pos, item);
                all.insert(all.end(), s.begin(), s.end());
            }
            std::set<std::string> seen;
            std::vector<BoardTriangle> deduped;
            for (auto& t : all) if (seen.insert(tri_key(t)).second) deduped.push_back(t);
            return deduped;
        }
        case SelectorOp::Inter: {
            if (sel.items.empty()) return select_triangle(adj, pos, Selector{SelectorOp::All, SelectorType::Tri});
            auto acc = select_triangle(adj, pos, sel.items[0]);
            for (size_t i = 1; i < sel.items.size(); i++) {
                auto next = select_triangle(adj, pos, sel.items[i]);
                std::set<std::string> next_keys;
                for (auto& t : next) next_keys.insert(tri_key(t));
                std::vector<BoardTriangle> out;
                for (auto& t : acc) if (next_keys.count(tri_key(t))) out.push_back(t);
                acc = std::move(out);
            }
            return acc;
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
            for (auto& t : find_triangles(adj)) if (!a_keys.count(tri_key(t))) out.push_back(t);
            return out;
        }
        case SelectorOp::All:
            return find_triangles(adj);
        case SelectorOp::None:
            return {};
        case SelectorOp::Conva: case SelectorOp::Conve: {
            if (sel.from == SelectorType::Quad)
                throw std::runtime_error("select_triangle: no association is defined between 'tri' and 'quad'");
            bool require_all = sel.op == SelectorOp::Conva;
            if (sel.from == SelectorType::Tri) return select_triangle(adj, pos, *sel.a); // same-kind: no-op (defensive)
            auto all_to = select_triangle(adj, pos, Selector{SelectorOp::All, SelectorType::Tri});
            auto tri_members = +[](const BoardTriangle& t) -> std::vector<int> { return { t.n1, t.n2, t.n3 }; };
            if (sel.from == SelectorType::Node) {
                std::set<int> selected_keys;
                for (int n : select_node(adj, pos, *sel.a)) selected_keys.insert(n);
                std::vector<int> all_from(adj.size());
                for (size_t i = 0; i < adj.size(); i++) all_from[i] = static_cast<int>(i);
                return convert_objects<int, BoardTriangle, int>(
                    all_to, tri_members, all_from, node_members, node_key, selected_keys, require_all);
            }
            // sel.from == SelectorType::Edge
            auto all_from = select_edge(adj, pos, Selector{SelectorOp::All, SelectorType::Edge});
            std::set<std::string> selected_keys;
            for (auto& e : select_edge(adj, pos, *sel.a)) selected_keys.insert(edge_key(e));
            return convert_objects<BoardEdge, BoardTriangle, std::string>(
                all_to, tri_members, all_from,
                +[](const BoardEdge& e) -> std::vector<int> { return { e.n1, e.n2 }; },
                edge_key, selected_keys, require_all);
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

std::vector<BoardQuad> select_quad(const std::vector<std::vector<int>>& adj,
                                        const std::vector<std::vector<unsigned>>& pos,
                                        const Selector& sel) {
    if (sel.type != SelectorType::Quad)
        throw std::runtime_error(
            "select_quad: expected a quad selector, got " + describe_selector_type(sel.type));
    switch (sel.op) {
        case SelectorOp::Union: {
            std::vector<BoardQuad> all;
            for (auto& item : sel.items) {
                auto s = select_quad(adj, pos, item);
                all.insert(all.end(), s.begin(), s.end());
            }
            std::set<std::string> seen;
            std::vector<BoardQuad> deduped;
            for (auto& s : all) if (seen.insert(quad_key(s)).second) deduped.push_back(s);
            return deduped;
        }
        case SelectorOp::Inter: {
            if (sel.items.empty()) return select_quad(adj, pos, Selector{SelectorOp::All, SelectorType::Quad});
            auto acc = select_quad(adj, pos, sel.items[0]);
            for (size_t i = 1; i < sel.items.size(); i++) {
                auto next = select_quad(adj, pos, sel.items[i]);
                std::set<std::string> next_keys;
                for (auto& s : next) next_keys.insert(quad_key(s));
                std::vector<BoardQuad> out;
                for (auto& s : acc) if (next_keys.count(quad_key(s))) out.push_back(s);
                acc = std::move(out);
            }
            return acc;
        }
        case SelectorOp::Diff: {
            auto a = select_quad(adj, pos, *sel.a);
            std::set<std::string> b_keys;
            for (auto& s : select_quad(adj, pos, *sel.b)) b_keys.insert(quad_key(s));
            std::vector<BoardQuad> out;
            for (auto& s : a) if (!b_keys.count(quad_key(s))) out.push_back(s);
            return out;
        }
        case SelectorOp::Compl: {
            std::set<std::string> a_keys;
            for (auto& s : select_quad(adj, pos, *sel.a)) a_keys.insert(quad_key(s));
            std::vector<BoardQuad> out;
            for (auto& s : find_quads(adj)) if (!a_keys.count(quad_key(s))) out.push_back(s);
            return out;
        }
        case SelectorOp::All:
            return find_quads(adj);
        case SelectorOp::None:
            return {};
        case SelectorOp::Conva: case SelectorOp::Conve: {
            if (sel.from == SelectorType::Tri)
                throw std::runtime_error("select_quad: no association is defined between 'tri' and 'quad'");
            bool require_all = sel.op == SelectorOp::Conva;
            if (sel.from == SelectorType::Quad) return select_quad(adj, pos, *sel.a); // same-kind: no-op (defensive)
            auto all_to = select_quad(adj, pos, Selector{SelectorOp::All, SelectorType::Quad});
            auto quad_members = +[](const BoardQuad& s) -> std::vector<int> { return { s.n1, s.n2, s.n3, s.n4 }; };
            if (sel.from == SelectorType::Node) {
                std::set<int> selected_keys;
                for (int n : select_node(adj, pos, *sel.a)) selected_keys.insert(n);
                std::vector<int> all_from(adj.size());
                for (size_t i = 0; i < adj.size(); i++) all_from[i] = static_cast<int>(i);
                return convert_objects<int, BoardQuad, int>(
                    all_to, quad_members, all_from, node_members, node_key, selected_keys, require_all);
            }
            // sel.from == SelectorType::Edge
            auto all_from = select_edge(adj, pos, Selector{SelectorOp::All, SelectorType::Edge});
            std::set<std::string> selected_keys;
            for (auto& e : select_edge(adj, pos, *sel.a)) selected_keys.insert(edge_key(e));
            return convert_objects<BoardEdge, BoardQuad, std::string>(
                all_to, quad_members, all_from,
                +[](const BoardEdge& e) -> std::vector<int> { return { e.n1, e.n2 }; },
                edge_key, selected_keys, require_all);
        }
        case SelectorOp::Rrmn: {
            auto base = select_quad(adj, pos, *sel.a);
            return randomly_remove(base, sel.count);
        }
        case SelectorOp::Rrmp: {
            auto base = select_quad(adj, pos, *sel.a);
            int remove_count = static_cast<int>(std::floor(sel.frac * static_cast<double>(base.size())));
            return randomly_remove(base, remove_count);
        }
        default:
            throw std::runtime_error("select_quad: unexpected quad-selector op");
    }
}
