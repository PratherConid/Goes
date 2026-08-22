#include "game/selector.h"
#include <algorithm>
#include <cctype>
#include <cmath>
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

// Parses a node SEL - mutually recursive with parse_edge_sel_expr via n2e's own operand. Every
// Selector this returns has type == SelectorType::Node. Mirrors shared/selector.ts's
// parseNodeSelExpr().
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
    if (op == "n2e") {
        Selector sel;
        sel.op = SelectorOp::N2E;
        sel.type = SelectorType::Node;
        sel.a = std::make_shared<Selector>(parse_edge_sel_expr(c));
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

// Parses an edge SEL - mutually recursive with parse_node_sel_expr via e2n's own operand. Every
// Selector this returns has type == SelectorType::Edge. Mirrors shared/selector.ts's
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
    if (op == "e2n") {
        Selector sel;
        sel.op = SelectorOp::E2N;
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

// Shared by parse_node_selector/parse_edge_selector: tokenizes s, runs parse_expr over the whole
// thing, and rejects any leftover trailing input. Mirrors shared/selector.ts's parseTopLevel().
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
        case SelectorOp::E2N: return "(e2n " + format_selector(*sel.a) + ")";
        case SelectorOp::N2E: return "(n2e " + format_selector(*sel.a) + ")";
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
    return true;
}

// ── evaluation ───────────────────────────────────────────────────────────────

static int degree(const std::vector<std::vector<int>>& adj, int i) {
    int d = 0;
    for (int v : adj[i]) d += v ? 1 : 0;
    return d;
}

// BoardEdge itself isn't a valid std::set/std::unordered_map key (two structurally-equal BoardEdges
// are different objects, and it has no operator< or std::hash), so union/inter/diff/compl below key
// edges by this canonical numeric id (n1 < n2, so unique per edge) whenever they need set-like
// membership tests. Mirrors shared/selector.ts's edgeKey().
static long long edge_key(int N, const BoardEdge& e) {
    return static_cast<long long>(e.n1) * N + e.n2;
}

// Mirrors shared/selector.ts's dedupeEdges() - in particular, its use of a JS Map to dedupe: the
// LAST edge seen for a given key overwrites the value, but the FIRST occurrence's position in
// iteration order is kept (a Map.set on an existing key doesn't move it) - reproduced here via an
// index-into-`out` map alongside `out` itself, rather than plain std::map/unordered_map keyed by
// edge_key (which would drop that first-seen ordering).
static std::vector<BoardEdge> dedupe_edges(int N, const std::vector<BoardEdge>& edges) {
    std::unordered_map<long long, size_t> first_seen;
    std::vector<BoardEdge> out;
    for (auto& e : edges) {
        long long k = edge_key(N, e);
        auto it = first_seen.find(k);
        if (it == first_seen.end()) { first_seen.emplace(k, out.size()); out.push_back(e); }
        else out[it->second] = e;
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

std::set<int> select_node(const std::vector<std::vector<int>>& adj,
                           const std::vector<std::vector<unsigned>>& pos,
                           const Selector& sel) {
    if (sel.type != SelectorType::Node)
        throw std::runtime_error("select_node: expected a node selector, got an edge selector");
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
        case SelectorOp::N2E: {
            std::set<int> out;
            for (auto& e : select_edge(adj, pos, *sel.a)) { out.insert(e.n1); out.insert(e.n2); }
            return out;
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
        throw std::runtime_error("select_edge: expected an edge selector, got a node selector");
    int N = static_cast<int>(adj.size());
    switch (sel.op) {
        case SelectorOp::Union: {
            auto a = select_edge(adj, pos, *sel.a);
            auto b = select_edge(adj, pos, *sel.b);
            a.insert(a.end(), b.begin(), b.end());
            return dedupe_edges(N, a);
        }
        case SelectorOp::Inter: {
            auto a = select_edge(adj, pos, *sel.a);
            auto b = select_edge(adj, pos, *sel.b);
            std::set<long long> b_keys;
            for (auto& e : b) b_keys.insert(edge_key(N, e));
            std::vector<BoardEdge> out;
            for (auto& e : a) if (b_keys.count(edge_key(N, e))) out.push_back(e);
            return out;
        }
        case SelectorOp::Diff: {
            auto a = select_edge(adj, pos, *sel.a);
            auto b = select_edge(adj, pos, *sel.b);
            std::set<long long> b_keys;
            for (auto& e : b) b_keys.insert(edge_key(N, e));
            std::vector<BoardEdge> out;
            for (auto& e : a) if (!b_keys.count(edge_key(N, e))) out.push_back(e);
            return out;
        }
        case SelectorOp::Compl: {
            auto a = select_edge(adj, pos, *sel.a);
            std::set<long long> a_keys;
            for (auto& e : a) a_keys.insert(edge_key(N, e));
            std::vector<BoardEdge> out;
            for (int i = 0; i < N; i++)
                for (int j = i + 1; j < N; j++) {
                    if (!adj[i][j]) continue;
                    BoardEdge e = make_board_edge(i, j);
                    if (!a_keys.count(edge_key(N, e))) out.push_back(e);
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
            return dedupe_edges(N, out);
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
        case SelectorOp::E2N: {
            auto nodes = select_node(adj, pos, *sel.a);
            std::vector<BoardEdge> out;
            for (int i = 0; i < N; i++)
                for (int j = i + 1; j < N; j++)
                    if (adj[i][j] && nodes.count(i) && nodes.count(j)) out.push_back(make_board_edge(i, j));
            return out;
        }
        case SelectorOp::Rrmn: {
            auto base = select_edge(adj, pos, *sel.a);
            return randomly_remove(std::move(base), sel.count);
        }
        case SelectorOp::Rrmp: {
            auto base = select_edge(adj, pos, *sel.a);
            int remove_count = static_cast<int>(std::floor(sel.frac * static_cast<double>(base.size())));
            return randomly_remove(std::move(base), remove_count);
        }
        default:
            throw std::runtime_error("select_edge: unexpected edge-selector op");
    }
}
