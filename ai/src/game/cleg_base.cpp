#include "game/cleg_base.h"
#include <cmath>
#include <set>
#include <sstream>
#include <stdexcept>

// Mirrors shared/clegBase.ts - see game/cleg_base.h's own top comment for the full context. This
// file only implements the declarations in that header; every function here that isn't declared
// there (ctkind_word, type_to_string_for_suffix, format_number_exact, edge_key_str/tri_key_str/
// quad_key_str, set_union_vals/set_intersect_vals/set_diff_vals, and the number_overload()/etc.
// overload builders) is a private implementation detail of this one file, same as it was a private
// (`static`) detail of the single cleg.cpp this was split out of.

bool type_equals(const ClegType& a, const ClegType& b) {
    if (a.kind != b.kind) return false;
    if (a.kind == CTKind::Array || a.kind == CTKind::Set) return type_equals(*a.elem, *b.elem);
    if (a.kind == CTKind::Func) {
        if (a.params.size() != b.params.size()) return false;
        for (size_t i = 0; i < a.params.size(); i++) if (!type_equals(a.params[i], b.params[i])) return false;
        return type_equals(*a.return_type, *b.return_type);
    }
    return true;
}

static std::string ctkind_word(CTKind k) {
    switch (k) {
        case CTKind::Egr: return "egr";
        case CTKind::Number: return "number";
        case CTKind::String: return "string";
        case CTKind::Bool: return "bool";
        case CTKind::Edge: return "edge";
        case CTKind::Simp: return "simp";
        case CTKind::Quad: return "quad";
        case CTKind::Sel: return "sel";
        case CTKind::Mod: return "mod";
        case CTKind::Lrs: return "lrs";
        case CTKind::Msel: return "msel";
        case CTKind::Array: case CTKind::Set: case CTKind::Func: break; // handled by type_to_string below
    }
    throw std::runtime_error("cleg: ctkind_word: unexpected kind");
}

// A func type printed directly before a `[]`/`{}` suffix needs its own extra parens (matching
// game/cleg_parser.cpp's own parse_paren_type grouping rule) - otherwise the suffix would silently
// re-parse as binding to the func type's own return type instead of to the func type as a whole
// (`(number, number) -> bool[]` means "returns bool[]", not "an array of these functions" - see
// game/cleg_base.h's own top comment). Every other ClegType kind is unambiguous either way.
static std::string type_to_string_for_suffix(const ClegType& t) {
    return t.kind == CTKind::Func ? "(" + type_to_string(t) + ")" : type_to_string(t);
}

std::string type_to_string(const ClegType& t) {
    if (t.kind == CTKind::Array) return type_to_string_for_suffix(*t.elem) + "[]";
    if (t.kind == CTKind::Set) return type_to_string_for_suffix(*t.elem) + "{}";
    if (t.kind == CTKind::Func) {
        std::string params;
        for (size_t i = 0; i < t.params.size(); i++) { if (i) params += ", "; params += type_to_string(t.params[i]); }
        return "(" + params + ") -> " + type_to_string(*t.return_type);
    }
    return ctkind_word(t.kind);
}

ClegType cleg_value_type(const ClegValue& v) {
    if (v.kind == CTKind::Array || v.kind == CTKind::Set)
        return ClegType{v.kind, std::make_shared<ClegType>(v.elem)};
    if (v.kind == CTKind::Func)
        return ClegType{CTKind::Func, nullptr, v.func_params, std::make_shared<ClegType>(v.func_return_type)};
    return ClegType{v.kind, nullptr};
}

std::string format_number_display(double d) {
    if (d == std::floor(d) && std::abs(d) < 1e15) return std::to_string(static_cast<long long>(d));
    std::ostringstream oss;
    oss.precision(15);
    oss << d;
    return oss.str();
}

// Full round-trip precision, for cleg_set_key below only (an internal dedup key, never user-facing
// text) - unlike format_number_display, this must never collide two distinct doubles.
static std::string format_number_exact(double d) {
    std::ostringstream oss;
    oss.precision(17);
    oss << d;
    return oss.str();
}

static std::string edge_key_str(const BoardEdge& e) {
    return std::to_string(e.n1) + "," + std::to_string(e.n2);
}
// A plain comma-join of nodes.size()-many indices is already an unambiguous, collision-free key
// across every arity - mirrors shared/clegBase.ts's own clegSetKey 'simp' case doc comment on why no
// separate arity tag is needed.
static std::string simp_key_str(const BoardSimplex& t) {
    std::string key;
    for (size_t i = 0; i < t.nodes.size(); i++) {
        if (i) key += ",";
        key += std::to_string(t.nodes[i]);
    }
    return key;
}
static std::string quad_key_str(const BoardQuad& q) {
    return std::to_string(q.n1) + "," + std::to_string(q.n2) + "," + std::to_string(q.n3) + "," + std::to_string(q.n4);
}

std::string cleg_set_key(const ClegValue& v) {
    switch (v.kind) {
        case CTKind::Number: return "n:" + format_number_exact(v.number);
        case CTKind::String: return "s:" + v.str;
        case CTKind::Bool: return v.boolean ? "b:1" : "b:0";
        case CTKind::Edge: return "e:" + edge_key_str(v.edge_v);
        case CTKind::Simp: return "s:" + simp_key_str(v.simp_v);
        case CTKind::Quad: return "q:" + quad_key_str(v.quad_v);
        default:
            throw std::runtime_error("cleg: '" + type_to_string(cleg_value_type(v)) + "' cannot be a set element");
    }
}

ClegValue make_cleg_set(ClegType elem, std::vector<ClegValue> values) {
    std::unordered_map<std::string, size_t> seen;
    std::vector<ClegValue> out;
    for (auto& v : values) {
        std::string k = cleg_set_key(v);
        if (seen.find(k) == seen.end()) { seen.emplace(k, out.size()); out.push_back(std::move(v)); }
    }
    ClegValue result;
    result.kind = CTKind::Set;
    result.elem = std::move(elem);
    result.arr_v = std::move(out);
    return result;
}

static ClegValue set_union_vals(const ClegValue& a, const ClegValue& b) {
    std::vector<ClegValue> combined = a.arr_v;
    combined.insert(combined.end(), b.arr_v.begin(), b.arr_v.end());
    return make_cleg_set(a.elem, std::move(combined));
}
// Intersection/difference only ever remove elements from `a`, which is already deduplicated, so
// unlike set_union_vals neither needs to go through make_cleg_set again - mirrors
// shared/clegBase.ts's setIntersect()/setDiff().
static ClegValue set_intersect_vals(const ClegValue& a, const ClegValue& b) {
    std::set<std::string> b_keys;
    for (auto& v : b.arr_v) b_keys.insert(cleg_set_key(v));
    std::vector<ClegValue> out;
    for (auto& v : a.arr_v) if (b_keys.count(cleg_set_key(v))) out.push_back(v);
    ClegValue result; result.kind = CTKind::Set; result.elem = a.elem; result.arr_v = std::move(out);
    return result;
}
static ClegValue set_diff_vals(const ClegValue& a, const ClegValue& b) {
    std::set<std::string> b_keys;
    for (auto& v : b.arr_v) b_keys.insert(cleg_set_key(v));
    std::vector<ClegValue> out;
    for (auto& v : a.arr_v) if (!b_keys.count(cleg_set_key(v))) out.push_back(v);
    ClegValue result; result.kind = CTKind::Set; result.elem = a.elem; result.arr_v = std::move(out);
    return result;
}

static BinaryOverload number_overload(std::function<double(double, double)> compute) {
    return BinaryOverload{
        "number, number -> number",
        [compute](const ClegType& l, const ClegType& r) -> std::optional<MatchResult> {
            if (l.kind != CTKind::Number || r.kind != CTKind::Number) return std::nullopt;
            return MatchResult{
                ClegType{CTKind::Number, nullptr},
                [compute](const ClegValue& lv, const ClegValue& rv) { return make_number(compute(lv.number, rv.number)); },
            };
        },
    };
}

static BinaryOverload string_concat_overload() {
    return BinaryOverload{
        "string, string -> string (concatenation)",
        [](const ClegType& l, const ClegType& r) -> std::optional<MatchResult> {
            if (l.kind != CTKind::String || r.kind != CTKind::String) return std::nullopt;
            return MatchResult{
                ClegType{CTKind::String, nullptr},
                [](const ClegValue& lv, const ClegValue& rv) { return make_string(lv.str + rv.str); },
            };
        },
    };
}

static BinaryOverload number_string_concat_overload() {
    return BinaryOverload{
        "number, string -> string (or string, number -> string; concatenation)",
        [](const ClegType& l, const ClegType& r) -> std::optional<MatchResult> {
            if (l.kind == CTKind::Number && r.kind == CTKind::String)
                return MatchResult{
                    ClegType{CTKind::String, nullptr},
                    [](const ClegValue& lv, const ClegValue& rv) {
                        return make_string(format_number_display(lv.number) + rv.str);
                    },
                };
            if (l.kind == CTKind::String && r.kind == CTKind::Number)
                return MatchResult{
                    ClegType{CTKind::String, nullptr},
                    [](const ClegValue& lv, const ClegValue& rv) {
                        return make_string(lv.str + format_number_display(rv.number));
                    },
                };
            return std::nullopt;
        },
    };
}

static BinaryOverload array_concat_overload() {
    return BinaryOverload{
        "T[], T[] -> T[] (same T)",
        [](const ClegType& l, const ClegType& r) -> std::optional<MatchResult> {
            if (l.kind != CTKind::Array || r.kind != CTKind::Array || !type_equals(l, r)) return std::nullopt;
            ClegType t = l;
            return MatchResult{
                t,
                [t](const ClegValue& lv, const ClegValue& rv) {
                    ClegValue out; out.kind = CTKind::Array; out.elem = *t.elem;
                    out.arr_v = lv.arr_v;
                    out.arr_v.insert(out.arr_v.end(), rv.arr_v.begin(), rv.arr_v.end());
                    return out;
                },
            };
        },
    };
}

// Mirrors shared/clegBase.ts's requireRepeatCount() - `n` isn't statically known, checked here at
// evaluation time.
static int require_repeat_count(double n) {
    if (n != std::floor(n) || n < 0)
        throw std::runtime_error("cleg: '*' replication count must be a nonnegative integer, got " + format_number_display(n));
    return static_cast<int>(n);
}
static ClegValue repeat_array_value(const ClegValue& arr, double count) {
    int n = require_repeat_count(count);
    ClegValue out; out.kind = CTKind::Array; out.elem = arr.elem;
    out.arr_v.reserve(static_cast<size_t>(n) * arr.arr_v.size());
    for (int i = 0; i < n; i++) out.arr_v.insert(out.arr_v.end(), arr.arr_v.begin(), arr.arr_v.end());
    return out;
}
static BinaryOverload repeat_array_overload() {
    return BinaryOverload{
        "T[], number -> T[] (or number, T[] -> T[]; replication - n must be a nonnegative integer)",
        [](const ClegType& l, const ClegType& r) -> std::optional<MatchResult> {
            if (l.kind == CTKind::Array && r.kind == CTKind::Number) {
                ClegType t = l;
                return MatchResult{ t, [](const ClegValue& lv, const ClegValue& rv) { return repeat_array_value(lv, rv.number); } };
            }
            if (l.kind == CTKind::Number && r.kind == CTKind::Array) {
                ClegType t = r;
                return MatchResult{ t, [](const ClegValue& lv, const ClegValue& rv) { return repeat_array_value(rv, lv.number); } };
            }
            return std::nullopt;
        },
    };
}
static ClegValue repeat_string_value(const std::string& s, double count) {
    int n = require_repeat_count(count);
    std::string out; out.reserve(s.size() * static_cast<size_t>(n));
    for (int i = 0; i < n; i++) out += s;
    return make_string(std::move(out));
}
static BinaryOverload repeat_string_overload() {
    return BinaryOverload{
        "string, number -> string (or number, string -> string; replication - n must be a nonnegative integer)",
        [](const ClegType& l, const ClegType& r) -> std::optional<MatchResult> {
            if (l.kind == CTKind::String && r.kind == CTKind::Number)
                return MatchResult{
                    ClegType{CTKind::String, nullptr},
                    [](const ClegValue& lv, const ClegValue& rv) { return repeat_string_value(lv.str, rv.number); },
                };
            if (l.kind == CTKind::Number && r.kind == CTKind::String)
                return MatchResult{
                    ClegType{CTKind::String, nullptr},
                    [](const ClegValue& lv, const ClegValue& rv) { return repeat_string_value(rv.str, lv.number); },
                };
            return std::nullopt;
        },
    };
}

using SetCombineFn = ClegValue (*)(const ClegValue&, const ClegValue&);
static BinaryOverload set_overload(std::string signature, SetCombineFn combine) {
    return BinaryOverload{
        std::move(signature),
        [combine](const ClegType& l, const ClegType& r) -> std::optional<MatchResult> {
            if (l.kind != CTKind::Set || r.kind != CTKind::Set || !type_equals(l, r)) return std::nullopt;
            ClegType t = l;
            return MatchResult{ t, [combine](const ClegValue& lv, const ClegValue& rv) { return combine(lv, rv); } };
        },
    };
}

// Mirrors shared/clegBase.ts's toComparable() - bool compares via C++'s own false=0/true=1 convention.
static double to_comparable(const ClegValue& v) { return v.kind == CTKind::Bool ? (v.boolean ? 1.0 : 0.0) : v.number; }

static BinaryOverload comparison_overload(CTKind elem_kind, std::function<bool(double, double)> compute) {
    std::string word = ctkind_word(elem_kind);
    return BinaryOverload{
        word + ", " + word + " -> bool",
        [elem_kind, compute](const ClegType& l, const ClegType& r) -> std::optional<MatchResult> {
            if (l.kind != elem_kind || r.kind != elem_kind) return std::nullopt;
            return MatchResult{
                ClegType{CTKind::Bool, nullptr},
                [compute](const ClegValue& lv, const ClegValue& rv) { return make_bool(compute(to_comparable(lv), to_comparable(rv))); },
            };
        },
    };
}

// Mirrors shared/clegBase.ts's logicalOverload() - `eval` here is never actually reached at runtime
// (game/cleg_eval.cpp's own eval_expr BinaryExpr case short-circuits `&&`/`||` itself), kept only for
// interface consistency with every other overload and so check_expr can type both operands regardless.
static BinaryOverload logical_overload(std::function<bool(bool, bool)> compute) {
    return BinaryOverload{
        "bool, bool -> bool",
        [compute](const ClegType& l, const ClegType& r) -> std::optional<MatchResult> {
            if (l.kind != CTKind::Bool || r.kind != CTKind::Bool) return std::nullopt;
            return MatchResult{
                ClegType{CTKind::Bool, nullptr},
                [compute](const ClegValue& lv, const ClegValue& rv) { return make_bool(compute(lv.boolean, rv.boolean)); },
            };
        },
    };
}

const std::unordered_map<std::string, std::vector<BinaryOverload>>& binary_operator_overloads() {
    static const std::unordered_map<std::string, std::vector<BinaryOverload>> table = [] {
        std::unordered_map<std::string, std::vector<BinaryOverload>> m;
        m["+"] = { number_overload([](double a, double b) { return a + b; }), string_concat_overload(),
                   number_string_concat_overload(), array_concat_overload(),
                   set_overload("T{}, T{} -> T{} (same T; union)", set_union_vals) };
        m["-"] = { number_overload([](double a, double b) { return a - b; }),
                   set_overload("T{}, T{} -> T{} (same T; difference)", set_diff_vals) };
        m["*"] = { number_overload([](double a, double b) { return a * b; }),
                   set_overload("T{}, T{} -> T{} (same T; intersection)", set_intersect_vals),
                   repeat_array_overload(), repeat_string_overload() };
        m["/"] = { number_overload([](double a, double b) { return a / b; }) };
        m["%"] = { number_overload([](double a, double b) { return std::fmod(a, b); }) };
        m["=="] = { comparison_overload(CTKind::Number, [](double a, double b) { return a == b; }),
                    comparison_overload(CTKind::Bool, [](double a, double b) { return a == b; }) };
        m["!="] = { comparison_overload(CTKind::Number, [](double a, double b) { return a != b; }),
                    comparison_overload(CTKind::Bool, [](double a, double b) { return a != b; }) };
        m["<"] = { comparison_overload(CTKind::Number, [](double a, double b) { return a < b; }),
                   comparison_overload(CTKind::Bool, [](double a, double b) { return a < b; }) };
        m[">"] = { comparison_overload(CTKind::Number, [](double a, double b) { return a > b; }),
                   comparison_overload(CTKind::Bool, [](double a, double b) { return a > b; }) };
        m["<="] = { comparison_overload(CTKind::Number, [](double a, double b) { return a <= b; }),
                    comparison_overload(CTKind::Bool, [](double a, double b) { return a <= b; }) };
        m[">="] = { comparison_overload(CTKind::Number, [](double a, double b) { return a >= b; }),
                    comparison_overload(CTKind::Bool, [](double a, double b) { return a >= b; }) };
        m["&&"] = { logical_overload([](bool a, bool b) { return a && b; }) };
        m["||"] = { logical_overload([](bool a, bool b) { return a || b; }) };
        return m;
    }();
    return table;
}

std::string stmt_kind_word(StmtKind k) {
    switch (k) {
        case StmtKind::VarDecl: return "VarDecl";
        case StmtKind::AssignStmt: return "AssignStmt";
        case StmtKind::IfStmt: return "IfStmt";
        case StmtKind::ForStmt: return "ForStmt";
        case StmtKind::WhileStmt: return "WhileStmt";
        case StmtKind::BreakStmt: return "BreakStmt";
        case StmtKind::ContinueStmt: return "ContinueStmt";
        case StmtKind::ReturnStmt: return "ReturnStmt";
        case StmtKind::ExprStmt: return "ExprStmt";
        case StmtKind::Block: return "Block";
    }
    throw std::runtime_error("cleg: stmt_kind_word: unexpected kind");
}
