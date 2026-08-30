#include "game/cleg.h"
#include "game/selector.h"
#include "game/topology.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

// Mirrors shared/cleg.ts - see that file's own top comment for the full grammar/semantics this
// implements (types, operators, builtins, the multiProd algorithm). This file only points out where
// the C++ port differs from the TS source; the TS comments are the canonical reference for WHAT each
// piece does, not repeated here. Structural differences that hold throughout this file, not
// mentioned per-function below:
//   - `ClegProgram`/`ClegType`/`ClegValue`/the whole AST are private to this file (see game/cleg.h's
//     own top comment on why - unlike TS, no other C++ consumer ever inspects them).
//   - No garbage collector: recursive AST/value substructures use std::shared_ptr for genuine
//     sharing (Expr/Stmt children, ClegValue's `egr` payload) or plain std::vector/value members
//     where TS's own object-reference semantics never actually needed sharing (Array/Set contents,
//     Selector/BoardModifier/FormSelector/MultiSelector payloads - each already cheap to copy, since
//     Selector/MultiSelector use shared_ptr internally for their own recursive children).
//   - TS's `number` (a JS double) is a C++ `double` throughout; an int is only ever produced at the
//     C++/board-primitive boundary (mkEdge/mkTri/mkQuad, board-arg conversion, msBase's index), via
//     an explicit int conversion - a data-type constraint the TS side doesn't have.
//   - Every ClegType/ClegValue/AST struct gives EVERY field a default member initializer, even ones
//     TS's own type only ever reads after unconditionally setting it - a defensive C++-only
//     precaution making every default-constructed instance well-defined on its own, not a behavioral
//     difference.
//   - TypeEnv/ValueEnv's parent chain uses raw pointers into the enclosing call's own stack frames
//     (safe here: every env is stack-allocated and never outlives the recursive check/eval call that
//     owns it) rather than TS's GC'd object graph.
//   - unparseCleg has no port - nothing in ai/ ever needs to show a cleg program back as text (no
//     UI), unlike the TS side's Configure Board popup.

// ── Types ────────────────────────────────────────────────────────────────────

enum class CTKind {
    Egr, Number, String, Bool, Edge, Tri, Quad, Sel, FormSel, Mod, Msel, Array, Set
};

// Mirrors shared/cleg.ts's ClegType - `elem` meaningful iff kind == Array/Set.
struct ClegType {
    CTKind kind = CTKind::Number;
    std::shared_ptr<ClegType> elem;
};

static bool type_equals(const ClegType& a, const ClegType& b) {
    if (a.kind != b.kind) return false;
    if (a.kind == CTKind::Array || a.kind == CTKind::Set) return type_equals(*a.elem, *b.elem);
    return true;
}

static std::string ctkind_word(CTKind k) {
    switch (k) {
        case CTKind::Egr: return "egr";
        case CTKind::Number: return "number";
        case CTKind::String: return "string";
        case CTKind::Bool: return "bool";
        case CTKind::Edge: return "edge";
        case CTKind::Tri: return "tri";
        case CTKind::Quad: return "quad";
        case CTKind::Sel: return "sel";
        case CTKind::FormSel: return "formSel";
        case CTKind::Mod: return "mod";
        case CTKind::Msel: return "msel";
        case CTKind::Array: case CTKind::Set: break; // handled by type_to_string below
    }
    throw std::runtime_error("cleg: ctkind_word: unexpected kind");
}

static std::string type_to_string(const ClegType& t) {
    if (t.kind == CTKind::Array) return type_to_string(*t.elem) + "[]";
    if (t.kind == CTKind::Set) return type_to_string(*t.elem) + "{}";
    return ctkind_word(t.kind);
}

// Mirrors shared/cleg.ts's MultiSelector (see its own doc comment near multiProd, further below) -
// defined ahead of ClegValue since ClegValue holds one by value.
enum class MSelOp { All, Base, Union, Inter, Diff };
struct MultiSelector {
    MSelOp op = MSelOp::All;
    int number = 0;                       // Base
    std::shared_ptr<Selector> sel;        // Base
    std::vector<MultiSelector> items;     // Union/Inter
    std::shared_ptr<MultiSelector> a, b;  // Diff
};

// ── Values ───────────────────────────────────────────────────────────────────

// Mirrors shared/cleg.ts's ClegValue - one flat tagged struct (same "meaningful iff kind == X"
// convention board_config.h's own BoardModifier/Selector already use in this codebase), rather than
// TS's discriminated union, since C++ has no such union without std::variant's own added ceremony.
// `egr` is the one payload big enough to warrant its own indirection (std::shared_ptr<BoardConfig> -
// an N×N adjacency matrix) - every other payload is cheap enough to hold by value.
struct ClegValue {
    CTKind kind = CTKind::Number;
    ClegType elem;                                  // Array/Set only
    double number = 0;
    std::string str;
    bool boolean = false;
    BoardEdge edge_v{};
    BoardTriangle tri_v{};
    BoardQuad quad_v{};
    std::shared_ptr<BoardConfig> egr_v;
    SelectorType sel_type = SelectorType::Node;      // Sel only
    Selector sel_v;
    FormSelector formsel_v{FormSelectorKind::Tri, std::nullopt};
    BoardModifier mod_v{ModifierKind::Rectify};
    MultiSelector msel_v;
    std::vector<ClegValue> arr_v;                    // Array or Set (Set: deduplicated by cleg_set_key)
};

static ClegType cleg_value_type(const ClegValue& v) {
    if (v.kind == CTKind::Array || v.kind == CTKind::Set)
        return ClegType{v.kind, std::make_shared<ClegType>(v.elem)};
    return ClegType{v.kind, nullptr};
}

static ClegValue make_number(double n) { ClegValue v; v.kind = CTKind::Number; v.number = n; return v; }
static ClegValue make_string(std::string s) { ClegValue v; v.kind = CTKind::String; v.str = std::move(s); return v; }
static ClegValue make_bool(bool b) { ClegValue v; v.kind = CTKind::Bool; v.boolean = b; return v; }
static ClegValue make_edge_v(BoardEdge e) { ClegValue v; v.kind = CTKind::Edge; v.edge_v = e; return v; }
static ClegValue make_tri_v(BoardTriangle t) { ClegValue v; v.kind = CTKind::Tri; v.tri_v = t; return v; }
static ClegValue make_quad_v(BoardQuad q) { ClegValue v; v.kind = CTKind::Quad; v.quad_v = q; return v; }
static ClegValue make_egr(BoardConfig bc) {
    ClegValue v; v.kind = CTKind::Egr; v.egr_v = std::make_shared<BoardConfig>(std::move(bc)); return v;
}
static ClegValue make_mod(BoardModifier m) { ClegValue v; v.kind = CTKind::Mod; v.mod_v = std::move(m); return v; }
static ClegValue make_form_sel(FormSelector fs) { ClegValue v; v.kind = CTKind::FormSel; v.formsel_v = std::move(fs); return v; }
static ClegValue make_msel(MultiSelector m) { ClegValue v; v.kind = CTKind::Msel; v.msel_v = std::move(m); return v; }

// Renders a double for error messages/number-string concatenation - an integral value prints with
// no decimal point/trailing zeros (matching JS's own Number->string for the plain integers cleg
// programs actually interpolate, e.g. biTemple's `(d + 1)`); otherwise 15 significant digits.
static std::string format_number_display(double d) {
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
static std::string tri_key_str(const BoardTriangle& t) {
    return std::to_string(t.n1) + "," + std::to_string(t.n2) + "," + std::to_string(t.n3);
}
static std::string quad_key_str(const BoardQuad& q) {
    return std::to_string(q.n1) + "," + std::to_string(q.n2) + "," + std::to_string(q.n3) + "," + std::to_string(q.n4);
}

// Mirrors shared/cleg.ts's clegSetKey() - a canonical string key for a set element (edge/tri/quad
// have no reference equality of their own, same reasoning as that file's own doc comment).
static std::string cleg_set_key(const ClegValue& v) {
    switch (v.kind) {
        case CTKind::Number: return "n:" + format_number_exact(v.number);
        case CTKind::String: return "s:" + v.str;
        case CTKind::Bool: return v.boolean ? "b:1" : "b:0";
        case CTKind::Edge: return "e:" + edge_key_str(v.edge_v);
        case CTKind::Tri: return "t:" + tri_key_str(v.tri_v);
        case CTKind::Quad: return "q:" + quad_key_str(v.quad_v);
        default:
            throw std::runtime_error("cleg: '" + type_to_string(cleg_value_type(v)) + "' cannot be a set element");
    }
}

// Mirrors shared/cleg.ts's makeClegSet() - collapses duplicates by cleg_set_key, first occurrence wins.
static ClegValue make_cleg_set(ClegType elem, std::vector<ClegValue> values) {
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
// shared/cleg.ts's setIntersect()/setDiff().
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

// ── Binary operators ─────────────────────────────────────────────────────────

// Mirrors shared/cleg.ts's BinaryOverload - `match` returns nullopt (TS: null) instead of matching.
struct MatchResult {
    ClegType type;
    std::function<ClegValue(const ClegValue&, const ClegValue&)> eval;
};
struct BinaryOverload {
    std::string signature;
    std::function<std::optional<MatchResult>(const ClegType&, const ClegType&)> match;
};

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

// Mirrors shared/cleg.ts's requireRepeatCount() - `n` isn't statically known, checked here at
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

// Mirrors shared/cleg.ts's toComparable() - bool compares via C++'s own false=0/true=1 convention.
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

// Mirrors shared/cleg.ts's BINARY_OPERATOR_OVERLOADS - keyed by the operator's own punctuation text
// (matching the parser's own token, rather than introducing a separate BinOp enum).
static const std::unordered_map<std::string, std::vector<BinaryOverload>>& binary_operator_overloads() {
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
        m["<"] = { comparison_overload(CTKind::Number, [](double a, double b) { return a < b; }),
                   comparison_overload(CTKind::Bool, [](double a, double b) { return a < b; }) };
        m[">"] = { comparison_overload(CTKind::Number, [](double a, double b) { return a > b; }),
                   comparison_overload(CTKind::Bool, [](double a, double b) { return a > b; }) };
        m["<="] = { comparison_overload(CTKind::Number, [](double a, double b) { return a <= b; }),
                    comparison_overload(CTKind::Bool, [](double a, double b) { return a <= b; }) };
        m[">="] = { comparison_overload(CTKind::Number, [](double a, double b) { return a >= b; }),
                    comparison_overload(CTKind::Bool, [](double a, double b) { return a >= b; }) };
        return m;
    }();
    return table;
}

// ── AST ──────────────────────────────────────────────────────────────────────

struct Param { ClegType type; std::string name; };

enum class ExprKind { NumberLit, StringLit, BoolLit, ArrayLit, SetLit, Identifier, CallExpr, BinaryExpr, UnaryExpr, NilExpr };

// Mirrors shared/cleg.ts's Expr union - one flat tagged struct, same convention as ClegValue above.
// `string_value` doubles as StringLit's text, Identifier's name, and CallExpr's callee;
// `elements` doubles as ArrayLit/SetLit's own elements and CallExpr's own args (each kind uses only
// one of the two roles, never both).
struct Expr {
    ExprKind kind = ExprKind::NumberLit;
    double number_value = 0;
    std::string string_value;
    bool bool_value = false;
    std::vector<Expr> elements;
    std::string op;                              // BinaryExpr only
    std::shared_ptr<Expr> left, right, operand;   // BinaryExpr (left/right) / UnaryExpr (operand)
    ClegType nil_type;                            // NilExpr only
};

enum class StmtKind { VarDecl, AssignStmt, IfStmt, ForStmt, ReturnStmt, ExprStmt, Block };

// Mirrors shared/cleg.ts's Stmt union - same flat-struct convention as Expr above. `expr` doubles as
// VarDecl's init, AssignStmt's value, ReturnStmt's value, and ExprStmt's own expr.
struct Stmt {
    StmtKind kind = StmtKind::Block;
    ClegType decl_type;                     // VarDecl only
    std::string name;                       // VarDecl/AssignStmt only
    std::shared_ptr<Expr> expr;
    std::shared_ptr<Expr> cond;              // IfStmt/ForStmt (ForStmt: null = omitted)
    std::shared_ptr<Stmt> then_stmt;         // IfStmt (a Block)
    std::shared_ptr<Stmt> else_stmt;         // IfStmt (null | Block | IfStmt)
    std::shared_ptr<Stmt> for_init;          // ForStmt (VarDecl/AssignStmt/ExprStmt-shaped, null = omitted)
    std::shared_ptr<Stmt> for_update;        // ForStmt (AssignStmt/ExprStmt-shaped, null = omitted)
    std::shared_ptr<Stmt> body;              // ForStmt (a Block)
    std::vector<Stmt> stmts;                 // Block only
};

static std::string stmt_kind_word(StmtKind k) {
    switch (k) {
        case StmtKind::VarDecl: return "VarDecl";
        case StmtKind::AssignStmt: return "AssignStmt";
        case StmtKind::IfStmt: return "IfStmt";
        case StmtKind::ForStmt: return "ForStmt";
        case StmtKind::ReturnStmt: return "ReturnStmt";
        case StmtKind::ExprStmt: return "ExprStmt";
        case StmtKind::Block: return "Block";
    }
    throw std::runtime_error("cleg: stmt_kind_word: unexpected kind");
}

struct FunctionDecl {
    ClegType return_type;
    std::string name;
    std::vector<Param> params;
    Stmt body; // kind == Block
};

// Definition of the type forward-declared in game/cleg.h - functions is order-independent
// (see shared/cleg.ts's own doc comment), stmts (a TopStmt subset: VarDecl/AssignStmt/ExprStmt) is
// order-dependent.
struct ClegProgram {
    std::vector<FunctionDecl> functions;
    std::vector<Stmt> stmts;
};

// ── Lexer ────────────────────────────────────────────────────────────────────

enum class TokKind { Ident, Number, String, Punct, Eof };
struct Token { TokKind kind; std::string text; size_t pos; };

static const std::string PUNCTUATION = "(){}[],;+-*/%";

static bool is_ident_start(char c) { return std::isalpha(static_cast<unsigned char>(c)) || c == '_'; }
static bool is_ident_cont(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; }

// Mirrors shared/cleg.ts's tokenize().
static std::vector<Token> tokenize(const std::string& src) {
    std::vector<Token> tokens;
    size_t n = src.size(), i = 0;
    while (i < n) {
        char c = src[i];
        if (std::isspace(static_cast<unsigned char>(c))) { i++; continue; }
        if (c == '/' && i + 1 < n && src[i + 1] == '/') { while (i < n && src[i] != '\n') i++; continue; }
        if (is_ident_start(c)) {
            size_t j = i + 1;
            while (j < n && is_ident_cont(src[j])) j++;
            tokens.push_back({TokKind::Ident, src.substr(i, j - i), i});
            i = j;
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(c))) {
            size_t j = i + 1;
            while (j < n && (std::isdigit(static_cast<unsigned char>(src[j])) || src[j] == '.')) j++;
            tokens.push_back({TokKind::Number, src.substr(i, j - i), i});
            i = j;
            continue;
        }
        if (c == '"') {
            size_t j = i + 1;
            std::string out;
            while (j < n && src[j] != '"') {
                if (src[j] == '\\' && j + 1 < n) {
                    char esc = src[j + 1];
                    out += esc == 'n' ? '\n' : esc == 't' ? '\t' : esc;
                    j += 2;
                } else {
                    out += src[j];
                    j++;
                }
            }
            if (j >= n) throw std::runtime_error("cleg: unterminated string literal starting at position " + std::to_string(i));
            tokens.push_back({TokKind::String, out, i});
            i = j + 1;
            continue;
        }
        if (c == '=' || c == '<' || c == '>') {
            if (i + 1 < n && src[i + 1] == '=') {
                tokens.push_back({TokKind::Punct, std::string(1, c) + "=", i});
                i += 2;
                continue;
            }
            tokens.push_back({TokKind::Punct, std::string(1, c), i});
            i++;
            continue;
        }
        if (PUNCTUATION.find(c) != std::string::npos) {
            tokens.push_back({TokKind::Punct, std::string(1, c), i});
            i++;
            continue;
        }
        throw std::runtime_error(std::string("cleg: unexpected character '") + c + "' at position " + std::to_string(i));
    }
    tokens.push_back({TokKind::Eof, "", n});
    return tokens;
}

// ── Parser ───────────────────────────────────────────────────────────────────

// Mirrors shared/cleg.ts's TokenCursor. peek_at clamps an out-of-range offset to the trailing Eof
// token (tokenize() always appends exactly one) rather than TS's own unchecked array index (which
// would read `undefined` for a maximally short/malformed program) - strictly safer, and behaviorally
// identical for every syntactically-plausible lookahead this parser ever performs (peek_at is only
// used for 1-2 token disambiguation, where "ran out of tokens" and "next token is Eof" mean the same
// thing).
class TokenCursor {
public:
    explicit TokenCursor(std::vector<Token> tokens) : tokens_(std::move(tokens)) {}

    const Token& peek() const { return tokens_[pos_]; }
    const Token& peek_at(size_t offset) const {
        size_t idx = pos_ + offset;
        return idx < tokens_.size() ? tokens_[idx] : tokens_.back();
    }
    Token next() { return tokens_[pos_++]; }
    bool at_end() const { return peek().kind == TokKind::Eof; }
    bool is_punct(const std::string& p) const { auto& t = peek(); return t.kind == TokKind::Punct && t.text == p; }
    bool is_keyword(const std::string& k) const { auto& t = peek(); return t.kind == TokKind::Ident && t.text == k; }

    void expect_punct(const std::string& p) {
        Token t = next();
        if (t.kind != TokKind::Punct || t.text != p)
            throw std::runtime_error("cleg: expected '" + p + "', got '" + (t.text.empty() ? "<eof>" : t.text) +
                "' at position " + std::to_string(t.pos));
    }
    std::string expect_ident() {
        Token t = next();
        if (t.kind != TokKind::Ident)
            throw std::runtime_error("cleg: expected an identifier, got '" + (t.text.empty() ? "<eof>" : t.text) +
                "' at position " + std::to_string(t.pos));
        return t.text;
    }

private:
    std::vector<Token> tokens_;
    size_t pos_ = 0;
};

static const std::set<std::string>& type_keywords() {
    static const std::set<std::string> kws = {
        "egr", "number", "string", "bool", "edge", "tri", "quad", "sel", "formSel", "mod", "msel",
    };
    return kws;
}
// String-keyed variant of is_set_elem_kind (further below) - the parser's own parse_type needs to
// check the raw base-type TOKEN (a string) before any ClegType/CTKind exists yet, unlike the type
// checker's own SetLit case, which already has a resolved CTKind to check instead.
static const std::set<std::string>& set_elem_kind_words() {
    static const std::set<std::string> kws = {"number", "string", "bool", "edge", "tri", "quad"};
    return kws;
}

static CTKind base_type_kind(const std::string& base) {
    if (base == "egr") return CTKind::Egr;
    if (base == "number") return CTKind::Number;
    if (base == "string") return CTKind::String;
    if (base == "bool") return CTKind::Bool;
    if (base == "edge") return CTKind::Edge;
    if (base == "tri") return CTKind::Tri;
    if (base == "quad") return CTKind::Quad;
    if (base == "sel") return CTKind::Sel;
    if (base == "formSel") return CTKind::FormSel;
    if (base == "mod") return CTKind::Mod;
    if (base == "msel") return CTKind::Msel;
    throw std::runtime_error("cleg: base_type_kind: unknown base type '" + base + "'");
}

static ClegType parse_type(TokenCursor& c) {
    std::string base = c.expect_ident();
    if (!type_keywords().count(base))
        throw std::runtime_error("cleg: expected a type (egr/number/string/bool/edge/tri/quad/sel/formSel/mod), got '" + base + "'");
    ClegType type{base_type_kind(base), nullptr};
    if (c.is_punct("{")) {
        if (!set_elem_kind_words().count(base))
            throw std::runtime_error(
                "cleg: '" + base + "{}' is not a supported set type - sets of egr, sets of sets, and sets of "
                "arrays are not supported");
        c.next();
        c.expect_punct("}");
        type = ClegType{CTKind::Set, std::make_shared<ClegType>(std::move(type))};
    }
    while (c.is_punct("[")) {
        c.next();
        c.expect_punct("]");
        type = ClegType{CTKind::Array, std::make_shared<ClegType>(std::move(type))};
    }
    return type;
}

static bool is_type_start(TokenCursor& c) {
    auto& t = c.peek();
    return t.kind == TokKind::Ident && type_keywords().count(t.text) > 0;
}
static bool is_function_decl_start(TokenCursor& c) {
    return is_type_start(c) && c.peek_at(1).kind == TokKind::Ident &&
           c.peek_at(2).kind == TokKind::Punct && c.peek_at(2).text == "(";
}

template <typename T, typename F>
static std::vector<T> parse_comma_separated(TokenCursor& c, const std::string& close, F parse_one) {
    std::vector<T> items;
    if (!c.is_punct(close)) {
        items.push_back(parse_one());
        while (c.is_punct(",")) { c.next(); items.push_back(parse_one()); }
    }
    c.expect_punct(close);
    return items;
}

static Stmt parse_block(TokenCursor& c);
static Stmt parse_stmt(TokenCursor& c);
static Expr parse_expr(TokenCursor& c);

static FunctionDecl parse_function_decl(TokenCursor& c) {
    ClegType return_type = parse_type(c);
    std::string name = c.expect_ident();
    c.expect_punct("(");
    auto params = parse_comma_separated<Param>(c, ")", [&]() {
        ClegType type = parse_type(c);
        std::string pname = c.expect_ident();
        return Param{type, pname};
    });
    Stmt body = parse_block(c);
    return FunctionDecl{return_type, name, std::move(params), std::move(body)};
}

static Stmt parse_block(TokenCursor& c) {
    c.expect_punct("{");
    std::vector<Stmt> stmts;
    while (!c.is_punct("}")) stmts.push_back(parse_stmt(c));
    c.expect_punct("}");
    Stmt block; block.kind = StmtKind::Block; block.stmts = std::move(stmts);
    return block;
}

static bool is_assign_start(TokenCursor& c) {
    return c.peek().kind == TokKind::Ident && c.peek_at(1).kind == TokKind::Punct && c.peek_at(1).text == "=";
}

static Stmt parse_if_stmt(TokenCursor& c);
static Stmt parse_for_stmt(TokenCursor& c);
static Stmt parse_return_stmt(TokenCursor& c);

static Stmt parse_var_decl_no_semi(TokenCursor& c) {
    ClegType type = parse_type(c);
    std::string name = c.expect_ident();
    c.expect_punct("=");
    Expr init = parse_expr(c);
    Stmt s; s.kind = StmtKind::VarDecl; s.decl_type = type; s.name = name;
    s.expr = std::make_shared<Expr>(std::move(init));
    return s;
}
static Stmt parse_var_decl(TokenCursor& c) {
    Stmt decl = parse_var_decl_no_semi(c);
    c.expect_punct(";");
    return decl;
}
static Stmt parse_assign_stmt_no_semi(TokenCursor& c) {
    std::string name = c.expect_ident();
    c.expect_punct("=");
    Expr value = parse_expr(c);
    Stmt s; s.kind = StmtKind::AssignStmt; s.name = name;
    s.expr = std::make_shared<Expr>(std::move(value));
    return s;
}
static Stmt parse_assign_stmt(TokenCursor& c) {
    Stmt stmt = parse_assign_stmt_no_semi(c);
    c.expect_punct(";");
    return stmt;
}

static Stmt parse_stmt(TokenCursor& c) {
    if (c.is_punct("{")) return parse_block(c);
    if (c.is_keyword("if")) return parse_if_stmt(c);
    if (c.is_keyword("for")) return parse_for_stmt(c);
    if (c.is_keyword("return")) return parse_return_stmt(c);
    if (is_type_start(c)) return parse_var_decl(c);
    if (is_assign_start(c)) return parse_assign_stmt(c);
    Expr expr = parse_expr(c);
    c.expect_punct(";");
    Stmt s; s.kind = StmtKind::ExprStmt; s.expr = std::make_shared<Expr>(std::move(expr));
    return s;
}

static Stmt parse_if_stmt(TokenCursor& c) {
    c.next(); // 'if'
    c.expect_punct("(");
    Expr cond = parse_expr(c);
    c.expect_punct(")");
    Stmt then_b = parse_block(c);
    std::shared_ptr<Stmt> else_s;
    if (c.is_keyword("else")) {
        c.next();
        else_s = std::make_shared<Stmt>(c.is_keyword("if") ? parse_if_stmt(c) : parse_block(c));
    }
    Stmt s; s.kind = StmtKind::IfStmt;
    s.cond = std::make_shared<Expr>(std::move(cond));
    s.then_stmt = std::make_shared<Stmt>(std::move(then_b));
    s.else_stmt = else_s;
    return s;
}

static std::shared_ptr<Stmt> parse_for_init(TokenCursor& c) {
    if (c.is_punct(";")) return nullptr;
    if (is_type_start(c)) return std::make_shared<Stmt>(parse_var_decl_no_semi(c));
    if (is_assign_start(c)) return std::make_shared<Stmt>(parse_assign_stmt_no_semi(c));
    Expr e = parse_expr(c);
    Stmt s; s.kind = StmtKind::ExprStmt; s.expr = std::make_shared<Expr>(std::move(e));
    return std::make_shared<Stmt>(std::move(s));
}
static std::shared_ptr<Stmt> parse_for_update(TokenCursor& c) {
    if (c.is_punct(")")) return nullptr;
    if (is_assign_start(c)) return std::make_shared<Stmt>(parse_assign_stmt_no_semi(c));
    Expr e = parse_expr(c);
    Stmt s; s.kind = StmtKind::ExprStmt; s.expr = std::make_shared<Expr>(std::move(e));
    return std::make_shared<Stmt>(std::move(s));
}
static Stmt parse_for_stmt(TokenCursor& c) {
    c.next(); // 'for'
    c.expect_punct("(");
    auto init = parse_for_init(c);
    c.expect_punct(";");
    std::shared_ptr<Expr> cond = c.is_punct(";") ? nullptr : std::make_shared<Expr>(parse_expr(c));
    c.expect_punct(";");
    auto update = parse_for_update(c);
    c.expect_punct(")");
    Stmt body = parse_block(c);
    Stmt s; s.kind = StmtKind::ForStmt;
    s.for_init = init; s.cond = cond; s.for_update = update;
    s.body = std::make_shared<Stmt>(std::move(body));
    return s;
}

static Stmt parse_return_stmt(TokenCursor& c) {
    c.next(); // 'return'
    Expr value = parse_expr(c);
    c.expect_punct(";");
    Stmt s; s.kind = StmtKind::ReturnStmt; s.expr = std::make_shared<Expr>(std::move(value));
    return s;
}

static bool is_punct_in(TokenCursor& c, const std::set<std::string>& ops) {
    auto& t = c.peek();
    return t.kind == TokKind::Punct && ops.count(t.text) > 0;
}
static Expr make_binary(std::string op, Expr left, Expr right) {
    Expr e; e.kind = ExprKind::BinaryExpr; e.op = std::move(op);
    e.left = std::make_shared<Expr>(std::move(left));
    e.right = std::make_shared<Expr>(std::move(right));
    return e;
}

static Expr parse_relational(TokenCursor& c);
static Expr parse_additive(TokenCursor& c);
static Expr parse_multiplicative(TokenCursor& c);
static Expr parse_unary(TokenCursor& c);
static Expr parse_atom(TokenCursor& c);

// Precedence chain mirrors shared/cleg.ts's own parseExpr/parseRelational/parseAdditive/
// parseMultiplicative/parseUnary exactly (`==` loosest, then `< > <= >=`, then `+ -`, then `* / %`
// tightest, all left-associative).
// Each loop body consumes the operator into a local BEFORE calling the next precedence level down -
// C++ function-call argument evaluation order is unspecified (unlike JS's own strict left-to-right),
// so folding both into one make_binary(...) call let the compiler legally evaluate the recursive
// parse before c.next() ever advanced past the operator token, parsing the wrong thing entirely.
static Expr parse_expr(TokenCursor& c) {
    static const std::set<std::string> EQ_OPS = {"=="};
    Expr left = parse_relational(c);
    while (is_punct_in(c, EQ_OPS)) {
        std::string op = c.next().text;
        Expr right = parse_relational(c);
        left = make_binary(std::move(op), std::move(left), std::move(right));
    }
    return left;
}
static Expr parse_relational(TokenCursor& c) {
    static const std::set<std::string> REL_OPS = {"<", ">", "<=", ">="};
    Expr left = parse_additive(c);
    while (is_punct_in(c, REL_OPS)) {
        std::string op = c.next().text;
        Expr right = parse_additive(c);
        left = make_binary(std::move(op), std::move(left), std::move(right));
    }
    return left;
}
static Expr parse_additive(TokenCursor& c) {
    static const std::set<std::string> ADD_OPS = {"+", "-"};
    Expr left = parse_multiplicative(c);
    while (is_punct_in(c, ADD_OPS)) {
        std::string op = c.next().text;
        Expr right = parse_multiplicative(c);
        left = make_binary(std::move(op), std::move(left), std::move(right));
    }
    return left;
}
static Expr parse_multiplicative(TokenCursor& c) {
    static const std::set<std::string> MUL_OPS = {"*", "/", "%"};
    Expr left = parse_unary(c);
    while (is_punct_in(c, MUL_OPS)) {
        std::string op = c.next().text;
        Expr right = parse_unary(c);
        left = make_binary(std::move(op), std::move(left), std::move(right));
    }
    return left;
}
static Expr parse_unary(TokenCursor& c) {
    if (c.is_punct("-")) {
        c.next();
        Expr e; e.kind = ExprKind::UnaryExpr; e.operand = std::make_shared<Expr>(parse_unary(c));
        return e;
    }
    return parse_atom(c);
}

static Expr parse_atom(TokenCursor& c) {
    const Token& tok = c.peek();
    if (tok.kind == TokKind::Number) {
        c.next();
        Expr e; e.kind = ExprKind::NumberLit; e.number_value = std::stod(tok.text);
        return e;
    }
    if (tok.kind == TokKind::String) {
        c.next();
        Expr e; e.kind = ExprKind::StringLit; e.string_value = tok.text;
        return e;
    }
    if (c.is_punct("[")) {
        c.next();
        Expr e; e.kind = ExprKind::ArrayLit;
        e.elements = parse_comma_separated<Expr>(c, "]", [&]() { return parse_expr(c); });
        return e;
    }
    if (c.is_punct("{")) {
        // Unambiguous with a Block's own '{' - parse_stmt/parse_block never call parse_expr where a
        // Block could appear instead (function/if/for bodies).
        c.next();
        Expr e; e.kind = ExprKind::SetLit;
        e.elements = parse_comma_separated<Expr>(c, "}", [&]() { return parse_expr(c); });
        return e;
    }
    if (c.is_punct("(")) {
        c.next();
        Expr inner = parse_expr(c);
        c.expect_punct(")");
        return inner;
    }
    if (tok.kind == TokKind::Ident) {
        if (tok.text == "true") { c.next(); Expr e; e.kind = ExprKind::BoolLit; e.bool_value = true; return e; }
        if (tok.text == "false") { c.next(); Expr e; e.kind = ExprKind::BoolLit; e.bool_value = false; return e; }
        if (tok.text == "nil") {
            c.next();
            c.expect_punct("(");
            ClegType type = parse_type(c);
            c.expect_punct(")");
            Expr e; e.kind = ExprKind::NilExpr; e.nil_type = type;
            return e;
        }
        std::string name = c.expect_ident();
        if (c.is_punct("(")) {
            c.next();
            Expr e; e.kind = ExprKind::CallExpr; e.string_value = name;
            e.elements = parse_comma_separated<Expr>(c, ")", [&]() { return parse_expr(c); });
            return e;
        }
        Expr e; e.kind = ExprKind::Identifier; e.string_value = name;
        return e;
    }
    throw std::runtime_error(
        "cleg: unexpected token '" + (tok.text.empty() ? "<eof>" : tok.text) + "' at position " + std::to_string(tok.pos));
}

// Mirrors shared/cleg.ts's parseCleg().
static ClegProgram parse_cleg_impl(const std::string& source) {
    TokenCursor c(tokenize(source));
    ClegProgram program;
    while (!c.at_end()) {
        if (is_function_decl_start(c)) { program.functions.push_back(parse_function_decl(c)); continue; }
        if (is_type_start(c)) { program.stmts.push_back(parse_var_decl(c)); continue; }
        if (is_assign_start(c)) { program.stmts.push_back(parse_assign_stmt(c)); continue; }
        Expr expr = parse_expr(c);
        c.expect_punct(";");
        Stmt s; s.kind = StmtKind::ExprStmt; s.expr = std::make_shared<Expr>(std::move(expr));
        program.stmts.push_back(std::move(s));
    }
    return program;
}

// ── Predefined (board-construction) functions ─────────────────────────────────

// Mirrors shared/cleg.ts's BuiltinFunction.
using CheckCallFn = std::function<ClegType(const std::string&, const std::vector<ClegType>&)>;
using CallFn = std::function<ClegValue(const std::vector<ClegValue>&)>;
struct BuiltinFunction { CheckCallFn check_call; CallFn call; };

// Mirrors shared/cleg.ts's fixedSignature().
static CheckCallFn fixed_signature(std::vector<ClegType> params, ClegType return_type) {
    return [params, return_type](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
        if (arg_types.size() != params.size())
            throw std::runtime_error(
                "cleg: '" + callee + "' expects " + std::to_string(params.size()) + " argument(s), got " +
                std::to_string(arg_types.size()));
        for (size_t i = 0; i < arg_types.size(); i++)
            if (!type_equals(arg_types[i], params[i]))
                throw std::runtime_error(
                    "cleg: '" + callee + "' argument " + std::to_string(i + 1) + ": expected " +
                    type_to_string(params[i]) + ", got " + type_to_string(arg_types[i]));
        return return_type;
    };
}

static const ClegType NUMBER_TYPE{CTKind::Number, nullptr};
static const ClegType STRING_TYPE{CTKind::String, nullptr};
static const ClegType EGR_TYPE{CTKind::Egr, nullptr};
static const ClegType EDGE_TYPE{CTKind::Edge, nullptr};
static const ClegType TRI_TYPE{CTKind::Tri, nullptr};
static const ClegType QUAD_TYPE{CTKind::Quad, nullptr};
static const ClegType MOD_TYPE{CTKind::Mod, nullptr};
static const ClegType FORMSEL_TYPE{CTKind::FormSel, nullptr};
static const ClegType MSEL_TYPE{CTKind::Msel, nullptr};

// CTKind-based counterpart of set_elem_kind_words() (parser) - used by the type checker's SetLit case.
static bool is_set_elem_kind(CTKind k) {
    return k == CTKind::Number || k == CTKind::String || k == CTKind::Bool ||
           k == CTKind::Edge || k == CTKind::Tri || k == CTKind::Quad;
}

static std::string selector_type_word(SelectorType t) {
    switch (t) {
        case SelectorType::Node: return "node";
        case SelectorType::Edge: return "edge";
        case SelectorType::Tri: return "tri";
        case SelectorType::Quad: return "quad";
    }
    throw std::runtime_error("cleg: selector_type_word: unexpected SelectorType");
}
static SelectorType selector_type_from_word(const std::string& w) {
    if (w == "node") return SelectorType::Node;
    if (w == "edge") return SelectorType::Edge;
    if (w == "tri") return SelectorType::Tri;
    if (w == "quad") return SelectorType::Quad;
    throw std::runtime_error("cleg: mkSel: unknown selector kind '" + w + "' - expected node/edge/tri/quad");
}
using SelectorParseFn = Selector (*)(const std::string&);
static SelectorParseFn selector_parser_for(SelectorType t) {
    switch (t) {
        case SelectorType::Node: return parse_node_selector;
        case SelectorType::Edge: return parse_edge_selector;
        case SelectorType::Tri: return parse_triangle_selector;
        case SelectorType::Quad: return parse_quad_selector;
    }
    throw std::runtime_error("cleg: selector_parser_for: unexpected SelectorType");
}
// Mirrors shared/cleg.ts's SELECTOR_SET_ELEM_KIND.
static CTKind selector_set_elem_kind(SelectorType t) {
    switch (t) {
        case SelectorType::Node: return CTKind::Number;
        case SelectorType::Edge: return CTKind::Edge;
        case SelectorType::Tri: return CTKind::Tri;
        case SelectorType::Quad: return CTKind::Quad;
    }
    throw std::runtime_error("cleg: selector_set_elem_kind: unexpected SelectorType");
}

// Mirrors shared/cleg.ts's setValueToSelectedVals() - builds a `raw` Selector wrapping `values`
// (a `set`'s own deduplicated ClegValue vector) directly, populating whichever of Selector's own
// raw_nodes/raw_edges/raw_tris/raw_quads (game/selector.h) matches want_kind.
static Selector raw_selector_from_set(SelectorType want_kind, const std::vector<ClegValue>& values) {
    Selector sel; sel.op = SelectorOp::Raw; sel.type = want_kind;
    switch (want_kind) {
        case SelectorType::Node:
            for (auto& v : values) sel.raw_nodes.insert(static_cast<int>(std::llround(v.number)));
            break;
        case SelectorType::Edge:
            for (auto& v : values) sel.raw_edges.push_back(v.edge_v);
            break;
        case SelectorType::Tri:
            for (auto& v : values) sel.raw_tris.push_back(v.tri_v);
            break;
        case SelectorType::Quad:
            for (auto& v : values) sel.raw_quads.push_back(v.quad_v);
            break;
    }
    return sel;
}

// Mirrors shared/cleg.ts's resolveSelectorArg().
static Selector resolve_selector_arg(
    const std::string& callee, const ClegValue& arg, SelectorType want_kind, SelectorParseFn parse_fn)
{
    if (arg.kind == CTKind::String) return parse_fn(arg.str);
    if (arg.kind == CTKind::Set) {
        if (arg.elem.kind != selector_set_elem_kind(want_kind))
            throw std::runtime_error(
                "cleg: '" + callee + "' expects a " + selector_type_word(want_kind) + " selector, got a set of " +
                type_to_string(arg.elem));
        return raw_selector_from_set(want_kind, arg.arr_v);
    }
    // arg.kind == CTKind::Sel
    if (arg.sel_type != want_kind)
        throw std::runtime_error(
            "cleg: '" + callee + "' expects a " + selector_type_word(want_kind) + " selector, got a '" +
            selector_type_word(arg.sel_type) + "' selector");
    return arg.sel_v;
}

// One entry per shared/boardConfig.ts's own PrescribedBoardMap row - old_kind is build_board_config's
// own dispatch string (e.g. "rect"); the cleg name registered below is always old_kind + "B" (e.g.
// "rectB"), matching PrescribedBoardMap's own cleg-name field exactly, so this table alone (rather
// than a separate PrescribedBoard enum/PrescribedBoardMap/PrescribedBoardFns trio like the TS side
// has) is enough to drive registration - no UI here needs PrescribedBoardMap's own display-only
// argStr/desc fields.
struct PrescribedEntry { std::string old_kind; std::vector<BoardArgKind> arg_kinds; };
static const std::vector<PrescribedEntry>& prescribed_boards() {
    static const std::vector<PrescribedEntry> table = {
        {"line", {BoardArgKind::Number}},
        {"rect", {BoardArgKind::Number, BoardArgKind::Number}},
        {"rectd", {BoardArgKind::Number, BoardArgKind::Number, BoardArgKind::Number}},
        {"cublat", {BoardArgKind::Number, BoardArgKind::Number, BoardArgKind::Number}},
        {"hcub", {BoardArgKind::Number, BoardArgKind::CommaSeparatedNumbers}},
        {"tri", {BoardArgKind::Number}},
        {"regpoly", {BoardArgKind::Number}},
        {"tetra", {}},
        {"dodeca", {}},
        {"icosa", {}},
        {"trihex", {BoardArgKind::Number}},
        {"hex", {BoardArgKind::Number}},
        {"hexdel", {BoardArgKind::Number}},
        {"snubsq", {BoardArgKind::Number, BoardArgKind::Number, BoardArgKind::Number}},
        {"snubsqtri", {BoardArgKind::Number, BoardArgKind::Number, BoardArgKind::Number}},
        {"twsq", {BoardArgKind::Number, BoardArgKind::Number, BoardArgKind::Number}},
        {"gtsq", {BoardArgKind::Number, BoardArgKind::Number, BoardArgKind::Number}},
        {"star", {BoardArgKind::Number}},
        {"octa", {}},
        {"sier", {BoardArgKind::Number, BoardArgKind::Number}},
        {"ortho", {BoardArgKind::Number}},
        {"dodflake", {BoardArgKind::Number}},
        {"icoflake", {BoardArgKind::Number}},
        {"octaflake", {BoardArgKind::Number}},
        {"polyflake", {BoardArgKind::Number, BoardArgKind::Number}},
        {"cpolyflake", {BoardArgKind::Number, BoardArgKind::Number}},
        {"cpentflake", {BoardArgKind::Number}},
        {"menger", {BoardArgKind::Number, BoardArgKind::Number, BoardArgKind::ZeroOneList}},
        {"ap", {BoardArgKind::Number}},
    };
    return table;
}

static ClegType arg_kind_to_cleg_type(BoardArgKind k) {
    switch (k) {
        case BoardArgKind::Number: return NUMBER_TYPE;
        case BoardArgKind::CommaSeparatedNumbers: return ClegType{CTKind::Array, std::make_shared<ClegType>(NUMBER_TYPE)};
        case BoardArgKind::ZeroOneList: return STRING_TYPE;
    }
    throw std::runtime_error("cleg: arg_kind_to_cleg_type: unexpected BoardArgKind");
}

// Mirrors shared/cleg.ts's valueToBoardArgEntry().
static BoardArgEntry value_to_board_arg_entry(BoardArgKind kind, const ClegValue& val) {
    switch (kind) {
        case BoardArgKind::Number:
            if (val.kind != CTKind::Number)
                throw std::runtime_error("cleg: expected a number argument, got " + type_to_string(cleg_value_type(val)));
            return num_arg(static_cast<int>(val.number));
        case BoardArgKind::CommaSeparatedNumbers: {
            if (val.kind != CTKind::Array || val.elem.kind != CTKind::Number)
                throw std::runtime_error("cleg: expected a number[] argument, got " + type_to_string(cleg_value_type(val)));
            std::vector<int> nums;
            for (auto& v : val.arr_v) nums.push_back(static_cast<int>(v.number));
            return csv_arg(std::move(nums));
        }
        case BoardArgKind::ZeroOneList: {
            if (val.kind != CTKind::String)
                throw std::runtime_error("cleg: expected a string argument, got " + type_to_string(cleg_value_type(val)));
            // Reuses shared/types.ts's own ZeroOneList validation, mirrored directly here (a plain
            // 0/1-character string, each char becoming one int element).
            std::vector<int> values;
            for (char ch : val.str) {
                if (ch != '0' && ch != '1')
                    throw std::runtime_error("cleg: expected a string of 0s and 1s, got '" + val.str + "'");
                values.push_back(ch - '0');
            }
            return zol_arg(std::move(values));
        }
    }
    throw std::runtime_error("cleg: value_to_board_arg_entry: unexpected BoardArgKind");
}

// ── multiProd: N-ary Cartesian board product, restricted by a MultiSelector ────

// Inverse of selector_set_elem_kind above - used only by resolve_any_kind_selector_arg below, which
// (unlike resolve_selector_arg) has no fixed want_kind to check a set's element type against.
static bool is_selector_set_elem_kind(CTKind k) {
    return k == CTKind::Number || k == CTKind::Edge || k == CTKind::Tri || k == CTKind::Quad;
}
static SelectorType selector_type_from_set_elem_kind(CTKind k) {
    switch (k) {
        case CTKind::Number: return SelectorType::Node;
        case CTKind::Edge: return SelectorType::Edge;
        case CTKind::Tri: return SelectorType::Tri;
        case CTKind::Quad: return SelectorType::Quad;
        default: throw std::runtime_error("cleg: selector_type_from_set_elem_kind: not a selector-set element kind");
    }
}

// Mirrors shared/cleg.ts's resolveAnyKindSelectorArg().
static Selector resolve_any_kind_selector_arg(const std::string& callee, const ClegValue& arg) {
    if (arg.kind == CTKind::Sel) return arg.sel_v;
    if (arg.kind == CTKind::Set) {
        if (!is_selector_set_elem_kind(arg.elem.kind))
            throw std::runtime_error(
                "cleg: '" + callee + "': a selector set must be a set of number/edge/tri/quad, got a set of " +
                type_to_string(arg.elem));
        SelectorType want_kind = selector_type_from_set_elem_kind(arg.elem.kind);
        return raw_selector_from_set(want_kind, arg.arr_v);
    }
    throw std::runtime_error("cleg: '" + callee + "': expected sel or set, got " + type_to_string(cleg_value_type(arg)));
}

// Mirrors shared/cleg.ts's FullProductIndex/makeFullProductIndex/fullIndexOf/tupleOfFullIndex - a
// long long index space (rather than TS's own double, which is exact up to 2^53) since the full,
// unrestricted product of several sizable boards can exceed 32-bit int range.
struct FullProductIndex { std::vector<int> Ns; std::vector<long long> stride; long long total; };

static FullProductIndex make_full_product_index(const std::vector<BoardConfig>& boards) {
    FullProductIndex fpi;
    fpi.Ns.resize(boards.size());
    for (size_t i = 0; i < boards.size(); i++) fpi.Ns[i] = boards[i].N;
    fpi.stride.resize(fpi.Ns.size());
    fpi.stride.back() = 1;
    for (int k = static_cast<int>(fpi.Ns.size()) - 2; k >= 0; k--) fpi.stride[k] = fpi.stride[k + 1] * fpi.Ns[k + 1];
    long long total = 1;
    for (int n : fpi.Ns) total *= n;
    fpi.total = total;
    return fpi;
}
static long long full_index_of(const FullProductIndex& fpi, const std::vector<int>& tuple) {
    long long sum = 0;
    for (size_t k = 0; k < tuple.size(); k++) sum += static_cast<long long>(tuple[k]) * fpi.stride[k];
    return sum;
}
static std::vector<int> tuple_of_full_index(const FullProductIndex& fpi, long long idx) {
    std::vector<int> tuple(fpi.Ns.size());
    for (size_t k = 0; k < fpi.Ns.size(); k++) tuple[k] = static_cast<int>((idx / fpi.stride[k]) % fpi.Ns[k]);
    return tuple;
}

// Mirrors shared/cleg.ts's restrictBoardBySelector().
struct RestrictResult { BoardConfig bc; std::vector<int> survivors; };
static RestrictResult restrict_board_by_selector(const BoardConfig& board, const Selector& sel) {
    if (sel.type == SelectorType::Node) {
        auto kept = select_node(board.adj, board.embed, sel);
        std::vector<int> survivors(kept.begin(), kept.end()); // std::set already ascending
        return {node_induced_subgraph(board, kept), std::move(survivors)};
    }
    if (sel.type == SelectorType::Edge) {
        auto edges = select_edge(board.adj, board.embed, sel);
        std::set<int> kept;
        for (auto& e : edges) { kept.insert(e.n1); kept.insert(e.n2); }
        std::vector<int> survivors(kept.begin(), kept.end());
        return {edge_induced_subgraph(board, edges), std::move(survivors)};
    }
    throw std::runtime_error(
        "cleg: multiProd: msBase's own selector must be a node or edge selector, got a '" +
        selector_type_word(sel.type) + "' selector");
}

static std::set<long long> universal_original_indices(const FullProductIndex& fpi) {
    std::set<long long> all;
    for (long long i = 0; i < fpi.total; i++) all.insert(i);
    return all;
}

// Mirrors shared/cleg.ts's buildFromOriginalIndices() - no defaultProductProjMat/Embedding here
// (unlike TS, C++'s BoardConfig has no projection-matrix field at all - it never renders, see
// board_config.h's own top comment), so the fresh BoardConfig is built directly.
struct BuiltResult { BoardConfig bc; std::vector<long long> orig_index; };
static BuiltResult build_from_original_indices(
    const std::vector<BoardConfig>& boards, const FullProductIndex& fpi, const std::set<long long>& kept_original)
{
    std::vector<long long> orig_index(kept_original.begin(), kept_original.end()); // std::set already ascending
    std::vector<std::vector<int>> tuples;
    tuples.reserve(orig_index.size());
    for (long long idx : orig_index) tuples.push_back(tuple_of_full_index(fpi, idx));
    unsigned emb_dim = 0;
    for (auto& b : boards) emb_dim += b.emb_dim;
    size_t K = orig_index.size();
    std::vector<std::vector<unsigned>> pos(K);
    for (size_t i = 0; i < K; i++) {
        pos[i].reserve(emb_dim);
        for (size_t k = 0; k < boards.size(); k++) {
            auto& bp = boards[k].embed[tuples[i][k]];
            pos[i].insert(pos[i].end(), bp.begin(), bp.end());
        }
    }
    auto adj = zero_adj(static_cast<int>(K));
    for (size_t a = 0; a < K; a++) {
        for (size_t b = a + 1; b < K; b++) {
            int diff_coord = -1;
            bool too_many_diffs = false;
            for (size_t k = 0; k < boards.size(); k++) {
                if (tuples[a][k] != tuples[b][k]) {
                    if (diff_coord != -1) { too_many_diffs = true; break; }
                    diff_coord = static_cast<int>(k);
                }
            }
            if (!too_many_diffs && diff_coord >= 0 &&
                boards[diff_coord].adj[tuples[a][diff_coord]][tuples[b][diff_coord]]) {
                adj[a][b] = 1;
                adj[b][a] = 1;
            }
        }
    }
    BoardConfig bc{static_cast<int>(K), std::move(adj), emb_dim, std::move(pos)};
    return {std::move(bc), std::move(orig_index)};
}

// Mirrors shared/cleg.ts's evalMultiSelector() - see its own doc comment for the full algorithm.
static BuiltResult eval_multi_selector(
    const std::vector<BoardConfig>& boards, const FullProductIndex& fpi, const MultiSelector& msel)
{
    switch (msel.op) {
        case MSelOp::All:
            return build_from_original_indices(boards, fpi, universal_original_indices(fpi));
        case MSelOp::Base: {
            auto restricted = restrict_board_by_selector(boards[msel.number], *msel.sel);
            std::vector<BoardConfig> factor_boards = boards;
            factor_boards[msel.number] = restricted.bc;
            BoardConfig bc = factor_boards[0];
            for (size_t i = 1; i < factor_boards.size(); i++) bc = product(bc, factor_boards[i]);
            std::vector<int> local_ns(factor_boards.size());
            for (size_t i = 0; i < factor_boards.size(); i++) local_ns[i] = factor_boards[i].N;
            std::vector<long long> local_stride(local_ns.size());
            local_stride.back() = 1;
            for (int k = static_cast<int>(local_ns.size()) - 2; k >= 0; k--)
                local_stride[k] = local_stride[k + 1] * local_ns[k + 1];
            std::vector<long long> orig_index(bc.N);
            for (int local = 0; local < bc.N; local++) {
                std::vector<int> tuple(local_ns.size());
                for (size_t k = 0; k < local_ns.size(); k++)
                    tuple[k] = static_cast<int>((local / local_stride[k]) % local_ns[k]);
                tuple[msel.number] = restricted.survivors[tuple[msel.number]];
                orig_index[local] = full_index_of(fpi, tuple);
            }
            return {std::move(bc), std::move(orig_index)};
        }
        case MSelOp::Union: {
            std::set<long long> kept;
            for (auto& item : msel.items) {
                auto r = eval_multi_selector(boards, fpi, item);
                kept.insert(r.orig_index.begin(), r.orig_index.end());
            }
            return build_from_original_indices(boards, fpi, kept);
        }
        case MSelOp::Inter: {
            if (msel.items.empty()) return build_from_original_indices(boards, fpi, universal_original_indices(fpi));
            auto first = eval_multi_selector(boards, fpi, msel.items[0]);
            std::set<long long> kept(first.orig_index.begin(), first.orig_index.end());
            for (size_t i = 1; i < msel.items.size(); i++) {
                auto next = eval_multi_selector(boards, fpi, msel.items[i]);
                std::set<long long> next_set(next.orig_index.begin(), next.orig_index.end());
                std::set<long long> out;
                for (long long idx : kept) if (next_set.count(idx)) out.insert(idx);
                kept = std::move(out);
            }
            return build_from_original_indices(boards, fpi, kept);
        }
        case MSelOp::Diff: {
            auto ra = eval_multi_selector(boards, fpi, *msel.a);
            auto rb = eval_multi_selector(boards, fpi, *msel.b);
            std::set<long long> b_set(rb.orig_index.begin(), rb.orig_index.end());
            std::set<long long> out;
            for (long long idx : ra.orig_index) if (!b_set.count(idx)) out.insert(idx);
            return build_from_original_indices(boards, fpi, out);
        }
    }
    throw std::runtime_error("cleg: eval_multi_selector: unexpected MSelOp");
}

// Mirrors shared/cleg.ts's BUILTIN_FUNCTIONS - built once via a function-local static (thread-safety
// is a non-issue for this single-threaded CLI tooling), one lambda per row instead of TS's top-level
// `BUILTIN_FUNCTIONS['x'] = {...}` assignments.
static const std::unordered_map<std::string, BuiltinFunction>& builtin_functions() {
    static const std::unordered_map<std::string, BuiltinFunction> table = [] {
        std::unordered_map<std::string, BuiltinFunction> m;

        // One builtin per shared/boardConfig.ts's own PrescribedBoardMap/PrescribedBoardFns entry -
        // see prescribed_boards()'s own doc comment above.
        for (auto& entry : prescribed_boards()) {
            std::string cleg_name = entry.old_kind + "B";
            std::vector<ClegType> params;
            for (auto k : entry.arg_kinds) params.push_back(arg_kind_to_cleg_type(k));
            std::string old_kind = entry.old_kind;
            std::vector<BoardArgKind> arg_kinds = entry.arg_kinds;
            m[cleg_name] = BuiltinFunction{
                fixed_signature(params, EGR_TYPE),
                [old_kind, arg_kinds](const std::vector<ClegValue>& args) {
                    std::vector<BoardArgEntry> board_args;
                    for (size_t i = 0; i < arg_kinds.size(); i++)
                        board_args.push_back(value_to_board_arg_entry(arg_kinds[i], args[i]));
                    return make_egr(build_board_config(old_kind, board_args));
                },
            };
        }

        m["len"] = BuiltinFunction{
            [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
                if (arg_types.size() != 1)
                    throw std::runtime_error("cleg: '" + callee + "' expects 1 argument(s), got " + std::to_string(arg_types.size()));
                if (arg_types[0].kind != CTKind::Array && arg_types[0].kind != CTKind::Set)
                    throw std::runtime_error("cleg: '" + callee + "' argument 1: expected an array or set, got " + type_to_string(arg_types[0]));
                return NUMBER_TYPE;
            },
            [](const std::vector<ClegValue>& args) { return make_number(static_cast<double>(args[0].arr_v.size())); },
        };

        CheckCallFn rand_rm_check_call = [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
            if (arg_types.size() != 2)
                throw std::runtime_error("cleg: '" + callee + "' expects 2 argument(s), got " + std::to_string(arg_types.size()));
            if (arg_types[0].kind != CTKind::Set)
                throw std::runtime_error("cleg: '" + callee + "' argument 1: expected a set, got " + type_to_string(arg_types[0]));
            if (arg_types[1].kind != CTKind::Number)
                throw std::runtime_error("cleg: '" + callee + "' argument 2: expected number, got " + type_to_string(arg_types[1]));
            return arg_types[0];
        };
        m["randRmN"] = BuiltinFunction{
            rand_rm_check_call,
            [](const std::vector<ClegValue>& args) {
                double count = args[1].number;
                if (count != std::floor(count) || count < 0)
                    throw std::runtime_error("cleg: 'randRmN' count must be a nonnegative integer, got " + format_number_display(count));
                auto kept = randomly_remove(args[0].arr_v, static_cast<int>(count));
                ClegValue v; v.kind = CTKind::Set; v.elem = args[0].elem; v.arr_v = std::move(kept);
                return v;
            },
        };
        m["randRmP"] = BuiltinFunction{
            rand_rm_check_call,
            [](const std::vector<ClegValue>& args) {
                double frac = args[1].number;
                if (!std::isfinite(frac) || frac < 0)
                    throw std::runtime_error("cleg: 'randRmP' portion must be a nonnegative number, got " + format_number_display(frac));
                int remove_count = static_cast<int>(std::floor(frac * static_cast<double>(args[0].arr_v.size())));
                auto kept = randomly_remove(args[0].arr_v, remove_count);
                ClegValue v; v.kind = CTKind::Set; v.elem = args[0].elem; v.arr_v = std::move(kept);
                return v;
            },
        };

        m["mkEdge"] = BuiltinFunction{
            fixed_signature({NUMBER_TYPE, NUMBER_TYPE}, EDGE_TYPE),
            [](const std::vector<ClegValue>& args) {
                return make_edge_v(make_board_edge(static_cast<int>(args[0].number), static_cast<int>(args[1].number)));
            },
        };
        m["mkTri"] = BuiltinFunction{
            fixed_signature({NUMBER_TYPE, NUMBER_TYPE, NUMBER_TYPE}, TRI_TYPE),
            [](const std::vector<ClegValue>& args) {
                return make_tri_v(make_board_triangle(
                    static_cast<int>(args[0].number), static_cast<int>(args[1].number), static_cast<int>(args[2].number)));
            },
        };
        m["mkQuad"] = BuiltinFunction{
            fixed_signature({NUMBER_TYPE, NUMBER_TYPE, NUMBER_TYPE, NUMBER_TYPE}, QUAD_TYPE),
            [](const std::vector<ClegValue>& args) {
                return make_quad_v(make_board_quad(
                    static_cast<int>(args[0].number), static_cast<int>(args[1].number),
                    static_cast<int>(args[2].number), static_cast<int>(args[3].number)));
            },
        };

        m["prod"] = BuiltinFunction{
            fixed_signature({EGR_TYPE, EGR_TYPE}, EGR_TYPE),
            [](const std::vector<ClegValue>& args) { return make_egr(product(*args[0].egr_v, *args[1].egr_v)); },
        };

        m["mkSel"] = BuiltinFunction{
            [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
                if (arg_types.size() != 2)
                    throw std::runtime_error("cleg: '" + callee + "' expects 2 argument(s), got " + std::to_string(arg_types.size()));
                if (arg_types[0].kind != CTKind::String)
                    throw std::runtime_error("cleg: '" + callee + "' argument 1: expected string, got " + type_to_string(arg_types[0]));
                if (arg_types[1].kind != CTKind::String && arg_types[1].kind != CTKind::Set)
                    throw std::runtime_error("cleg: '" + callee + "' argument 2: expected string or set, got " + type_to_string(arg_types[1]));
                return ClegType{CTKind::Sel, nullptr};
            },
            [](const std::vector<ClegValue>& args) {
                SelectorType kind = selector_type_from_word(args[0].str);
                ClegValue v; v.kind = CTKind::Sel; v.sel_type = kind;
                v.sel_v = resolve_selector_arg("mkSel", args[1], kind, selector_parser_for(kind));
                return v;
            },
        };

        CheckCallFn mk_form_sel_check_call = [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
            if (arg_types.size() != 1 && arg_types.size() != 2)
                throw std::runtime_error("cleg: '" + callee + "' expects 1 or 2 argument(s), got " + std::to_string(arg_types.size()));
            if (arg_types[0].kind != CTKind::String)
                throw std::runtime_error("cleg: '" + callee + "' argument 1: expected string, got " + type_to_string(arg_types[0]));
            if (arg_types.size() == 2 && arg_types[1].kind != CTKind::Sel && arg_types[1].kind != CTKind::String && arg_types[1].kind != CTKind::Set)
                throw std::runtime_error("cleg: '" + callee + "' argument 2: expected sel, string, or set, got " + type_to_string(arg_types[1]));
            return FORMSEL_TYPE;
        };
        m["mkFormSel"] = BuiltinFunction{
            mk_form_sel_check_call,
            [](const std::vector<ClegValue>& args) {
                const std::string& kind = args[0].str;
                if (kind != "tri" && kind != "quad")
                    throw std::runtime_error("cleg: mkFormSel: unknown form-selector kind '" + kind + "' - expected tri/quad");
                SelectorType st = kind == "tri" ? SelectorType::Tri : SelectorType::Quad;
                FormSelector fs; fs.kind = kind == "tri" ? FormSelectorKind::Tri : FormSelectorKind::Quad;
                if (args.size() == 2) fs.sel = resolve_selector_arg("mkFormSel", args[1], st, selector_parser_for(st));
                return make_form_sel(fs);
            },
        };

        m["rectify"] = BuiltinFunction{
            fixed_signature({}, MOD_TYPE), [](const std::vector<ClegValue>&) { return make_mod(BoardModifier{ModifierKind::Rectify}); },
        };
        m["globalCentralize"] = BuiltinFunction{
            fixed_signature({}, MOD_TYPE), [](const std::vector<ClegValue>&) { return make_mod(BoardModifier{ModifierKind::GlobalCentralize}); },
        };
        m["quadOctarize"] = BuiltinFunction{
            fixed_signature({}, MOD_TYPE), [](const std::vector<ClegValue>&) { return make_mod(BoardModifier{ModifierKind::QuadOctarize}); },
        };
        m["edgeSplit"] = BuiltinFunction{
            fixed_signature({NUMBER_TYPE}, MOD_TYPE),
            [](const std::vector<ClegValue>& args) {
                BoardModifier bm{ModifierKind::EdgeSplit}; bm.split_n = static_cast<int>(args[0].number);
                return make_mod(bm);
            },
        };
        m["mergeClose"] = BuiltinFunction{
            fixed_signature({NUMBER_TYPE}, MOD_TYPE),
            [](const std::vector<ClegValue>& args) {
                BoardModifier bm{ModifierKind::MergeClose}; bm.dist = args[0].number;
                return make_mod(bm);
            },
        };
        m["scale"] = BuiltinFunction{
            fixed_signature({NUMBER_TYPE}, MOD_TYPE),
            [](const std::vector<ClegValue>& args) {
                BoardModifier bm{ModifierKind::Scale}; bm.dist = args[0].number;
                return make_mod(bm);
            },
        };

        CheckCallFn induced_subgraph_mod_check_call = [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
            if (arg_types.size() != 1)
                throw std::runtime_error("cleg: '" + callee + "' expects 1 argument(s), got " + std::to_string(arg_types.size()));
            if (arg_types[0].kind != CTKind::Sel && arg_types[0].kind != CTKind::String && arg_types[0].kind != CTKind::Set)
                throw std::runtime_error("cleg: '" + callee + "' argument 1: expected sel, string, or set, got " + type_to_string(arg_types[0]));
            return MOD_TYPE;
        };
        m["nis"] = BuiltinFunction{
            induced_subgraph_mod_check_call,
            [](const std::vector<ClegValue>& args) {
                BoardModifier bm{ModifierKind::NodeInducedSubgraph};
                bm.sel = resolve_selector_arg("nis", args[0], SelectorType::Node, parse_node_selector);
                return make_mod(bm);
            },
        };
        m["eis"] = BuiltinFunction{
            induced_subgraph_mod_check_call,
            [](const std::vector<ClegValue>& args) {
                BoardModifier bm{ModifierKind::EdgeInducedSubgraph};
                bm.sel = resolve_selector_arg("eis", args[0], SelectorType::Edge, parse_edge_selector);
                return make_mod(bm);
            },
        };

        auto select_set_check_call = [](ClegType elem_type) -> CheckCallFn {
            return [elem_type](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
                if (arg_types.size() != 2)
                    throw std::runtime_error("cleg: '" + callee + "' expects 2 argument(s), got " + std::to_string(arg_types.size()));
                if (arg_types[0].kind != CTKind::Sel && arg_types[0].kind != CTKind::String && arg_types[0].kind != CTKind::Set)
                    throw std::runtime_error("cleg: '" + callee + "' argument 1: expected sel, string, or set, got " + type_to_string(arg_types[0]));
                if (arg_types[1].kind != CTKind::Egr)
                    throw std::runtime_error("cleg: '" + callee + "' argument 2: expected egr, got " + type_to_string(arg_types[1]));
                return ClegType{CTKind::Set, std::make_shared<ClegType>(elem_type)};
            };
        };
        m["selectNode"] = BuiltinFunction{
            select_set_check_call(NUMBER_TYPE),
            [](const std::vector<ClegValue>& args) {
                Selector sel = resolve_selector_arg("selectNode", args[0], SelectorType::Node, parse_node_selector);
                const BoardConfig& bc = *args[1].egr_v;
                auto nodes = select_node(bc.adj, bc.embed, sel);
                std::vector<ClegValue> values;
                for (int n : nodes) values.push_back(make_number(n));
                return make_cleg_set(NUMBER_TYPE, std::move(values));
            },
        };
        m["selectEdge"] = BuiltinFunction{
            select_set_check_call(EDGE_TYPE),
            [](const std::vector<ClegValue>& args) {
                Selector sel = resolve_selector_arg("selectEdge", args[0], SelectorType::Edge, parse_edge_selector);
                const BoardConfig& bc = *args[1].egr_v;
                auto edges = select_edge(bc.adj, bc.embed, sel);
                std::vector<ClegValue> values;
                for (auto& e : edges) values.push_back(make_edge_v(e));
                return make_cleg_set(EDGE_TYPE, std::move(values));
            },
        };
        m["selectTriangle"] = BuiltinFunction{
            select_set_check_call(TRI_TYPE),
            [](const std::vector<ClegValue>& args) {
                Selector sel = resolve_selector_arg("selectTriangle", args[0], SelectorType::Tri, parse_triangle_selector);
                const BoardConfig& bc = *args[1].egr_v;
                auto tris = select_triangle(bc.adj, bc.embed, sel);
                std::vector<ClegValue> values;
                for (auto& t : tris) values.push_back(make_tri_v(t));
                return make_cleg_set(TRI_TYPE, std::move(values));
            },
        };
        m["selectQuad"] = BuiltinFunction{
            select_set_check_call(QUAD_TYPE),
            [](const std::vector<ClegValue>& args) {
                Selector sel = resolve_selector_arg("selectQuad", args[0], SelectorType::Quad, parse_quad_selector);
                const BoardConfig& bc = *args[1].egr_v;
                auto quads = select_quad(bc.adj, bc.embed, sel);
                std::vector<ClegValue> values;
                for (auto& q : quads) values.push_back(make_quad_v(q));
                return make_cleg_set(QUAD_TYPE, std::move(values));
            },
        };

        CheckCallFn form_mod_check_call = [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
            if (arg_types.size() != 1 && arg_types.size() != 2)
                throw std::runtime_error("cleg: '" + callee + "' expects 1 or 2 argument(s), got " + std::to_string(arg_types.size()));
            if (arg_types[0].kind != CTKind::Number)
                throw std::runtime_error("cleg: '" + callee + "' argument 1: expected number, got " + type_to_string(arg_types[0]));
            if (arg_types.size() == 2 && arg_types[1].kind != CTKind::Sel && arg_types[1].kind != CTKind::String && arg_types[1].kind != CTKind::Set)
                throw std::runtime_error("cleg: '" + callee + "' argument 2: expected sel, string, or set, got " + type_to_string(arg_types[1]));
            return MOD_TYPE;
        };
        m["triangleForm"] = BuiltinFunction{
            form_mod_check_call,
            [](const std::vector<ClegValue>& args) {
                BoardModifier bm{ModifierKind::TriangleForm};
                bm.split_n = static_cast<int>(args[0].number);
                if (args.size() == 2) bm.form_sel = resolve_selector_arg("triangleForm", args[1], SelectorType::Tri, parse_triangle_selector);
                return make_mod(bm);
            },
        };
        m["quadForm"] = BuiltinFunction{
            form_mod_check_call,
            [](const std::vector<ClegValue>& args) {
                BoardModifier bm{ModifierKind::QuadForm};
                bm.split_n = static_cast<int>(args[0].number);
                if (args.size() == 2) bm.form_sel = resolve_selector_arg("quadForm", args[1], SelectorType::Quad, parse_quad_selector);
                return make_mod(bm);
            },
        };

        m["form"] = BuiltinFunction{
            [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
                if (arg_types.size() < 2)
                    throw std::runtime_error(
                        "cleg: '" + callee + "' expects at least 2 argument(s) (w, and >= 1 formSel), got " + std::to_string(arg_types.size()));
                if (arg_types[0].kind != CTKind::Number)
                    throw std::runtime_error("cleg: '" + callee + "' argument 1: expected number, got " + type_to_string(arg_types[0]));
                for (size_t i = 1; i < arg_types.size(); i++)
                    if (arg_types[i].kind != CTKind::FormSel)
                        throw std::runtime_error("cleg: '" + callee + "' argument " + std::to_string(i + 1) + ": expected formSel, got " + type_to_string(arg_types[i]));
                return MOD_TYPE;
            },
            [](const std::vector<ClegValue>& args) {
                BoardModifier bm{ModifierKind::Form};
                bm.split_n = static_cast<int>(args[0].number);
                for (size_t i = 1; i < args.size(); i++) bm.form_sels.push_back(args[i].formsel_v);
                return make_mod(bm);
            },
        };

        m["modify"] = BuiltinFunction{
            fixed_signature({ClegType{CTKind::Array, std::make_shared<ClegType>(MOD_TYPE)}, EGR_TYPE}, EGR_TYPE),
            [](const std::vector<ClegValue>& args) {
                std::vector<BoardModifier> mods;
                for (auto& v : args[0].arr_v) mods.push_back(v.mod_v);
                return make_egr(apply_modifiers(*args[1].egr_v, mods));
            },
        };

        m["msAll"] = BuiltinFunction{
            fixed_signature({}, MSEL_TYPE), [](const std::vector<ClegValue>&) { return make_msel(MultiSelector{MSelOp::All}); },
        };
        m["msBase"] = BuiltinFunction{
            [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
                if (arg_types.size() != 2)
                    throw std::runtime_error("cleg: '" + callee + "' expects 2 argument(s), got " + std::to_string(arg_types.size()));
                if (arg_types[0].kind != CTKind::Number)
                    throw std::runtime_error("cleg: '" + callee + "' argument 1: expected number, got " + type_to_string(arg_types[0]));
                if (arg_types[1].kind != CTKind::Sel && arg_types[1].kind != CTKind::Set)
                    throw std::runtime_error("cleg: '" + callee + "' argument 2: expected sel or set, got " + type_to_string(arg_types[1]));
                return MSEL_TYPE;
            },
            [](const std::vector<ClegValue>& args) {
                double number = args[0].number;
                if (number != std::floor(number) || number < 0)
                    throw std::runtime_error("cleg: msBase: number must be a nonnegative integer, got " + format_number_display(number));
                MultiSelector ms; ms.op = MSelOp::Base; ms.number = static_cast<int>(number);
                ms.sel = std::make_shared<Selector>(resolve_any_kind_selector_arg("msBase", args[1]));
                return make_msel(ms);
            },
        };
        m["msUnion"] = BuiltinFunction{
            fixed_signature({ClegType{CTKind::Array, std::make_shared<ClegType>(MSEL_TYPE)}}, MSEL_TYPE),
            [](const std::vector<ClegValue>& args) {
                MultiSelector ms; ms.op = MSelOp::Union;
                for (auto& v : args[0].arr_v) ms.items.push_back(v.msel_v);
                return make_msel(ms);
            },
        };
        m["msInter"] = BuiltinFunction{
            fixed_signature({ClegType{CTKind::Array, std::make_shared<ClegType>(MSEL_TYPE)}}, MSEL_TYPE),
            [](const std::vector<ClegValue>& args) {
                MultiSelector ms; ms.op = MSelOp::Inter;
                for (auto& v : args[0].arr_v) ms.items.push_back(v.msel_v);
                return make_msel(ms);
            },
        };
        m["msDiff"] = BuiltinFunction{
            fixed_signature({MSEL_TYPE, MSEL_TYPE}, MSEL_TYPE),
            [](const std::vector<ClegValue>& args) {
                MultiSelector ms; ms.op = MSelOp::Diff;
                ms.a = std::make_shared<MultiSelector>(args[0].msel_v);
                ms.b = std::make_shared<MultiSelector>(args[1].msel_v);
                return make_msel(ms);
            },
        };
        m["multiProd"] = BuiltinFunction{
            fixed_signature({ClegType{CTKind::Array, std::make_shared<ClegType>(EGR_TYPE)}, MSEL_TYPE}, EGR_TYPE),
            [](const std::vector<ClegValue>& args) {
                std::vector<BoardConfig> boards;
                for (auto& v : args[0].arr_v) boards.push_back(*v.egr_v);
                if (boards.empty()) throw std::runtime_error("cleg: multiProd: boards must be non-empty");
                FullProductIndex fpi = make_full_product_index(boards);
                auto result = eval_multi_selector(boards, fpi, args[1].msel_v);
                return make_egr(std::move(result.bc));
            },
        };

        return m;
    }();
    return table;
}

// ── Type checking ──────────────────────────────────────────────────────────────

// Mirrors shared/cleg.ts's TypeEnv - `parent` is a raw pointer into an enclosing stack frame (see
// this file's own top comment on why that's safe here), not a GC'd reference.
struct TypeEnv {
    std::unordered_map<std::string, ClegType> vars;
    TypeEnv* parent = nullptr;
};
static const ClegType* lookup_var_type(TypeEnv& env, const std::string& name) {
    for (TypeEnv* e = &env; e; e = e->parent) {
        auto it = e->vars.find(name);
        if (it != e->vars.end()) return &it->second;
    }
    return nullptr;
}

struct FunctionSignature { std::vector<ClegType> params; ClegType return_type; };
using FuncTable = std::unordered_map<std::string, FunctionSignature>;

static void check_block(const Stmt& block, TypeEnv* parent, FuncTable& funcs, const ClegType& return_type);
static void check_stmt(const Stmt& stmt, TypeEnv& env, FuncTable& funcs, const ClegType& return_type);
static ClegType check_expr(const Expr& expr, TypeEnv& env, FuncTable& funcs);

static void check_block(const Stmt& block, TypeEnv* parent, FuncTable& funcs, const ClegType& return_type) {
    TypeEnv env; env.parent = parent;
    for (auto& stmt : block.stmts) check_stmt(stmt, env, funcs, return_type);
}

static void check_stmt(const Stmt& stmt, TypeEnv& env, FuncTable& funcs, const ClegType& return_type) {
    switch (stmt.kind) {
        case StmtKind::VarDecl: {
            if (env.vars.count(stmt.name))
                throw std::runtime_error("cleg: '" + stmt.name + "' is already declared in this scope");
            ClegType init_type = check_expr(*stmt.expr, env, funcs);
            if (!type_equals(init_type, stmt.decl_type))
                throw std::runtime_error(
                    "cleg: cannot initialize '" + stmt.name + "' of type " + type_to_string(stmt.decl_type) +
                    " with a value of type " + type_to_string(init_type));
            env.vars[stmt.name] = stmt.decl_type;
            return;
        }
        case StmtKind::AssignStmt: {
            const ClegType* var_type = lookup_var_type(env, stmt.name);
            if (!var_type) throw std::runtime_error("cleg: assignment to undeclared variable '" + stmt.name + "'");
            ClegType value_type = check_expr(*stmt.expr, env, funcs);
            if (!type_equals(value_type, *var_type))
                throw std::runtime_error(
                    "cleg: cannot assign a value of type " + type_to_string(value_type) + " to '" + stmt.name +
                    "' of type " + type_to_string(*var_type));
            return;
        }
        case StmtKind::IfStmt: {
            ClegType cond_type = check_expr(*stmt.cond, env, funcs);
            if (cond_type.kind != CTKind::Bool)
                throw std::runtime_error("cleg: if condition must be bool, got " + type_to_string(cond_type));
            check_block(*stmt.then_stmt, &env, funcs, return_type);
            if (stmt.else_stmt) {
                if (stmt.else_stmt->kind == StmtKind::Block) check_block(*stmt.else_stmt, &env, funcs, return_type);
                else check_stmt(*stmt.else_stmt, env, funcs, return_type);
            }
            return;
        }
        case StmtKind::ForStmt: {
            // A fresh scope for init's own declared variable (if any), enclosing cond/update/body -
            // NOT the same scope as body's own (check_block below gives body its own further-nested
            // scope, same as every other BLOCK) - see ForStmt's own doc comment (shared/cleg.ts).
            TypeEnv loop_env; loop_env.parent = &env;
            if (stmt.for_init) check_stmt(*stmt.for_init, loop_env, funcs, return_type);
            if (stmt.cond) {
                ClegType cond_type = check_expr(*stmt.cond, loop_env, funcs);
                if (cond_type.kind != CTKind::Bool)
                    throw std::runtime_error("cleg: for-loop condition must be bool, got " + type_to_string(cond_type));
            }
            if (stmt.for_update) check_stmt(*stmt.for_update, loop_env, funcs, return_type);
            check_block(*stmt.body, &loop_env, funcs, return_type);
            return;
        }
        case StmtKind::ReturnStmt: {
            ClegType t = check_expr(*stmt.expr, env, funcs);
            if (!type_equals(t, return_type))
                throw std::runtime_error(
                    "cleg: return type mismatch - function returns " + type_to_string(return_type) + ", got " + type_to_string(t));
            return;
        }
        case StmtKind::ExprStmt:
            check_expr(*stmt.expr, env, funcs);
            return;
        case StmtKind::Block:
            check_block(stmt, &env, funcs, return_type);
            return;
    }
}

static ClegType check_expr(const Expr& expr, TypeEnv& env, FuncTable& funcs) {
    switch (expr.kind) {
        case ExprKind::NumberLit: return ClegType{CTKind::Number, nullptr};
        case ExprKind::StringLit: return ClegType{CTKind::String, nullptr};
        case ExprKind::BoolLit: return ClegType{CTKind::Bool, nullptr};
        case ExprKind::Identifier: {
            const ClegType* t = lookup_var_type(env, expr.string_value);
            if (!t) throw std::runtime_error("cleg: undeclared variable '" + expr.string_value + "'");
            return *t;
        }
        case ExprKind::ArrayLit: {
            if (expr.elements.empty()) throw std::runtime_error("cleg: cannot infer the element type of an empty array literal '[]'");
            std::vector<ClegType> elem_types;
            for (auto& e : expr.elements) elem_types.push_back(check_expr(e, env, funcs));
            for (size_t i = 1; i < elem_types.size(); i++)
                if (!type_equals(elem_types[i], elem_types[0]))
                    throw std::runtime_error(
                        "cleg: array literal mixes " + type_to_string(elem_types[0]) + " and " + type_to_string(elem_types[i]) + " elements");
            return ClegType{CTKind::Array, std::make_shared<ClegType>(elem_types[0])};
        }
        case ExprKind::SetLit: {
            if (expr.elements.empty()) throw std::runtime_error("cleg: cannot infer the element type of an empty set literal '{}'");
            std::vector<ClegType> elem_types;
            for (auto& e : expr.elements) elem_types.push_back(check_expr(e, env, funcs));
            for (size_t i = 1; i < elem_types.size(); i++)
                if (!type_equals(elem_types[i], elem_types[0]))
                    throw std::runtime_error(
                        "cleg: set literal mixes " + type_to_string(elem_types[0]) + " and " + type_to_string(elem_types[i]) + " elements");
            if (!is_set_elem_kind(elem_types[0].kind))
                throw std::runtime_error(
                    "cleg: '" + type_to_string(elem_types[0]) + "' is not a supported set element type - only "
                    "number/string/bool/edge/tri/quad may be set elements");
            return ClegType{CTKind::Set, std::make_shared<ClegType>(elem_types[0])};
        }
        case ExprKind::CallExpr: {
            std::vector<ClegType> arg_types;
            for (auto& a : expr.elements) arg_types.push_back(check_expr(a, env, funcs));
            auto& builtins = builtin_functions();
            auto bit = builtins.find(expr.string_value);
            if (bit != builtins.end()) return bit->second.check_call(expr.string_value, arg_types);
            auto fit = funcs.find(expr.string_value);
            if (fit == funcs.end()) throw std::runtime_error("cleg: call to undeclared function '" + expr.string_value + "'");
            auto& sig = fit->second;
            if (arg_types.size() != sig.params.size())
                throw std::runtime_error(
                    "cleg: '" + expr.string_value + "' expects " + std::to_string(sig.params.size()) + " argument(s), got " +
                    std::to_string(arg_types.size()));
            for (size_t i = 0; i < arg_types.size(); i++)
                if (!type_equals(arg_types[i], sig.params[i]))
                    throw std::runtime_error(
                        "cleg: '" + expr.string_value + "' argument " + std::to_string(i + 1) + ": expected " +
                        type_to_string(sig.params[i]) + ", got " + type_to_string(arg_types[i]));
            return sig.return_type;
        }
        case ExprKind::BinaryExpr: {
            ClegType l = check_expr(*expr.left, env, funcs);
            ClegType r = check_expr(*expr.right, env, funcs);
            auto& overloads = binary_operator_overloads().at(expr.op);
            for (auto& overload : overloads) {
                auto m = overload.match(l, r);
                if (m) return m->type;
            }
            std::string sigs;
            for (size_t i = 0; i < overloads.size(); i++) { if (i) sigs += "; "; sigs += overloads[i].signature; }
            throw std::runtime_error(
                "cleg: operator '" + expr.op + "' has no overload for operand types " + type_to_string(l) + " and " +
                type_to_string(r) + " (expected one of: " + sigs + ")");
        }
        case ExprKind::UnaryExpr: {
            ClegType t = check_expr(*expr.operand, env, funcs);
            if (t.kind != CTKind::Number) throw std::runtime_error("cleg: unary '-' requires a number operand, got " + type_to_string(t));
            return ClegType{CTKind::Number, nullptr};
        }
        case ExprKind::NilExpr:
            return ClegType{CTKind::Array, std::make_shared<ClegType>(expr.nil_type)};
    }
    throw std::runtime_error("cleg: check_expr: unexpected ExprKind");
}

// Mirrors shared/cleg.ts's typecheckCleg() - internal (unlike TS, no consumer here needs the checked
// result type directly; typecheck_cleg_as_board (public, below) just needs to know it's egr).
static ClegType typecheck_cleg(const ClegProgram& program) {
    FuncTable funcs;
    for (auto& fn : program.functions) {
        if (funcs.count(fn.name) || builtin_functions().count(fn.name))
            throw std::runtime_error("cleg: function '" + fn.name + "' is declared more than once (or shadows a builtin function)");
        FunctionSignature sig;
        for (auto& p : fn.params) sig.params.push_back(p.type);
        sig.return_type = fn.return_type;
        funcs[fn.name] = sig;
    }
    for (auto& fn : program.functions) {
        TypeEnv param_env;
        for (auto& p : fn.params) param_env.vars[p.name] = p.type;
        check_block(fn.body, &param_env, funcs, fn.return_type);
    }

    if (program.stmts.empty()) throw std::runtime_error("cleg: program has no top-level statement");
    if (program.stmts.back().kind != StmtKind::ExprStmt)
        throw std::runtime_error(
            "cleg: the last top-level statement must be an expression, got " + stmt_kind_word(program.stmts.back().kind));

    TypeEnv env;
    std::optional<ClegType> result_type;
    for (auto& stmt : program.stmts) {
        if (stmt.kind == StmtKind::ExprStmt) result_type = check_expr(*stmt.expr, env, funcs);
        else check_stmt(stmt, env, funcs, EGR_TYPE);
    }
    return *result_type;
}

void typecheck_cleg_as_board(const ClegProgram& program) {
    ClegType t = typecheck_cleg(program);
    if (t.kind != CTKind::Egr) throw std::runtime_error("cleg: a board description must produce an egr, got " + type_to_string(t));
}

// ── Evaluation ───────────────────────────────────────────────────────────────

// Mirrors shared/cleg.ts's ValueEnv - same raw-pointer parent-chain convention as TypeEnv above.
struct ValueEnv {
    std::unordered_map<std::string, ClegValue> vars;
    ValueEnv* parent = nullptr;
};
static ClegValue* lookup_value_ptr(ValueEnv& env, const std::string& name) {
    for (ValueEnv* e = &env; e; e = e->parent) {
        auto it = e->vars.find(name);
        if (it != e->vars.end()) return &it->second;
    }
    return nullptr;
}
// Unreachable in a program that has passed typecheck_cleg - every Identifier there already resolved
// to a declared variable. Mirrors shared/cleg.ts's lookupValue().
static ClegValue lookup_value(ValueEnv& env, const std::string& name) {
    ClegValue* v = lookup_value_ptr(env, name);
    if (!v) throw std::runtime_error("cleg: undeclared variable '" + name + "'");
    return *v;
}
// Mirrors shared/cleg.ts's setValue() - mutates `name`'s existing binding in whichever env of the
// chain declared it.
static void set_value(ValueEnv& env, const std::string& name, ClegValue value) {
    ClegValue* v = lookup_value_ptr(env, name);
    if (!v) throw std::runtime_error("cleg: undeclared variable '" + name + "'");
    *v = std::move(value);
}

using UserFuncTable = std::unordered_map<std::string, const FunctionDecl*>;

// Mirrors shared/cleg.ts's ReturnSignal - thrown to unwind out of nested blocks/if-statements on
// `return`, always caught by call_user_function below.
struct ReturnSignal { ClegValue value; };

static void eval_block(const Stmt& block, ValueEnv* parent, UserFuncTable& funcs);
static void eval_stmt(const Stmt& stmt, ValueEnv& env, UserFuncTable& funcs);
static ClegValue eval_expr(const Expr& expr, ValueEnv& env, UserFuncTable& funcs);
static ClegValue call_user_function(const FunctionDecl& fn, const std::vector<ClegValue>& args, UserFuncTable& funcs);

static void eval_block(const Stmt& block, ValueEnv* parent, UserFuncTable& funcs) {
    ValueEnv env; env.parent = parent;
    for (auto& stmt : block.stmts) eval_stmt(stmt, env, funcs);
}

static void eval_stmt(const Stmt& stmt, ValueEnv& env, UserFuncTable& funcs) {
    switch (stmt.kind) {
        case StmtKind::VarDecl:
            env.vars[stmt.name] = eval_expr(*stmt.expr, env, funcs);
            return;
        case StmtKind::AssignStmt:
            set_value(env, stmt.name, eval_expr(*stmt.expr, env, funcs));
            return;
        case StmtKind::IfStmt: {
            ClegValue cond = eval_expr(*stmt.cond, env, funcs);
            if (cond.boolean) eval_block(*stmt.then_stmt, &env, funcs);
            else if (stmt.else_stmt) {
                if (stmt.else_stmt->kind == StmtKind::Block) eval_block(*stmt.else_stmt, &env, funcs);
                else eval_stmt(*stmt.else_stmt, env, funcs);
            }
            return;
        }
        case StmtKind::ForStmt: {
            // One scope for the whole loop (init's own variable, if any, persists across every
            // iteration) - body gets its own further-nested scope each iteration via eval_block, same
            // as any other BLOCK - see ForStmt's own doc comment (shared/cleg.ts).
            ValueEnv loop_env; loop_env.parent = &env;
            if (stmt.for_init) eval_stmt(*stmt.for_init, loop_env, funcs);
            while (!stmt.cond || eval_expr(*stmt.cond, loop_env, funcs).boolean) {
                eval_block(*stmt.body, &loop_env, funcs);
                if (stmt.for_update) eval_stmt(*stmt.for_update, loop_env, funcs);
            }
            return;
        }
        case StmtKind::ReturnStmt:
            throw ReturnSignal{eval_expr(*stmt.expr, env, funcs)};
        case StmtKind::ExprStmt:
            eval_expr(*stmt.expr, env, funcs);
            return;
        case StmtKind::Block:
            eval_block(stmt, &env, funcs);
            return;
    }
}

static ClegValue eval_expr(const Expr& expr, ValueEnv& env, UserFuncTable& funcs) {
    switch (expr.kind) {
        case ExprKind::NumberLit: return make_number(expr.number_value);
        case ExprKind::StringLit: return make_string(expr.string_value);
        case ExprKind::BoolLit: return make_bool(expr.bool_value);
        case ExprKind::Identifier: return lookup_value(env, expr.string_value);
        case ExprKind::ArrayLit: {
            std::vector<ClegValue> values;
            for (auto& e : expr.elements) values.push_back(eval_expr(e, env, funcs));
            // typecheck_cleg already rejected an empty or mixed-element-type literal, so the first
            // value's own type is always the array's element type.
            ClegValue v; v.kind = CTKind::Array; v.elem = cleg_value_type(values[0]); v.arr_v = std::move(values);
            return v;
        }
        case ExprKind::SetLit: {
            std::vector<ClegValue> values;
            for (auto& e : expr.elements) values.push_back(eval_expr(e, env, funcs));
            // Same evaluation-order hazard as parse_expr's own precedence chain (see its own
            // comment) - cleg_value_type(values[0]) must be computed in its own statement, before
            // std::move(values) is ever handed to make_cleg_set as a sibling argument.
            ClegType elem = cleg_value_type(values[0]);
            return make_cleg_set(std::move(elem), std::move(values));
        }
        case ExprKind::CallExpr: {
            std::vector<ClegValue> args;
            for (auto& a : expr.elements) args.push_back(eval_expr(a, env, funcs));
            auto& builtins = builtin_functions();
            auto bit = builtins.find(expr.string_value);
            if (bit != builtins.end()) return bit->second.call(args);
            return call_user_function(*funcs.at(expr.string_value), args, funcs);
        }
        case ExprKind::BinaryExpr: {
            ClegValue l = eval_expr(*expr.left, env, funcs);
            ClegValue r = eval_expr(*expr.right, env, funcs);
            ClegType lt = cleg_value_type(l), rt = cleg_value_type(r);
            for (auto& overload : binary_operator_overloads().at(expr.op)) {
                auto m = overload.match(lt, rt);
                if (m) return m->eval(l, r);
            }
            // Unreachable in a program that has passed typecheck_cleg.
            throw std::runtime_error("cleg: operator '" + expr.op + "' has no overload for these operand types at runtime");
        }
        case ExprKind::UnaryExpr:
            return make_number(-eval_expr(*expr.operand, env, funcs).number);
        case ExprKind::NilExpr: {
            ClegValue v; v.kind = CTKind::Array; v.elem = expr.nil_type;
            return v;
        }
    }
    throw std::runtime_error("cleg: eval_expr: unexpected ExprKind");
}

static ClegValue call_user_function(const FunctionDecl& fn, const std::vector<ClegValue>& args, UserFuncTable& funcs) {
    ValueEnv env;
    for (size_t i = 0; i < fn.params.size(); i++) env.vars[fn.params[i].name] = args[i];
    try {
        eval_block(fn.body, &env, funcs);
    } catch (ReturnSignal& r) {
        return std::move(r.value);
    }
    throw std::runtime_error("cleg: function '" + fn.name + "' fell off its own end without a 'return'");
}

// Mirrors shared/cleg.ts's runClegProgram() - always re-typechecks even though every public entry
// point here (typecheck_cleg_as_board, build_board_from_cleg) already does too on its own path, same
// "cheap relative to actually evaluating" reasoning as the TS side's own doc comment.
static ClegValue run_cleg_program(const ClegProgram& program) {
    typecheck_cleg(program);
    UserFuncTable funcs;
    for (auto& fn : program.functions) funcs[fn.name] = &fn;

    ValueEnv env;
    std::optional<ClegValue> result;
    for (auto& stmt : program.stmts) {
        if (stmt.kind == StmtKind::ExprStmt) result = eval_expr(*stmt.expr, env, funcs);
        else eval_stmt(stmt, env, funcs);
    }
    return std::move(*result); // typecheck_cleg already required the last top-level statement to be an ExprStmt
}

// ── Public API ───────────────────────────────────────────────────────────────

std::shared_ptr<ClegProgram> parse_cleg(const std::string& source) {
    return std::make_shared<ClegProgram>(parse_cleg_impl(source));
}

BoardConfig build_board_from_cleg(const ClegProgram& program) {
    typecheck_cleg_as_board(program);
    ClegValue result = run_cleg_program(program);
    return std::move(*result.egr_v);
}
