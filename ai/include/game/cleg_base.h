#pragma once
#include "game/board_config.h"
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// Mirrors shared/clegBase.ts - the data-structure declarations shared by game/cleg_parser.*,
// game/cleg_check.*, and game/cleg_eval.* (see shared/clegBase.ts's own top comment for the cleg
// language itself - grammar, semantics, the four-file split this is one part of; this header only
// points out where the C++ port differs from the TS source, same convention as every other cleg_*
// file). Structural differences that hold throughout cleg's own four C++ files, not mentioned
// per-function below:
//   - `ClegProgram`/`ClegType`/`ClegValue`/the whole AST are declared here (rather than kept private
//     to a single .cpp, as this port did before this split) purely so game/cleg_parser.cpp,
//     game/cleg_check.cpp, and game/cleg_eval.cpp can all share them - exactly like TS's own
//     clegBase.ts, nothing OUTSIDE cleg's own four files ever includes this header or inspects them
//     (game/cleg_parser.h/game/cleg_check.h/game/cleg_eval.h forward-declare `ClegProgram` only, for
//     their own public entry points - see each header's own top comment).
//   - No garbage collector: recursive AST/value substructures use std::shared_ptr for genuine
//     sharing (Expr/Stmt children, ClegValue's `egr` payload) or plain std::vector/value members
//     where TS's own object-reference semantics never actually needed sharing (Array/Set contents,
//     Selector/BoardModifier/MultiSelector payloads - each already cheap to copy, since
//     Selector/MultiSelector use shared_ptr internally for their own recursive children).
//   - TS's `number` (a JS double) is a C++ `double` throughout; an int is only ever produced at the
//     C++/board-primitive boundary (mkEdge/mkTri/mkQuad, board-arg conversion, msBase's index), via
//     an explicit int conversion - a data-type constraint the TS side doesn't have.
//   - Every ClegType/ClegValue/AST struct gives EVERY field a default member initializer, even ones
//     TS's own type only ever reads after unconditionally setting it - a defensive C++-only
//     precaution making every default-constructed instance well-defined on its own, not a behavioral
//     difference.
//   - unparseCleg has no port - nothing in ai/ ever needs to show a cleg program back as text (no
//     UI), unlike the TS side's Configure Board popup.

// ── Types ────────────────────────────────────────────────────────────────────

enum class CTKind {
    Egr, Number, String, Bool, Edge, Simp, Quad, Sel, Mod, Formsel, Lrs, Msel, Array, Set, Func
};

// Mirrors shared/clegBase.ts's ClegType - `elem` meaningful iff kind == Array/Set; `params`/
// `return_type` meaningful iff kind == Func (a monomorphic function-pointer type, e.g.
// `(number, number) -> bool`).
struct ClegType {
    CTKind kind = CTKind::Number;
    std::shared_ptr<ClegType> elem;
    std::vector<ClegType> params;
    std::shared_ptr<ClegType> return_type;
};

bool type_equals(const ClegType& a, const ClegType& b);
std::string type_to_string(const ClegType& t);

// Mirrors shared/clegBase.ts's MultiSelector - defined ahead of ClegValue since ClegValue holds one
// by value.
enum class MSelOp { All, Base, Union, Inter, Diff };
struct MultiSelector {
    MSelOp op = MSelOp::All;
    int number = 0;                       // Base
    std::shared_ptr<Selector> sel;        // Base
    std::vector<MultiSelector> items;     // Union/Inter
    std::shared_ptr<MultiSelector> a, b;  // Diff
};

// ── Values ───────────────────────────────────────────────────────────────────

// Mirrors shared/clegBase.ts's ClegValue - one flat tagged struct (same "meaningful iff kind == X"
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
    BoardSimplex simp_v{};
    BoardQuad quad_v{};
    std::shared_ptr<BoardConfig> egr_v;
    SelectorType sel_type = SelectorType{SelectorKind::Node}; // Sel only
    Selector sel_v;
    BoardModifier mod_v{ModifierKind::Rectify};
    FormSelector form_sel_v; // Formsel only
    LocalReplaceSelector lrs_v; // Lrs only
    MultiSelector msel_v;
    std::vector<ClegValue> arr_v;                    // Array or Set (Set: deduplicated by cleg_set_key)
    // Func only - mirrors shared/clegBase.ts's ClegValue own 'func' variant: a function-pointer value
    // is a reference to one of `program`'s own top-level functions, held by `func_name` (looked up in
    // `funcs`/UserFuncTable again at call time - see game/cleg_eval.cpp's eval_expr own CallExpr
    // case) rather than a direct FunctionDecl pointer, matching the TS side's own reasoning for doing
    // the same (keeps this type free of any dependency on the AST shape below). `func_bound_args` has
    // one slot per entry of the ORIGINAL function's own full parameter list - std::nullopt at every
    // still-uninstantiated ('#') position, the actual (already-evaluated) argument everywhere else -
    // so a plain, uncalled reference (built from a bare Identifier) is simply the all-nullopt case,
    // and a partial application (`f(a, #, b)`, see CallExpr's own doc comment) is the general one;
    // calling either kind of value later interleaves the caller's own supplied arguments into the
    // nullopt slots, in order (see game/cleg_eval.cpp's fill_holes). `func_params`/`func_return_type`
    // describe this VALUE's own callable signature, not the original function's - for a plain
    // reference the two coincide, but a partial application's `func_params` is only the '#'
    // positions' types, in order. Cached here (rather than re-derived on every use) purely so
    // cleg_value_type can report this value's own ClegType without needing a funcs-table lookup.
    std::vector<ClegType> func_params;
    ClegType func_return_type;
    std::string func_name;
    std::vector<std::optional<ClegValue>> func_bound_args;
};

ClegType cleg_value_type(const ClegValue& v);

inline ClegValue make_number(double n) { ClegValue v; v.kind = CTKind::Number; v.number = n; return v; }
inline ClegValue make_string(std::string s) { ClegValue v; v.kind = CTKind::String; v.str = std::move(s); return v; }
inline ClegValue make_bool(bool b) { ClegValue v; v.kind = CTKind::Bool; v.boolean = b; return v; }
inline ClegValue make_edge_v(BoardEdge e) { ClegValue v; v.kind = CTKind::Edge; v.edge_v = e; return v; }
inline ClegValue make_simp_v(BoardSimplex t) { ClegValue v; v.kind = CTKind::Simp; v.simp_v = std::move(t); return v; }
inline ClegValue make_quad_v(BoardQuad q) { ClegValue v; v.kind = CTKind::Quad; v.quad_v = q; return v; }
inline ClegValue make_egr(BoardConfig bc) {
    ClegValue v; v.kind = CTKind::Egr; v.egr_v = std::make_shared<BoardConfig>(std::move(bc)); return v;
}
inline ClegValue make_mod(BoardModifier m) { ClegValue v; v.kind = CTKind::Mod; v.mod_v = std::move(m); return v; }
inline ClegValue make_form_sel_v(FormSelector s) { ClegValue v; v.kind = CTKind::Formsel; v.form_sel_v = std::move(s); return v; }
inline ClegValue make_lrs_v(LocalReplaceSelector s) { ClegValue v; v.kind = CTKind::Lrs; v.lrs_v = std::move(s); return v; }
inline ClegValue make_msel(MultiSelector m) { ClegValue v; v.kind = CTKind::Msel; v.msel_v = std::move(m); return v; }

// Renders a double for error messages/number-string concatenation - an integral value prints with
// no decimal point/trailing zeros (matching JS's own Number->string for the plain integers cleg
// programs actually interpolate, e.g. biTemple's `(d + 1)`); otherwise 15 significant digits.
std::string format_number_display(double d);

// Mirrors shared/clegBase.ts's clegSetKey() - a canonical string key for a set element (edge/tri/quad
// have no reference equality of their own, same reasoning as that file's own doc comment).
std::string cleg_set_key(const ClegValue& v);

// Mirrors shared/clegBase.ts's makeClegSet() - collapses duplicates by cleg_set_key, first occurrence
// wins.
ClegValue make_cleg_set(ClegType elem, std::vector<ClegValue> values);

// ── Binary operators ─────────────────────────────────────────────────────────

// Mirrors shared/clegBase.ts's BinaryOverload - `match` returns nullopt (TS: null) instead of matching.
struct MatchResult {
    ClegType type;
    std::function<ClegValue(const ClegValue&, const ClegValue&)> eval;
};
struct BinaryOverload {
    std::string signature;
    std::function<std::optional<MatchResult>(const ClegType&, const ClegType&)> match;
};

// Mirrors shared/clegBase.ts's BINARY_OPERATOR_OVERLOADS - keyed by the operator's own punctuation
// text (matching the parser's own token, rather than introducing a separate BinOp enum).
const std::unordered_map<std::string, std::vector<BinaryOverload>>& binary_operator_overloads();

// ── AST ──────────────────────────────────────────────────────────────────────

struct Param { ClegType type; std::string name; };

enum class ExprKind { NumberLit, StringLit, BoolLit, ArrayLit, SetLit, Identifier, CallExpr, BinaryExpr, UnaryExpr, NilExpr, IndexExpr, HoleExpr };

// Mirrors shared/clegBase.ts's Expr union - one flat tagged struct, same convention as ClegValue
// above. `string_value` doubles as StringLit's text, Identifier's name, and CallExpr's callee;
// `elements` doubles as ArrayLit/SetLit's own elements and CallExpr's own args; `left`/`right` double
// as BinaryExpr's own operands and IndexExpr's own array/index respectively (each kind uses only one
// of these roles, never two at once).
struct Expr {
    ExprKind kind = ExprKind::NumberLit;
    double number_value = 0;
    std::string string_value;
    bool bool_value = false;
    std::vector<Expr> elements;
    std::string op;                              // BinaryExpr ('+'/etc.) / UnaryExpr ('-' or '!')
    std::shared_ptr<Expr> left, right, operand;   // BinaryExpr/IndexExpr (left/right) / UnaryExpr (operand)
    ClegType nil_type;                            // NilExpr only
};

enum class StmtKind { VarDecl, AssignStmt, IfStmt, ForStmt, WhileStmt, BreakStmt, ContinueStmt, ReturnStmt, ExprStmt, Block };

// Mirrors shared/clegBase.ts's Stmt union - same flat-struct convention as Expr above. `expr` doubles
// as VarDecl's init, AssignStmt's value, ReturnStmt's value, and ExprStmt's own expr. WhileStmt
// reuses ForStmt's own `cond`/`body` fields (a WhileStmt has no init/update). AssignStmt's own
// `indices` (empty for a plain `x = expr`) - see shared/clegBase.ts's AssignStmt doc comment - lets
// it mutate one element of an already-declared array in place (`arr[i] = expr`, `arr[i][j] = expr`,
// ...).
struct Stmt {
    StmtKind kind = StmtKind::Block;
    ClegType decl_type;                     // VarDecl only
    std::string name;                       // VarDecl/AssignStmt only
    std::vector<Expr> indices;               // AssignStmt only
    std::shared_ptr<Expr> expr;
    std::shared_ptr<Expr> cond;              // IfStmt/ForStmt/WhileStmt (ForStmt: null = omitted)
    std::shared_ptr<Stmt> then_stmt;         // IfStmt (a Block)
    std::shared_ptr<Stmt> else_stmt;         // IfStmt (null | Block | IfStmt)
    std::shared_ptr<Stmt> for_init;          // ForStmt (VarDecl/AssignStmt/ExprStmt-shaped, null = omitted)
    std::shared_ptr<Stmt> for_update;        // ForStmt (AssignStmt/ExprStmt-shaped, null = omitted)
    std::shared_ptr<Stmt> body;              // ForStmt/WhileStmt (a Block)
    std::vector<Stmt> stmts;                 // Block only
};

std::string stmt_kind_word(StmtKind k);

struct FunctionDecl {
    ClegType return_type;
    std::string name;
    std::vector<Param> params;
    Stmt body; // kind == Block
};

// A whole cleg program - functions is order-independent (see shared/clegBase.ts's own doc comment),
// stmts (a TopStmt subset: VarDecl/AssignStmt/ExprStmt) is order-dependent. game/cleg_parser.h,
// game/cleg_check.h, and game/cleg_eval.h each forward-declare this same struct (pimpl-style) for
// their own public entry points, rather than including this header - see each one's own top comment.
struct ClegProgram {
    std::vector<FunctionDecl> functions;
    std::vector<Stmt> stmts;
};
