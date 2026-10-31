#pragma once
#include "game/board_config.h"
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// Mirrors shared/clegBase.ts. C++ differences that hold across all of game/cleg_*:
//   - Only cleg's own files include this header; game/cleg_parser.h/game/cleg_eval.h (what
//     train.cpp/server.cpp include) forward-declare the AST types they need instead.
//   - No garbage collector: std::shared_ptr where substructures are genuinely shared (Expr/Stmt
//     children, ClegValue's `egr_v`), plain std::vector/value members elsewhere (Array/Set contents,
//     Selector/BoardModifier/ProdSelector payloads).
//   - TS's `number` is a C++ `double`; an int is only produced at the board-primitive boundary
//     (mkEdge/mkTri/mkQuad, board-arg conversion, psBase's index), via an explicit conversion.
//   - Every ClegType/ClegValue/AST field has a default member initializer, so a default-constructed
//     instance is always well-defined.

// ── Types ────────────────────────────────────────────────────────────────────

enum class CTKind {
    Egr, Number, String, Bool, Edge, Simp, Quad, Sel, Mod, Formsel, Lrs, Psel, Array, Set, Func
};

// Mirrors shared/clegBase.ts's ClegType as one flat struct - `elem` meaningful iff kind ==
// Array/Set, `params`/`return_type` iff kind == Func.
struct ClegType {
    CTKind kind = CTKind::Number;
    std::shared_ptr<ClegType> elem;
    std::vector<ClegType> params;
    std::shared_ptr<ClegType> return_type;
};

bool type_equals(const ClegType& a, const ClegType& b);
std::string type_to_string(const ClegType& t);

// Mirrors shared/clegBase.ts's ProdSelector - defined ahead of ClegValue since ClegValue holds one
// by value.
enum class PSelOp { All, Base, BaseNE, Union, Inter, Diff };
struct ProdSelector {
    PSelOp op = PSelOp::All;
    int number = 0;                       // Base/BaseNE
    std::shared_ptr<Selector> sel;        // Base
    std::shared_ptr<Selector> node_sel;   // BaseNE
    std::shared_ptr<Selector> edge_sel;   // BaseNE
    std::vector<ProdSelector> items;      // Union/Inter
    std::shared_ptr<ProdSelector> a, b;   // Diff
};

// ── Values ───────────────────────────────────────────────────────────────────

// Mirrors shared/clegBase.ts's ClegValue - one flat tagged struct ("meaningful iff kind == X", like
// board_config.h's BoardModifier/Selector) instead of a discriminated union. Only `egr_v` is held
// via std::shared_ptr (a whole BoardConfig); every other payload is held by value.
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
    ProdSelector psel_v;
    std::vector<ClegValue> arr_v;                    // Array or Set
    // Func only - the 'func' variant's params/returnType/name/boundArgs (std::nullopt where TS uses
    // null for a '#' slot).
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
inline ClegValue make_psel(ProdSelector p) { ClegValue v; v.kind = CTKind::Psel; v.psel_v = std::move(p); return v; }

// Stands in for JS's Number->string (error messages, number/string concatenation) - an integral
// value prints with no decimal point, matching JS; otherwise 15 significant digits.
std::string format_number_display(double d);

// Mirrors shared/clegBase.ts's clegSetKey().
std::string cleg_set_key(const ClegValue& v);

// Mirrors shared/clegBase.ts's makeClegSet().
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

// Mirrors shared/clegBase.ts's BINARY_OPERATOR_OVERLOADS - keyed by the operator's token text (no
// BinOp enum).
const std::unordered_map<std::string, std::vector<BinaryOverload>>& binary_operator_overloads();

// ── AST ──────────────────────────────────────────────────────────────────────

struct Param { ClegType type; std::string name; };

enum class ExprKind { NumberLit, StringLit, BoolLit, ArrayLit, SetLit, Identifier, CallExpr, BinaryExpr, UnaryExpr, NilExpr, IndexExpr, HoleExpr };

// Mirrors shared/clegBase.ts's Expr union - one flat tagged struct, same convention as ClegValue
// above. `string_value` doubles as StringLit's text, Identifier's name, and CallExpr's callee;
// `elements` as ArrayLit/SetLit's elements and CallExpr's args; `left`/`right` as BinaryExpr's
// operands and IndexExpr's array/index.
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
// as VarDecl's init, AssignStmt's value, ReturnStmt's value, and ExprStmt's expr; WhileStmt reuses
// ForStmt's `cond`/`body`.
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

// Mirrors shared/clegBase.ts's ClegProgram - `stmts` holds only TopStmt kinds
// (VarDecl/AssignStmt/ExprStmt), as plain Stmts.
struct ClegProgram {
    std::vector<FunctionDecl> functions;
    std::vector<Stmt> stmts;
};
