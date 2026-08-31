#include "game/cleg_check.h"
#include "game/cleg_eval_internal.h"
#include <algorithm>
#include <optional>
#include <stdexcept>
#include <unordered_map>

// Mirrors shared/clegCheck.ts - the static type checker - see game/cleg_check.h's own top comment
// for the full context. This file only points out where the C++ port differs from the TS source;
// the TS comments are the canonical reference for WHAT each piece does, not repeated here.

// ── Type checking ──────────────────────────────────────────────────────────────

// Mirrors shared/clegCheck.ts's TypeEnv - `parent` is a raw pointer into an enclosing stack frame
// (safe here: every env is stack-allocated and never outlives the recursive check call that owns
// it), not a GC'd reference.
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

// Shared arg-count/arg-type check for calling a (params, return_type) signature by name - used by
// check_expr's own CallExpr case for both a top-level function and a local func-typed variable's
// value, so the two share one error-message format instead of duplicating it.
static void check_call_args(const std::string& callee, const std::vector<ClegType>& arg_types, const std::vector<ClegType>& params) {
    if (arg_types.size() != params.size())
        throw std::runtime_error(
            "cleg: '" + callee + "' expects " + std::to_string(params.size()) + " argument(s), got " +
            std::to_string(arg_types.size()));
    for (size_t i = 0; i < arg_types.size(); i++)
        if (!type_equals(arg_types[i], params[i]))
            throw std::runtime_error(
                "cleg: '" + callee + "' argument " + std::to_string(i + 1) + ": expected " +
                type_to_string(params[i]) + ", got " + type_to_string(arg_types[i]));
}

static void check_block(const Stmt& block, TypeEnv* parent, FuncTable& funcs, const ClegType& return_type, bool in_loop);
static void check_stmt(const Stmt& stmt, TypeEnv& env, FuncTable& funcs, const ClegType& return_type, bool in_loop);
static ClegType check_expr(const Expr& expr, TypeEnv& env, FuncTable& funcs);

// Mirrors shared/clegCheck.ts's checkPartialApplication() - type-checks a partial-application
// CallExpr's own `args` (at least one is a HoleExpr) against `ref_params`, the currently-callable
// parameter list of whatever `callee` refers to, one entry per `args` position. Shared by both
// partial-application sources (see CallExpr's own doc comment, shared/clegBase.ts): a bare top-level
// function name (`ref_params` is its own full signature, since nothing is bound yet) and an existing
// local variable holding a func value, itself possibly already a partial application (`ref_params`
// is that value's own, already-reduced, params - the positions still open). Returns the resulting
// closure's own new params - the subset of `ref_params` at exactly the hole positions, in order.
static std::vector<ClegType> check_partial_application(
    const std::string& callee, const std::vector<Expr>& args, const std::vector<ClegType>& ref_params,
    TypeEnv& env, FuncTable& funcs)
{
    if (args.size() != ref_params.size())
        throw std::runtime_error(
            "cleg: '" + callee + "' expects " + std::to_string(ref_params.size()) + " argument(s), got " +
            std::to_string(args.size()));
    std::vector<ClegType> hole_params;
    for (size_t i = 0; i < args.size(); i++) {
        if (args[i].kind == ExprKind::HoleExpr) { hole_params.push_back(ref_params[i]); continue; }
        ClegType t = check_expr(args[i], env, funcs);
        if (!type_equals(t, ref_params[i]))
            throw std::runtime_error(
                "cleg: '" + callee + "' argument " + std::to_string(i + 1) + ": expected " +
                type_to_string(ref_params[i]) + ", got " + type_to_string(t));
    }
    return hole_params;
}

static void check_block(const Stmt& block, TypeEnv* parent, FuncTable& funcs, const ClegType& return_type, bool in_loop) {
    TypeEnv env; env.parent = parent;
    for (auto& stmt : block.stmts) check_stmt(stmt, env, funcs, return_type, in_loop);
}

// Mirrors shared/clegCheck.ts's checkStmt() - `in_loop` is true while checking a ForStmt/WhileStmt's
// own `body` (or anything nested inside it), required by BreakStmt/ContinueStmt (see their own cases).
static void check_stmt(const Stmt& stmt, TypeEnv& env, FuncTable& funcs, const ClegType& return_type, bool in_loop) {
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
            // Walk one 'array' level per index (see shared/clegBase.ts's own AssignStmt doc comment)
            // - `target_type` ends up being `*var_type` itself when `indices` is empty (the plain
            // whole-value reassignment case), exactly like before this field existed.
            ClegType target_type = *var_type;
            for (auto& idx : stmt.indices) {
                if (target_type.kind != CTKind::Array)
                    throw std::runtime_error(
                        "cleg: too many indices assigning to '" + stmt.name + "' - " + type_to_string(*var_type) +
                        " is not nested that deep");
                ClegType idx_type = check_expr(idx, env, funcs);
                if (idx_type.kind != CTKind::Number)
                    throw std::runtime_error("cleg: array index must be a number, got " + type_to_string(idx_type));
                target_type = *target_type.elem;
            }
            ClegType value_type = check_expr(*stmt.expr, env, funcs);
            if (!type_equals(value_type, target_type)) {
                std::string target_name = stmt.name;
                for (size_t i = 0; i < stmt.indices.size(); i++) target_name += "[]";
                throw std::runtime_error(
                    "cleg: cannot assign a value of type " + type_to_string(value_type) + " to '" + target_name +
                    "' of type " + type_to_string(target_type));
            }
            return;
        }
        case StmtKind::IfStmt: {
            ClegType cond_type = check_expr(*stmt.cond, env, funcs);
            if (cond_type.kind != CTKind::Bool)
                throw std::runtime_error("cleg: if condition must be bool, got " + type_to_string(cond_type));
            check_block(*stmt.then_stmt, &env, funcs, return_type, in_loop);
            if (stmt.else_stmt) {
                if (stmt.else_stmt->kind == StmtKind::Block) check_block(*stmt.else_stmt, &env, funcs, return_type, in_loop);
                else check_stmt(*stmt.else_stmt, env, funcs, return_type, in_loop);
            }
            return;
        }
        case StmtKind::ForStmt: {
            // A fresh scope for init's own declared variable (if any), enclosing cond/update/body -
            // NOT the same scope as body's own (check_block below gives body its own further-nested
            // scope, same as every other BLOCK) - see ForStmt's own doc comment (shared/clegBase.ts).
            TypeEnv loop_env; loop_env.parent = &env;
            if (stmt.for_init) check_stmt(*stmt.for_init, loop_env, funcs, return_type, in_loop);
            if (stmt.cond) {
                ClegType cond_type = check_expr(*stmt.cond, loop_env, funcs);
                if (cond_type.kind != CTKind::Bool)
                    throw std::runtime_error("cleg: for-loop condition must be bool, got " + type_to_string(cond_type));
            }
            if (stmt.for_update) check_stmt(*stmt.for_update, loop_env, funcs, return_type, in_loop);
            check_block(*stmt.body, &loop_env, funcs, return_type, true);
            return;
        }
        case StmtKind::WhileStmt: {
            ClegType cond_type = check_expr(*stmt.cond, env, funcs);
            if (cond_type.kind != CTKind::Bool)
                throw std::runtime_error("cleg: while condition must be bool, got " + type_to_string(cond_type));
            check_block(*stmt.body, &env, funcs, return_type, true);
            return;
        }
        case StmtKind::BreakStmt:
            if (!in_loop) throw std::runtime_error("cleg: 'break' outside a loop");
            return;
        case StmtKind::ContinueStmt:
            if (!in_loop) throw std::runtime_error("cleg: 'continue' outside a loop");
            return;
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
            check_block(stmt, &env, funcs, return_type, in_loop);
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
            if (t) return *t;
            // Not a variable - maybe a bare reference to one of program's own top-level functions,
            // used as a function-pointer value (e.g. passing a comparator by name) rather than being
            // called directly. A builtin can't be referenced this way (see ClegType's own Func doc
            // comment on why), so it gets its own clearer error instead of falling through to the
            // generic "undeclared variable" below.
            auto fit = funcs.find(expr.string_value);
            if (fit != funcs.end())
                return ClegType{CTKind::Func, nullptr, fit->second.params, std::make_shared<ClegType>(fit->second.return_type)};
            if (builtin_functions().count(expr.string_value))
                throw std::runtime_error(
                    "cleg: builtin function '" + expr.string_value + "' cannot be used as a function pointer - only "
                    "a cleg-declared function can");
            throw std::runtime_error("cleg: undeclared variable '" + expr.string_value + "'");
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
            if (std::any_of(expr.elements.begin(), expr.elements.end(),
                             [](const Expr& a) { return a.kind == ExprKind::HoleExpr; })) {
                // Partial application - a bare top-level function name, or an existing local
                // variable holding a func value (a plain pointer, or itself already a partial
                // application - see check_partial_application's own doc comment) - never a builtin
                // (no single fixed signature to close over for the generic/overloaded ones).
                const ClegType* var_type = lookup_var_type(env, expr.string_value);
                if (var_type) {
                    if (var_type->kind != CTKind::Func)
                        throw std::runtime_error("cleg: '" + expr.string_value + "' is not callable (" + type_to_string(*var_type) + ")");
                    auto hole_params = check_partial_application(expr.string_value, expr.elements, var_type->params, env, funcs);
                    return ClegType{CTKind::Func, nullptr, std::move(hole_params), std::make_shared<ClegType>(*var_type->return_type)};
                }
                if (builtin_functions().count(expr.string_value))
                    throw std::runtime_error(
                        "cleg: partial application ('#') is only supported for a cleg-declared function or a "
                        "func-typed variable, not builtin '" + expr.string_value + "'");
                auto fit = funcs.find(expr.string_value);
                if (fit == funcs.end()) throw std::runtime_error("cleg: call to undeclared function '" + expr.string_value + "'");
                auto hole_params = check_partial_application(expr.string_value, expr.elements, fit->second.params, env, funcs);
                return ClegType{CTKind::Func, nullptr, std::move(hole_params), std::make_shared<ClegType>(fit->second.return_type)};
            }
            std::vector<ClegType> arg_types;
            for (auto& a : expr.elements) arg_types.push_back(check_expr(a, env, funcs));
            auto& builtins = builtin_functions();
            auto bit = builtins.find(expr.string_value);
            if (bit != builtins.end()) return bit->second.check_call(expr.string_value, arg_types);
            // A local variable (almost always a parameter) of func type shadows a same-named
            // top-level function here - the whole point of passing a comparator by name is to call
            // it through the parameter that received it (see shared/clegBase.ts's own top comment).
            const ClegType* var_type = lookup_var_type(env, expr.string_value);
            if (var_type) {
                if (var_type->kind != CTKind::Func)
                    throw std::runtime_error("cleg: '" + expr.string_value + "' is not callable (" + type_to_string(*var_type) + ")");
                check_call_args(expr.string_value, arg_types, var_type->params);
                return *var_type->return_type;
            }
            auto fit = funcs.find(expr.string_value);
            if (fit == funcs.end()) throw std::runtime_error("cleg: call to undeclared function '" + expr.string_value + "'");
            check_call_args(expr.string_value, arg_types, fit->second.params);
            return fit->second.return_type;
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
            if (expr.op == "-") {
                if (t.kind != CTKind::Number) throw std::runtime_error("cleg: unary '-' requires a number operand, got " + type_to_string(t));
                return ClegType{CTKind::Number, nullptr};
            }
            if (t.kind != CTKind::Bool) throw std::runtime_error("cleg: unary '!' requires a bool operand, got " + type_to_string(t));
            return ClegType{CTKind::Bool, nullptr};
        }
        case ExprKind::NilExpr:
            return ClegType{CTKind::Array, std::make_shared<ClegType>(expr.nil_type)};
        case ExprKind::IndexExpr: {
            ClegType arr_type = check_expr(*expr.left, env, funcs);
            if (arr_type.kind != CTKind::Array)
                throw std::runtime_error("cleg: '[]' requires an array, got " + type_to_string(arr_type));
            ClegType idx_type = check_expr(*expr.right, env, funcs);
            if (idx_type.kind != CTKind::Number)
                throw std::runtime_error("cleg: array index must be a number, got " + type_to_string(idx_type));
            return *arr_type.elem;
        }
    }
    throw std::runtime_error("cleg: check_expr: unexpected ExprKind");
}

// Mirrors shared/clegCheck.ts's typecheckCleg() - internal (unlike TS, no consumer here needs the
// checked result type directly; typecheck_cleg_as_board (public, below) just needs to know it's egr).
ClegType typecheck_cleg(const ClegProgram& program) {
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
        check_block(fn.body, &param_env, funcs, fn.return_type, false);
    }

    if (program.stmts.empty()) throw std::runtime_error("cleg: program has no top-level statement");
    if (program.stmts.back().kind != StmtKind::ExprStmt)
        throw std::runtime_error(
            "cleg: the last top-level statement must be an expression, got " + stmt_kind_word(program.stmts.back().kind));

    TypeEnv env;
    std::optional<ClegType> result_type;
    for (auto& stmt : program.stmts) {
        if (stmt.kind == StmtKind::ExprStmt) result_type = check_expr(*stmt.expr, env, funcs);
        else check_stmt(stmt, env, funcs, EGR_TYPE, false);
    }
    return *result_type;
}

void typecheck_cleg_as_board(const ClegProgram& program) {
    ClegType t = typecheck_cleg(program);
    if (t.kind != CTKind::Egr) throw std::runtime_error("cleg: a board description must produce an egr, got " + type_to_string(t));
}
