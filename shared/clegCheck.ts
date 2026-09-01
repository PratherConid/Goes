/**
 * cleg's static type checker - see shared/clegBase.ts's own top comment for the cleg language
 * itself (grammar, semantics, the four-file split this is one part of).
 *
 * Imports BUILTIN_FUNCTIONS from shared/clegEval.ts for its own `checkCall` half (see
 * BuiltinFunction's own doc comment there for why checkCall/call are bundled together) - this is a
 * real (non-type-only) circular import (clegEval.ts imports typecheckCleg back from this file to
 * re-validate before running a program), safe because every use on both sides is deferred to inside
 * a function body - see clegEval.ts's own top comment for the full explanation.
 */

import {
    type ClegType, typeEquals, typeToString, SET_ELEM_KINDS, BINARY_OPERATOR_OVERLOADS,
    type Block, type Stmt, type Expr, type ClegProgram,
} from './clegBase.js';
import { unparseAssignTarget } from './clegParser.js';
import { BUILTIN_FUNCTIONS, EGR_TYPE } from './clegEval.js';

// ── Type checking ──────────────────────────────────────────────────────────────

interface TypeEnv { vars: Map<string, ClegType>; parent: TypeEnv | null; }
function lookupVarType(env: TypeEnv, name: string): ClegType | undefined {
    for (let e: TypeEnv | null = env; e; e = e.parent) { const t = e.vars.get(name); if (t) return t; }
    return undefined;
}

export interface FunctionSignature { params: ClegType[]; returnType: ClegType; }
type FuncTable = Record<string, FunctionSignature>;

// Shared arg-count/arg-type check for calling a (params, returnType) signature by name - used by
// checkExpr's own CallExpr case for both a top-level function and a local func-typed variable's
// value, so the two share one error-message format instead of duplicating it.
function checkCallArgs(callee: string, argTypes: ClegType[], params: ClegType[]): void {
    if (argTypes.length !== params.length)
        throw new Error(`cleg: '${callee}' expects ${params.length} argument(s), got ${argTypes.length}`);
    argTypes.forEach((t, i) => {
        if (!typeEquals(t, params[i]))
            throw new Error(`cleg: '${callee}' argument ${i + 1}: expected ${typeToString(params[i])}, got ${typeToString(t)}`);
    });
}

/** Type-checks a partial-application CallExpr's own `args` (at least one is a HoleExpr) against
 * `refParams` - the currently-callable parameter list of whatever `callee` refers to, one entry per
 * `args` position. Shared by both partial-application sources (see CallExpr's own doc comment): a
 * bare top-level function name (`refParams` is its own full signature, since nothing is bound yet)
 * and an existing local variable holding a `func` value, itself possibly already a partial
 * application (`refParams` is that value's own, already-reduced, `params` - the positions still
 * open) - "some args are holes, producing a new/further func value" is exactly the same check
 * either way, only which parameter list it's checked against differs. Returns the resulting
 * closure's own new `params` - the subset of `refParams` at exactly the hole positions, in order. */
function checkPartialApplication(callee: string, args: Expr[], refParams: ClegType[], env: TypeEnv, funcs: FuncTable): ClegType[] {
    if (args.length !== refParams.length)
        throw new Error(`cleg: '${callee}' expects ${refParams.length} argument(s), got ${args.length}`);
    const holeParams: ClegType[] = [];
    args.forEach((a, i) => {
        if (a.kind === 'HoleExpr') { holeParams.push(refParams[i]); return; }
        const t = checkExpr(a, env, funcs);
        if (!typeEquals(t, refParams[i]))
            throw new Error(
                `cleg: '${callee}' argument ${i + 1}: expected ${typeToString(refParams[i])}, got ${typeToString(t)}`);
    });
    return holeParams;
}

/**
 * Statically checks `program`: every function's own body is checked against its declared
 * parameter/return types, with one flat, program-wide function namespace shared between
 * BUILTIN_FUNCTIONS and `program`'s own top-level declarations (a user function redeclaring a
 * builtin's name is rejected, not shadowed). Also checks every top-level TopStmt in one shared scope
 * (see ClegProgram's own doc comment - entirely separate from any function's own), and requires at
 * least one to exist, the last of which must be an ExprStmt (there is no other way for a cleg program
 * to produce a value at all). Throws descriptively on the first error found; does not attempt to
 * collect more than one. Returns the checked type of `program`'s own LAST top-level statement - since
 * there's no branching at top level, this is always exactly the type runClegProgram will actually
 * produce, computable without evaluating anything (see shared/clegEval.ts's own
 * typecheckClegAsBoard/buildBoardFromCleg, which use this to validate a program's result type before
 * ever running it).
 *
 * Simplification: does not check that every path through a function actually reaches a `return` -
 * a function whose body falls off the end without one is only caught at evaluation time (see
 * shared/clegEval.ts's own callUserFunction), not here.
 */
export function typecheckCleg(program: ClegProgram): ClegType {
    const funcs: FuncTable = {};
    for (const fn of program.functions) {
        if (funcs[fn.name] || BUILTIN_FUNCTIONS[fn.name])
            throw new Error(`cleg: function '${fn.name}' is declared more than once (or shadows a builtin function)`);
        funcs[fn.name] = { params: fn.params.map(p => p.type), returnType: fn.returnType };
    }

    for (const fn of program.functions) {
        const env: TypeEnv = { vars: new Map(fn.params.map(p => [p.name, p.type])), parent: null };
        checkBlock(fn.body, env, funcs, fn.returnType, false);
    }

    if (program.stmts.length === 0) throw new Error(`cleg: program has no top-level statement`);
    const lastStmt = program.stmts[program.stmts.length - 1];
    if (lastStmt.kind !== 'ExprStmt')
        throw new Error(`cleg: the last top-level statement must be an expression, got ${lastStmt.kind}`);

    // One env shared across every top-level TopStmt (never passed into a function call's own env -
    // see ClegProgram's own doc comment on why a function can't see these). checkStmt's own
    // returnType param is never actually read here - a TopStmt is never a ReturnStmt - EGR_TYPE is
    // just a harmless placeholder.
    const env: TypeEnv = { vars: new Map(), parent: null };
    let resultType: ClegType | null = null;
    for (const stmt of program.stmts) {
        if (stmt.kind === 'ExprStmt') resultType = checkExpr(stmt.expr, env, funcs);
        else checkStmt(stmt, env, funcs, EGR_TYPE, false);
    }
    return resultType!;
}

function checkBlock(block: Block, parent: TypeEnv, funcs: FuncTable, returnType: ClegType, inLoop: boolean): void {
    const env: TypeEnv = { vars: new Map(), parent };
    for (const stmt of block.stmts) checkStmt(stmt, env, funcs, returnType, inLoop);
}

// `inLoop` is true while checking a ForStmt/WhileStmt's own `body` (or anything nested inside it,
// e.g. an IfStmt's own then/else, or a further-nested loop's body) - BreakStmt/ContinueStmt require
// it, matching real C++'s own "break/continue outside a loop" rejection. A nested loop's own body
// re-passes `true` (already true, or newly so); everything else (IfStmt's then/else, a bare Block)
// just threads the caller's own value through unchanged, since neither adds nor removes loop-ness.
function checkStmt(stmt: Stmt, env: TypeEnv, funcs: FuncTable, returnType: ClegType, inLoop: boolean): void {
    switch (stmt.kind) {
        case 'VarDecl': {
            if (env.vars.has(stmt.name)) throw new Error(`cleg: '${stmt.name}' is already declared in this scope`);
            const initType = checkExpr(stmt.init, env, funcs);
            if (!typeEquals(initType, stmt.type))
                throw new Error(
                    `cleg: cannot initialize '${stmt.name}' of type ${typeToString(stmt.type)} ` +
                    `with a value of type ${typeToString(initType)}`);
            env.vars.set(stmt.name, stmt.type);
            return;
        }
        case 'AssignStmt': {
            const varType = lookupVarType(env, stmt.name);
            if (!varType) throw new Error(`cleg: assignment to undeclared variable '${stmt.name}'`);
            // Walk one 'array' level per index (see AssignStmt's own doc comment) - `targetType`
            // ends up being `varType` itself when `indices` is empty (the plain whole-value
            // reassignment case), exactly like before this field existed.
            let targetType = varType;
            for (const idx of stmt.indices) {
                if (targetType.kind !== 'array')
                    throw new Error(
                        `cleg: too many indices assigning to '${stmt.name}' - ${typeToString(varType)} is not ` +
                        `nested that deep`);
                const idxType = checkExpr(idx, env, funcs);
                if (idxType.kind !== 'number')
                    throw new Error(`cleg: array index must be a number, got ${typeToString(idxType)}`);
                targetType = targetType.elem;
            }
            const valueType = checkExpr(stmt.value, env, funcs);
            if (!typeEquals(valueType, targetType))
                throw new Error(
                    `cleg: cannot assign a value of type ${typeToString(valueType)} to '${unparseAssignTarget(stmt, 0)}' ` +
                    `of type ${typeToString(targetType)}`);
            return;
        }
        case 'IfStmt': {
            const condType = checkExpr(stmt.cond, env, funcs);
            if (condType.kind !== 'bool') throw new Error(`cleg: if condition must be bool, got ${typeToString(condType)}`);
            checkBlock(stmt.then, env, funcs, returnType, inLoop);
            if (stmt.else_)
                stmt.else_.kind === 'Block'
                    ? checkBlock(stmt.else_, env, funcs, returnType, inLoop)
                    : checkStmt(stmt.else_, env, funcs, returnType, inLoop);
            return;
        }
        case 'ForStmt': {
            // A fresh scope for init's own declared variable (if any), enclosing cond/update/body -
            // NOT the same scope as body's own (checkBlock below gives body its own further-nested
            // scope, same as every other BLOCK) - see ForStmt's own doc comment.
            const loopEnv: TypeEnv = { vars: new Map(), parent: env };
            if (stmt.init) checkStmt(stmt.init, loopEnv, funcs, returnType, inLoop);
            if (stmt.cond) {
                const condType = checkExpr(stmt.cond, loopEnv, funcs);
                if (condType.kind !== 'bool')
                    throw new Error(`cleg: for-loop condition must be bool, got ${typeToString(condType)}`);
            }
            if (stmt.update) checkStmt(stmt.update, loopEnv, funcs, returnType, inLoop);
            checkBlock(stmt.body, loopEnv, funcs, returnType, true);
            return;
        }
        case 'WhileStmt': {
            const condType = checkExpr(stmt.cond, env, funcs);
            if (condType.kind !== 'bool')
                throw new Error(`cleg: while condition must be bool, got ${typeToString(condType)}`);
            checkBlock(stmt.body, env, funcs, returnType, true);
            return;
        }
        case 'BreakStmt':
            if (!inLoop) throw new Error(`cleg: 'break' outside a loop`);
            return;
        case 'ContinueStmt':
            if (!inLoop) throw new Error(`cleg: 'continue' outside a loop`);
            return;
        case 'ReturnStmt': {
            const t = checkExpr(stmt.value, env, funcs);
            if (!typeEquals(t, returnType))
                throw new Error(`cleg: return type mismatch - function returns ${typeToString(returnType)}, got ${typeToString(t)}`);
            return;
        }
        case 'ExprStmt':
            checkExpr(stmt.expr, env, funcs);
            return;
        case 'Block':
            checkBlock(stmt, env, funcs, returnType, inLoop);
            return;
    }
}

function checkExpr(expr: Expr, env: TypeEnv, funcs: FuncTable): ClegType {
    switch (expr.kind) {
        case 'NumberLit': return { kind: 'number' };
        case 'StringLit': return { kind: 'string' };
        case 'BoolLit': return { kind: 'bool' };
        case 'Identifier': {
            const t = lookupVarType(env, expr.name);
            if (t) return t;
            // Not a variable - maybe a bare reference to one of program's own top-level functions,
            // used as a function-pointer value (e.g. passing a comparator by name) rather than being
            // called directly. A builtin can't be referenced this way (see ClegType's own 'func' doc
            // comment on why), so it gets its own clearer error instead of falling through to the
            // generic "undeclared variable" below.
            const fn = funcs[expr.name];
            if (fn) return { kind: 'func', params: fn.params, returnType: fn.returnType };
            if (BUILTIN_FUNCTIONS[expr.name])
                throw new Error(
                    `cleg: builtin function '${expr.name}' cannot be used as a function pointer - only a ` +
                    `cleg-declared function can`);
            throw new Error(`cleg: undeclared variable '${expr.name}'`);
        }
        case 'ArrayLit': {
            if (expr.elements.length === 0)
                throw new Error(`cleg: cannot infer the element type of an empty array literal '[]'`);
            const elemTypes = expr.elements.map(e => checkExpr(e, env, funcs));
            for (let i = 1; i < elemTypes.length; i++)
                if (!typeEquals(elemTypes[i], elemTypes[0]))
                    throw new Error(
                        `cleg: array literal mixes ${typeToString(elemTypes[0])} and ${typeToString(elemTypes[i])} elements`);
            return { kind: 'array', elem: elemTypes[0] };
        }
        case 'SetLit': {
            if (expr.elements.length === 0)
                throw new Error(`cleg: cannot infer the element type of an empty set literal '{}'`);
            const elemTypes = expr.elements.map(e => checkExpr(e, env, funcs));
            for (let i = 1; i < elemTypes.length; i++)
                if (!typeEquals(elemTypes[i], elemTypes[0]))
                    throw new Error(
                        `cleg: set literal mixes ${typeToString(elemTypes[0])} and ${typeToString(elemTypes[i])} elements`);
            if (!SET_ELEM_KINDS.has(elemTypes[0].kind))
                throw new Error(
                    `cleg: '${typeToString(elemTypes[0])}' is not a supported set element type - only ` +
                    `number/string/bool/edge/simp/quad may be set elements`);
            return { kind: 'set', elem: elemTypes[0] };
        }
        case 'CallExpr': {
            if (expr.args.some(a => a.kind === 'HoleExpr')) {
                // Partial application - a bare top-level function name, or an existing local
                // variable holding a func value (a plain pointer, or itself already a partial
                // application - see checkPartialApplication's own doc comment) - never a builtin (no
                // single fixed signature to close over for the generic/overloaded ones).
                const varType = lookupVarType(env, expr.callee);
                if (varType) {
                    if (varType.kind !== 'func')
                        throw new Error(`cleg: '${expr.callee}' is not callable (${typeToString(varType)})`);
                    const holeParams = checkPartialApplication(expr.callee, expr.args, varType.params, env, funcs);
                    return { kind: 'func', params: holeParams, returnType: varType.returnType };
                }
                if (BUILTIN_FUNCTIONS[expr.callee])
                    throw new Error(
                        `cleg: partial application ('#') is only supported for a cleg-declared function or a ` +
                        `func-typed variable, not builtin '${expr.callee}'`);
                const sig = funcs[expr.callee];
                if (!sig) throw new Error(`cleg: call to undeclared function '${expr.callee}'`);
                const holeParams = checkPartialApplication(expr.callee, expr.args, sig.params, env, funcs);
                return { kind: 'func', params: holeParams, returnType: sig.returnType };
            }
            const argTypes = expr.args.map(a => checkExpr(a, env, funcs));
            const builtin = BUILTIN_FUNCTIONS[expr.callee];
            if (builtin) return builtin.checkCall(expr.callee, argTypes);
            // A local variable (almost always a parameter) of func type shadows a same-named
            // top-level function here - the whole point of passing a comparator by name is to call
            // it through the parameter that received it (see this file's own top comment).
            const varType = lookupVarType(env, expr.callee);
            if (varType) {
                if (varType.kind !== 'func')
                    throw new Error(`cleg: '${expr.callee}' is not callable (${typeToString(varType)})`);
                checkCallArgs(expr.callee, argTypes, varType.params);
                return varType.returnType;
            }
            const sig = funcs[expr.callee];
            if (!sig) throw new Error(`cleg: call to undeclared function '${expr.callee}'`);
            checkCallArgs(expr.callee, argTypes, sig.params);
            return sig.returnType;
        }
        case 'BinaryExpr': {
            const l = checkExpr(expr.left, env, funcs);
            const r = checkExpr(expr.right, env, funcs);
            for (const overload of BINARY_OPERATOR_OVERLOADS[expr.op]) {
                const m = overload.match(l, r);
                if (m) return m.type;
            }
            throw new Error(
                `cleg: operator '${expr.op}' has no overload for operand types ${typeToString(l)} and ${typeToString(r)} ` +
                `(expected one of: ${BINARY_OPERATOR_OVERLOADS[expr.op].map(o => o.signature).join('; ')})`);
        }
        case 'UnaryExpr': {
            const t = checkExpr(expr.operand, env, funcs);
            if (expr.op === '-') {
                if (t.kind !== 'number') throw new Error(`cleg: unary '-' requires a number operand, got ${typeToString(t)}`);
                return { kind: 'number' };
            }
            if (t.kind !== 'bool') throw new Error(`cleg: unary '!' requires a bool operand, got ${typeToString(t)}`);
            return { kind: 'bool' };
        }
        case 'NilExpr': return { kind: 'array', elem: expr.type };
        case 'IndexExpr': {
            const arrType = checkExpr(expr.array, env, funcs);
            if (arrType.kind !== 'array')
                throw new Error(`cleg: '[]' requires an array, got ${typeToString(arrType)}`);
            const idxType = checkExpr(expr.index, env, funcs);
            if (idxType.kind !== 'number')
                throw new Error(`cleg: array index must be a number, got ${typeToString(idxType)}`);
            return arrType.elem;
        }
        // Unreachable - CallExpr's own case above always filters HoleExpr args out before recursing
        // into checkExpr for the rest; the parser never produces one anywhere else.
        case 'HoleExpr':
            throw new Error(`cleg: '#' is only valid as an argument in a partial-application call`);
    }
}
