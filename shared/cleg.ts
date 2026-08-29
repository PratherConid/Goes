/**
 * CLEG - "Construction Language for Embedded Graphs": a small typed language for describing
 * boards, built on top of shared/boardConfig.ts's own board-construction functions. This is the
 * first, deliberately minimal version - modifiers and selectors are not represented at all yet
 * (every cleg program can only combine the "prescribed board" constructors below), and the
 * language itself has no loops, no comparison or logical operators, and no way to construct a
 * `bool` value at all beyond the `true`/`false` literals - see the design notes scattered through
 * this file (each marked "Simplification:") for what's deliberately left out for now and would need
 * revisiting to grow the language further. The five arithmetic operators (`+ - * / %`) are supported
 * (`number` operands only, `number` result), with `()` for grouping/precedence.
 *
 * A cleg program is a sequence of top-level function declarations. Every function must declare its
 * own return type and always returns a value via `return EXPR;` (there is no `void`). Exactly one
 * function, `main`, is the program's entry point - runCleg() below evaluates it, given a
 * caller-supplied argument list (there is no other way for a cleg program to receive external
 * input yet), and returns whatever it returns.
 *
 * Concrete syntax (deliberately C++-like, per this language's own design brief):
 *
 *   TYPE       := 'egr' | 'number' | 'string' | 'bool' | TYPE '[' ']'
 *   PROGRAM    := FUNCDECL*
 *   FUNCDECL   := TYPE IDENT '(' (PARAM (',' PARAM)*)? ')' BLOCK
 *   PARAM      := TYPE IDENT
 *   BLOCK      := '{' STMT* '}'
 *   STMT       := VARDECL | ASSIGNSTMT | IFSTMT | RETURNSTMT | EXPRSTMT | BLOCK
 *   VARDECL    := TYPE IDENT '=' EXPR ';'
 *   ASSIGNSTMT := IDENT '=' EXPR ';'
 *   IFSTMT     := 'if' '(' EXPR ')' BLOCK ('else' (IFSTMT | BLOCK))?
 *   RETURNSTMT := 'return' EXPR ';'
 *   EXPRSTMT   := EXPR ';'
 *   EXPR       := TERM (('+' | '-') TERM)*
 *   TERM       := UNARY (('*' | '/' | '%') UNARY)*
 *   UNARY      := '-' UNARY | ATOM
 *   ATOM       := NUMBER | STRING | 'true' | 'false' | ARRAYLIT | IDENT | CALL | '(' EXPR ')'
 *   ARRAYLIT   := '[' (EXPR (',' EXPR)*)? ']'
 *   CALL       := IDENT '(' (EXPR (',' EXPR)*)? ')'
 *
 * `//` line comments are supported. There is no comparison or logical operator of any kind (no
 * `==`, `&&`, ...) - besides the five arithmetic operators (all `number -> number -> number`,
 * standard precedence, left-associative, `()` overrides precedence), the only way to combine or
 * inspect values is by calling a function (either a pre-defined board-construction function, see
 * PREDEFINED_FUNCTIONS below, or another cleg function).
 * Simplification: since `if` needs a `bool` condition, and nothing yet produces one besides a bare
 * `true`/`false` literal, an `if` in a real program can only ever branch on a literal constant -
 * genuinely conditional programs need comparison/logic functions (or operators) added later.
 *
 * Example:
 *   egr main() {
 *       egr x = menger(3, 3, "0011");
 *       return x;
 *   }
 */

import {
    BoardArgType, numArg, csvArg, zolArg, parseBoardArgToken,
    type BoardArgEntry, type BoardConfig,
} from './types.js';
import { PrescribedBoard, PrescribedBoardMap, PrescribedBoardFns } from './boardConfig.js';

// ── Types ────────────────────────────────────────────────────────────────────

/** A cleg type: one of the four basic types, or an array of some other cleg type (nestable, e.g.
 * `number[][]`). This is a value (not just a compile-time-only construct) - ClegValue below carries
 * one of these at runtime too, so an array's own element type is always known even for `[]`-shaped
 * values passed across a function boundary. */
export type ClegType =
    | { kind: 'egr' }
    | { kind: 'number' }
    | { kind: 'string' }
    | { kind: 'bool' }
    | { kind: 'array'; elem: ClegType };

function typeEquals(a: ClegType, b: ClegType): boolean {
    if (a.kind !== b.kind) return false;
    return a.kind === 'array' ? typeEquals(a.elem, (b as { kind: 'array'; elem: ClegType }).elem) : true;
}

function typeToString(t: ClegType): string {
    return t.kind === 'array' ? `${typeToString(t.elem)}[]` : t.kind;
}

// ── Values ───────────────────────────────────────────────────────────────────

/** A runtime cleg value, tagged with its own ClegType (same 'kind' discriminant convention as
 * ClegType itself, so a value's type is always `clegValueType(v)` without needing a separate
 * lookup). `egr` wraps a real shared/boardConfig.ts BoardConfig - the actual board a program
 * builds. */
export type ClegValue =
    | { kind: 'egr'; value: BoardConfig }
    | { kind: 'number'; value: number }
    | { kind: 'string'; value: string }
    | { kind: 'bool'; value: boolean }
    | { kind: 'array'; elem: ClegType; value: ClegValue[] };

function clegValueType(v: ClegValue): ClegType {
    return v.kind === 'array' ? { kind: 'array', elem: v.elem } : { kind: v.kind };
}

// ── AST ──────────────────────────────────────────────────────────────────────

export interface Param { type: ClegType; name: string; }

export interface FunctionDecl {
    kind: 'FunctionDecl';
    returnType: ClegType;
    name: string;
    params: Param[];
    body: Block;
}

export type Stmt = VarDecl | AssignStmt | IfStmt | ReturnStmt | ExprStmt | Block;

/** Declares and initializes a new local; see AssignStmt below for mutating an already-declared one. */
export interface VarDecl { kind: 'VarDecl'; type: ClegType; name: string; init: Expr; }
/** Reassigns an already-declared local (`x = expr;`) - mutates the binding in whichever enclosing
 * scope originally declared `name` (see evaluation's setValue), it does not shadow it with a new
 * one in the current block. `name` must already be declared with `value`'s exact type - there is no
 * way to introduce a new binding via assignment, only VarDecl does that. */
export interface AssignStmt { kind: 'AssignStmt'; name: string; value: Expr; }
/** `else_` is null (no else clause), a Block (`else { ... }`), or another IfStmt (`else if (...)`). */
export interface IfStmt { kind: 'IfStmt'; cond: Expr; then: Block; else_: Block | IfStmt | null; }
/** Every function must return a value (there is no `void`), so unlike C++ this is never bare. */
export interface ReturnStmt { kind: 'ReturnStmt'; value: Expr; }
export interface ExprStmt { kind: 'ExprStmt'; expr: Expr; }
export interface Block { kind: 'Block'; stmts: Stmt[]; }

export type Expr =
    | NumberLit | StringLit | BoolLit | ArrayLit | Identifier | CallExpr | BinaryExpr | UnaryExpr;

export interface NumberLit { kind: 'NumberLit'; value: number; }
export interface StringLit { kind: 'StringLit'; value: string; }
export interface BoolLit { kind: 'BoolLit'; value: boolean; }
/** Simplification: an empty `[]` has no way to say what its element type is, so it's rejected by
 * typecheckCleg (see checkExpr's own ArrayLit case) rather than silently guessed at. */
export interface ArrayLit { kind: 'ArrayLit'; elements: Expr[]; }
export interface Identifier { kind: 'Identifier'; name: string; }
/** `callee` names either a pre-defined board-construction function (PREDEFINED_FUNCTIONS below) or
 * another function declared in the same program - one flat namespace, see typecheckCleg. */
export interface CallExpr { kind: 'CallExpr'; callee: string; args: Expr[]; }
/** One of the five arithmetic operators - `number -> number -> number` only (see checkExpr's own
 * BinaryExpr case). `(...)` grouping isn't its own AST node - parseAtom just returns the
 * parenthesized subexpression directly, so precedence is fully resolved by the time the AST exists. */
export interface BinaryExpr { kind: 'BinaryExpr'; op: '+' | '-' | '*' | '/' | '%'; left: Expr; right: Expr; }
/** Unary negation (e.g. `-x`, `-f()`) - also how a negative number literal is written now (`-3`
 * parses as UnaryExpr wrapping NumberLit(3); the lexer itself never produces a signed number). */
export interface UnaryExpr { kind: 'UnaryExpr'; op: '-'; operand: Expr; }

/** A whole cleg program: its own top-level function declarations, in the order written. Functions
 * may call each other regardless of declaration order (forward references are fine) and may
 * recurse (directly or mutually) - the only form of repetition this language has at all, since it
 * has no loops. */
export interface ClegProgram { kind: 'ClegProgram'; functions: FunctionDecl[]; }

// ── Lexer ────────────────────────────────────────────────────────────────────

type TokenKind = 'ident' | 'number' | 'string' | 'punct' | 'eof';
interface Token { kind: TokenKind; text: string; pos: number; }

const PUNCTUATION = '(){}[],;=+-*/%';

/** Splits `src` into tokens - identifiers (including keywords, disambiguated later by the parser,
 * same convention as shared/selector.ts's own tokenize()/parser split), unsigned integer/decimal
 * number literals (negative numbers are the parser's unary '-' applied to one of these, see
 * parseUnary - the lexer itself never produces a signed number token), double-quoted string
 * literals (`\\`, `\"`, `\n`, `\t` escapes only), and single-character punctuation (including the
 * five arithmetic operators). `//` starts a line comment. */
function tokenize(src: string): Token[] {
    const tokens: Token[] = [];
    const n = src.length;
    let i = 0;
    while (i < n) {
        const c = src[i];
        if (/\s/.test(c)) { i++; continue; }
        if (c === '/' && src[i + 1] === '/') { while (i < n && src[i] !== '\n') i++; continue; }
        if (/[A-Za-z_]/.test(c)) {
            let j = i + 1;
            while (j < n && /[A-Za-z0-9_]/.test(src[j])) j++;
            tokens.push({ kind: 'ident', text: src.slice(i, j), pos: i });
            i = j;
            continue;
        }
        if (/[0-9]/.test(c)) {
            let j = i + 1;
            while (j < n && /[0-9.]/.test(src[j])) j++;
            tokens.push({ kind: 'number', text: src.slice(i, j), pos: i });
            i = j;
            continue;
        }
        if (c === '"') {
            let j = i + 1;
            let out = '';
            while (j < n && src[j] !== '"') {
                if (src[j] === '\\' && j + 1 < n) {
                    const esc = src[j + 1];
                    out += esc === 'n' ? '\n' : esc === 't' ? '\t' : esc;
                    j += 2;
                } else {
                    out += src[j];
                    j++;
                }
            }
            if (j >= n) throw new Error(`cleg: unterminated string literal starting at position ${i}`);
            tokens.push({ kind: 'string', text: out, pos: i });
            i = j + 1;
            continue;
        }
        if (PUNCTUATION.includes(c)) { tokens.push({ kind: 'punct', text: c, pos: i }); i++; continue; }
        throw new Error(`cleg: unexpected character '${c}' at position ${i}`);
    }
    tokens.push({ kind: 'eof', text: '', pos: n });
    return tokens;
}

// ── Parser ───────────────────────────────────────────────────────────────────

class TokenCursor {
    private pos = 0;
    constructor(private tokens: Token[]) {}

    peek(): Token { return this.tokens[this.pos]; }
    peekAt(offset: number): Token { return this.tokens[this.pos + offset]; }
    next(): Token { return this.tokens[this.pos++]; }
    atEnd(): boolean { return this.peek().kind === 'eof'; }
    isPunct(p: string): boolean { const t = this.peek(); return t.kind === 'punct' && t.text === p; }
    isKeyword(k: string): boolean { const t = this.peek(); return t.kind === 'ident' && t.text === k; }

    expectPunct(p: string): void {
        const t = this.next();
        if (t.kind !== 'punct' || t.text !== p)
            throw new Error(`cleg: expected '${p}', got '${t.text || '<eof>'}' at position ${t.pos}`);
    }
    expectIdent(): string {
        const t = this.next();
        if (t.kind !== 'ident') throw new Error(`cleg: expected an identifier, got '${t.text || '<eof>'}' at position ${t.pos}`);
        return t.text;
    }
}

const TYPE_KEYWORDS = new Set(['egr', 'number', 'string', 'bool']);

function parseType(c: TokenCursor): ClegType {
    const base = c.expectIdent();
    if (!TYPE_KEYWORDS.has(base)) throw new Error(`cleg: expected a type (egr/number/string/bool), got '${base}'`);
    let type: ClegType = { kind: base as 'egr' | 'number' | 'string' | 'bool' };
    while (c.isPunct('[')) { c.next(); c.expectPunct(']'); type = { kind: 'array', elem: type }; }
    return type;
}

function isTypeStart(c: TokenCursor): boolean {
    const t = c.peek();
    return t.kind === 'ident' && TYPE_KEYWORDS.has(t.text);
}

function parseCommaSeparated<T>(c: TokenCursor, close: string, parseOne: () => T): T[] {
    const items: T[] = [];
    if (!c.isPunct(close)) {
        items.push(parseOne());
        while (c.isPunct(',')) { c.next(); items.push(parseOne()); }
    }
    c.expectPunct(close);
    return items;
}

function parseFunctionDecl(c: TokenCursor): FunctionDecl {
    const returnType = parseType(c);
    const name = c.expectIdent();
    c.expectPunct('(');
    const params = parseCommaSeparated(c, ')', () => {
        const type = parseType(c);
        const paramName = c.expectIdent();
        return { type, name: paramName };
    });
    const body = parseBlock(c);
    return { kind: 'FunctionDecl', returnType, name, params, body };
}

function parseBlock(c: TokenCursor): Block {
    c.expectPunct('{');
    const stmts: Stmt[] = [];
    while (!c.isPunct('}')) stmts.push(parseStmt(c));
    c.expectPunct('}');
    return { kind: 'Block', stmts };
}

function parseStmt(c: TokenCursor): Stmt {
    if (c.isPunct('{')) return parseBlock(c);
    if (c.isKeyword('if')) return parseIfStmt(c);
    if (c.isKeyword('return')) return parseReturnStmt(c);
    if (isTypeStart(c)) return parseVarDecl(c);
    // Only an identifier immediately followed by '=' is an assignment (as opposed to, say, a bare
    // call-expression statement) - look ahead one extra token to tell them apart, since parseExpr's
    // own Identifier case doesn't consume '='.
    if (c.peek().kind === 'ident' && c.peekAt(1).kind === 'punct' && c.peekAt(1).text === '=')
        return parseAssignStmt(c);
    const expr = parseExpr(c);
    c.expectPunct(';');
    return { kind: 'ExprStmt', expr };
}

function parseVarDecl(c: TokenCursor): VarDecl {
    const type = parseType(c);
    const name = c.expectIdent();
    c.expectPunct('=');
    const init = parseExpr(c);
    c.expectPunct(';');
    return { kind: 'VarDecl', type, name, init };
}

function parseAssignStmt(c: TokenCursor): AssignStmt {
    const name = c.expectIdent();
    c.expectPunct('=');
    const value = parseExpr(c);
    c.expectPunct(';');
    return { kind: 'AssignStmt', name, value };
}

function parseIfStmt(c: TokenCursor): IfStmt {
    c.next(); // 'if'
    c.expectPunct('(');
    const cond = parseExpr(c);
    c.expectPunct(')');
    const then = parseBlock(c);
    let else_: Block | IfStmt | null = null;
    if (c.isKeyword('else')) { c.next(); else_ = c.isKeyword('if') ? parseIfStmt(c) : parseBlock(c); }
    return { kind: 'IfStmt', cond, then, else_ };
}

function parseReturnStmt(c: TokenCursor): ReturnStmt {
    c.next(); // 'return'
    const value = parseExpr(c);
    c.expectPunct(';');
    return { kind: 'ReturnStmt', value };
}

const ADDITIVE_OPS = new Set(['+', '-']);
const MULTIPLICATIVE_OPS = new Set(['*', '/', '%']);

function isPunctIn(c: TokenCursor, ops: Set<string>): boolean {
    const t = c.peek();
    return t.kind === 'punct' && ops.has(t.text);
}

/** Expression entry point, lowest precedence (`+`/`-`), left-associative - every existing call site
 * (VarDecl init, AssignStmt value, if condition, return value, call arguments, array elements)
 * already calls parseExpr, so arithmetic works everywhere an expression was already accepted
 * without any caller changes. */
function parseExpr(c: TokenCursor): Expr {
    let left = parseMultiplicative(c);
    while (isPunctIn(c, ADDITIVE_OPS)) {
        const op = c.next().text as '+' | '-';
        left = { kind: 'BinaryExpr', op, left, right: parseMultiplicative(c) };
    }
    return left;
}

/** `*`/`/`/`%` - binds tighter than `+`/`-`, left-associative. */
function parseMultiplicative(c: TokenCursor): Expr {
    let left = parseUnary(c);
    while (isPunctIn(c, MULTIPLICATIVE_OPS)) {
        const op = c.next().text as '*' | '/' | '%';
        left = { kind: 'BinaryExpr', op, left, right: parseUnary(c) };
    }
    return left;
}

/** Unary `-` (`-x`, `-f()`, ...) - binds tighter than any binary operator, and right-recursive so
 * `--x` (double negation) parses too. */
function parseUnary(c: TokenCursor): Expr {
    if (c.isPunct('-')) { c.next(); return { kind: 'UnaryExpr', op: '-', operand: parseUnary(c) }; }
    return parseAtom(c);
}

function parseAtom(c: TokenCursor): Expr {
    const tok = c.peek();
    if (tok.kind === 'number') { c.next(); return { kind: 'NumberLit', value: Number(tok.text) }; }
    if (tok.kind === 'string') { c.next(); return { kind: 'StringLit', value: tok.text }; }
    if (c.isPunct('[')) {
        c.next();
        const elements = parseCommaSeparated(c, ']', () => parseExpr(c));
        return { kind: 'ArrayLit', elements };
    }
    if (c.isPunct('(')) {
        c.next();
        const inner = parseExpr(c);
        c.expectPunct(')');
        return inner;
    }
    if (tok.kind === 'ident') {
        if (tok.text === 'true') { c.next(); return { kind: 'BoolLit', value: true }; }
        if (tok.text === 'false') { c.next(); return { kind: 'BoolLit', value: false }; }
        const name = c.expectIdent();
        if (c.isPunct('(')) {
            c.next();
            const args = parseCommaSeparated(c, ')', () => parseExpr(c));
            return { kind: 'CallExpr', callee: name, args };
        }
        return { kind: 'Identifier', name };
    }
    throw new Error(`cleg: unexpected token '${tok.text || '<eof>'}' at position ${tok.pos}`);
}

/** Parses a whole cleg program (see this file's own top comment for the grammar) - throws if
 * `source` doesn't follow it, or if anything is left over after the last top-level function
 * declaration. */
export function parseCleg(source: string): ClegProgram {
    const c = new TokenCursor(tokenize(source));
    const functions: FunctionDecl[] = [];
    while (!c.atEnd()) functions.push(parseFunctionDecl(c));
    return { kind: 'ClegProgram', functions };
}

// ── Predefined (board-construction) functions ─────────────────────────────────

function argTypeToClegType(t: BoardArgType): ClegType {
    switch (t) {
        case BoardArgType.Number: return { kind: 'number' };
        case BoardArgType.CommaSeparatedNumbers: return { kind: 'array', elem: { kind: 'number' } };
        case BoardArgType.ZeroOneList: return { kind: 'string' };
    }
}

function valueToBoardArgEntry(argType: BoardArgType, val: ClegValue): BoardArgEntry {
    switch (argType) {
        case BoardArgType.Number:
            if (val.kind !== 'number') throw new Error(`cleg: expected a number argument, got ${typeToString(clegValueType(val))}`);
            return numArg(val.value);
        case BoardArgType.CommaSeparatedNumbers:
            if (val.kind !== 'array' || val.elem.kind !== 'number')
                throw new Error(`cleg: expected a number[] argument, got ${typeToString(clegValueType(val))}`);
            return csvArg(val.value.map(v => (v as { kind: 'number'; value: number }).value));
        case BoardArgType.ZeroOneList:
            if (val.kind !== 'string') throw new Error(`cleg: expected a string argument, got ${typeToString(clegValueType(val))}`);
            // Reuses shared/types.ts's own ZeroOneList validation (rejects anything but a string of
            // 0/1 characters) rather than re-implementing it here.
            return parseBoardArgToken(BoardArgType.ZeroOneList, val.value);
    }
}

export interface FunctionSignature { params: ClegType[]; returnType: ClegType; }
interface PredefinedFunction extends FunctionSignature { call: (args: ClegValue[]) => ClegValue; }

// One predefined function per shared/boardConfig.ts's own PrescribedBoardMap/PrescribedBoardFns
// entry, named after its own command-line token (PrescribedBoardMap[pb][1], e.g. "menger", "rect",
// "cublat") so a cleg program's board-construction calls read exactly like this project's own
// command syntax - modifiers/selectors aside (see this file's own top comment). Built generically
// from that existing table (rather than one hand-written cleg function per board type) so this
// list never drifts out of sync with it.
const PREDEFINED_FUNCTIONS: Record<string, PredefinedFunction> = {};
for (const [pbKey, [argTypes, cmdName]] of
    Object.entries(PrescribedBoardMap) as [string, [BoardArgType[], string, string, string]][]) {
    const pb = Number(pbKey) as PrescribedBoard;
    PREDEFINED_FUNCTIONS[cmdName] = {
        params: argTypes.map(argTypeToClegType),
        returnType: { kind: 'egr' },
        call: (args: ClegValue[]): ClegValue =>
            ({ kind: 'egr', value: PrescribedBoardFns[pb](...argTypes.map((t, i) => valueToBoardArgEntry(t, args[i]))) }),
    };
}

// ── Type checking ──────────────────────────────────────────────────────────────

interface TypeEnv { vars: Map<string, ClegType>; parent: TypeEnv | null; }
function lookupVarType(env: TypeEnv, name: string): ClegType | undefined {
    for (let e: TypeEnv | null = env; e; e = e.parent) { const t = e.vars.get(name); if (t) return t; }
    return undefined;
}

type FuncTable = Record<string, FunctionSignature>;

/**
 * Statically checks `program`: every function's own body is checked against its declared
 * parameter/return types, with one flat, program-wide function namespace shared between
 * PREDEFINED_FUNCTIONS and `program`'s own top-level declarations (a user function redeclaring a
 * predefined name is rejected, not shadowed). Requires a `main` function to exist. Throws
 * descriptively on the first error found; does not attempt to collect more than one.
 *
 * Simplification: does not check that every path through a function actually reaches a `return` -
 * a function whose body falls off the end without one is only caught at evaluation time (see
 * callUserFunction below), not here.
 */
export function typecheckCleg(program: ClegProgram): void {
    const funcs: FuncTable = { ...PREDEFINED_FUNCTIONS };
    for (const fn of program.functions) {
        if (funcs[fn.name])
            throw new Error(`cleg: function '${fn.name}' is declared more than once (or shadows a predefined function)`);
        funcs[fn.name] = { params: fn.params.map(p => p.type), returnType: fn.returnType };
    }
    if (!funcs['main']) throw new Error(`cleg: program has no 'main' function`);

    for (const fn of program.functions) {
        const env: TypeEnv = { vars: new Map(fn.params.map(p => [p.name, p.type])), parent: null };
        checkBlock(fn.body, env, funcs, fn.returnType);
    }
}

function checkBlock(block: Block, parent: TypeEnv, funcs: FuncTable, returnType: ClegType): void {
    const env: TypeEnv = { vars: new Map(), parent };
    for (const stmt of block.stmts) checkStmt(stmt, env, funcs, returnType);
}

function checkStmt(stmt: Stmt, env: TypeEnv, funcs: FuncTable, returnType: ClegType): void {
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
            const valueType = checkExpr(stmt.value, env, funcs);
            if (!typeEquals(valueType, varType))
                throw new Error(
                    `cleg: cannot assign a value of type ${typeToString(valueType)} to '${stmt.name}' ` +
                    `of type ${typeToString(varType)}`);
            return;
        }
        case 'IfStmt': {
            const condType = checkExpr(stmt.cond, env, funcs);
            if (condType.kind !== 'bool') throw new Error(`cleg: if condition must be bool, got ${typeToString(condType)}`);
            checkBlock(stmt.then, env, funcs, returnType);
            if (stmt.else_)
                stmt.else_.kind === 'Block'
                    ? checkBlock(stmt.else_, env, funcs, returnType)
                    : checkStmt(stmt.else_, env, funcs, returnType);
            return;
        }
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
            checkBlock(stmt, env, funcs, returnType);
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
            if (!t) throw new Error(`cleg: undeclared variable '${expr.name}'`);
            return t;
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
        case 'CallExpr': {
            const sig = funcs[expr.callee];
            if (!sig) throw new Error(`cleg: call to undeclared function '${expr.callee}'`);
            if (expr.args.length !== sig.params.length)
                throw new Error(`cleg: '${expr.callee}' expects ${sig.params.length} argument(s), got ${expr.args.length}`);
            expr.args.forEach((a, i) => {
                const t = checkExpr(a, env, funcs);
                if (!typeEquals(t, sig.params[i]))
                    throw new Error(
                        `cleg: '${expr.callee}' argument ${i + 1}: expected ${typeToString(sig.params[i])}, got ${typeToString(t)}`);
            });
            return sig.returnType;
        }
        case 'BinaryExpr': {
            const l = checkExpr(expr.left, env, funcs);
            const r = checkExpr(expr.right, env, funcs);
            if (l.kind !== 'number' || r.kind !== 'number')
                throw new Error(
                    `cleg: operator '${expr.op}' requires number operands, got ${typeToString(l)} and ${typeToString(r)}`);
            return { kind: 'number' };
        }
        case 'UnaryExpr': {
            const t = checkExpr(expr.operand, env, funcs);
            if (t.kind !== 'number') throw new Error(`cleg: unary '-' requires a number operand, got ${typeToString(t)}`);
            return { kind: 'number' };
        }
    }
}

// ── Evaluation ───────────────────────────────────────────────────────────────

interface ValueEnv { vars: Map<string, ClegValue>; parent: ValueEnv | null; }
function lookupValue(env: ValueEnv, name: string): ClegValue {
    for (let e: ValueEnv | null = env; e; e = e.parent) { const v = e.vars.get(name); if (v) return v; }
    // Unreachable in a program that has passed typecheckCleg - every Identifier there already
    // resolved to a declared variable.
    throw new Error(`cleg: undeclared variable '${name}'`);
}

/** Mutates `name`'s existing binding in place, in whichever env of the chain declared it - unlike
 * VarDecl's own `env.vars.set` (which always creates a fresh binding in the innermost scope), this
 * walks up to the declaring scope first so an assignment inside a nested block/if-branch is visible
 * to the enclosing scope that declared the variable, not just shadowed locally. */
function setValue(env: ValueEnv, name: string, value: ClegValue): void {
    for (let e: ValueEnv | null = env; e; e = e.parent) {
        if (e.vars.has(name)) { e.vars.set(name, value); return; }
    }
    // Unreachable in a program that has passed typecheckCleg - see lookupValue above.
    throw new Error(`cleg: undeclared variable '${name}'`);
}

type UserFuncTable = Record<string, FunctionDecl>;

// Thrown to unwind out of nested blocks/if-statements on `return` - always caught by
// callUserFunction below, never escapes runCleg itself.
class ReturnSignal { constructor(public value: ClegValue) {} }

function evalBlock(block: Block, parent: ValueEnv, funcs: UserFuncTable): void {
    const env: ValueEnv = { vars: new Map(), parent };
    for (const stmt of block.stmts) evalStmt(stmt, env, funcs);
}

function evalStmt(stmt: Stmt, env: ValueEnv, funcs: UserFuncTable): void {
    switch (stmt.kind) {
        case 'VarDecl':
            env.vars.set(stmt.name, evalExpr(stmt.init, env, funcs));
            return;
        case 'AssignStmt':
            setValue(env, stmt.name, evalExpr(stmt.value, env, funcs));
            return;
        case 'IfStmt': {
            const cond = evalExpr(stmt.cond, env, funcs) as { kind: 'bool'; value: boolean };
            if (cond.value) evalBlock(stmt.then, env, funcs);
            else if (stmt.else_) stmt.else_.kind === 'Block' ? evalBlock(stmt.else_, env, funcs) : evalStmt(stmt.else_, env, funcs);
            return;
        }
        case 'ReturnStmt':
            throw new ReturnSignal(evalExpr(stmt.value, env, funcs));
        case 'ExprStmt':
            evalExpr(stmt.expr, env, funcs);
            return;
        case 'Block':
            evalBlock(stmt, env, funcs);
            return;
    }
}

function evalExpr(expr: Expr, env: ValueEnv, funcs: UserFuncTable): ClegValue {
    switch (expr.kind) {
        case 'NumberLit': return { kind: 'number', value: expr.value };
        case 'StringLit': return { kind: 'string', value: expr.value };
        case 'BoolLit': return { kind: 'bool', value: expr.value };
        case 'Identifier': return lookupValue(env, expr.name);
        case 'ArrayLit': {
            const values = expr.elements.map(e => evalExpr(e, env, funcs));
            // typecheckCleg already rejected an empty or mixed-element-type literal, so the first
            // value's own type is always the array's element type.
            return { kind: 'array', elem: clegValueType(values[0]), value: values };
        }
        case 'CallExpr': {
            const args = expr.args.map(a => evalExpr(a, env, funcs));
            const predefined = PREDEFINED_FUNCTIONS[expr.callee];
            return predefined ? predefined.call(args) : callUserFunction(funcs[expr.callee], args, funcs);
        }
        case 'BinaryExpr': {
            const l = (evalExpr(expr.left, env, funcs) as { kind: 'number'; value: number }).value;
            const r = (evalExpr(expr.right, env, funcs) as { kind: 'number'; value: number }).value;
            const value = expr.op === '+' ? l + r
                : expr.op === '-' ? l - r
                : expr.op === '*' ? l * r
                : expr.op === '/' ? l / r
                : l % r;
            return { kind: 'number', value };
        }
        case 'UnaryExpr': {
            const v = (evalExpr(expr.operand, env, funcs) as { kind: 'number'; value: number }).value;
            return { kind: 'number', value: -v };
        }
    }
}

function callUserFunction(fn: FunctionDecl, args: ClegValue[], funcs: UserFuncTable): ClegValue {
    const env: ValueEnv = { vars: new Map(fn.params.map((p, i) => [p.name, args[i]])), parent: null };
    try {
        evalBlock(fn.body, env, funcs);
    } catch (e) {
        if (e instanceof ReturnSignal) return e.value;
        throw e;
    }
    throw new Error(`cleg: function '${fn.name}' fell off its own end without a 'return'`);
}

/**
 * Parses, type-checks, then runs `source`'s own `main` function with `args` as its parameters -
 * the only way a cleg program receives external input right now (see ClegProgram's own doc
 * comment). `args` is checked against `main`'s declared parameter types the same way an ordinary
 * call's arguments are. Returns whatever `main` returns.
 */
export function runCleg(source: string, args: ClegValue[] = []): ClegValue {
    const program = parseCleg(source);
    typecheckCleg(program);
    const funcs: UserFuncTable = {};
    for (const fn of program.functions) funcs[fn.name] = fn;
    const main = funcs['main']; // typecheckCleg already required this to exist

    if (args.length !== main.params.length)
        throw new Error(`cleg: main expects ${main.params.length} argument(s), got ${args.length}`);
    main.params.forEach((p, i) => {
        const t = clegValueType(args[i]);
        if (!typeEquals(t, p.type))
            throw new Error(`cleg: main argument ${i + 1} ('${p.name}'): expected ${typeToString(p.type)}, got ${typeToString(t)}`);
    });

    return callUserFunction(main, args, funcs);
}
