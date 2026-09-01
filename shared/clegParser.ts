/**
 * cleg's lexer, recursive-descent parser (source text -> AST), and unparser (AST -> source text,
 * the inverse) - see shared/clegBase.ts's own top comment for the cleg language itself (grammar,
 * semantics, the four-file split this is one part of).
 */

import {
    type ClegType, SET_ELEM_KINDS, typeToString,
    type FunctionDecl, type Stmt, type VarDecl, type AssignStmt, type IfStmt,
    type ForStmt, type WhileStmt, type BreakStmt, type ContinueStmt, type ReturnStmt, type ExprStmt,
    type Block, type Expr, type TopStmt, type ClegProgram,
} from './clegBase.js';

// ── Lexer ────────────────────────────────────────────────────────────────────

type TokenKind = 'ident' | 'number' | 'string' | 'punct' | 'eof';
interface Token { kind: TokenKind; text: string; pos: number; }

const PUNCTUATION = '(){}[],;+-*/%!#';

/** Splits `src` into tokens - identifiers (including keywords, disambiguated later by the parser,
 * same convention as shared/selector.ts's own tokenize()/parser split), unsigned integer/decimal
 * number literals (negative numbers are the parser's unary '-' applied to one of these, see
 * parseUnary - the lexer itself never produces a signed number token), double-quoted string
 * literals (`\\`, `\"`, `\n`, `\t` escapes only), single-character punctuation (including the five
 * arithmetic operators, unary '!', and '#' - the partial-application placeholder, see CallExpr's
 * own doc comment), '='/'<'/'>'/'!' - each its own one-character token unless
 * immediately followed by another '=' (making '==', '<=', '>=', or '!=' instead - '!' alone is
 * still unary/logical negation), and '&&'/'||' - each requires its own doubled character (a lone
 * '&' or '|' is a lexer error - there is no bitwise operator of any kind). All needing a second
 * character of lookahead are handled separately from `PUNCTUATION` above. `//` starts a line
 * comment. */
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
        if (c === '=' || c === '<' || c === '>' || c === '!') {
            if (src[i + 1] === '=') { tokens.push({ kind: 'punct', text: c + '=', pos: i }); i += 2; continue; }
            tokens.push({ kind: 'punct', text: c, pos: i });
            i++;
            continue;
        }
        if (c === '&' && src[i + 1] === '&') { tokens.push({ kind: 'punct', text: '&&', pos: i }); i += 2; continue; }
        if (c === '|' && src[i + 1] === '|') { tokens.push({ kind: 'punct', text: '||', pos: i }); i += 2; continue; }
        // '->' (FUNCTYPE's own arrow, e.g. `(number, number) -> bool`) - checked before the generic
        // PUNCTUATION fallback below, same as '&&'/'||' above, since '-' alone is already in
        // PUNCTUATION (arithmetic/unary minus).
        if (c === '-' && src[i + 1] === '>') { tokens.push({ kind: 'punct', text: '->', pos: i }); i += 2; continue; }
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

    /** Saves the cursor's current position, to be handed back to restore() later - lets a caller
     * (isFunctionDeclStart below) speculatively run a real parse function purely to see what follows,
     * then roll back as if it had never looked, regardless of whether that parse succeeded. */
    save(): number { return this.pos; }
    restore(pos: number): void { this.pos = pos; }

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

const TYPE_KEYWORDS = new Set([
    'egr', 'number', 'string', 'bool', 'edge', 'simp', 'tri', 'quad', 'sel', 'mod', 'msel',
]);

function parseType(c: TokenCursor): ClegType {
    let type: ClegType;
    if (c.isPunct('(')) {
        type = parseParenType(c);
    } else {
        const base = c.expectIdent();
        if (!TYPE_KEYWORDS.has(base))
            throw new Error(`cleg: expected a type (egr/number/string/bool/edge/simp/quad/sel/mod), got '${base}'`);
        // 'tri' is just an older spelling of 'simp' (both erased, no arity - see
        // shared/clegBase.ts's own top comment on ClegType's 'simp' variant), so both map to the
        // same kind here.
        type = {
            kind: (base === 'tri' ? 'simp' : base) as
                'egr' | 'number' | 'string' | 'bool' | 'edge' | 'simp' | 'quad' | 'sel' | 'mod' | 'msel',
        };
        if (c.isPunct('{')) {
            if (!SET_ELEM_KINDS.has(type.kind))
                throw new Error(
                    `cleg: '${base}{}' is not a supported set type - sets of egr, sets of sets, and sets of ` +
                    `arrays are not supported`);
            c.next();
            c.expectPunct('}');
            type = { kind: 'set', elem: type };
        }
    }
    while (c.isPunct('[')) { c.next(); c.expectPunct(']'); type = { kind: 'array', elem: type }; }
    return type;
}

// A leading '(' starts either FUNCTYPE's own param list (immediately followed by '->' once the list
// closes, e.g. `(number, number) -> bool`) or a parenthesized GROUPING of a single already-complete
// func type - needed so a trailing `[]` can bind to a func type as a WHOLE rather than (per FUNCTYPE's own
// recursive-return-type parsing, see below) to its return type: `((number) -> number)[]` is an array
// of comparator-shaped functions, vs. `(number) -> number[]` (no outer grouping), a single function
// returning `number[]`. The two are told apart only after the closing ')' of the parenthesized list:
// '->' immediately after means FUNCTYPE (consume it and parse the return type - recursing through
// parseType this way, rather than a separate parseFuncType, is also what lets a func type itself
// take/return another func type, higher-order, still fully concrete/non-generic, for free); anything
// else means the list must have held exactly one, itself func-typed, item, unwrapped as plain
// grouping. Deliberately NOT extended to grouping any other single type (`(number)` alone is
// rejected, not accepted as a redundant-parens `number`) even though that would be unambiguous in
// isolation - isTypeStart's own speculative parseType call (see below) needs "this fully parses as a
// TYPE" to never accidentally also describe a valid EXPR, and a bare grouped BASETYPE like `(number)`
// can collide with `(number) + 1` where `number` names an ordinary variable that happens to share a
// type keyword's spelling; a grouped FUNCTYPE can't collide this way, since no EXPR production can
// ever contain FUNCTYPE's own mandatory '->' token.
function parseParenType(c: TokenCursor): ClegType {
    c.expectPunct('(');
    const items = parseCommaSeparated<ClegType>(c, ')', () => parseType(c));
    if (c.isPunct('->')) {
        c.next();
        const returnType = parseType(c);
        return { kind: 'func', params: items, returnType };
    }
    if (items.length === 1 && items[0].kind === 'func') return items[0];
    throw new Error(`cleg: expected '->' after a parenthesized parameter list`);
}

// True at a TYPE's own first token - either a BASETYPE keyword, or '(' starting a FUNCTYPE (see
// parseType/parseFuncType above). A bare '(' can also start a grouped EXPR at some of isTypeStart's
// own call sites (parseForInit's/parseCleg's own bare-EXPR fallback) - since a real EXPR can never
// contain FUNCTYPE's own mandatory '->' token, the two can only be told apart by actually trying to
// parse a TYPE. Rather than re-deriving parseType's own grammar here, speculatively run parseType for
// real via TokenCursor's own save/restore, always rolling back afterward (success or failure) so the
// cursor ends up exactly where it started either way.
function isTypeStart(c: TokenCursor): boolean {
    const t = c.peek();
    if (t.kind === 'ident' && TYPE_KEYWORDS.has(t.text)) return true;
    if (!c.isPunct('(')) return false;
    const pos = c.save();
    try {
        parseType(c);
        return true;
    } catch {
        return false;
    } finally {
        c.restore(pos);
    }
}

// A FUNCDECL and a top-level VARDECL both start with a TYPE (isTypeStart above, true for either) -
// the tokens right after the TYPE tell them apart: IDENT '(' for a function declaration ('tri
// makeTri() {...}', 'number[] mk() {...}', '(number)->bool makeCmp() {...}'), IDENT '=' for a
// variable declaration ('tri x = ...;'). Only meaningful at top level - parseStmt (inside a function
// body) never sees a FUNCDECL, so isTypeStart alone already means VARDECL there. Speculatively runs
// the real parseType directly (same save/restore technique as isTypeStart above, rather than
// re-deriving its own grammar) since a TYPE isn't always one token; a malformed type here just means
// "not a function decl start" - whichever of parseVarDecl/parseFunctionDecl actually runs next will
// surface the same error for real.
function isFunctionDeclStart(c: TokenCursor): boolean {
    const pos = c.save();
    try {
        parseType(c);
        return c.peek().kind === 'ident' && c.peekAt(1).kind === 'punct' && c.peekAt(1).text === '(';
    } catch {
        return false;
    } finally {
        c.restore(pos);
    }
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

// An identifier, optionally followed by one or more '[' EXPR ']' index brackets, immediately
// followed by '=' is an assignment (as opposed to, say, a bare call-expression statement or a bare
// indexing expression like `arr[i];`) - shared by parseStmt and the for-loop's own
// parseForInit/parseForUpdate. The zero-index case (`x =`) is checked by a cheap fixed 2-token
// lookahead; since the number of indices isn't bounded, telling `arr[i] = ...` apart from a bare
// `arr[i];`/`arr[i] + 1;` expression needs the same speculative-parse-then-restore technique as
// isTypeStart/isFunctionDeclStart above (consuming the index brackets for real via TokenCursor's own
// save/restore, rather than re-deriving their own grammar here) - a malformed index expression here
// just means "not an assignment start"; whichever of parseAssignStmt/the EXPRSTMT fallback actually
// runs next will surface the same error for real.
function isAssignStart(c: TokenCursor): boolean {
    if (c.peek().kind !== 'ident') return false;
    if (c.peekAt(1).kind === 'punct' && c.peekAt(1).text === '=') return true;
    if (!(c.peekAt(1).kind === 'punct' && c.peekAt(1).text === '[')) return false;
    const pos = c.save();
    try {
        c.next();
        while (c.isPunct('[')) { c.next(); parseExpr(c); c.expectPunct(']'); }
        return c.isPunct('=');
    } catch {
        return false;
    } finally {
        c.restore(pos);
    }
}

function parseStmt(c: TokenCursor): Stmt {
    if (c.isPunct('{')) return parseBlock(c);
    if (c.isKeyword('if')) return parseIfStmt(c);
    if (c.isKeyword('for')) return parseForStmt(c);
    if (c.isKeyword('while')) return parseWhileStmt(c);
    if (c.isKeyword('break')) return parseBreakStmt(c);
    if (c.isKeyword('continue')) return parseContinueStmt(c);
    if (c.isKeyword('return')) return parseReturnStmt(c);
    if (isTypeStart(c)) return parseVarDecl(c);
    if (isAssignStart(c)) return parseAssignStmt(c);
    const expr = parseExpr(c);
    c.expectPunct(';');
    return { kind: 'ExprStmt', expr };
}

// Doesn't consume a trailing ';' - shared by parseVarDecl (which does) and parseForInit (which
// instead leaves it for parseForStmt's own explicit c.expectPunct(';') between FORINIT and cond).
function parseVarDeclNoSemi(c: TokenCursor): VarDecl {
    const type = parseType(c);
    const name = c.expectIdent();
    c.expectPunct('=');
    const init = parseExpr(c);
    return { kind: 'VarDecl', type, name, init };
}

function parseVarDecl(c: TokenCursor): VarDecl {
    const decl = parseVarDeclNoSemi(c);
    c.expectPunct(';');
    return decl;
}

// Doesn't consume a trailing ';' - shared by parseAssignStmt (which does) and parseForInit/
// parseForUpdate (which instead leave it for parseForStmt's own explicit delimiters).
function parseAssignStmtNoSemi(c: TokenCursor): AssignStmt {
    const name = c.expectIdent();
    const indices: Expr[] = [];
    while (c.isPunct('[')) {
        c.next();
        indices.push(parseExpr(c));
        c.expectPunct(']');
    }
    c.expectPunct('=');
    const value = parseExpr(c);
    return { kind: 'AssignStmt', name, indices, value };
}

function parseAssignStmt(c: TokenCursor): AssignStmt {
    const stmt = parseAssignStmtNoSemi(c);
    c.expectPunct(';');
    return stmt;
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

// FORINIT (the segment before the first ';' inside 'for (...)') - VarDecl/AssignStmt/ExprStmt-
// shaped but consumes neither its own trailing ';' (parseForStmt's own explicit one delimits it)
// nor, for the ExprStmt case, wraps a bare Expr any differently than parseStmt's own fallback does.
function parseForInit(c: TokenCursor): VarDecl | AssignStmt | ExprStmt | null {
    if (c.isPunct(';')) return null;
    if (isTypeStart(c)) return parseVarDeclNoSemi(c);
    if (isAssignStart(c)) return parseAssignStmtNoSemi(c);
    return { kind: 'ExprStmt', expr: parseExpr(c) };
}

// FORUPDATE (the segment before the closing ')') - like FORINIT but never a VarDecl (declaring a
// fresh variable in a for-loop's update clause isn't meaningful - real C++ rejects it too).
function parseForUpdate(c: TokenCursor): AssignStmt | ExprStmt | null {
    if (c.isPunct(')')) return null;
    if (isAssignStart(c)) return parseAssignStmtNoSemi(c);
    return { kind: 'ExprStmt', expr: parseExpr(c) };
}

function parseForStmt(c: TokenCursor): ForStmt {
    c.next(); // 'for'
    c.expectPunct('(');
    const init = parseForInit(c);
    c.expectPunct(';');
    const cond = c.isPunct(';') ? null : parseExpr(c);
    c.expectPunct(';');
    const update = parseForUpdate(c);
    c.expectPunct(')');
    const body = parseBlock(c);
    return { kind: 'ForStmt', init, cond, update, body };
}

function parseWhileStmt(c: TokenCursor): WhileStmt {
    c.next(); // 'while'
    c.expectPunct('(');
    const cond = parseExpr(c);
    c.expectPunct(')');
    const body = parseBlock(c);
    return { kind: 'WhileStmt', cond, body };
}

function parseBreakStmt(c: TokenCursor): BreakStmt {
    c.next(); // 'break'
    c.expectPunct(';');
    return { kind: 'BreakStmt' };
}

function parseContinueStmt(c: TokenCursor): ContinueStmt {
    c.next(); // 'continue'
    c.expectPunct(';');
    return { kind: 'ContinueStmt' };
}

function parseReturnStmt(c: TokenCursor): ReturnStmt {
    c.next(); // 'return'
    const value = parseExpr(c);
    c.expectPunct(';');
    return { kind: 'ReturnStmt', value };
}

const LOGICAL_OR_OPS = new Set(['||']);
const LOGICAL_AND_OPS = new Set(['&&']);
const EQUALITY_OPS = new Set(['==', '!=']);
const RELATIONAL_OPS = new Set(['<', '>', '<=', '>=']);
const ADDITIVE_OPS = new Set(['+', '-']);
const MULTIPLICATIVE_OPS = new Set(['*', '/', '%']);

function isPunctIn(c: TokenCursor, ops: Set<string>): boolean {
    const t = c.peek();
    return t.kind === 'punct' && ops.has(t.text);
}

/** Expression entry point, lowest precedence (`||`), left-associative - every existing call site
 * (VarDecl init, AssignStmt value, if/for condition, return value, call arguments, array/set
 * elements) already calls parseExpr, so every new precedence level works everywhere an expression
 * was already accepted without any caller changes. */
function parseExpr(c: TokenCursor): Expr {
    let left = parseLogicalAnd(c);
    while (isPunctIn(c, LOGICAL_OR_OPS)) {
        const op = c.next().text as '||';
        left = { kind: 'BinaryExpr', op, left, right: parseLogicalAnd(c) };
    }
    return left;
}

/** `&&` - binds tighter than `||`, looser than `==`, left-associative (matches real C++'s own
 * precedence between the two). */
function parseLogicalAnd(c: TokenCursor): Expr {
    let left = parseEquality(c);
    while (isPunctIn(c, LOGICAL_AND_OPS)) {
        const op = c.next().text as '&&';
        left = { kind: 'BinaryExpr', op, left, right: parseEquality(c) };
    }
    return left;
}

/** `== !=` - binds tighter than `&&`, looser than `< > <= >=`, left-associative. */
function parseEquality(c: TokenCursor): Expr {
    let left = parseRelational(c);
    while (isPunctIn(c, EQUALITY_OPS)) {
        const op = c.next().text as '==' | '!=';
        left = { kind: 'BinaryExpr', op, left, right: parseRelational(c) };
    }
    return left;
}

/** `< > <= >=` - binds tighter than `==`, looser than `+`/`-`, left-associative (matches real
 * C++'s own precedence between the two, e.g. `a + b < c == d` parses as `(a + b < c) == d`). */
function parseRelational(c: TokenCursor): Expr {
    let left = parseAdditive(c);
    while (isPunctIn(c, RELATIONAL_OPS)) {
        const op = c.next().text as '<' | '>' | '<=' | '>=';
        left = { kind: 'BinaryExpr', op, left, right: parseAdditive(c) };
    }
    return left;
}

/** `+`/`-` - binds tighter than comparison, looser than `*`/`/`/`%`, left-associative. */
function parseAdditive(c: TokenCursor): Expr {
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

/** Unary `-` (`-x`, `-f()`, ...) or unary `!` (`!x`, `!f()`, ...) - both bind tighter than any
 * binary operator, and both are right-recursive so `--x`/`!!x` (double negation) parse too. */
function parseUnary(c: TokenCursor): Expr {
    if (c.isPunct('-')) { c.next(); return { kind: 'UnaryExpr', op: '-', operand: parseUnary(c) }; }
    if (c.isPunct('!')) { c.next(); return { kind: 'UnaryExpr', op: '!', operand: parseUnary(c) }; }
    return parsePostfix(c);
}

/** Postfix `[]` indexing (`arr[i]`, `arr[i][j]`, `f()[0]`, ...) - binds tighter than unary `-`/`!`,
 * left-associative (each `[...]` wraps the result of everything to its left so far). */
function parsePostfix(c: TokenCursor): Expr {
    let e = parseAtom(c);
    while (c.isPunct('[')) {
        c.next();
        const index = parseExpr(c);
        c.expectPunct(']');
        e = { kind: 'IndexExpr', array: e, index };
    }
    return e;
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
    if (c.isPunct('{')) {
        // Unambiguous with a Block's own '{' - that only ever appears where parseStmt/parseBlock
        // are called (function/if bodies), never where an expression like this one is expected.
        c.next();
        const elements = parseCommaSeparated(c, '}', () => parseExpr(c));
        return { kind: 'SetLit', elements };
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
        // 'nil' is a reserved atom-starting keyword, like 'true'/'false' above, not an ordinary
        // identifier - its own '(' introduces a TYPE (parseType), never an argument list of
        // expressions like a real CallExpr's own '(' does (see NilExpr's own doc comment).
        if (tok.text === 'nil') {
            c.next();
            c.expectPunct('(');
            const type = parseType(c);
            c.expectPunct(')');
            return { kind: 'NilExpr', type };
        }
        const name = c.expectIdent();
        if (c.isPunct('(')) {
            c.next();
            // Each argument is either a real EXPR or a bare '#' (HoleExpr) - only ever meaningful
            // when `name` is a plain top-level function (partial application, see CallExpr's own doc
            // comment), but that isn't known yet at parse time, so '#' is always syntactically
            // accepted here and rejected later by checkExpr if `name` doesn't qualify.
            const args = parseCommaSeparated(c, ')', (): Expr => {
                if (c.isPunct('#')) { c.next(); return { kind: 'HoleExpr' }; }
                return parseExpr(c);
            });
            return { kind: 'CallExpr', callee: name, args };
        }
        return { kind: 'Identifier', name };
    }
    throw new Error(`cleg: unexpected token '${tok.text || '<eof>'}' at position ${tok.pos}`);
}

/**
 * Parses a whole cleg program (see shared/clegBase.ts's own top comment for the grammar) - throws if
 * `source` doesn't follow it, or if anything is left over after the last top-level item. Each
 * top-level item is a FUNCDECL if it starts a function declaration (isFunctionDeclStart), a VARDECL
 * if it merely starts a type (isTypeStart) without being one, an ASSIGNSTMT if it starts an
 * assignment (isAssignStart), or an EXPRSTMT otherwise - the same per-kind dispatch parseStmt already
 * uses inside a function body, minus IFSTMT/FORSTMT/RETURNSTMT (there is no branching, looping, or
 * `return` at top level) and plus FUNCDECL (which parseStmt never has to consider, since a function
 * body can't itself contain a nested function declaration).
 */
export function parseCleg(source: string): ClegProgram {
    const c = new TokenCursor(tokenize(source));
    const functions: FunctionDecl[] = [];
    const stmts: TopStmt[] = [];
    while (!c.atEnd()) {
        if (isFunctionDeclStart(c)) { functions.push(parseFunctionDecl(c)); continue; }
        if (isTypeStart(c)) { stmts.push(parseVarDecl(c)); continue; }
        if (isAssignStart(c)) { stmts.push(parseAssignStmt(c)); continue; }
        const expr = parseExpr(c);
        c.expectPunct(';');
        stmts.push({ kind: 'ExprStmt', expr });
    }
    return { kind: 'ClegProgram', functions, stmts };
}

// ── Unparsing ────────────────────────────────────────────────────────────────

/**
 * Regenerates cleg source text from `program` - the inverse of parseCleg, used so a value like
 * GameConfig.boardDescr (a ClegProgram, not source text - see that field's own doc comment) can
 * still be shown back to a user for editing (e.g. re-opening the Configure Board popup). Not a
 * pretty-printer in the sense of preserving the exact original formatting/comments (those aren't
 * part of the AST at all) - only that `parseCleg(unparseCleg(p))` produces a program equivalent to
 * `p` (same result type, same behavior). Every binary/unary sub-expression is always parenthesized
 * (`unparseExpr`'s own BinaryExpr/UnaryExpr cases) rather than re-deriving the parser's own
 * precedence rules to decide when parens are actually needed - always correct, just not minimal.
 * An ArrayLit/SetLit/CallExpr's own bracketed list (see unparseBracketed) is laid out one element
 * per line, indented, whenever its flat rendering wouldn't fit within LINE_WIDTH_BUDGET columns of
 * its own indent - a rough, not column-exact, readability heuristic (an inner list's own
 * flat-vs-wrapped choice is re-evaluated at its actual indent once its enclosing list has already
 * decided to wrap).
 */
export function unparseCleg(program: ClegProgram): string {
    const parts: string[] = [
        ...program.functions.map(fn => unparseFunctionDecl(fn, 0)),
        ...program.stmts.map(s => unparseStmt(s, 0)),
    ];
    return parts.join('\n');
}

function unparseFunctionDecl(fn: FunctionDecl, indent: number): string {
    const params = fn.params.map(p => `${typeToString(p.type)} ${p.name}`).join(', ');
    return `${typeToString(fn.returnType)} ${fn.name}(${params}) ${unparseBlock(fn.body, indent)}`;
}

function unparseBlock(block: Block, indent: number): string {
    if (block.stmts.length === 0) return '{}';
    const inner = indent + 2;
    const pad = ' '.repeat(inner);
    const lines = block.stmts.map(s => `${pad}${unparseStmt(s, inner)}`);
    return `{\n${lines.join('\n')}\n${' '.repeat(indent)}}`;
}

// `name` followed by zero or more `[EXPR]` indices - shared by unparseStmt's and unparseForClause's
// own AssignStmt cases. Exported for shared/clegCheck.ts's own use (a nicer error message than the
// bare variable name for an indexed-assignment type mismatch - see checkStmt's own AssignStmt case).
export function unparseAssignTarget(stmt: AssignStmt, indent: number): string {
    return `${stmt.name}${stmt.indices.map(idx => `[${unparseExpr(idx, indent)}]`).join('')}`;
}

function unparseStmt(stmt: Stmt, indent: number): string {
    switch (stmt.kind) {
        case 'VarDecl': return `${typeToString(stmt.type)} ${stmt.name} = ${unparseExpr(stmt.init, indent)};`;
        case 'AssignStmt': return `${unparseAssignTarget(stmt, indent)} = ${unparseExpr(stmt.value, indent)};`;
        case 'IfStmt': {
            const elsePart = stmt.else_ === null ? ''
                : stmt.else_.kind === 'Block' ? ` else ${unparseBlock(stmt.else_, indent)}`
                    : ` else ${unparseStmt(stmt.else_, indent)}`;
            return `if (${unparseExpr(stmt.cond, indent)}) ${unparseBlock(stmt.then, indent)}${elsePart}`;
        }
        case 'ForStmt': {
            const init = stmt.init === null ? '' : unparseForClause(stmt.init, indent);
            const cond = stmt.cond === null ? '' : unparseExpr(stmt.cond, indent);
            const update = stmt.update === null ? '' : unparseForClause(stmt.update, indent);
            return `for (${init}; ${cond}; ${update}) ${unparseBlock(stmt.body, indent)}`;
        }
        case 'WhileStmt': return `while (${unparseExpr(stmt.cond, indent)}) ${unparseBlock(stmt.body, indent)}`;
        case 'BreakStmt': return 'break;';
        case 'ContinueStmt': return 'continue;';
        case 'ReturnStmt': return `return ${unparseExpr(stmt.value, indent)};`;
        case 'ExprStmt': return `${unparseExpr(stmt.expr, indent)};`;
        case 'Block': return unparseBlock(stmt, indent);
    }
}

// FORINIT/FORUPDATE - same as unparseStmt's own VarDecl/AssignStmt/ExprStmt cases, minus the
// trailing ';' (parseForStmt's own explicit ';'/')' delimit these, see ForStmt's own doc comment).
function unparseForClause(stmt: VarDecl | AssignStmt | ExprStmt, indent: number): string {
    switch (stmt.kind) {
        case 'VarDecl': return `${typeToString(stmt.type)} ${stmt.name} = ${unparseExpr(stmt.init, indent)}`;
        case 'AssignStmt': return `${unparseAssignTarget(stmt, indent)} = ${unparseExpr(stmt.value, indent)}`;
        case 'ExprStmt': return unparseExpr(stmt.expr, indent);
    }
}

function unparseExpr(expr: Expr, indent: number): string {
    switch (expr.kind) {
        case 'NumberLit': return String(expr.value);
        case 'StringLit': return `"${escapeClegString(expr.value)}"`;
        case 'BoolLit': return expr.value ? 'true' : 'false';
        case 'ArrayLit': return unparseBracketed('[', ']', expr.elements, indent, 0);
        case 'SetLit': return unparseBracketed('{', '}', expr.elements, indent, 0);
        case 'Identifier': return expr.name;
        case 'CallExpr': return `${expr.callee}${unparseBracketed('(', ')', expr.args, indent, expr.callee.length)}`;
        case 'BinaryExpr': return `(${unparseExpr(expr.left, indent)} ${expr.op} ${unparseExpr(expr.right, indent)})`;
        case 'UnaryExpr': return `(${expr.op}${unparseExpr(expr.operand, indent)})`;
        case 'NilExpr': return `nil(${typeToString(expr.type)})`;
        case 'IndexExpr': return `${unparseExpr(expr.array, indent)}[${unparseExpr(expr.index, indent)}]`;
        case 'HoleExpr': return '#';
    }
}

// How much room a bracketed list's own flat rendering gets beyond its indent - i.e. the line-width
// limit is `indent + LINE_WIDTH_BUDGET`, not one fixed column for every nesting depth (equivalent
// to comparing just extraWidth + flat.length against this constant, indent cancelling out of both
// sides - see unparseBracketed below).
const LINE_WIDTH_BUDGET = 64;

// A bracketed, comma-separated element list - an ArrayLit/SetLit's own elements, or a CallExpr's
// own args - rendered flat on one line if that fits within `indent + LINE_WIDTH_BUDGET` columns
// (`extraWidth` accounts for a CallExpr's own callee name, which sits before its own '(' on the
// same line; 0 for ArrayLit/SetLit, which have no such prefix), one element per indented line
// otherwise.
function unparseBracketed(open: string, close: string, items: Expr[], indent: number, extraWidth: number): string {
    if (items.length === 0) return `${open}${close}`;
    const flat = `${open}${items.map(e => unparseExpr(e, indent)).join(', ')}${close}`;
    if (!flat.includes('\n') && extraWidth + flat.length <= LINE_WIDTH_BUDGET) return flat;
    const inner = indent + 2;
    const pad = ' '.repeat(inner);
    const lines = items.map(e => `${pad}${unparseExpr(e, inner)}`);
    return `${open}\n${lines.join(',\n')}\n${' '.repeat(indent)}${close}`;
}

// The lexer's own string-literal escapes (see tokenize's own doc comment) - '\\'  must come first,
// so it doesn't double-escape the backslashes this function itself just inserted for '"'/'\n'/'\t'.
function escapeClegString(s: string): string {
    return s.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, '\\n').replace(/\t/g, '\\t');
}
