/**
 * CLEG - "Construction Language for Embedded Graphs": a small typed language for describing
 * boards, built on top of shared/boardConfig.ts's own board-construction functions. This is the
 * first, deliberately minimal version - `nis`/`eis` (below) are the only two board modifiers with
 * their own dedicated builtin so far; every other one (`rectify`, `quadform`, `repeat`, ...) is only
 * reachable via `applyMod` (below), which parses and applies one at a time from its own command-line
 * string syntax (shared/boardConfig.ts's own parseModifier/parseModifiers) rather than cleg having a
 * dedicated function or AST node per modifier. The language has a single C++-style `for` loop (see
 * ForStmt below) - its only loop construct; recursion is otherwise still the only other way to
 * repeat anything. There is still no logical operator (no `&&`/`||`/`!`) and no way to construct a
 * `bool` value from a literal other than `true`/`false` - see the design notes scattered through
 * this file (each marked "Simplification:") for what's deliberately left out for now and would need
 * revisiting to grow the language further. The five arithmetic operators (`+ - * / %`) and five
 * comparison operators (`== < > <= >=`) are supported, with `()` for grouping/precedence; each is
 * resolved against a small overload table (BINARY_OPERATOR_OVERLOADS below) rather than being
 * hardcoded to one signature, so operators can be polymorphic - `+` currently has three overloads
 * (`number, number -> number`; array concatenation, `T[], T[] -> T[]`; set union, `T{}, T{} ->
 * T{}`, see below), `-` has two (`number, number -> number` and set difference), `*` has two
 * (`number, number -> number` and set intersection), `/ %` currently have only the one `number,
 * number -> number` overload each, and each comparison operator has two (`number, number -> bool`
 * and `bool, bool -> bool`, the latter via C++'s own false=0/true=1 convention, e.g. `false < true`
 * is `true` - see toComparable/comparisonOverload).
 * Besides the per-prescribed-board functions, `prod` (shared/boardConfig.ts's own product() - the
 * graph/tensor product of two `egr`s), and `applyMod` (parses its `string` argument as exactly one
 * board modifier via parseModifiers, then applies it via applyModifier - all fixed-signature like
 * `mkEdge`/`mkTri`/`mkQuad` below), there's also a small set of generic built-ins whose
 * result type depends on their actual argument types rather than one fixed signature
 * (BUILTIN_FUNCTIONS below covers both kinds under one interface) - `len` (an array's or set's
 * length, as a `number`), `randRmN` (a set with `count` elements removed uniformly at random),
 * `randRmP` (a set with a `frac`-sized portion removed uniformly at random, mirroring
 * shared/selector.ts's own `(rrmn <num> SEL)`/`(rrmp <num> SEL)`), and `nis`/`eis` (the first real
 * board modifiers: shared/boardConfig.ts's own nodeInducedSubgraph/edgeInducedSubgraph, taking an
 * `egr` and either a `number{}`/`edge{}` set, a `sel` (kind-checked at runtime), or a `string`
 * parsed as a selector at call time).
 *
 * Three more basic types mirror shared/types.ts's own board primitives: `edge`, `tri`, and `quad`
 * (wrapping a BoardEdge/BoardTriangle/BoardQuad respectively), built via the `mkEdge(n1, n2)`,
 * `mkTri(n1, n2, n3)`, `mkQuad(n1, n2, n3, n4)` builtins. `mkTri`/`mkQuad` canonicalize their
 * arguments the same way shared/types.ts's own makeBoardTriangle/makeBoardQuad do (`mkQuad`'s
 * arguments must already be in cycle order, exactly like makeBoardQuad's own) - there is no way yet
 * to read `n1`/`n2`/etc. back out of one of these values (no field access of any kind exists), so
 * for now they're only useful as opaque values to pass to a future selector/modifier API.
 *
 * A second type constructor, `{}` (set), pairs with `[]` (array) - `number{}` is a set of numbers.
 * Unlike `[]`, `{}` may only directly follow one of `number`/`string`/`bool`/`edge`/`tri`/`quad` (see
 * SET_ELEM_KINDS) - sets of `egr`, sets of sets, and sets of arrays are all rejected, the first two
 * as a parse error (the grammar has no production for them) and the third (e.g. a set literal mixing
 * in an array-typed element) at typecheck time. An array of sets (`number{}[]`) is fine - only a
 * set's own element type is restricted. A set literal `{x1, ..., xn}` (SetLit) duplicate-collapses
 * its elements by value (so `{1, 1, 2}` is the same set as `{1, 2}`) - see makeClegSet.
 *
 * One more basic type, `sel`, wraps a real shared/types.ts Selector - built via `mkSel(kind, str)`,
 * where `kind` is `"node"`/`"edge"`/`"tri"`/`"quad"` and `str` is parsed exactly as
 * shared/selector.ts's own grammar/semantics define (see that file's own top comment), via whichever
 * of its four real parse*Selector functions matches `kind` (see SELECTOR_PARSERS below). `sel` itself
 * carries no kind at the type level - `kind` is an ordinary runtime string argument, not something
 * the type checker can see ahead of a call - so two `sel`-typed locals can hold selectors of two
 * different actual kinds; ClegValue's own 'sel' variant carries the real kind (`selType`) once a
 * value actually exists. There is no selector literal syntax and (like `edge`/`tri`/`quad`) no way
 * to read a `sel` value's contents back out - it's passed straight through to whichever of `nis`/
 * `eis`'s own `sel` argument case consumes it (see BUILTIN_FUNCTIONS below), the only thing that
 * does yet.
 *
 * A cleg program is a sequence of top-level function declarations. Every function must declare its
 * own return type and always returns a value via `return EXPR;` (there is no `void`). Exactly one
 * function, `main`, is the program's entry point - runCleg() below evaluates it, given a
 * caller-supplied argument list (there is no other way for a cleg program to receive external
 * input yet), and returns whatever it returns.
 *
 * Concrete syntax (deliberately C++-like, per this language's own design brief):
 *
 *   TYPE       := BASETYPE ('{' '}')? ('[' ']')*
 *   BASETYPE   := 'egr' | 'number' | 'string' | 'bool' | 'edge' | 'tri' | 'quad'
 *   PROGRAM    := FUNCDECL*
 *   FUNCDECL   := TYPE IDENT '(' (PARAM (',' PARAM)*)? ')' BLOCK
 *   PARAM      := TYPE IDENT
 *   BLOCK      := '{' STMT* '}'
 *   STMT       := VARDECL | ASSIGNSTMT | IFSTMT | FORSTMT | RETURNSTMT | EXPRSTMT | BLOCK
 *   VARDECL    := TYPE IDENT '=' EXPR ';'
 *   ASSIGNSTMT := IDENT '=' EXPR ';'
 *   IFSTMT     := 'if' '(' EXPR ')' BLOCK ('else' (IFSTMT | BLOCK))?
 *   FORSTMT    := 'for' '(' FORINIT? ';' EXPR? ';' FORUPDATE? ')' BLOCK
 *   FORINIT    := TYPE IDENT '=' EXPR | IDENT '=' EXPR | EXPR
 *   FORUPDATE  := IDENT '=' EXPR | EXPR
 *   RETURNSTMT := 'return' EXPR ';'
 *   EXPRSTMT   := EXPR ';'
 *   EXPR       := RELATIONAL (('==') RELATIONAL)*
 *   RELATIONAL := ADDITIVE (('<' | '>' | '<=' | '>=') ADDITIVE)*
 *   ADDITIVE   := TERM (('+' | '-') TERM)*
 *   TERM       := UNARY (('*' | '/' | '%') UNARY)*
 *   UNARY      := '-' UNARY | ATOM
 *   ATOM       := NUMBER | STRING | 'true' | 'false' | ARRAYLIT | SETLIT | IDENT | CALL | '(' EXPR ')'
 *   ARRAYLIT   := '[' (EXPR (',' EXPR)*)? ']'
 *   SETLIT     := '{' (EXPR (',' EXPR)*)? '}'
 *   CALL       := IDENT '(' (EXPR (',' EXPR)*)? ')'
 *
 * `//` line comments are supported. There is still no logical operator of any kind (no `&&`, `||`,
 * `!`) - besides the ten (possibly overloaded, see BINARY_OPERATOR_OVERLOADS above) arithmetic/
 * comparison operators (standard C++ precedence - `==` loosest, then `< > <= >=`, then `+ -`, then
 * `* / %` tightest - all left-associative, `()` overrides precedence), the only way to combine or
 * inspect values is by calling a function (either a builtin, see BUILTIN_FUNCTIONS below, or another
 * cleg function). Since comparison operators now produce `bool`, `if`/`for` conditions are no longer
 * limited to bare literals - `if (x < 10) { ... }` works exactly as it looks.
 *
 * Example:
 *   egr main() {
 *       egr x = menger(3, 3, "0011");
 *       return x;
 *   }
 */

import {
    BoardArgType, numArg, csvArg, zolArg, parseBoardArgToken,
    makeBoardEdge, makeBoardTriangle, makeBoardQuad,
    type BoardArgEntry, type BoardConfig, type BoardEdge, type BoardTriangle, type BoardQuad,
    type Selector, type SelectorType,
} from './types.js';
import {
    PrescribedBoard, PrescribedBoardMap, PrescribedBoardFns, nodeInducedSubgraph, edgeInducedSubgraph,
    product, parseModifiers, applyModifier,
} from './boardConfig.js';
import {
    randomlyRemove, parseNodeSelector, parseEdgeSelector, parseTriangleSelector, parseQuadSelector,
    selectNode, selectEdge,
} from './selector.js';

// ── Types ────────────────────────────────────────────────────────────────────

/** A cleg type: one of the seven basic types, an array of some other cleg type (nestable, e.g.
 * `number[][]`), or a set of one of the SET_ELEM_KINDS types (not nestable - see that constant's
 * own doc comment). This is a value (not just a compile-time-only construct) - ClegValue below
 * carries one of these at runtime too, so an array/set's own element type is always known even for
 * an empty-looking runtime value passed across a function boundary. */
export type ClegType =
    | { kind: 'egr' }
    | { kind: 'number' }
    | { kind: 'string' }
    | { kind: 'bool' }
    | { kind: 'edge' }
    | { kind: 'tri' }
    | { kind: 'quad' }
    /** One flat type for a selector of any of the four SelectorType kinds (node/edge/tri/quad) -
     * unlike 'array'/'set', not parameterized by that kind, since mkSel's own kind argument is an
     * ordinary runtime string (see BUILTIN_FUNCTIONS['mkSel']), not something the type checker can
     * know ahead of a call. ClegValue's own 'sel' variant carries the actual kind (`selType`) at
     * runtime instead. */
    | { kind: 'sel' }
    | { kind: 'array'; elem: ClegType }
    | { kind: 'set'; elem: ClegType };

/** The ClegType kinds a set (`{}`) may directly hold, per this language's own design brief - `egr`
 * (no natural equality/hashing for a whole board), sets (no nested `{}`), and arrays (no nested
 * `[]` inside a `{}`) are all excluded. Doubles as the parser's own check for what may precede a
 * `{}` type suffix (parseType) and the type-checker's check on a SetLit's inferred element type
 * (checkExpr) - one canonical list for both. */
const SET_ELEM_KINDS = new Set(['number', 'string', 'bool', 'edge', 'tri', 'quad']);

function typeEquals(a: ClegType, b: ClegType): boolean {
    if (a.kind !== b.kind) return false;
    if (a.kind === 'array' || a.kind === 'set')
        return typeEquals(a.elem, (b as { kind: 'array' | 'set'; elem: ClegType }).elem);
    return true;
}

function typeToString(t: ClegType): string {
    if (t.kind === 'array') return `${typeToString(t.elem)}[]`;
    if (t.kind === 'set') return `${typeToString(t.elem)}{}`;
    return t.kind;
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
    | { kind: 'edge'; value: BoardEdge }
    | { kind: 'tri'; value: BoardTriangle }
    | { kind: 'quad'; value: BoardQuad }
    /** `selType` records which of the four SelectorType kinds `value` actually is (see ClegType's
     * own 'sel' doc comment) - always set from whichever parse*Selector function built `value`, so
     * it's never out of sync with `value.type`. */
    | { kind: 'sel'; selType: SelectorType; value: Selector }
    | { kind: 'array'; elem: ClegType; value: ClegValue[] }
    /** A set's `value` is always deduplicated by clegSetKey (see makeClegSet) - unlike 'array',
     * where `value` may hold anything an ArrayLit/array-typed value can, `value` here never holds
     * two elements with the same key. Represented as a plain array (not a JS Set/Map) since
     * edge/tri/quad don't have reference equality, so every set operation already needs its own
     * clegSetKey-based comparison regardless of the backing container - see setUnion/etc. below. */
    | { kind: 'set'; elem: ClegType; value: ClegValue[] };

function clegValueType(v: ClegValue): ClegType {
    if (v.kind === 'array') return { kind: 'array', elem: v.elem };
    if (v.kind === 'set') return { kind: 'set', elem: v.elem };
    return { kind: v.kind };
}

// ── Binary operators ─────────────────────────────────────────────────────────

type BinOp = '+' | '-' | '*' | '/' | '%' | '==' | '<' | '>' | '<=' | '>=';

/**
 * One candidate signature for a binary operator - operators are polymorphic (see `+`'s overloads
 * below), resolved by trying each operator's own overload list in order and taking the first
 * match. `match` is the single source of truth for "do these operand types satisfy this
 * overload": both checkExpr (which only needs the result `type`) and evalExpr (which only needs
 * `eval`) call it, so the two can never disagree about which overload applies to a given pair of
 * operand types. Returns null if `l`/`r` don't match this overload.
 */
interface BinaryOverload {
    /** Shown in the error message when no overload of an operator matches, e.g.
     * "number, number -> number" or "T[], T[] -> T[] (same T)". */
    signature: string;
    match: (l: ClegType, r: ClegType) =>
        { type: ClegType; eval: (l: ClegValue, r: ClegValue) => ClegValue } | null;
}

/** `number, number -> number` via `compute` - shared by all five arithmetic operators below,
 * only `compute` itself differs per operator. */
function numberOverload(compute: (a: number, b: number) => number): BinaryOverload {
    return {
        signature: 'number, number -> number',
        match: (l, r) => (l.kind === 'number' && r.kind === 'number')
            ? {
                type: { kind: 'number' },
                eval: (lv, rv) => ({
                    kind: 'number',
                    value: compute((lv as { value: number }).value, (rv as { value: number }).value),
                }),
            }
            : null,
    };
}

/** `T[], T[] -> T[]` (concatenation) - only matches when both operands are arrays of the exact
 * same element type (via typeEquals), so e.g. `number[] + string[]` is still rejected. */
const arrayConcatOverload: BinaryOverload = {
    signature: 'T[], T[] -> T[] (same T)',
    match: (l, r) => (l.kind === 'array' && r.kind === 'array' && typeEquals(l, r))
        ? {
            type: l,
            eval: (lv, rv) => ({
                kind: 'array',
                elem: (l as { elem: ClegType }).elem,
                value: [...(lv as { value: ClegValue[] }).value, ...(rv as { value: ClegValue[] }).value],
            }),
        }
        : null,
};

/** A canonical string key for a set element - two ClegValues of a SET_ELEM_KINDS type represent
 * the same set member iff their keys are equal, since edge/tri/quad (unlike number/string/bool)
 * have no reference equality of their own, so every set operation (makeClegSet/setUnion/
 * setIntersect/setDiff) goes through this rather than `===`/JS Set/Map identity. */
function clegSetKey(v: ClegValue): string {
    switch (v.kind) {
        case 'number': return `n:${v.value}`;
        case 'string': return `s:${JSON.stringify(v.value)}`;
        case 'bool': return `b:${v.value}`;
        case 'edge': return `e:${v.value.n1},${v.value.n2}`;
        case 'tri': return `t:${v.value.n1},${v.value.n2},${v.value.n3}`;
        case 'quad': return `q:${v.value.n1},${v.value.n2},${v.value.n3},${v.value.n4}`;
        default:
            // Unreachable in a program that has passed typecheckCleg - SET_ELEM_KINDS already
            // rejects every other kind as a set element.
            throw new Error(`cleg: '${v.kind}' cannot be a set element`);
    }
}

/** Builds a 'set' ClegValue from `values`, collapsing duplicates by clegSetKey (first occurrence
 * wins) - the one place a possibly-duplicate ClegValue[] becomes the deduplicated invariant every
 * other 'set' value maintains (see ClegValue's own doc comment). Used by evalExpr's SetLit case
 * and by setUnion below (the only operation that can introduce a fresh duplicate, since it merges
 * two already-deduplicated arrays that may overlap). */
function makeClegSet(elem: ClegType, values: ClegValue[]): ClegValue {
    const seen = new Map<string, ClegValue>();
    for (const v of values) { const k = clegSetKey(v); if (!seen.has(k)) seen.set(k, v); }
    return { kind: 'set', elem, value: [...seen.values()] };
}

function setUnion(a: { elem: ClegType; value: ClegValue[] }, b: { value: ClegValue[] }): ClegValue {
    return makeClegSet(a.elem, [...a.value, ...b.value]);
}

/** Intersection/difference only ever remove elements from `a`, which is already deduplicated, so
 * unlike setUnion neither needs to go through makeClegSet again. */
function setIntersect(a: { elem: ClegType; value: ClegValue[] }, b: { value: ClegValue[] }): ClegValue {
    const bKeys = new Set(b.value.map(clegSetKey));
    return { kind: 'set', elem: a.elem, value: a.value.filter(v => bKeys.has(clegSetKey(v))) };
}

function setDiff(a: { elem: ClegType; value: ClegValue[] }, b: { value: ClegValue[] }): ClegValue {
    const bKeys = new Set(b.value.map(clegSetKey));
    return { kind: 'set', elem: a.elem, value: a.value.filter(v => !bKeys.has(clegSetKey(v))) };
}

/** Builds a `T{}, T{} -> T{}` BinaryOverload for one of the three set operators from its own
 * `combine` (setUnion/setIntersect/setDiff) - only matches when both operands are sets of the
 * exact same element type (via typeEquals), mirroring arrayConcatOverload's own same-element-type
 * requirement above. */
function setOverload(signature: string, combine: typeof setUnion): BinaryOverload {
    return {
        signature,
        match: (l, r) => (l.kind === 'set' && r.kind === 'set' && typeEquals(l, r))
            ? {
                type: l,
                eval: (lv, rv) => combine(
                    lv as { elem: ClegType; value: ClegValue[] }, rv as { elem: ClegType; value: ClegValue[] }),
            }
            : null,
    };
}

const setUnionOverload = setOverload('T{}, T{} -> T{} (same T; union)', setUnion);
const setDiffOverload = setOverload('T{}, T{} -> T{} (same T; difference)', setDiff);
const setIntersectOverload = setOverload('T{}, T{} -> T{} (same T; intersection)', setIntersect);

/** A `number`/`bool` operand as a plain JS number for comparison purposes - `bool` compares via
 * C++'s own false=0/true=1 convention (e.g. `false < true` is `true`), `number` passes through
 * unchanged. Only ever called on a value whose kind a comparisonOverload's own `match` has already
 * confirmed is `elemKind` (`number` or `bool`), so the two are the only cases handled. */
function toComparable(v: ClegValue): number {
    return v.kind === 'bool' ? (v.value ? 1 : 0) : (v as { value: number }).value;
}

/** `elemKind, elemKind -> bool` for one of `==`/`<`/`>`/`<=`/`>=` - `compute` receives both
 * operands already normalized to a plain JS number via toComparable. */
function comparisonOverload(elemKind: 'number' | 'bool', compute: (a: number, b: number) => boolean): BinaryOverload {
    return {
        signature: `${elemKind}, ${elemKind} -> bool`,
        match: (l, r) => (l.kind === elemKind && r.kind === elemKind)
            ? {
                type: { kind: 'bool' },
                eval: (lv, rv) => ({ kind: 'bool', value: compute(toComparable(lv), toComparable(rv)) }),
            }
            : null,
    };
}

const BINARY_OPERATOR_OVERLOADS: Record<BinOp, BinaryOverload[]> = {
    '+': [numberOverload((a, b) => a + b), arrayConcatOverload, setUnionOverload],
    '-': [numberOverload((a, b) => a - b), setDiffOverload],
    '*': [numberOverload((a, b) => a * b), setIntersectOverload],
    '/': [numberOverload((a, b) => a / b)],
    '%': [numberOverload((a, b) => a % b)],
    '==': [comparisonOverload('number', (a, b) => a === b), comparisonOverload('bool', (a, b) => a === b)],
    '<': [comparisonOverload('number', (a, b) => a < b), comparisonOverload('bool', (a, b) => a < b)],
    '>': [comparisonOverload('number', (a, b) => a > b), comparisonOverload('bool', (a, b) => a > b)],
    '<=': [comparisonOverload('number', (a, b) => a <= b), comparisonOverload('bool', (a, b) => a <= b)],
    '>=': [comparisonOverload('number', (a, b) => a >= b), comparisonOverload('bool', (a, b) => a >= b)],
};

// ── AST ──────────────────────────────────────────────────────────────────────

export interface Param { type: ClegType; name: string; }

export interface FunctionDecl {
    kind: 'FunctionDecl';
    returnType: ClegType;
    name: string;
    params: Param[];
    body: Block;
}

export type Stmt = VarDecl | AssignStmt | IfStmt | ForStmt | ReturnStmt | ExprStmt | Block;

/** Declares and initializes a new local; see AssignStmt below for mutating an already-declared one. */
export interface VarDecl { kind: 'VarDecl'; type: ClegType; name: string; init: Expr; }
/** Reassigns an already-declared local (`x = expr;`) - mutates the binding in whichever enclosing
 * scope originally declared `name` (see evaluation's setValue), it does not shadow it with a new
 * one in the current block. `name` must already be declared with `value`'s exact type - there is no
 * way to introduce a new binding via assignment, only VarDecl does that. */
export interface AssignStmt { kind: 'AssignStmt'; name: string; value: Expr; }
/** `else_` is null (no else clause), a Block (`else { ... }`), or another IfStmt (`else if (...)`). */
export interface IfStmt { kind: 'IfStmt'; cond: Expr; then: Block; else_: Block | IfStmt | null; }
/** `init`/`cond`/`update` are each independently optional, exactly like real C++'s `for (;;)` -
 * `init`/`update` are VARDECL/ASSIGNSTMT/EXPRSTMT-shaped but never consume a trailing `;` themselves
 * (the for-loop's own two `;` tokens are the delimiters, see parseForStmt), and `update` specifically
 * excludes VarDecl (declaring a fresh variable in the update clause isn't meaningful - real C++
 * rejects it too). A missing `cond` means "always true," matching C++. `init`'s own declared
 * variable (if any) is scoped to the whole loop (header + body, across every iteration) - a fresh
 * child scope of whichever scope the `for` itself appears in, NOT the same scope as `body`'s own
 * (`body` gets its own further-nested scope per BLOCK's usual rule, fresh each iteration) - see
 * checkStmt's/evalStmt's own ForStmt cases. */
export interface ForStmt {
    kind: 'ForStmt';
    init: VarDecl | AssignStmt | ExprStmt | null;
    cond: Expr | null;
    update: AssignStmt | ExprStmt | null;
    body: Block;
}
/** Every function must return a value (there is no `void`), so unlike C++ this is never bare. */
export interface ReturnStmt { kind: 'ReturnStmt'; value: Expr; }
export interface ExprStmt { kind: 'ExprStmt'; expr: Expr; }
export interface Block { kind: 'Block'; stmts: Stmt[]; }

export type Expr =
    | NumberLit | StringLit | BoolLit | ArrayLit | SetLit | Identifier | CallExpr | BinaryExpr
    | UnaryExpr;

export interface NumberLit { kind: 'NumberLit'; value: number; }
export interface StringLit { kind: 'StringLit'; value: string; }
export interface BoolLit { kind: 'BoolLit'; value: boolean; }
/** Simplification: an empty `[]` has no way to say what its element type is, so it's rejected by
 * typecheckCleg (see checkExpr's own ArrayLit case) rather than silently guessed at. */
export interface ArrayLit { kind: 'ArrayLit'; elements: Expr[]; }
/** Same empty-literal simplification as ArrayLit above, plus its own restriction: the inferred
 * element type must be one of SET_ELEM_KINDS (see checkExpr's own SetLit case) - a set of `egr`,
 * of arrays, or of sets is rejected here even though the individual elements typecheck fine on
 * their own. Duplicate elements (by clegSetKey) collapse to one - see makeClegSet. */
export interface SetLit { kind: 'SetLit'; elements: Expr[]; }
export interface Identifier { kind: 'Identifier'; name: string; }
/** `callee` names either a builtin (BUILTIN_FUNCTIONS below) or another function declared in the
 * same program - one flat namespace, see typecheckCleg. */
export interface CallExpr { kind: 'CallExpr'; callee: string; args: Expr[]; }
/** One of the five arithmetic operators or five comparison operators (`== < > <= >=`), each
 * possibly overloaded beyond a single fixed signature - see BINARY_OPERATOR_OVERLOADS and
 * checkExpr's own BinaryExpr case. `(...)` grouping isn't its own AST node - parseAtom just returns
 * the parenthesized subexpression directly, so precedence is fully resolved by the time the AST
 * exists. */
export interface BinaryExpr {
    kind: 'BinaryExpr';
    op: '+' | '-' | '*' | '/' | '%' | '==' | '<' | '>' | '<=' | '>=';
    left: Expr;
    right: Expr;
}
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

const PUNCTUATION = '(){}[],;+-*/%';

/** Splits `src` into tokens - identifiers (including keywords, disambiguated later by the parser,
 * same convention as shared/selector.ts's own tokenize()/parser split), unsigned integer/decimal
 * number literals (negative numbers are the parser's unary '-' applied to one of these, see
 * parseUnary - the lexer itself never produces a signed number token), double-quoted string
 * literals (`\\`, `\"`, `\n`, `\t` escapes only), single-character punctuation (including the five
 * arithmetic operators), and '='/'<'/'>' - each its own one-character token unless immediately
 * followed by another '=' (making '==', '<=', or '>=' instead), handled separately from
 * `PUNCTUATION` above since it's the one lexer rule needing a second character of lookahead.
 * `//` starts a line comment. */
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
        if (c === '=' || c === '<' || c === '>') {
            if (src[i + 1] === '=') { tokens.push({ kind: 'punct', text: c + '=', pos: i }); i += 2; continue; }
            tokens.push({ kind: 'punct', text: c, pos: i });
            i++;
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

const TYPE_KEYWORDS = new Set(['egr', 'number', 'string', 'bool', 'edge', 'tri', 'quad', 'sel']);

function parseType(c: TokenCursor): ClegType {
    const base = c.expectIdent();
    if (!TYPE_KEYWORDS.has(base))
        throw new Error(`cleg: expected a type (egr/number/string/bool/edge/tri/quad/sel), got '${base}'`);
    let type: ClegType = { kind: base as 'egr' | 'number' | 'string' | 'bool' | 'edge' | 'tri' | 'quad' | 'sel' };
    if (c.isPunct('{')) {
        if (!SET_ELEM_KINDS.has(base))
            throw new Error(
                `cleg: '${base}{}' is not a supported set type - sets of egr, sets of sets, and sets of ` +
                `arrays are not supported`);
        c.next();
        c.expectPunct('}');
        type = { kind: 'set', elem: type };
    }
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

// Only an identifier immediately followed by '=' is an assignment (as opposed to, say, a bare
// call-expression statement) - look ahead one extra token to tell them apart, since parseExpr's
// own Identifier case doesn't consume '='. Shared by parseStmt and the for-loop's own
// parseForInit/parseForUpdate.
function isAssignStart(c: TokenCursor): boolean {
    return c.peek().kind === 'ident' && c.peekAt(1).kind === 'punct' && c.peekAt(1).text === '=';
}

function parseStmt(c: TokenCursor): Stmt {
    if (c.isPunct('{')) return parseBlock(c);
    if (c.isKeyword('if')) return parseIfStmt(c);
    if (c.isKeyword('for')) return parseForStmt(c);
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
    c.expectPunct('=');
    const value = parseExpr(c);
    return { kind: 'AssignStmt', name, value };
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

function parseReturnStmt(c: TokenCursor): ReturnStmt {
    c.next(); // 'return'
    const value = parseExpr(c);
    c.expectPunct(';');
    return { kind: 'ReturnStmt', value };
}

const EQUALITY_OPS = new Set(['==']);
const RELATIONAL_OPS = new Set(['<', '>', '<=', '>=']);
const ADDITIVE_OPS = new Set(['+', '-']);
const MULTIPLICATIVE_OPS = new Set(['*', '/', '%']);

function isPunctIn(c: TokenCursor, ops: Set<string>): boolean {
    const t = c.peek();
    return t.kind === 'punct' && ops.has(t.text);
}

/** Expression entry point, lowest precedence (`==`), left-associative - every existing call site
 * (VarDecl init, AssignStmt value, if/for condition, return value, call arguments, array/set
 * elements) already calls parseExpr, so every new precedence level works everywhere an expression
 * was already accepted without any caller changes. */
function parseExpr(c: TokenCursor): Expr {
    let left = parseRelational(c);
    while (isPunctIn(c, EQUALITY_OPS)) {
        const op = c.next().text as '==';
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

/**
 * A callable built into the language itself, as opposed to one of `program`'s own function
 * declarations (which always have a fixed FunctionSignature - see FuncTable). Covers two rather
 * different kinds of builtin under one shape:
 *   - the fixed-signature per-prescribed-board constructors (`menger`, `rect`, `cublat`, ...) -
 *     `checkCall` here is just fixedSignature(...)'s own arg-count/arg-type check against a fixed
 *     ClegType[]/ClegType, built once per PrescribedBoardMap entry below.
 *   - the small set of generic functions (currently just `len`) whose result type can depend on the
 *     actual argument types at a call site, so a fixed ClegType[]/ClegType can't describe them -
 *     `checkCall` is hand-written per function instead.
 * Both live in one flat BUILTIN_FUNCTIONS table, checked before `program`'s own functions in
 * checkExpr/evalExpr's CallExpr cases, and reserved against user redeclaration in typecheckCleg -
 * from a cleg program's own point of view there's no distinction between the two kinds.
 */
interface BuiltinFunction {
    /** Validates `argTypes` (throwing descriptively on a bad arg count/type) and returns the call's
     * result type. */
    checkCall: (callee: string, argTypes: ClegType[]) => ClegType;
    call: (args: ClegValue[]) => ClegValue;
}

/** Builds a BuiltinFunction's own checkCall from a fixed ClegType[] -> ClegType signature - shares
 * the arg-count/arg-type checking logic with checkExpr's own user-function CallExpr case (see
 * checkExpr) rather than duplicating it. */
function fixedSignature(params: ClegType[], returnType: ClegType): BuiltinFunction['checkCall'] {
    return (callee, argTypes) => {
        if (argTypes.length !== params.length)
            throw new Error(`cleg: '${callee}' expects ${params.length} argument(s), got ${argTypes.length}`);
        argTypes.forEach((t, i) => {
            if (!typeEquals(t, params[i]))
                throw new Error(`cleg: '${callee}' argument ${i + 1}: expected ${typeToString(params[i])}, got ${typeToString(t)}`);
        });
        return returnType;
    };
}

const BUILTIN_FUNCTIONS: Record<string, BuiltinFunction> = {};

// One builtin per shared/boardConfig.ts's own PrescribedBoardMap/PrescribedBoardFns entry, named
// after its own command-line token (PrescribedBoardMap[pb][1], e.g. "menger", "rect", "cublat") so a
// cleg program's board-construction calls read exactly like this project's own command syntax -
// modifiers/selectors aside (see this file's own top comment). Built generically from that existing
// table (rather than one hand-written cleg function per board type) so this list never drifts out of
// sync with it.
for (const [pbKey, [argTypes, cmdName]] of
    Object.entries(PrescribedBoardMap) as [string, [BoardArgType[], string, string, string]][]) {
    const pb = Number(pbKey) as PrescribedBoard;
    BUILTIN_FUNCTIONS[cmdName] = {
        checkCall: fixedSignature(argTypes.map(argTypeToClegType), { kind: 'egr' }),
        call: (args: ClegValue[]): ClegValue =>
            ({ kind: 'egr', value: PrescribedBoardFns[pb](...argTypes.map((t, i) => valueToBoardArgEntry(t, args[i]))) }),
    };
}

// `len`: an array's or set's length, as a `number` - its one argument may be an array or set of
// any element type, which a fixedSignature(...) can't express (there's no "any" ClegType), hence
// the hand-written checkCall/call pair here instead.
BUILTIN_FUNCTIONS['len'] = {
    checkCall(callee, argTypes) {
        if (argTypes.length !== 1)
            throw new Error(`cleg: '${callee}' expects 1 argument(s), got ${argTypes.length}`);
        if (argTypes[0].kind !== 'array' && argTypes[0].kind !== 'set')
            throw new Error(
                `cleg: '${callee}' argument 1: expected an array or set, got ${typeToString(argTypes[0])}`);
        return { kind: 'number' };
    },
    call(args) {
        const v = args[0] as { kind: 'array' | 'set'; elem: ClegType; value: ClegValue[] };
        return { kind: 'number', value: v.value.length };
    },
};

// `randRmN`/`randRmP`: both `(T{}, number) -> T{}`, differing only in how the second argument
// becomes a removal count - share this one checkCall rather than duplicating its arg-count/
// arg-type checks. Like `len`, their result type depends on the actual argument type (here, the
// whole input set type passes through unchanged), so they need a hand-written checkCall rather
// than fixedSignature(...).
function randRmCheckCall(callee: string, argTypes: ClegType[]): ClegType {
    if (argTypes.length !== 2)
        throw new Error(`cleg: '${callee}' expects 2 argument(s), got ${argTypes.length}`);
    if (argTypes[0].kind !== 'set')
        throw new Error(`cleg: '${callee}' argument 1: expected a set, got ${typeToString(argTypes[0])}`);
    if (argTypes[1].kind !== 'number')
        throw new Error(`cleg: '${callee}' argument 2: expected number, got ${typeToString(argTypes[1])}`);
    return argTypes[0];
}

// Randomly (uniformly) removes `count` elements from a set of any element type, mirroring
// shared/selector.ts's own `(rrmn <num> SEL)` selector semantics - reuses that file's own
// randomlyRemove() rather than reimplementing the same partial shuffle.
BUILTIN_FUNCTIONS['randRmN'] = {
    checkCall: randRmCheckCall,
    call(args) {
        const s = args[0] as { kind: 'set'; elem: ClegType; value: ClegValue[] };
        const count = (args[1] as { value: number }).value;
        if (!Number.isInteger(count) || count < 0)
            throw new Error(`cleg: 'randRmN' count must be a nonnegative integer, got ${count}`);
        return { kind: 'set', elem: s.elem, value: randomlyRemove(s.value, count) };
    },
};

// Randomly removes a fixed portion of a set - `frac` (a nonnegative float, not necessarily <= 1)
// times the set's own size, rounded down - mirroring shared/selector.ts's own `(rrmp <num> SEL)`
// exactly (including that a `frac` big enough to exceed the set's size just empties it, via
// randomlyRemove's own clamp, same as rrmp's own behavior).
BUILTIN_FUNCTIONS['randRmP'] = {
    checkCall: randRmCheckCall,
    call(args) {
        const s = args[0] as { kind: 'set'; elem: ClegType; value: ClegValue[] };
        const frac = (args[1] as { value: number }).value;
        if (!Number.isFinite(frac) || frac < 0)
            throw new Error(`cleg: 'randRmP' portion must be a nonnegative number, got ${frac}`);
        return { kind: 'set', elem: s.elem, value: randomlyRemove(s.value, Math.floor(frac * s.value.length)) };
    },
};

// `nis`/`eis`: both `(egr, X) -> egr`, where X may be a set of nodes/edges (`number{}`/`edge{}`),
// a `sel` (its actual kind checked at runtime, since 'sel' carries no kind at the type level - see
// ClegType's own 'sel' doc comment), or a `string` (parsed at call time via the real
// parseNodeSelector/parseEdgeSelector, following shared/selector.ts's own grammar exactly). Share
// this one checkCall (parameterized by which set element kind is valid), the same way randRmN/
// randRmP share randRmCheckCall above.
function inducedSubgraphCheckCall(elemKind: 'number' | 'edge'): BuiltinFunction['checkCall'] {
    return (callee, argTypes) => {
        if (argTypes.length !== 2)
            throw new Error(`cleg: '${callee}' expects 2 argument(s), got ${argTypes.length}`);
        if (argTypes[0].kind !== 'egr')
            throw new Error(`cleg: '${callee}' argument 1: expected egr, got ${typeToString(argTypes[0])}`);
        const t = argTypes[1];
        const okSet = t.kind === 'set' && t.elem.kind === elemKind;
        if (!okSet && t.kind !== 'sel' && t.kind !== 'string')
            throw new Error(
                `cleg: '${callee}' argument 2: expected ${elemKind}{}, sel, or string, got ${typeToString(t)}`);
        return { kind: 'egr' };
    };
}

// `nis(bc, nodes)`: shared/boardConfig.ts's own nodeInducedSubgraph, lowered to accept a plain
// Set<number> - `nodes` is derived from whichever of the three argument-2 shapes was actually
// given, always ending up as exactly that Set<number> before being handed to nodeInducedSubgraph.
BUILTIN_FUNCTIONS['nis'] = {
    checkCall: inducedSubgraphCheckCall('number'),
    call([egrVal, arg]) {
        const bc = (egrVal as { value: BoardConfig }).value;
        let nodes: Set<number>;
        if (arg.kind === 'set') {
            nodes = new Set(arg.value.map(v => (v as { value: number }).value));
        } else if (arg.kind === 'sel') {
            if (arg.selType !== 'node')
                throw new Error(`cleg: 'nis' argument 2: expected a node selector, got a '${arg.selType}' selector`);
            nodes = selectNode(bc.adj, bc.emb.pos, arg.value);
        } else {
            nodes = selectNode(bc.adj, bc.emb.pos, parseNodeSelector((arg as { value: string }).value));
        }
        return { kind: 'egr', value: nodeInducedSubgraph(bc, nodes) };
    },
};

// `eis(bc, edges)` - the edge-flavored counterpart of `nis` above, backed by
// shared/boardConfig.ts's own (equally lowered) edgeInducedSubgraph.
BUILTIN_FUNCTIONS['eis'] = {
    checkCall: inducedSubgraphCheckCall('edge'),
    call([egrVal, arg]) {
        const bc = (egrVal as { value: BoardConfig }).value;
        let edges: BoardEdge[];
        if (arg.kind === 'set') {
            edges = arg.value.map(v => (v as { value: BoardEdge }).value);
        } else if (arg.kind === 'sel') {
            if (arg.selType !== 'edge')
                throw new Error(`cleg: 'eis' argument 2: expected an edge selector, got a '${arg.selType}' selector`);
            edges = selectEdge(bc.adj, bc.emb.pos, arg.value);
        } else {
            edges = selectEdge(bc.adj, bc.emb.pos, parseEdgeSelector((arg as { value: string }).value));
        }
        return { kind: 'egr', value: edgeInducedSubgraph(bc, edges) };
    },
};

const NUMBER_TYPE: ClegType = { kind: 'number' };
const STRING_TYPE: ClegType = { kind: 'string' };
const EGR_TYPE: ClegType = { kind: 'egr' };

// `mkEdge`/`mkTri`/`mkQuad`: build an edge/tri/quad from node indices, canonicalized exactly as
// shared/types.ts's own makeBoardEdge/makeBoardTriangle/makeBoardQuad do (mkQuad's arguments must
// already be in cycle order, same requirement as makeBoardQuad's own - see its doc comment). All
// three are fixed-signature (number, ..., number) -> edge/tri/quad, so they're built with
// fixedSignature(...) like the board constructors above rather than needing a hand-written checkCall.
BUILTIN_FUNCTIONS['mkEdge'] = {
    checkCall: fixedSignature([NUMBER_TYPE, NUMBER_TYPE], { kind: 'edge' }),
    call: ([a, b]) => ({
        kind: 'edge',
        value: makeBoardEdge((a as { value: number }).value, (b as { value: number }).value),
    }),
};
BUILTIN_FUNCTIONS['mkTri'] = {
    checkCall: fixedSignature([NUMBER_TYPE, NUMBER_TYPE, NUMBER_TYPE], { kind: 'tri' }),
    call: ([a, b, c]) => ({
        kind: 'tri',
        value: makeBoardTriangle(
            (a as { value: number }).value, (b as { value: number }).value, (c as { value: number }).value),
    }),
};
BUILTIN_FUNCTIONS['mkQuad'] = {
    checkCall: fixedSignature([NUMBER_TYPE, NUMBER_TYPE, NUMBER_TYPE, NUMBER_TYPE], { kind: 'quad' }),
    call: ([a, b, c, d]) => ({
        kind: 'quad',
        value: makeBoardQuad(
            (a as { value: number }).value, (b as { value: number }).value,
            (c as { value: number }).value, (d as { value: number }).value),
    }),
};

// `prod(a, b)`: the graph (tensor) product of two boards - shared/boardConfig.ts's own product(),
// fixed-signature (egr, egr) -> egr like mkEdge/mkTri/mkQuad above.
BUILTIN_FUNCTIONS['prod'] = {
    checkCall: fixedSignature([EGR_TYPE, EGR_TYPE], { kind: 'egr' }),
    call: ([a, b]) => ({
        kind: 'egr',
        value: product((a as { value: BoardConfig }).value, (b as { value: BoardConfig }).value),
    }),
};

// `applyMod(bc, str)`: parses `str` as exactly one board modifier (shared/boardConfig.ts's own
// parseModifiers - the same "<name> <args>" syntax this project's own modifier-list textboxes take,
// see parseModifier's own doc comment for the per-command grammar) and applies it to `bc` via
// applyModifier. `str` naming 'beginprod'/'repeat' is fine too (parseModifiers folds a whole nested
// span into one Prod/Repeat BoardModifier), but must still resolve to exactly one modifier overall.
BUILTIN_FUNCTIONS['applyMod'] = {
    checkCall: fixedSignature([EGR_TYPE, STRING_TYPE], { kind: 'egr' }),
    call: ([egrVal, strVal]) => {
        const bc = (egrVal as { value: BoardConfig }).value;
        const modifiers = parseModifiers((strVal as { value: string }).value);
        if (modifiers.length !== 1)
            throw new Error(`cleg: 'applyMod' expects its string argument to name exactly one modifier, got ${modifiers.length}`);
        return { kind: 'egr', value: applyModifier(bc, modifiers[0]) };
    },
};

// One real parse*Selector function per SelectorType - shared/selector.ts itself has no single
// kind-agnostic parse entry point (see that file's own top comment: parsing is type-directed,
// since e.g. `(all)` means a different thing depending on which of these is called), so mkSel's
// own `call` below dispatches through this table on its first (runtime string) argument.
const SELECTOR_PARSERS: Record<SelectorType, (s: string) => Selector> = {
    node: parseNodeSelector,
    edge: parseEdgeSelector,
    tri: parseTriangleSelector,
    quad: parseQuadSelector,
};

// `mkSel(kind, str)`: parses `str` as a selector of the given `kind` ("node"/"edge"/"tri"/"quad"),
// via the matching real parse*Selector function above - `str` follows shared/selector.ts's own
// grammar/semantics exactly (see that file's own top comment), including its own error messages on
// a malformed `str`. Fixed-signature (`string, string -> sel`, see fixedSignature/BUILTIN_FUNCTIONS
// above) since `sel` itself isn't parameterized by kind at the type level (see ClegType's own 'sel'
// doc comment) - `kind` is only validated/dispatched on at call time, not check time.
BUILTIN_FUNCTIONS['mkSel'] = {
    checkCall: fixedSignature([STRING_TYPE, STRING_TYPE], { kind: 'sel' }),
    call: ([kindVal, strVal]) => {
        const kind = (kindVal as { value: string }).value;
        const parse = SELECTOR_PARSERS[kind as SelectorType];
        if (!parse) throw new Error(`cleg: mkSel: unknown selector kind '${kind}' - expected node/edge/tri/quad`);
        return { kind: 'sel', selType: kind as SelectorType, value: parse((strVal as { value: string }).value) };
    },
};

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
 * BUILTIN_FUNCTIONS and `program`'s own top-level declarations (a user function redeclaring a
 * builtin's name is rejected, not shadowed). Requires a `main` function to exist. Throws
 * descriptively on the first error found; does not attempt to collect more than one.
 *
 * Simplification: does not check that every path through a function actually reaches a `return` -
 * a function whose body falls off the end without one is only caught at evaluation time (see
 * callUserFunction below), not here.
 */
export function typecheckCleg(program: ClegProgram): void {
    const funcs: FuncTable = {};
    for (const fn of program.functions) {
        if (funcs[fn.name] || BUILTIN_FUNCTIONS[fn.name])
            throw new Error(`cleg: function '${fn.name}' is declared more than once (or shadows a builtin function)`);
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
        case 'ForStmt': {
            // A fresh scope for init's own declared variable (if any), enclosing cond/update/body -
            // NOT the same scope as body's own (checkBlock below gives body its own further-nested
            // scope, same as every other BLOCK) - see ForStmt's own doc comment.
            const loopEnv: TypeEnv = { vars: new Map(), parent: env };
            if (stmt.init) checkStmt(stmt.init, loopEnv, funcs, returnType);
            if (stmt.cond) {
                const condType = checkExpr(stmt.cond, loopEnv, funcs);
                if (condType.kind !== 'bool')
                    throw new Error(`cleg: for-loop condition must be bool, got ${typeToString(condType)}`);
            }
            if (stmt.update) checkStmt(stmt.update, loopEnv, funcs, returnType);
            checkBlock(stmt.body, loopEnv, funcs, returnType);
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
                    `number/string/bool/edge/tri/quad may be set elements`);
            return { kind: 'set', elem: elemTypes[0] };
        }
        case 'CallExpr': {
            const argTypes = expr.args.map(a => checkExpr(a, env, funcs));
            const builtin = BUILTIN_FUNCTIONS[expr.callee];
            if (builtin) return builtin.checkCall(expr.callee, argTypes);
            const sig = funcs[expr.callee];
            if (!sig) throw new Error(`cleg: call to undeclared function '${expr.callee}'`);
            if (argTypes.length !== sig.params.length)
                throw new Error(`cleg: '${expr.callee}' expects ${sig.params.length} argument(s), got ${argTypes.length}`);
            argTypes.forEach((t, i) => {
                if (!typeEquals(t, sig.params[i]))
                    throw new Error(
                        `cleg: '${expr.callee}' argument ${i + 1}: expected ${typeToString(sig.params[i])}, got ${typeToString(t)}`);
            });
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
        case 'ForStmt': {
            // One scope for the whole loop (init's own variable, if any, persists across every
            // iteration) - body gets its own further-nested scope each iteration via evalBlock,
            // same as any other BLOCK - see ForStmt's own doc comment.
            const loopEnv: ValueEnv = { vars: new Map(), parent: env };
            if (stmt.init) evalStmt(stmt.init, loopEnv, funcs);
            while (!stmt.cond || (evalExpr(stmt.cond, loopEnv, funcs) as { kind: 'bool'; value: boolean }).value) {
                evalBlock(stmt.body, loopEnv, funcs);
                if (stmt.update) evalStmt(stmt.update, loopEnv, funcs);
            }
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
        case 'SetLit': {
            const values = expr.elements.map(e => evalExpr(e, env, funcs));
            // typecheckCleg already rejected an empty, mixed-element-type, or non-SET_ELEM_KINDS
            // literal, so the first value's own type is always the set's element type.
            return makeClegSet(clegValueType(values[0]), values);
        }
        case 'CallExpr': {
            const args = expr.args.map(a => evalExpr(a, env, funcs));
            const builtin = BUILTIN_FUNCTIONS[expr.callee];
            return builtin ? builtin.call(args) : callUserFunction(funcs[expr.callee], args, funcs);
        }
        case 'BinaryExpr': {
            const l = evalExpr(expr.left, env, funcs);
            const r = evalExpr(expr.right, env, funcs);
            for (const overload of BINARY_OPERATOR_OVERLOADS[expr.op]) {
                const m = overload.match(clegValueType(l), clegValueType(r));
                if (m) return m.eval(l, r);
            }
            // Unreachable in a program that has passed typecheckCleg.
            throw new Error(`cleg: operator '${expr.op}' has no overload for these operand types at runtime`);
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
