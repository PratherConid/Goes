/**
 * CLEG - "Construction Language for Embedded Graphs": a small typed language for describing
 * boards, built on top of shared/boardConfig.ts's own board-construction functions. This is the
 * first, deliberately minimal version. Every BoardModifier kind except `Prod`/`Repeat` (handled
 * elsewhere) now has its own builtin - all of them (`nis`, `eis`, `rectify`, `edgeSplit`,
 * `mergeClose`, `triangleForm`, `quadForm`, `form`, `globalCentralize`, `quadOctarize`, `scale`)
 * BUILD a `mod` value (see that type's own doc comment below) rather than applying it to a board
 * immediately - there is no cleg builtin yet that takes a `mod` and an `egr` and applies one to the
 * other. The language has two C++-style loop constructs - `for` (see ForStmt below) and a plain
 * pretest `while` (see WhileStmt below, equivalent to a `for` with empty init/update clauses) - each
 * with its own `break`/`continue` (BreakStmt/ContinueStmt, rejected by checkStmt outside a loop) -
 * recursion is otherwise still the only other way to repeat anything. An array value's elements can
 * now be read back out via indexing (`arr[i]`, IndexExpr below) - a postfix operator, binding tighter
 * than unary `-`/`!` (so `arr[i][j]`/`f()[0]`/`-arr[0]` all parse as expected) - `arr` must be
 * array-typed (never set-typed - sets are unordered, so indexing one has no defined meaning) and `i`
 * a `number`, checked to be a nonnegative in-bounds integer at evaluation time (not statically
 * knowable). An AssignStmt's own left-hand side may now carry zero or more of these same `[...]`
 * indices (`arr[i] = x;`, `arr[i][j] = x;`, ... - see AssignStmt's own doc comment), mutating one
 * element of an already-declared array IN PLACE rather than rebinding `arr` itself to a whole new
 * value (checkStmt's own AssignStmt case walks one `array` level per index, requiring `name`'s own
 * declared type to be nested at least that deep). Arrays are VALUE types, not references, despite
 * this in-place mutation - assigning an array to another variable, or passing one as a function
 * argument, always copies it first (cloneArrayValue, called at every VarDecl init / whole-value
 * AssignStmt / function-argument binding site), so `b = a; b[0] = 9;` can never affect `a` even
 * though nothing in cleg's own AST distinguishes "the same array" from "an equal-looking copy" -
 * this is the one place evaluation performs a defensive copy that TS's own object-reference
 * semantics wouldn't otherwise need, mirroring this language's own "deliberately C++-like" design
 * brief (std::vector-style containers, not shared references). A function
 * can now also be passed around as a value - a monomorphic (fully concrete, no type variables/
 * generics) function-pointer type, `(T1, T2, ...) -> R` (FUNCTYPE, ClegType's own 'func' variant),
 * usable anywhere a TYPE is (a param, a VARDECL, a return type, even another FUNCTYPE's own param/
 * return for a higher-order function). A bare reference to one of `program`'s own top-level
 * functions by name (not immediately followed by `(`, which would instead call it directly) IS a
 * value of this type - the whole mechanism riding entirely on Identifier/CallExpr's EXISTING
 * resolution logic (see checkExpr/evalExpr's own cases) rather than needing new AST nodes: passing
 * `cmp` where a `(number, number) -> bool` is expected, then later calling `cmp(a, b)` inside that
 * function's own body, both already worked once Identifier could resolve to a function and CallExpr
 * could dispatch through a local variable of func type. Only a cleg-declared function can be
 * referenced this way - a builtin can't, since BUILTIN_FUNCTIONS' own checkCall/call pair has no
 * single static signature for the generic/overloaded ones (`+`, `len`, ...) to describe as a `func`
 * ClegType. A bare Identifier reference to a top-level function names it by itself - see
 * ClegValue's own 'func' variant - there is no way to construct a `func` value from an arbitrary
 * expression, only to name an existing top-level function, either directly or (below) partially
 * applied. A trailing `[]` on a bare FUNCTYPE binds to its return type, not
 * the whole thing (`(number) -> bool[]` is a function returning `bool[]`) - wrapping it in an extra
 * pair of parens first (`((number) -> bool)[]`, parseType's own grouping alternative for a func type
 * specifically, see parseParenType) makes the array apply to the func type as a whole instead, an
 * array of comparator-shaped functions (a func type can never have a `{}` set suffix either way - it
 * isn't one of SET_ELEM_KINDS, same as `egr`). cleg now has a limited form of closure: writing `#`
 * (HoleExpr) in place of one or more of a CALL's own arguments - `f(a, #, b)` - is a PARTIAL
 * APPLICATION rather than an ordinary call, producing a `func` value that closes over the non-`#`
 * arguments (each evaluated once, right there) and expects only the `#` positions' own values later,
 * interleaved back into their original slots (see CallExpr's own doc comment, and ClegValue's own
 * 'func'/`boundArgs` for the representation). This is deliberately narrow, not general
 * closures-over-arbitrary-expressions: `f` must be either a bare reference to one of `program`'s own
 * top-level functions, or an existing local variable already holding a `func` value (itself a plain
 * pointer or an already-partial closure - further-applying it narrows its own remaining open
 * positions, rather than starting over) - never a builtin (most have no single fixed signature to
 * close over). There is still no way to construct a `bool` value from a literal
 * other than `true`/`false` - see the design notes scattered through this file (each marked "Simplification:")
 * for what's deliberately left out for now and would need revisiting to grow the language further.
 * The five arithmetic operators (`+ - * / %`), six comparison operators (`== != < > <= >=`), and two
 * logical operators (`&& ||`, both short-circuiting - see evalExpr's own BinaryExpr case) are
 * supported, with `()` for grouping/precedence; each is
 * resolved against a small overload table (BINARY_OPERATOR_OVERLOADS below) rather than being
 * hardcoded to one signature, so operators can be polymorphic - `+` currently has five overloads
 * (`number, number -> number`; string concatenation, `string, string -> string`; number/string
 * concatenation, `number, string -> string` and `string, number -> string`, converting the number
 * operand to a string - see stringConcatOverload/numberStringConcatOverload; array concatenation,
 * `T[], T[] -> T[]`; set union, `T{}, T{} -> T{}`, see below), `-` has two (`number, number ->
 * number` and set difference), `*` has four
 * (`number, number -> number`; set intersection; array/string replication, `T[], number -> T[]` and
 * `string, number -> string`, each also accepted with the two operands swapped - `n` must be a
 * nonnegative integer, checked at evaluation time since it isn't knowable from `number`'s type alone
 * - see repeatArrayOverload/repeatStringOverload), `/ %` currently have only the one `number,
 * number -> number` overload each, and each comparison operator has two (`number, number -> bool`
 * and `bool, bool -> bool`, the latter via C++'s own false=0/true=1 convention, e.g. `false < true`
 * is `true` - see toComparable/comparisonOverload).
 * Besides the per-prescribed-board functions and `prod` (shared/boardConfig.ts's own product() -
 * the graph/tensor product of two `egr`s, fixed-signature like `mkEdge`/`mkTri`/`mkQuad` below),
 * there's also a small set of generic built-ins whose result type depends on their actual argument
 * types rather than one fixed signature (BUILTIN_FUNCTIONS below covers both kinds under one
 * interface) - `len` (an array's or set's length, as a `number`), `has(x, e)` (whether `x`, a `T[]`
 * or `T{}`, contains `e` - `T` restricted to SET_ELEM_KINDS, the same equality-bearing types a set's
 * own elements are already restricted to, since nothing else in the language has a defined equality -
 * see hasCheckCall below), `randRmN`/`randRmP` (a set with
 * elements removed uniformly at random, by count or by portion, mirroring shared/selector.ts's own
 * `(rrmn <num> SEL)`/`(rrmp <num> SEL)`), and `nis`/`eis`/`triangleForm`/`quadForm`/`mkFormSel`
 * (each accepts a `sel`, a `string`, or a `set` of the matching element type, resolved via their one
 * shared resolveSelectorArg). `selectNode`/`selectEdge`/`selectTriangle`/`selectQuad` (X, bc)
 * similarly each accept a `sel`, `string`, or `set`, but evaluate it immediately against a real
 * board `bc` (an `egr`) and return the exact set of nodes/edges/triangles/quads selected -
 * `number{}`/`edge{}`/`tri{}`/`quad{}` respectively -
 * rather than building something to apply later. One builtin per kind rather than a single
 * overloaded name, since a `sel` value's own actual kind (see below) is only known at evaluation
 * time, never from its ClegType alone - there would be no way for checkCall to know which of the
 * four set types a single `select(X, bc)` should statically return.
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
 * One more basic type, `sel`, wraps a real shared/types.ts Selector - built via `mkSel(kind, X)`,
 * where `kind` is `"node"`/`"edge"`/`"tri"`/`"quad"` and `X` is either a `string`, parsed exactly as
 * shared/selector.ts's own grammar/semantics define (see that file's own top comment) via whichever
 * of its four real parse*Selector functions matches `kind` (see SELECTOR_PARSERS below), or a `set`
 * of the ClegType matching `kind` (see SELECTOR_SET_ELEM_KIND), wrapped directly into a `raw`
 * Selector with no parsing at all. `sel` itself carries no kind at the type level - `kind` is an
 * ordinary runtime string argument, not something the type checker can see ahead of a call - so two
 * `sel`-typed locals can hold selectors of two different actual kinds; ClegValue's own 'sel' variant
 * carries the real kind (`selType`) once a value actually exists. There is no selector literal
 * syntax and (like `edge`/`tri`/`quad`) no way to read a `sel` value's contents back out - it's
 * passed straight through to whichever consuming builtin's own selector-shaped argument resolves it
 * (`nis`/`eis`/`triangleForm`/`quadForm`/`mkFormSel` - see resolveSelectorArg below, their one shared
 * "sel, string, or set" resolution).
 *
 * Two more basic types round out the board-modifier builtins: `formSel` wraps a real
 * shared/types.ts FormSelector (`(tri [SEL])`/`(quad [SEL])`, see that type's own doc comment) -
 * built via `mkFormSel(kind, [selArg])`, the `form`-modifier counterpart of `sel`/`mkSel` (`kind` is
 * `"tri"`/`"quad"`; `selArg`, like `triangleForm`/`quadForm`'s own, is optional - a `sel` or
 * `string`, resolved the same way, restricting which tri/quads qualify, default every one found).
 * `mod` wraps a real shared/types.ts BoardModifier - one flat type covering every kind (`Rectify`,
 * `EdgeSplit`, `TriangleForm`, `Form`, ...), built by whichever of the constructor builtins listed
 * two paragraphs up matches. Both are opaque the same way `sel`/`edge`/`tri`/`quad` are - no literal
 * syntax, no way to read fields back out.
 *
 * A cleg program is a sequence of top-level items: function declarations, and TOPSTMTs (a VARDECL,
 * an ASSIGNSTMT, or a bare EXPRSTMT) - there is no `main` and no other designated entry-point
 * function. Every function must declare its own return type and always returns a value via `return
 * EXPR;` (there is no `void`). runCleg() runs every top-level TOPSTMT in order (function
 * declarations aren't executed, just collected - order-independent, see ClegProgram's own doc
 * comment), sharing one flat scope across all of them (so a VARDECL near the top of the program is
 * visible to every TOPSTMT after it) - but that scope is entirely separate from every function's own
 * (cleg has no closures at all: a function body only ever sees its own parameters, never any
 * top-level variable), and returns whatever the LAST one (which must itself be an EXPRSTMT) evaluated
 * to - the only way a cleg program produces an overall value, and (since there's no `main` to hand
 * external input to as parameters) also currently the only thing a cleg program actually "does".
 *
 * Concrete syntax (deliberately C++-like, per this language's own design brief):
 *
 *   TYPE       := (BASETYPE ('{' '}')? | FUNCTYPE | '(' FUNCTYPE ')') ('[' ']')*
 *   BASETYPE   := 'egr' | 'number' | 'string' | 'bool' | 'edge' | 'tri' | 'quad'
 *   FUNCTYPE   := '(' (TYPE (',' TYPE)*)? ')' '->' TYPE
 *   PROGRAM    := (FUNCDECL | TOPSTMT)*
 *   TOPSTMT    := VARDECL | ASSIGNSTMT | EXPRSTMT
 *   FUNCDECL   := TYPE IDENT '(' (PARAM (',' PARAM)*)? ')' BLOCK
 *   PARAM      := TYPE IDENT
 *   BLOCK      := '{' STMT* '}'
 *   STMT       := VARDECL | ASSIGNSTMT | IFSTMT | FORSTMT | WHILESTMT | BREAKSTMT | CONTINUESTMT
 *               | RETURNSTMT | EXPRSTMT | BLOCK
 *   VARDECL    := TYPE IDENT '=' EXPR ';'
 *   ASSIGNSTMT := IDENT ('[' EXPR ']')* '=' EXPR ';'
 *   IFSTMT     := 'if' '(' EXPR ')' BLOCK ('else' (IFSTMT | BLOCK))?
 *   FORSTMT    := 'for' '(' FORINIT? ';' EXPR? ';' FORUPDATE? ')' BLOCK
 *   FORINIT    := TYPE IDENT '=' EXPR | IDENT ('[' EXPR ']')* '=' EXPR | EXPR
 *   FORUPDATE  := IDENT ('[' EXPR ']')* '=' EXPR | EXPR
 *   WHILESTMT  := 'while' '(' EXPR ')' BLOCK
 *   BREAKSTMT  := 'break' ';'
 *   CONTINUESTMT := 'continue' ';'
 *   RETURNSTMT := 'return' EXPR ';'
 *   EXPRSTMT   := EXPR ';'
 *   EXPR       := LOGIC_OR
 *   LOGIC_OR   := LOGIC_AND (('||') LOGIC_AND)*
 *   LOGIC_AND  := EQUALITY (('&&') EQUALITY)*
 *   EQUALITY   := RELATIONAL (('==' | '!=') RELATIONAL)*
 *   RELATIONAL := ADDITIVE (('<' | '>' | '<=' | '>=') ADDITIVE)*
 *   ADDITIVE   := TERM (('+' | '-') TERM)*
 *   TERM       := UNARY (('*' | '/' | '%') UNARY)*
 *   UNARY      := ('-' | '!') UNARY | POSTFIX
 *   POSTFIX    := ATOM ('[' EXPR ']')*
 *   ATOM       := NUMBER | STRING | 'true' | 'false' | ARRAYLIT | SETLIT | NIL | IDENT | CALL
 *               | '(' EXPR ')'
 *   ARRAYLIT   := '[' (EXPR (',' EXPR)*)? ']'
 *   SETLIT     := '{' (EXPR (',' EXPR)*)? '}'
 *   NIL        := 'nil' '(' TYPE ')'
 *   CALL       := IDENT '(' (CALLARG (',' CALLARG)*)? ')'
 *   CALLARG    := EXPR | '#'
 *
 * `//` line comments are supported. Besides the thirteen (possibly overloaded, see
 * BINARY_OPERATOR_OVERLOADS above) arithmetic/comparison/logical operators plus unary `!` (standard
 * C++ precedence - `||` loosest, then `&&`, then `== !=`, then `< > <= >=`, then `+ -`, then `* / %`
 * tightest, unary `-`/`!` tighter still, postfix `[]` indexing tighter still - all left-associative,
 * `()` overrides precedence), the only other way to combine or inspect values is by calling a
 * function (either a builtin, see BUILTIN_FUNCTIONS below, or another cleg function). Since
 * comparison operators now produce `bool`, `if`/`for`
 * conditions are no longer limited to bare literals - `if (x < 10) { ... }` works exactly as it
 * looks, and `&&`/`||` short-circuit exactly like C++/JS (the right operand is only evaluated if the
 * left doesn't already determine the result) - relevant since a right operand can have visible
 * effects (e.g. `randRmN`/`randRmP`'s RNG draw).
 *
 * Example - the program's own value is whatever its last top-level statement evaluates to:
 *   egr helper() {
 *       return mengerB(3, 3, "0011");
 *   }
 *   helper();
 */

import {
    BoardArgType, numArg, csvArg, zolArg, parseBoardArgToken,
    makeBoardEdge, makeBoardTriangle, makeBoardQuad, Embedding,
    type BoardArgEntry, type BoardConfig, type BoardEdge, type BoardTriangle, type BoardQuad,
    type Selector, type SelectorType, type SelectedVals, type FormSelector, type BoardModifier,
} from './types.js';
import {
    PrescribedBoard, PrescribedBoardMap, PrescribedBoardFns, product, applyModifiers,
    make, nodeInducedSubgraph, edgeInducedSubgraph,
} from './boardConfig.js';
import {
    randomlyRemove, parseNodeSelector, parseEdgeSelector, parseTriangleSelector, parseQuadSelector,
    selectNode, selectEdge, selectTriangle, selectQuad,
} from './selector.js';
import { zeroAdj } from './topology.js';

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
    /** Wraps a real shared/types.ts FormSelector (`(tri [SEL])`/`(quad [SEL])`) - built via
     * `mkFormSel(kind, [selArg])`, the `form`-modifier counterpart of `sel`/`mkSel`. Not itself a
     * `sel` - a FormSelector isn't selecting FROM an existing known-kind set, it's declaring which
     * kind (tri/quad) to look for in the first place (see FormSelector's own doc comment). */
    | { kind: 'formSel' }
    /** Wraps a real shared/types.ts BoardModifier - built via one of the modifier-constructor
     * builtins below (`rectify`, `edgeSplit`, `triangleForm`, `form`, ...). One flat type covering
     * every BoardModifier kind, the same way `sel`/`egr` are each one flat type regardless of which
     * SelectorType/PrescribedBoard they actually hold. */
    | { kind: 'mod' }
    /** Wraps a MultiSelector (defined further below, near the multiProd/msBase/msUnion/msInter/
     * msDiff builtins that are its only consumers) - a cleg-internal-only concept, unlike
     * sel/formSel/mod, none of which are exposed via shared/types.ts either but wrap something a
     * consumer OUTSIDE cleg.ts also builds/uses (a real Selector/FormSelector/BoardModifier) -
     * nothing outside this file ever builds or consumes a MultiSelector directly. */
    | { kind: 'msel' }
    | { kind: 'array'; elem: ClegType }
    | { kind: 'set'; elem: ClegType }
    /** A monomorphic function-pointer type - `(number, number) -> bool` syntax (parseType's own
     * FUNCTYPE production), fully concrete (no type variables/generics - see this file's own top
     * comment). Only ever refers to one of `program`'s own top-level FunctionDecls (see ClegValue's
     * own 'func' variant) - a builtin can't be referenced this way, since BUILTIN_FUNCTIONS' own
     * checkCall/call pair has no single static ClegType[]/ClegType signature to describe for the
     * generic/overloaded ones (`+`, `nis`, `len`, ...), and every fixed-signature one is trivially
     * callable directly by name anyway (see checkExpr/evalExpr's own CallExpr cases). */
    | { kind: 'func'; params: ClegType[]; returnType: ClegType };

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
    if (a.kind === 'func') {
        const bf = b as { kind: 'func'; params: ClegType[]; returnType: ClegType };
        return a.params.length === bf.params.length
            && a.params.every((p, i) => typeEquals(p, bf.params[i]))
            && typeEquals(a.returnType, bf.returnType);
    }
    return true;
}

export function typeToString(t: ClegType): string {
    if (t.kind === 'array') return `${typeToStringForSuffix(t.elem)}[]`;
    if (t.kind === 'set') return `${typeToStringForSuffix(t.elem)}{}`;
    if (t.kind === 'func') return `(${t.params.map(typeToString).join(', ')}) -> ${typeToString(t.returnType)}`;
    return t.kind;
}

// A func type printed directly before a `[]`/`{}` suffix needs its own extra parens (matching
// parseParenType's own grouping rule) - otherwise the suffix would silently re-parse as binding to
// the func type's own return type instead of to the func type as a whole (`(number, number) ->
// bool[]` means "returns bool[]", not "an array of these functions" - see this file's own top
// comment). Every other ClegType kind is unambiguous either way, so this only differs from
// typeToString itself for 'func'.
function typeToStringForSuffix(t: ClegType): string {
    return t.kind === 'func' ? `(${typeToString(t)})` : typeToString(t);
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
    | { kind: 'formSel'; value: FormSelector }
    | { kind: 'mod'; value: BoardModifier }
    | { kind: 'msel'; value: MultiSelector }
    | { kind: 'array'; elem: ClegType; value: ClegValue[] }
    /** A set's `value` is always deduplicated by clegSetKey (see makeClegSet) - unlike 'array',
     * where `value` may hold anything an ArrayLit/array-typed value can, `value` here never holds
     * two elements with the same key. Represented as a plain array (not a JS Set/Map) since
     * edge/tri/quad don't have reference equality, so every set operation already needs its own
     * clegSetKey-based comparison regardless of the backing container - see setUnion/etc. below. */
    | { kind: 'set'; elem: ClegType; value: ClegValue[] }
    /** A function-pointer value - a reference to one of `program`'s own top-level functions, held by
     * `name` (looked up in `funcs`/UserFuncTable again at call time - see evalExpr's own CallExpr
     * case) rather than a direct FunctionDecl reference: keeps this type free of any dependency on
     * cleg's own AST shape, which matters for a future consumer (e.g. a Selector - shared/types.ts)
     * that needs to hold "a named callback to resolve later" as plain, evaluator-agnostic data,
     * without importing anything cleg-internal. `boundArgs` has one slot per entry of the ORIGINAL
     * function's own full parameter list - `null` at every still-uninstantiated ('#') position, the
     * actual (already-evaluated) argument everywhere else - so a plain, uncalled reference (built
     * from a bare Identifier, see evalExpr's own case) is simply the all-`null` case, and a partial
     * application (`f(a, #, b)`, see CallExpr's own doc comment) is the general one; calling either
     * kind of value later interleaves the caller's own supplied arguments into the `null` slots, in
     * order (see evalExpr's own CallExpr case). `params`/`returnType` describe this VALUE's own
     * callable signature, not the original function's - for a plain reference the two coincide, but
     * a partial application's `params` is only the `#` positions' types, in order (its `returnType`
     * is always unchanged, since cleg has no currying of the return value itself). Cached here
     * (rather than re-derived on every use) purely so clegValueType can report this value's own
     * ClegType without needing a funcs-table lookup. */
    | { kind: 'func'; params: ClegType[]; returnType: ClegType; name: string; boundArgs: (ClegValue | null)[] };

function clegValueType(v: ClegValue): ClegType {
    if (v.kind === 'array') return { kind: 'array', elem: v.elem };
    if (v.kind === 'set') return { kind: 'set', elem: v.elem };
    if (v.kind === 'func') return { kind: 'func', params: v.params, returnType: v.returnType };
    return { kind: v.kind };
}

// ── Binary operators ─────────────────────────────────────────────────────────

type BinOp = '+' | '-' | '*' | '/' | '%' | '==' | '!=' | '<' | '>' | '<=' | '>=' | '&&' | '||';

/**
 * One candidate signature for a binary operator - operators are polymorphic (see `+`'s overloads
 * below), resolved by trying each operator's own overload list in order and taking the first
 * match. `match` is the single source of truth for "do these operand types satisfy this
 * overload": both checkExpr (which only needs the result `type`) and evalExpr (which only needs
 * `eval`) call it, so the two can never disagree about which overload applies to a given pair of
 * operand types. Returns null if `l`/`r` don't match this overload.
 *
 * `&&`/`||` are the one exception to evalExpr actually calling `eval` here - see logicalOverload's
 * own doc comment for why they're short-circuited by evalExpr's own BinaryExpr case instead, before
 * ever reaching this table at evaluation time (checkExpr still goes through this table exactly like
 * every other operator, since type-checking both operands doesn't depend on short-circuiting).
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

/** `string, string -> string` (concatenation) - needed alongside numberStringConcatOverload below
 * so a chain like `"a" + (n + 1) + "b"` (string, number -> string, then that result, string ->
 * string) actually typechecks end to end. */
const stringConcatOverload: BinaryOverload = {
    signature: 'string, string -> string (concatenation)',
    match: (l, r) => (l.kind === 'string' && r.kind === 'string')
        ? {
            type: { kind: 'string' },
            eval: (lv, rv) => ({
                kind: 'string',
                value: `${(lv as { value: string }).value}${(rv as { value: string }).value}`,
            }),
        }
        : null,
};

/** `number, string -> string`/`string, number -> string` (concatenation, converting the number
 * operand to a string first) - unlike repeatArrayOverload/repeatStringOverload's `*` swap (which
 * shares one meaning either way round), these two orderings just concatenate in the order the
 * operands appear (`1 + "x"` is `"1x"`, `"x" + 1` is `"x1"`), so both directions are spelled out
 * rather than sharing a single symmetric `eval`. */
const numberStringConcatOverload: BinaryOverload = {
    signature: 'number, string -> string (or string, number -> string; concatenation)',
    match: (l, r) => {
        if (l.kind === 'number' && r.kind === 'string')
            return {
                type: { kind: 'string' },
                eval: (lv, rv) => ({
                    kind: 'string',
                    value: `${(lv as { value: number }).value}${(rv as { value: string }).value}`,
                }),
            };
        if (l.kind === 'string' && r.kind === 'number')
            return {
                type: { kind: 'string' },
                eval: (lv, rv) => ({
                    kind: 'string',
                    value: `${(lv as { value: string }).value}${(rv as { value: number }).value}`,
                }),
            };
        return null;
    },
};

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

// Shared by repeatArrayOverload/repeatStringOverload below - `n` isn't statically known (it's a
// `number`-typed expression, not necessarily a literal), so "nonnegative integer" can only be
// checked once an actual value exists, at evaluation time - never by a BinaryOverload's own `match`
// (which only ever sees types, not values).
function requireRepeatCount(n: number): number {
    if (!Number.isInteger(n) || n < 0)
        throw new Error(`cleg: '*' replication count must be a nonnegative integer, got ${n}`);
    return n;
}

function repeatArrayValue(arr: { elem: ClegType; value: ClegValue[] }, count: number): ClegValue {
    const n = requireRepeatCount(count);
    const value: ClegValue[] = [];
    for (let i = 0; i < n; i++) value.push(...arr.value);
    return { kind: 'array', elem: arr.elem, value };
}

/** `T[], number -> T[]` (replication - concatenates n copies of the array), also accepted with the
 * operands swapped (`number, T[] -> T[]`) so both `arr * 3` and `3 * arr` work. */
const repeatArrayOverload: BinaryOverload = {
    signature: 'T[], number -> T[] (or number, T[] -> T[]; replication - n must be a nonnegative integer)',
    match: (l, r) => {
        if (l.kind === 'array' && r.kind === 'number')
            return {
                type: l,
                eval: (lv, rv) =>
                    repeatArrayValue(lv as { elem: ClegType; value: ClegValue[] }, (rv as { value: number }).value),
            };
        if (l.kind === 'number' && r.kind === 'array')
            return {
                type: r,
                eval: (lv, rv) =>
                    repeatArrayValue(rv as { elem: ClegType; value: ClegValue[] }, (lv as { value: number }).value),
            };
        return null;
    },
};

function repeatStringValue(s: string, count: number): ClegValue {
    return { kind: 'string', value: s.repeat(requireRepeatCount(count)) };
}

/** `string, number -> string` (replication), also accepted with the operands swapped (`number,
 * string -> string`) so both `s * 3` and `3 * s` work. */
const repeatStringOverload: BinaryOverload = {
    signature: 'string, number -> string (or number, string -> string; replication - n must be a nonnegative integer)',
    match: (l, r) => {
        if (l.kind === 'string' && r.kind === 'number')
            return {
                type: { kind: 'string' },
                eval: (lv, rv) => repeatStringValue((lv as { value: string }).value, (rv as { value: number }).value),
            };
        if (l.kind === 'number' && r.kind === 'string')
            return {
                type: { kind: 'string' },
                eval: (lv, rv) => repeatStringValue((rv as { value: string }).value, (lv as { value: number }).value),
            };
        return null;
    },
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

/** `bool, bool -> bool` for `&&`/`||` - only used by checkExpr (both operands still need typing
 * regardless of runtime short-circuiting) and as documentation of the operator's own signature;
 * evalExpr's own BinaryExpr case short-circuits these two operators itself, before ever consulting
 * BINARY_OPERATOR_OVERLOADS, so `eval` here is never actually reached at runtime - it's still a
 * genuinely correct (just non-short-circuiting) implementation, kept for interface consistency with
 * every other overload rather than a stub. */
function logicalOverload(compute: (a: boolean, b: boolean) => boolean): BinaryOverload {
    return {
        signature: 'bool, bool -> bool',
        match: (l, r) => (l.kind === 'bool' && r.kind === 'bool')
            ? {
                type: { kind: 'bool' },
                eval: (lv, rv) => ({
                    kind: 'bool',
                    value: compute((lv as { value: boolean }).value, (rv as { value: boolean }).value),
                }),
            }
            : null,
    };
}

const BINARY_OPERATOR_OVERLOADS: Record<BinOp, BinaryOverload[]> = {
    '+': [numberOverload((a, b) => a + b), stringConcatOverload, numberStringConcatOverload, arrayConcatOverload, setUnionOverload],
    '-': [numberOverload((a, b) => a - b), setDiffOverload],
    '*': [numberOverload((a, b) => a * b), setIntersectOverload, repeatArrayOverload, repeatStringOverload],
    '/': [numberOverload((a, b) => a / b)],
    '%': [numberOverload((a, b) => a % b)],
    '==': [comparisonOverload('number', (a, b) => a === b), comparisonOverload('bool', (a, b) => a === b)],
    '!=': [comparisonOverload('number', (a, b) => a !== b), comparisonOverload('bool', (a, b) => a !== b)],
    '<': [comparisonOverload('number', (a, b) => a < b), comparisonOverload('bool', (a, b) => a < b)],
    '>': [comparisonOverload('number', (a, b) => a > b), comparisonOverload('bool', (a, b) => a > b)],
    '<=': [comparisonOverload('number', (a, b) => a <= b), comparisonOverload('bool', (a, b) => a <= b)],
    '>=': [comparisonOverload('number', (a, b) => a >= b), comparisonOverload('bool', (a, b) => a >= b)],
    '&&': [logicalOverload((a, b) => a && b)],
    '||': [logicalOverload((a, b) => a || b)],
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

export type Stmt =
    | VarDecl | AssignStmt | IfStmt | ForStmt | WhileStmt | BreakStmt | ContinueStmt | ReturnStmt
    | ExprStmt | Block;

/** Declares and initializes a new local; see AssignStmt below for mutating an already-declared one. */
export interface VarDecl { kind: 'VarDecl'; type: ClegType; name: string; init: Expr; }
/** Reassigns an already-declared local (`x = expr;`), or - if `indices` is nonempty - mutates one
 * element of an already-declared array in place (`arr[i] = expr;`, `arr[i][j] = expr;`, ...): each
 * entry of `indices` must be a `number`, and `name`'s own declared type must be an array nested at
 * least `indices.length` deep (checkStmt's own AssignStmt case walks one `array` level per index,
 * rejecting anything that runs out of array levels before `indices` does - see this file's own top
 * comment on why arrays are value types, not references, for what this mutation can and can't affect
 * through an alias). With no indices, mutates the binding in whichever enclosing scope originally
 * declared `name` (see evaluation's setValue), it does not shadow it with a new one in the current
 * block. `name` must already be declared, and the assigned value's type must exactly match either
 * `name`'s own declared type (no indices) or the array-element type `indices.length` levels down
 * (with indices) - there is no way to introduce a new binding via assignment, only VarDecl does
 * that. */
export interface AssignStmt { kind: 'AssignStmt'; name: string; indices: Expr[]; value: Expr; }
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
/** A plain pretest loop: `while (EXPR) BLOCK` - equivalent to `for (; EXPR; ) BLOCK`, just spelled
 * directly rather than requiring the empty init/update clauses. Unlike ForStmt, there's no separate
 * "loop header scope" - with no init clause to declare a loop-scoped variable, `cond` is checked/
 * evaluated in the same scope the `while` itself appears in, and `body` gets its own further-nested
 * scope each iteration per BLOCK's usual rule (see checkStmt's/evalStmt's own WhileStmt cases). */
export interface WhileStmt { kind: 'WhileStmt'; cond: Expr; body: Block; }
/** Exits the innermost enclosing ForStmt/WhileStmt immediately (checkStmt rejects one outside a
 * loop - see its own `inLoop` doc comment). Mirrors real C++'s own `break;`. */
export interface BreakStmt { kind: 'BreakStmt'; }
/** Skips the rest of the innermost enclosing ForStmt/WhileStmt's current iteration and moves on to
 * the next one - for a ForStmt this still runs `update` before re-checking `cond`, exactly like real
 * C++'s own `continue;` (see evalStmt's own ForStmt/WhileStmt cases); checkStmt rejects one outside
 * a loop, same as BreakStmt. */
export interface ContinueStmt { kind: 'ContinueStmt'; }
/** Every function must return a value (there is no `void`), so unlike C++ this is never bare. */
export interface ReturnStmt { kind: 'ReturnStmt'; value: Expr; }
export interface ExprStmt { kind: 'ExprStmt'; expr: Expr; }
export interface Block { kind: 'Block'; stmts: Stmt[]; }

export type Expr =
    | NumberLit | StringLit | BoolLit | ArrayLit | SetLit | Identifier | CallExpr | BinaryExpr
    | UnaryExpr | NilExpr | IndexExpr | HoleExpr;

export interface NumberLit { kind: 'NumberLit'; value: number; }
export interface StringLit { kind: 'StringLit'; value: string; }
export interface BoolLit { kind: 'BoolLit'; value: boolean; }
/** Simplification: an empty `[]` has no way to say what its element type is, so it's rejected by
 * typecheckCleg (see checkExpr's own ArrayLit case) rather than silently guessed at - `nil(TYPE)`
 * below is the escape hatch: `nil(number)` is an empty `number[]`, exactly like `[]` would be if its
 * element type could somehow be inferred. */
export interface ArrayLit { kind: 'ArrayLit'; elements: Expr[]; }
/** Same empty-literal simplification as ArrayLit above, plus its own restriction: the inferred
 * element type must be one of SET_ELEM_KINDS (see checkExpr's own SetLit case) - a set of `egr`,
 * of arrays, or of sets is rejected here even though the individual elements typecheck fine on
 * their own. Duplicate elements (by clegSetKey) collapse to one - see makeClegSet. */
export interface SetLit { kind: 'SetLit'; elements: Expr[]; }
export interface Identifier { kind: 'Identifier'; name: string; }
/** `callee` names either a builtin (BUILTIN_FUNCTIONS below), another function declared in the same
 * program, or (see checkExpr/evalExpr's own CallExpr cases) a local variable holding a `func` value
 * - one flat namespace, see typecheckCleg. If any of `args` is a HoleExpr (`#`), this is a PARTIAL
 * APPLICATION rather than an ordinary call: `f(a, #, b)` evaluates to a function-pointer value
 * closing over the non-`#` arguments (evaluated once, right here), whose own parameter list is just
 * the `#` positions' types, in order - calling that value later supplies only those, interleaved
 * back into their original positions (see ClegValue's own 'func'/`boundArgs` doc comment). `callee`
 * here is either a bare reference to one of `program`'s own top-level functions, or an existing
 * local variable already holding a `func` value - a plain pointer, or itself already a partial
 * application, in which case this narrows its own remaining open positions further rather than
 * starting over (see checkPartialApplication/mergeBoundArgs) - never a builtin (no single fixed
 * signature to close over for the generic/overloaded ones). */
export interface CallExpr { kind: 'CallExpr'; callee: string; args: Expr[]; }
/** One of the five arithmetic operators, six comparison operators (`== != < > <= >=`), or two
 * logical operators (`&& ||`, short-circuiting - see evalExpr's own BinaryExpr case), each possibly
 * overloaded beyond a single fixed signature - see BINARY_OPERATOR_OVERLOADS and checkExpr's own
 * BinaryExpr case. `(...)` grouping isn't its own AST node - parseAtom just returns the
 * parenthesized subexpression directly, so precedence is fully resolved by the time the AST exists. */
export interface BinaryExpr {
    kind: 'BinaryExpr';
    op: '+' | '-' | '*' | '/' | '%' | '==' | '!=' | '<' | '>' | '<=' | '>=' | '&&' | '||';
    left: Expr;
    right: Expr;
}
/** Unary negation (`-x`, `-f()` - also how a negative number literal is written now: `-3` parses as
 * UnaryExpr wrapping NumberLit(3), the lexer itself never produces a signed number) or unary logical
 * negation (`!x`, requires a `bool` operand - see checkExpr's own UnaryExpr case). */
export interface UnaryExpr { kind: 'UnaryExpr'; op: '-' | '!'; operand: Expr; }
/** `nil(TYPE)` - an empty array whose element type is TYPE, e.g. `nil(number)` is an empty
 * `number[]`, `nil(msel)` an empty `msel[]`, `nil(number[])` an empty `number[][]` - the escape
 * hatch for ArrayLit's own empty-literal simplification (see its own doc comment): unlike every
 * other Expr, `TYPE` is a real ClegType (parsed via parseType, not parseExpr), not itself an Expr. */
export interface NilExpr { kind: 'NilExpr'; type: ClegType; }
/** Array indexing (`arr[i]`) - `array` must be array-typed (never set-typed - see this file's own
 * top comment on why), `index` a `number`, checked to be a nonnegative in-bounds integer only at
 * evaluation time (not statically knowable - see checkExpr's own IndexExpr case vs evalExpr's own).
 * Parses as a postfix operator (see POSTFIX in this file's own grammar comment) - binds tighter than
 * unary `-`/`!`, and chains naturally for nested arrays (`arr[i][j]`). */
export interface IndexExpr { kind: 'IndexExpr'; array: Expr; index: Expr; }
/** `#` - an uninstantiated-argument placeholder, valid only inside a CallExpr's own `args` (the
 * parser never produces one anywhere else - see parseAtom's own CallExpr branch). A CallExpr with
 * one or more HoleExpr args is a partial application, not an ordinary call - see CallExpr's own doc
 * comment for the restriction (a plain top-level function name only) and ClegValue's own 'func'
 * variant for how the resulting closure is represented. */
export interface HoleExpr { kind: 'HoleExpr'; }

/**
 * A top-level statement: a VARDECL, an ASSIGNSTMT, or a bare EXPRSTMT - unlike inside a function
 * BLOCK, never an IFSTMT/FORSTMT/WHILESTMT/RETURNSTMT (there is no branching, looping, or `return`
 * at top level - see ClegProgram's own doc comment on why). checkStmt/evalStmt (which handle every
 * Stmt kind, not just these three) still process a TopStmt correctly, since TopStmt is a subset of
 * Stmt - their own ForStmt/WhileStmt/IfStmt/ReturnStmt cases are simply never reached from top level.
 */
export type TopStmt = VarDecl | AssignStmt | ExprStmt;

/**
 * A whole cleg program: its own top-level function declarations (order-independent - functions may
 * call each other regardless of declaration order, forward references are fine, and may recurse
 * directly or mutually) plus its own top-level statements, `stmts` (order-DOES-matter - see
 * runCleg's own doc comment for why). There is no `main` and no other designated entry-point
 * function - a cleg program's own top-level statements, not any one function, are what actually run.
 * A VARDECL/ASSIGNSTMT here shares one flat scope across every TopStmt in `stmts` (so a variable
 * declared by one is visible to every one after it), but that scope is entirely separate from any
 * function's own - cleg has no closures, so a function body can never see a top-level variable,
 * only its own parameters (see typecheckCleg/runClegProgram's own top-level `env`). The LAST entry
 * in `stmts` must be an EXPRSTMT (checked by typecheckCleg) - that is the program's own overall
 * value, so a program can't usefully end on a bare declaration/assignment with nothing left to
 * produce a result.
 */
export interface ClegProgram { kind: 'ClegProgram'; functions: FunctionDecl[]; stmts: TopStmt[]; }

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
    'egr', 'number', 'string', 'bool', 'edge', 'tri', 'quad', 'sel', 'formSel', 'mod', 'msel',
]);

function parseType(c: TokenCursor): ClegType {
    let type: ClegType;
    if (c.isPunct('(')) {
        type = parseParenType(c);
    } else {
        const base = c.expectIdent();
        if (!TYPE_KEYWORDS.has(base))
            throw new Error(`cleg: expected a type (egr/number/string/bool/edge/tri/quad/sel/formSel/mod), got '${base}'`);
        type = {
            kind: base as 'egr' | 'number' | 'string' | 'bool' | 'edge' | 'tri' | 'quad' | 'sel' | 'formSel' | 'mod',
        };
        if (c.isPunct('{')) {
            if (!SET_ELEM_KINDS.has(base))
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
 * Parses a whole cleg program (see this file's own top comment for the grammar) - throws if
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
// own AssignStmt cases.
function unparseAssignTarget(stmt: AssignStmt, indent: number): string {
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
    /** `funcs` is only ever needed by a builtin that itself calls a `func`-typed argument back
     * (e.g. subHcublat's own `cond`, via callUserFunction) - every other builtin's own `call` simply
     * ignores it, which TypeScript allows (a function declared with fewer parameters than a call
     * signature requires is still a valid implementation of it). */
    call: (args: ClegValue[], funcs: UserFuncTable) => ClegValue;
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
// after PrescribedBoardMap's own cleg-name field (e.g. "mengerB", "rectB" - already carrying its own
// trailing "B", so no name-mangling happens here) - built generically from that existing table
// (rather than one hand-written cleg function per board type) so this list never drifts out of sync
// with it. The "B" suffix (baked into PrescribedBoardMap itself) keeps every one of these names
// clear of TYPE_KEYWORDS by construction (rather than special-casing the one existing collision,
// "tri" vs. the `tri` triangle-value type - a future board name could collide too, e.g. "mod" or
// "egr").
for (const [pbKey, [argTypes, clegName]] of
    Object.entries(PrescribedBoardMap) as [string, [BoardArgType[], string, string, string]][]) {
    const pb = Number(pbKey) as PrescribedBoard;
    BUILTIN_FUNCTIONS[clegName] = {
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

// `has(x, e)`: whether `x` (a `T[]` or `T{}`) contains `e` (a `T`), as a `bool` - like `len`, its
// result depends on the actual argument types (here, argument 2's own required type, taken from
// argument 1's element type), hence the hand-written checkCall/call pair. `T` is restricted to
// SET_ELEM_KINDS (number/string/bool/edge/tri/quad) for BOTH `T[]` and `T{}` - even though an
// array's own element type is normally unrestricted, nothing outside SET_ELEM_KINDS has a defined
// equality in this language (e.g. `egr`: "no natural equality/hashing for a whole board", see this
// file's own top comment on SET_ELEM_KINDS), so `has` can't be given a well-defined meaning for one
// either. Compares by clegSetKey, the same equality every set operation already uses.
function hasCheckCall(callee: string, argTypes: ClegType[]): ClegType {
    if (argTypes.length !== 2)
        throw new Error(`cleg: '${callee}' expects 2 argument(s), got ${argTypes.length}`);
    if (argTypes[0].kind !== 'array' && argTypes[0].kind !== 'set')
        throw new Error(
            `cleg: '${callee}' argument 1: expected an array or set, got ${typeToString(argTypes[0])}`);
    const elem = (argTypes[0] as { elem: ClegType }).elem;
    if (!SET_ELEM_KINDS.has(elem.kind))
        throw new Error(
            `cleg: '${callee}' argument 1: element type ${typeToString(elem)} has no defined equality - only ` +
            `number/string/bool/edge/tri/quad elements are supported`);
    if (!typeEquals(argTypes[1], elem))
        throw new Error(
            `cleg: '${callee}' argument 2: expected ${typeToString(elem)} (the element type of argument 1), got ` +
            `${typeToString(argTypes[1])}`);
    return { kind: 'bool' };
}
BUILTIN_FUNCTIONS['has'] = {
    checkCall: hasCheckCall,
    call(args) {
        const container = args[0] as { kind: 'array' | 'set'; elem: ClegType; value: ClegValue[] };
        const key = clegSetKey(args[1]);
        return { kind: 'bool', value: container.value.some(v => clegSetKey(v) === key) };
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

// The ClegType 'kind' a set of SelectorType `k` is made of - 'node' selections are plain numbers
// (node indices), the other three match their own SelectorType name exactly. Shared by
// resolveSelectorArg/mkSel below (both need to validate/convert a `set`-typed selector argument)
// and Selector's own 'raw' variant (shared/types.ts), which this builds.
const SELECTOR_SET_ELEM_KIND: Record<SelectorType, ClegType['kind']> = {
    node: 'number', edge: 'edge', tri: 'tri', quad: 'quad',
};

// Unwraps a `set` ClegValue's own elements into the matching SelectedVals branch - `kind` must
// already be known to match the set's own elem type (checked by resolveSelectorArg's caller before
// this runs). 'node' collects into a real Set<number> (numbers have genuine equality); the other
// three stay plain arrays, matching SelectedVals' own doc comment on why.
function setValueToSelectedVals(kind: SelectorType, values: ClegValue[]): SelectedVals {
    switch (kind) {
        case 'node': return { kind: 'node', value: new Set(values.map(v => (v as { value: number }).value)) };
        case 'edge': return { kind: 'edge', value: values.map(v => (v as { value: BoardEdge }).value) };
        case 'tri': return { kind: 'tri', value: values.map(v => (v as { value: BoardTriangle }).value) };
        case 'quad': return { kind: 'quad', value: values.map(v => (v as { value: BoardQuad }).value) };
    }
}

// Resolves a nis/eis/triangleForm/quadForm-style "selector-like" argument into a real Selector -
// a `sel` value (its actual kind checked against `wantKind` at runtime, since 'sel' carries no kind
// at the type level - see ClegType's own 'sel' doc comment), a `string` (parsed via `parseFn`,
// following shared/selector.ts's own grammar exactly), or a `set` (of the ClegType matching
// `wantKind` - see SELECTOR_SET_ELEM_KIND - wrapped directly into a `raw` Selector, no parsing
// involved). Shared by every builtin that accepts this shape, so the "string, sel, or set - kind-
// checked" logic exists in exactly one place.
function resolveSelectorArg(
    callee: string, arg: ClegValue, wantKind: SelectorType, parseFn: (s: string) => Selector,
): Selector {
    if (arg.kind === 'string') return parseFn(arg.value);
    if (arg.kind === 'set') {
        if (arg.elem.kind !== SELECTOR_SET_ELEM_KIND[wantKind])
            throw new Error(
                `cleg: '${callee}' expects a ${wantKind} selector, got a set of ${typeToString(arg.elem)}`);
        return { op: 'raw', type: wantKind, items: setValueToSelectedVals(wantKind, arg.value) };
    }
    const sel = arg as { kind: 'sel'; selType: SelectorType; value: Selector };
    if (sel.selType !== wantKind)
        throw new Error(`cleg: '${callee}' expects a ${wantKind} selector, got a '${sel.selType}' selector`);
    return sel.value;
}

const NUMBER_TYPE: ClegType = { kind: 'number' };
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

// `mkSel(kind, X)`: builds a selector of the given `kind` ("node"/"edge"/"tri"/"quad") from `X` - a
// `string` (parsed via the matching real parse*Selector function above, following
// shared/selector.ts's own grammar/semantics exactly, including its own error messages on a
// malformed string) or a `set` (of the ClegType matching `kind` - see SELECTOR_SET_ELEM_KIND -
// wrapped directly into a `raw` Selector, no parsing involved), resolved via resolveSelectorArg
// exactly like nis/eis/etc below, once `kind` is known. Hand-written checkCall (rather than
// fixedSignature(...)) since `X`'s own accepted type isn't just one fixed ClegType, and `sel` itself
// isn't parameterized by kind at the type level (see ClegType's own 'sel' doc comment) - `kind` is
// only validated/dispatched on at call time, not check time.
function mkSelCheckCall(callee: string, argTypes: ClegType[]): ClegType {
    if (argTypes.length !== 2)
        throw new Error(`cleg: '${callee}' expects 2 argument(s), got ${argTypes.length}`);
    if (argTypes[0].kind !== 'string')
        throw new Error(`cleg: '${callee}' argument 1: expected string, got ${typeToString(argTypes[0])}`);
    if (argTypes[1].kind !== 'string' && argTypes[1].kind !== 'set')
        throw new Error(`cleg: '${callee}' argument 2: expected string or set, got ${typeToString(argTypes[1])}`);
    return { kind: 'sel' };
}
BUILTIN_FUNCTIONS['mkSel'] = {
    checkCall: mkSelCheckCall,
    call: ([kindVal, arg]) => {
        const kind = (kindVal as { value: string }).value as SelectorType;
        const parse = SELECTOR_PARSERS[kind];
        if (!parse) throw new Error(`cleg: mkSel: unknown selector kind '${kind}' - expected node/edge/tri/quad`);
        return { kind: 'sel', selType: kind, value: resolveSelectorArg('mkSel', arg, kind, parse) };
    },
};

const MOD_TYPE: ClegType = { kind: 'mod' };
const FORMSEL_TYPE: ClegType = { kind: 'formSel' };

// `mkFormSel(kind, [selArg])`: builds a real shared/types.ts FormSelector - `kind` is
// "tri"/"quad" (validated/dispatched at call time, same convention as mkSel's own `kind`), and the
// optional `selArg` (a `sel`, `string`, or `set`, resolved via resolveSelectorArg - see FormSelector's own
// optional `sel?: Selector` field) restricts which tri/quads of that kind qualify (omitted: every
// one found). Variable-arity (1 or 2 args) rather than fixedSignature(...), to mirror that
// optionality exactly.
function mkFormSelCheckCall(callee: string, argTypes: ClegType[]): ClegType {
    if (argTypes.length !== 1 && argTypes.length !== 2)
        throw new Error(`cleg: '${callee}' expects 1 or 2 argument(s), got ${argTypes.length}`);
    if (argTypes[0].kind !== 'string')
        throw new Error(`cleg: '${callee}' argument 1: expected string, got ${typeToString(argTypes[0])}`);
    if (argTypes.length === 2 && argTypes[1].kind !== 'sel' && argTypes[1].kind !== 'string' && argTypes[1].kind !== 'set')
        throw new Error(`cleg: '${callee}' argument 2: expected sel, string, or set, got ${typeToString(argTypes[1])}`);
    return FORMSEL_TYPE;
}
BUILTIN_FUNCTIONS['mkFormSel'] = {
    checkCall: mkFormSelCheckCall,
    call: (args) => {
        const kind = (args[0] as { value: string }).value;
        if (kind !== 'tri' && kind !== 'quad')
            throw new Error(`cleg: mkFormSel: unknown form-selector kind '${kind}' - expected tri/quad`);
        if (args.length === 1) return { kind: 'formSel', value: { kind } };
        const sel = resolveSelectorArg('mkFormSel', args[1], kind, SELECTOR_PARSERS[kind]);
        return { kind: 'formSel', value: { kind, sel } };
    },
};

// `rectify`/`globalCentralize`/`quadOctarize`: zero-argument BoardModifier constructors, one per
// shared/types.ts's own like-named BoardModifier kind - build the value directly (`{ kind: 'X' }`)
// rather than calling shared/boardConfig.ts's own rectify()/globalCentralize()/quadOctarize()
// (those APPLY a modifier to a board immediately; these instead build the modifier value itself,
// to be applied later - see this file's own top comment on the `mod` type).
BUILTIN_FUNCTIONS['rectify'] = {
    checkCall: fixedSignature([], MOD_TYPE),
    call: () => ({ kind: 'mod', value: { kind: 'Rectify' } }),
};
BUILTIN_FUNCTIONS['globalCentralize'] = {
    checkCall: fixedSignature([], MOD_TYPE),
    call: () => ({ kind: 'mod', value: { kind: 'GlobalCentralize' } }),
};
BUILTIN_FUNCTIONS['quadOctarize'] = {
    checkCall: fixedSignature([], MOD_TYPE),
    call: () => ({ kind: 'mod', value: { kind: 'QuadOctarize' } }),
};

// `edgeSplit`/`mergeClose`/`scale`: one-number-argument BoardModifier constructors, same
// "build the value, don't apply it" rationale as rectify/globalCentralize/quadOctarize above.
BUILTIN_FUNCTIONS['edgeSplit'] = {
    checkCall: fixedSignature([NUMBER_TYPE], MOD_TYPE),
    call: ([n]) => ({ kind: 'mod', value: { kind: 'EdgeSplit', splitN: (n as { value: number }).value } }),
};
BUILTIN_FUNCTIONS['mergeClose'] = {
    checkCall: fixedSignature([NUMBER_TYPE], MOD_TYPE),
    call: ([d]) => ({ kind: 'mod', value: { kind: 'MergeClose', dist: (d as { value: number }).value } }),
};
BUILTIN_FUNCTIONS['scale'] = {
    checkCall: fixedSignature([NUMBER_TYPE], MOD_TYPE),
    call: ([f]) => ({ kind: 'mod', value: { kind: 'Scale', factor: (f as { value: number }).value } }),
};

// `nis(X)`/`eis(X)`: build a NodeInducedSubgraph/EdgeInducedSubgraph BoardModifier - `X` (a `sel` or
// `string`, resolved via resolveSelectorArg above) becomes that modifier's own `sel: Selector`
// field. Same family as triangleForm/quadForm just below (construct the value, don't apply it), but
// simpler - no `w`, and the selector is mandatory rather than optional (NodeInducedSubgraph/
// EdgeInducedSubgraph's own `sel` field isn't `?`). Unlike a `number{}`/`edge{}` set (nis/eis's own
// earlier, since-removed third accepted shape), there's no Selector grammar production for "exactly
// this literal set of nodes/edges" (see this file's own top comment history), so a real
// NodeInducedSubgraph/EdgeInducedSubgraph modifier value can only ever hold a genuine Selector.
function inducedSubgraphModCheckCall(callee: string, argTypes: ClegType[]): ClegType {
    if (argTypes.length !== 1)
        throw new Error(`cleg: '${callee}' expects 1 argument(s), got ${argTypes.length}`);
    if (argTypes[0].kind !== 'sel' && argTypes[0].kind !== 'string' && argTypes[0].kind !== 'set')
        throw new Error(`cleg: '${callee}' argument 1: expected sel, string, or set, got ${typeToString(argTypes[0])}`);
    return MOD_TYPE;
}
BUILTIN_FUNCTIONS['nis'] = {
    checkCall: inducedSubgraphModCheckCall,
    call: ([arg]) => (
        { kind: 'mod', value: { kind: 'NodeInducedSubgraph', sel: resolveSelectorArg('nis', arg, 'node', parseNodeSelector) } }
    ),
};
BUILTIN_FUNCTIONS['eis'] = {
    checkCall: inducedSubgraphModCheckCall,
    call: ([arg]) => (
        { kind: 'mod', value: { kind: 'EdgeInducedSubgraph', sel: resolveSelectorArg('eis', arg, 'edge', parseEdgeSelector) } }
    ),
};

const EDGE_TYPE: ClegType = { kind: 'edge' };
const TRI_TYPE: ClegType = { kind: 'tri' };
const QUAD_TYPE: ClegType = { kind: 'quad' };

// `selectNode(X, bc)`/`selectEdge(X, bc)`/`selectTriangle(X, bc)`/`selectQuad(X, bc)`: evaluates a
// selector (`X`, a `sel`, `string`, or `set` - resolved via resolveSelectorArg above, same convention as
// nis/eis/triangleForm/quadForm) against a real board `bc`, returning the exact set of
// nodes/edges/triangles/quads it selects (shared/selector.ts's own selectNode/selectEdge/
// selectTriangle/selectQuad do the actual work) - unlike nis/eis (which build a
// NodeInducedSubgraph/EdgeInducedSubgraph BoardModifier to apply LATER via modify(...)), this
// evaluates the selector immediately against a real board and hands back the result as an ordinary
// cleg set, so a program can inspect/combine/count (len) it directly. One builtin per selector kind,
// rather than a single overloaded name, because `sel`'s own ClegType carries no kind at the type
// level (see this file's own top comment) - checkCall only ever sees argument TYPES, never their
// runtime values, so there's no way for one `select(X, bc)` to know ahead of time which of
// number{}/edge{}/tri{}/quad{} it should return.
function selectSetCheckCall(elemType: ClegType): BuiltinFunction['checkCall'] {
    return (callee, argTypes) => {
        if (argTypes.length !== 2)
            throw new Error(`cleg: '${callee}' expects 2 argument(s), got ${argTypes.length}`);
        if (argTypes[0].kind !== 'sel' && argTypes[0].kind !== 'string' && argTypes[0].kind !== 'set')
            throw new Error(`cleg: '${callee}' argument 1: expected sel, string, or set, got ${typeToString(argTypes[0])}`);
        if (argTypes[1].kind !== 'egr')
            throw new Error(`cleg: '${callee}' argument 2: expected egr, got ${typeToString(argTypes[1])}`);
        return { kind: 'set', elem: elemType };
    };
}
BUILTIN_FUNCTIONS['selectNode'] = {
    checkCall: selectSetCheckCall(NUMBER_TYPE),
    call: ([arg, egrVal]) => {
        const sel = resolveSelectorArg('selectNode', arg, 'node', parseNodeSelector);
        const bc = (egrVal as { value: BoardConfig }).value;
        const nodes = [...selectNode(bc.adj, bc.emb.pos, sel)].map((n): ClegValue => ({ kind: 'number', value: n }));
        return makeClegSet(NUMBER_TYPE, nodes);
    },
};
BUILTIN_FUNCTIONS['selectEdge'] = {
    checkCall: selectSetCheckCall(EDGE_TYPE),
    call: ([arg, egrVal]) => {
        const sel = resolveSelectorArg('selectEdge', arg, 'edge', parseEdgeSelector);
        const bc = (egrVal as { value: BoardConfig }).value;
        const edges = selectEdge(bc.adj, bc.emb.pos, sel).map((e): ClegValue => ({ kind: 'edge', value: e }));
        return makeClegSet(EDGE_TYPE, edges);
    },
};
BUILTIN_FUNCTIONS['selectTriangle'] = {
    checkCall: selectSetCheckCall(TRI_TYPE),
    call: ([arg, egrVal]) => {
        const sel = resolveSelectorArg('selectTriangle', arg, 'tri', parseTriangleSelector);
        const bc = (egrVal as { value: BoardConfig }).value;
        const tris = selectTriangle(bc.adj, bc.emb.pos, sel).map((t): ClegValue => ({ kind: 'tri', value: t }));
        return makeClegSet(TRI_TYPE, tris);
    },
};
BUILTIN_FUNCTIONS['selectQuad'] = {
    checkCall: selectSetCheckCall(QUAD_TYPE),
    call: ([arg, egrVal]) => {
        const sel = resolveSelectorArg('selectQuad', arg, 'quad', parseQuadSelector);
        const bc = (egrVal as { value: BoardConfig }).value;
        const quads = selectQuad(bc.adj, bc.emb.pos, sel).map((q): ClegValue => ({ kind: 'quad', value: q }));
        return makeClegSet(QUAD_TYPE, quads);
    },
};

// `triangleForm(w, [selArg])`/`quadForm(w, [selArg])`: builds a TriangleForm/QuadForm
// BoardModifier - `selArg` (a `sel`, `string`, or `set`, resolved via resolveSelectorArg) restricts which
// triangles/quads get replaced, mirroring TriangleForm/QuadForm's own optional `sel?: Selector`
// field exactly - omitted, every one found gets replaced. Variable-arity like mkFormSel above.
function formModCheckCall(callee: string, argTypes: ClegType[]): ClegType {
    if (argTypes.length !== 1 && argTypes.length !== 2)
        throw new Error(`cleg: '${callee}' expects 1 or 2 argument(s), got ${argTypes.length}`);
    if (argTypes[0].kind !== 'number')
        throw new Error(`cleg: '${callee}' argument 1: expected number, got ${typeToString(argTypes[0])}`);
    if (argTypes.length === 2 && argTypes[1].kind !== 'sel' && argTypes[1].kind !== 'string' && argTypes[1].kind !== 'set')
        throw new Error(`cleg: '${callee}' argument 2: expected sel, string, or set, got ${typeToString(argTypes[1])}`);
    return MOD_TYPE;
}
BUILTIN_FUNCTIONS['triangleForm'] = {
    checkCall: formModCheckCall,
    call: (args) => {
        const w = (args[0] as { value: number }).value;
        if (args.length === 1) return { kind: 'mod', value: { kind: 'TriangleForm', w } };
        const sel = resolveSelectorArg('triangleForm', args[1], 'tri', parseTriangleSelector);
        return { kind: 'mod', value: { kind: 'TriangleForm', w, sel } };
    },
};
BUILTIN_FUNCTIONS['quadForm'] = {
    checkCall: formModCheckCall,
    call: (args) => {
        const w = (args[0] as { value: number }).value;
        if (args.length === 1) return { kind: 'mod', value: { kind: 'QuadForm', w } };
        const sel = resolveSelectorArg('quadForm', args[1], 'quad', parseQuadSelector);
        return { kind: 'mod', value: { kind: 'QuadForm', w, sel } };
    },
};

// `form(w, ...sels)`: builds a Form BoardModifier - `w` (the shared lattice width) followed by one
// or more `formSel` arguments, mirroring genericForm's own (bc, w, sels) signature and
// parseModifier's own `assert(sels.length >= 1, ...)` requirement (see its 'form' case).
function formCheckCall(callee: string, argTypes: ClegType[]): ClegType {
    if (argTypes.length < 2)
        throw new Error(`cleg: '${callee}' expects at least 2 argument(s) (w, and >= 1 formSel), got ${argTypes.length}`);
    if (argTypes[0].kind !== 'number')
        throw new Error(`cleg: '${callee}' argument 1: expected number, got ${typeToString(argTypes[0])}`);
    for (let i = 1; i < argTypes.length; i++)
        if (argTypes[i].kind !== 'formSel')
            throw new Error(`cleg: '${callee}' argument ${i + 1}: expected formSel, got ${typeToString(argTypes[i])}`);
    return MOD_TYPE;
}
BUILTIN_FUNCTIONS['form'] = {
    checkCall: formCheckCall,
    call: (args) => {
        const w = (args[0] as { value: number }).value;
        const sels = args.slice(1).map(a => (a as { value: FormSelector }).value);
        return { kind: 'mod', value: { kind: 'Form', w, sels } };
    },
};

// `modify(mods, bc)`: applies every modifier in `mods`, in order, to `bc` - shared/boardConfig.ts's
// own applyModifiers(), the one builtin that actually turns a list of `mod` values into a
// transformed board (every rectify/edgeSplit/.../form/nis/eis builtin above only constructs an
// opaque `mod` value, never applies one). `mods` is a plain array (`mod[]`), not a set - modifiers
// are meaningfully ordered and can repeat (e.g. `[scale(2), scale(2)]` is not the same as one
// `scale(4)`), neither of which a set would preserve.
BUILTIN_FUNCTIONS['modify'] = {
    checkCall: fixedSignature([{ kind: 'array', elem: MOD_TYPE }, EGR_TYPE], EGR_TYPE),
    call: ([modsVal, egrVal]) => {
        const mods = (modsVal as { value: ClegValue[] }).value.map(v => (v as { value: BoardModifier }).value);
        const bc = (egrVal as { value: BoardConfig }).value;
        return { kind: 'egr', value: applyModifiers(bc, mods) };
    },
};

// ── multiProd: N-ary Cartesian board product, restricted by a MultiSelector ────

// Denotes a subset of the FULL N-ary Cartesian product's own node space (fixed once per multiProd
// call, from `boards`' own ORIGINAL, unrestricted sizes - see FullProductIndex below) - cleg-
// internal only (see ClegType's own 'msel' doc comment), built by msAll/msBase/msUnion/msInter/
// msDiff and consumed only by multiProd (evalMultiSelector). `base`'s own `sel` may syntactically be
// any Selector - msBase itself doesn't check its SelectorType at all, only multiProd's own
// evaluation does, once it actually needs to restrict a specific board (see
// restrictBoardBySelector) - so a tri/quad selector parses/builds fine as an msBase argument and
// only fails later, when actually evaluated. `all` is every original index, unrestricted - the same
// universal set `msInter(nil(msel))` already denotes (the usual absorbing-element identity for an
// empty intersection fold), just spelled directly rather than via that idiom.
type MultiSelector =
    | { op: 'all' }
    | { op: 'base'; number: number; sel: Selector }
    | { op: 'union' | 'inter'; items: MultiSelector[] }
    | { op: 'diff'; a: MultiSelector; b: MultiSelector };

const MSEL_TYPE: ClegType = { kind: 'msel' };

// Inverse of SELECTOR_SET_ELEM_KIND above (ClegType elem kind -> SelectorType) - used only by
// resolveAnyKindSelectorArg below, which (unlike resolveSelectorArg) has no fixed wantKind of its
// own to check a set's element type against, so it has to recover a SelectorType FROM the set's own
// element kind instead.
const SELECTOR_TYPE_BY_SET_ELEM_KIND: Partial<Record<ClegType['kind'], SelectorType>> = {
    number: 'node', edge: 'edge', tri: 'tri', quad: 'quad',
};

// Resolves an msBase-style selector argument into a real Selector - a `sel` value (used directly,
// whatever SelectorType it is) or a `set` (of number/edge/tri/quad, wrapped into a `raw` Selector
// the same way resolveSelectorArg's own `set` case does) - but NOT a bare `string`, unlike
// resolveSelectorArg: every resolveSelectorArg call site has one fixed wantKind to parse a string
// against, but msBase doesn't know its own kind ahead of time (that's the whole point - see
// MultiSelector's own doc comment), so there is no parser to pick for a plain string here.
function resolveAnyKindSelectorArg(callee: string, arg: ClegValue): Selector {
    if (arg.kind === 'sel') return arg.value;
    if (arg.kind === 'set') {
        const wantKind = SELECTOR_TYPE_BY_SET_ELEM_KIND[arg.elem.kind];
        if (!wantKind)
            throw new Error(
                `cleg: '${callee}': a selector set must be a set of number/edge/tri/quad, got a set of ${typeToString(arg.elem)}`);
        return { op: 'raw', type: wantKind, items: setValueToSelectedVals(wantKind, arg.value) };
    }
    throw new Error(`cleg: '${callee}': expected sel or set, got ${typeToString(clegValueType(arg))}`);
}

function msBaseCheckCall(callee: string, argTypes: ClegType[]): ClegType {
    if (argTypes.length !== 2)
        throw new Error(`cleg: '${callee}' expects 2 argument(s), got ${argTypes.length}`);
    if (argTypes[0].kind !== 'number')
        throw new Error(`cleg: '${callee}' argument 1: expected number, got ${typeToString(argTypes[0])}`);
    if (argTypes[1].kind !== 'sel' && argTypes[1].kind !== 'set')
        throw new Error(`cleg: '${callee}' argument 2: expected sel or set, got ${typeToString(argTypes[1])}`);
    return MSEL_TYPE;
}
// `msAll()`: every node of the full product, unrestricted - see MultiSelector's own 'all' doc
// comment.
BUILTIN_FUNCTIONS['msAll'] = {
    checkCall: fixedSignature([], MSEL_TYPE),
    call: () => ({ kind: 'msel', value: { op: 'all' } }),
};

// `msBase(number, X)`: "every full-product node whose `number`-th coordinate is kept by X, every
// other coordinate unrestricted" - see MultiSelector's own doc comment for what X may be.
BUILTIN_FUNCTIONS['msBase'] = {
    checkCall: msBaseCheckCall,
    call: ([numberVal, arg]) => {
        const number = (numberVal as { value: number }).value;
        if (!Number.isInteger(number) || number < 0)
            throw new Error(`cleg: msBase: number must be a nonnegative integer, got ${number}`);
        const sel = resolveAnyKindSelectorArg('msBase', arg);
        return { kind: 'msel', value: { op: 'base', number, sel } };
    },
};

// `msUnion(items)`/`msInter(items)`: fixed-signature (msel[] -> msel) - a plain array, not a set,
// mirroring modify's own mods array (see its own doc comment) and the underlying Selector grammar's
// own (union SEL...)/(inter SEL...), which are similarly plain lists.
BUILTIN_FUNCTIONS['msUnion'] = {
    checkCall: fixedSignature([{ kind: 'array', elem: MSEL_TYPE }], MSEL_TYPE),
    call: ([itemsVal]) => ({
        kind: 'msel',
        value: {
            op: 'union',
            items: (itemsVal as { value: ClegValue[] }).value.map(v => (v as { value: MultiSelector }).value),
        },
    }),
};
BUILTIN_FUNCTIONS['msInter'] = {
    checkCall: fixedSignature([{ kind: 'array', elem: MSEL_TYPE }], MSEL_TYPE),
    call: ([itemsVal]) => ({
        kind: 'msel',
        value: {
            op: 'inter',
            items: (itemsVal as { value: ClegValue[] }).value.map(v => (v as { value: MultiSelector }).value),
        },
    }),
};
// `msDiff(a, b)`: fixed-signature (msel, msel -> msel).
BUILTIN_FUNCTIONS['msDiff'] = {
    checkCall: fixedSignature([MSEL_TYPE, MSEL_TYPE], MSEL_TYPE),
    call: ([a, b]) => ({
        kind: 'msel',
        value: {
            op: 'diff',
            a: (a as { value: MultiSelector }).value,
            b: (b as { value: MultiSelector }).value,
        },
    }),
};

// Fixed once per multiProd call, from `boards`' own ORIGINAL (unrestricted) sizes: `Ns[k]` is
// boards[k]'s own node count, and `stride[k]` lets any tuple of per-board node indices flatten
// to/from one "original" flat index into the full (never fully materialized)
// Ns[0] x Ns[1] x ... x Ns[N-1] product space - the one shared universe every MultiSelector
// combinator (msUnion/msInter/msDiff) has to combine its own operands against. This has to be fixed
// UP FRONT, before any msBase/msUnion/msInter/msDiff runs: two differently-restricted intermediate
// boards (e.g. one msBase restricting board 0, another restricting board 1) would otherwise have no
// common index space to combine against at all.
interface FullProductIndex { Ns: number[]; stride: number[]; total: number; }

function makeFullProductIndex(boards: BoardConfig[]): FullProductIndex {
    const Ns = boards.map(b => b.N);
    const stride = new Array<number>(Ns.length);
    stride[Ns.length - 1] = 1;
    for (let k = Ns.length - 2; k >= 0; k--) stride[k] = stride[k + 1] * Ns[k + 1];
    const total = Ns.reduce((p, n) => p * n, 1);
    return { Ns, stride, total };
}

function fullIndexOf(fpi: FullProductIndex, tuple: number[]): number {
    return tuple.reduce((sum, n, k) => sum + n * fpi.stride[k], 0);
}

function tupleOfFullIndex(fpi: FullProductIndex, idx: number): number[] {
    return fpi.Ns.map((n, k) => Math.floor(idx / fpi.stride[k]) % n);
}

// Restricts `board` (always boards[msel.number], see evalMultiSelector's own 'base' case) to just
// the nodes `sel` keeps - nodeInducedSubgraph directly for a node selector, or (mirroring
// edgeInducedSubgraph's own "which nodes survive" rule) the nodes touched by at least one selected
// edge for an edge selector. Returns the restricted board AND which of `board`'s own ORIGINAL node
// indices survived, in the same ascending order nodeInducedSubgraph/edgeInducedSubgraph themselves
// compact to - evalMultiSelector's own full-product index bookkeeping needs that mapping to place
// the restricted board's own local nodes back into the fixed full index space (FullProductIndex).
// Throws for any SelectorType other than node/edge - unlike msBase itself (which accepts any
// SelectorType at the data-structure level - see MultiSelector's own doc comment), multiProd's own
// evaluation requires node or edge specifically, since there's no other sensible way to turn a
// tri/quad selection into "which nodes of this one factor board survive".
function restrictBoardBySelector(board: BoardConfig, sel: Selector): { bc: BoardConfig; survivors: number[] } {
    if (sel.type === 'node') {
        const kept = selectNode(board.adj, board.emb.pos, sel);
        return { bc: nodeInducedSubgraph(board, kept), survivors: [...kept].sort((a, b) => a - b) };
    }
    if (sel.type === 'edge') {
        const edges = selectEdge(board.adj, board.emb.pos, sel);
        const kept = new Set(edges.flatMap(e => [e.n1, e.n2]));
        return { bc: edgeInducedSubgraph(board, edges), survivors: [...kept].sort((a, b) => a - b) };
    }
    throw new Error(
        `cleg: multiProd: msBase's own selector must be a node or edge selector, got a '${sel.type}' selector`);
}

// Every original flat index, 0..fpi.total-1 - the universal set 'all' and 'inter' (with zero
// operands) both denote (see MultiSelector's own doc comment on why those two coincide).
function universalOriginalIndices(fpi: FullProductIndex): Set<number> {
    const all = new Set<number>();
    for (let i = 0; i < fpi.total; i++) all.add(i);
    return all;
}

// Builds a real BoardConfig for an arbitrary subset of the full product's node space, given only the
// kept ORIGINAL flat indices (`keptOriginal`) - decomposes each back into its own per-board tuple
// (via `fpi`) to compute adjacency (Cartesian product rule: two kept nodes are adjacent iff they
// differ in EXACTLY one coordinate k, adjacent there in boards[k]) and embedding (per-board
// positions concatenated) directly, never materializing the full product's own (possibly enormous)
// N x N adjacency matrix. Compacts to a fresh 0..K-1 range in ascending original-index order (same
// convention as nodeInducedSubgraph/edgeInducedSubgraph) - returns both the new BoardConfig and
// origIndex (new local index -> kept original flat index), so a further msUnion/msInter/msDiff can
// keep combining against the very same fixed full-product index space. Used by every
// evalMultiSelector case except 'base' (which instead reuses the real product() function directly -
// see its own comment on why that still ends up with the exact same adjacency/embedding).
function buildFromOriginalIndices(
    boards: BoardConfig[], fpi: FullProductIndex, keptOriginal: Set<number>,
): { bc: BoardConfig; origIndex: number[] } {
    const origIndex = [...keptOriginal].sort((a, b) => a - b);
    const tuples = origIndex.map(idx => tupleOfFullIndex(fpi, idx));
    const embDim = boards.reduce((s, b) => s + b.emb.embDim, 0);
    const pos = tuples.map(tuple => tuple.flatMap((n, k) => boards[k].emb.pos[n]));
    const K = origIndex.length;
    const adj = zeroAdj(K);
    for (let a = 0; a < K; a++) {
        for (let b = a + 1; b < K; b++) {
            let diffCoord = -1;
            let tooManyDiffs = false;
            for (let k = 0; k < boards.length; k++) {
                if (tuples[a][k] !== tuples[b][k]) {
                    if (diffCoord !== -1) { tooManyDiffs = true; break; }
                    diffCoord = k;
                }
            }
            if (!tooManyDiffs && diffCoord >= 0 && boards[diffCoord].adj[tuples[a][diffCoord]][tuples[b][diffCoord]]) {
                adj[a][b] = 1;
                adj[b][a] = 1;
            }
        }
    }
    return { bc: make(new Embedding(embDim, pos), adj), origIndex };
}

/**
 * Evaluates a MultiSelector against `boards`/`fpi` into a real BoardConfig plus origIndex (see
 * buildFromOriginalIndices) - every combinator ultimately reduces to a set operation over
 * origIndex's own shared "which of the full product's original flat indices survive" universe (see
 * FullProductIndex's own doc comment on why that has to be fixed up front, from `boards`' own
 * unrestricted sizes, rather than derived along the way).
 *
 * 'base' restricts boards[number] (see restrictBoardBySelector) and folds the real product()
 * pairwise, left to right, across every factor (boards[number] replaced by its restriction) -
 * mathematically identical adjacency/embedding to a direct N-ary construction, since product()'s own
 * row-major node indexing composes correctly under folding (`product(product(A,B),C)`'s own index
 * `(a*NB+b)*NC+c` already equals the direct 3-ary flattening `a*NB*NC+b*NC+c`). The resulting local
 * node indices are then translated back into the fixed full-product index space one at a time
 * (decompose via the RESTRICTED factors' own sizes, substitute the restricted coordinate's own local
 * index for its ORIGINAL boards[number] index via `survivors`, then flatten via `fpi`).
 *
 * 'union'/'inter'/'diff' recursively evaluate their own operands first (discarding each one's own
 * `bc`, since only its origIndex set matters for combining), combine via ordinary Set operations,
 * then materialize a fresh BoardConfig for exactly the combined set via buildFromOriginalIndices.
 * 'inter' with zero operands is the universal set (every original index) - the usual absorbing-
 * element identity for an empty intersection fold, matching Selector's own `(inter)`.
 */
function evalMultiSelector(
    boards: BoardConfig[], fpi: FullProductIndex, msel: MultiSelector,
): { bc: BoardConfig; origIndex: number[] } {
    switch (msel.op) {
        case 'all':
            return buildFromOriginalIndices(boards, fpi, universalOriginalIndices(fpi));
        case 'base': {
            const { bc: restricted, survivors } = restrictBoardBySelector(boards[msel.number], msel.sel);
            const factorBoards = boards.map((b, i) => i === msel.number ? restricted : b);
            const bc = factorBoards.reduce((acc, b) => product(acc, b));
            const localNs = factorBoards.map(b => b.N);
            const localStride = new Array<number>(localNs.length);
            localStride[localNs.length - 1] = 1;
            for (let k = localNs.length - 2; k >= 0; k--) localStride[k] = localStride[k + 1] * localNs[k + 1];
            const origIndex = new Array<number>(bc.N);
            for (let local = 0; local < bc.N; local++) {
                const tuple = localNs.map((n, k) => Math.floor(local / localStride[k]) % n);
                tuple[msel.number] = survivors[tuple[msel.number]];
                origIndex[local] = fullIndexOf(fpi, tuple);
            }
            return { bc, origIndex };
        }
        case 'union': {
            const kept = new Set<number>();
            for (const item of msel.items)
                for (const idx of evalMultiSelector(boards, fpi, item).origIndex) kept.add(idx);
            return buildFromOriginalIndices(boards, fpi, kept);
        }
        case 'inter': {
            if (msel.items.length === 0) return buildFromOriginalIndices(boards, fpi, universalOriginalIndices(fpi));
            let kept = new Set(evalMultiSelector(boards, fpi, msel.items[0]).origIndex);
            for (let i = 1; i < msel.items.length; i++) {
                const next = new Set(evalMultiSelector(boards, fpi, msel.items[i]).origIndex);
                kept = new Set([...kept].filter(idx => next.has(idx)));
            }
            return buildFromOriginalIndices(boards, fpi, kept);
        }
        case 'diff': {
            const a = new Set(evalMultiSelector(boards, fpi, msel.a).origIndex);
            const b = new Set(evalMultiSelector(boards, fpi, msel.b).origIndex);
            return buildFromOriginalIndices(boards, fpi, new Set([...a].filter(idx => !b.has(idx))));
        }
    }
}

// `multiProd(boards, msel)`: the N-ary Cartesian product of `boards` (an egr[]), restricted to
// exactly the subgraph `msel` denotes - see evalMultiSelector's own doc comment for the full
// algorithm. `boards` must be non-empty - an N-ary product of zero factors has no principled
// definition here.
BUILTIN_FUNCTIONS['multiProd'] = {
    checkCall: fixedSignature([{ kind: 'array', elem: EGR_TYPE }, MSEL_TYPE], EGR_TYPE),
    call: ([boardsVal, mselVal]) => {
        const boards = (boardsVal as { value: ClegValue[] }).value.map(v => (v as { value: BoardConfig }).value);
        if (boards.length === 0) throw new Error(`cleg: multiProd: boards must be non-empty`);
        const fpi = makeFullProductIndex(boards);
        const msel = (mselVal as { value: MultiSelector }).value;
        return { kind: 'egr', value: evalMultiSelector(boards, fpi, msel).bc };
    },
};

// `subHcublat(bounds, cond)`: a "sub-region" of an N-dimensional hypercubical lattice - `bounds` is
// an N-length array of `[lo, hi]` pairs (inclusive integer bounds, one pair per dimension)
// describing the bounding hyperrectangle; `cond` decides which lattice points inside it actually
// become nodes, called once per candidate point (as that point's own N coordinates, a `number[]`)
// via callUserFunction - the one builtin so far that needs to call back into a `func`-typed argument,
// which is why `funcs` is threaded through BuiltinFunction's own `call` signature. Surviving nodes
// keep the plain grid adjacency (connected iff their coordinates differ by exactly 1 in exactly one
// dimension) and their own lattice coordinates, re-centered (see below), as their N-dim embedding
// position - same convention, and the same full-lattice-index/stride bookkeeping to avoid an
// O(survivors^2) adjacency scan, as
// shared/boardConfig.ts's own hypercuboidBoard, just over an explicit per-dimension [lo, hi] rather
// than always starting at 0 - unlike hypercuboidBoard, the re-centering (see the end of `call`
// below) is computed from the SURVIVING nodes' own bounding box, not from `bounds` itself, since
// `cond` may keep a shape nowhere near centered within the hyperrectangle it was given.
BUILTIN_FUNCTIONS['subHcublat'] = {
    checkCall: fixedSignature(
        [
            { kind: 'array', elem: { kind: 'array', elem: NUMBER_TYPE } },
            { kind: 'func', params: [{ kind: 'array', elem: NUMBER_TYPE }], returnType: { kind: 'bool' } },
        ],
        EGR_TYPE,
    ),
    call([boundsVal, condVal], funcs) {
        const boundsArr = (boundsVal as { value: ClegValue[] }).value;
        const k = boundsArr.length;
        if (k === 0) throw new Error(`cleg: 'subHcublat' bounds must be non-empty`);
        const lo = new Array<number>(k);
        const dims = new Array<number>(k);
        boundsArr.forEach((pairVal, i) => {
            const pair = (pairVal as { value: ClegValue[] }).value;
            if (pair.length !== 2)
                throw new Error(`cleg: 'subHcublat' bounds[${i}] must have exactly 2 entries (lower, upper), got ${pair.length}`);
            const a = (pair[0] as { value: number }).value;
            const b = (pair[1] as { value: number }).value;
            if (!Number.isInteger(a) || !Number.isInteger(b) || a > b)
                throw new Error(
                    `cleg: 'subHcublat' bounds[${i}] must be integers with lower <= upper, got [${a}, ${b}]`);
            lo[i] = a;
            dims[i] = b - a + 1;
        });
        // `condVal` may be a plain top-level-function reference OR a partial application (e.g.
        // `goDeskCond(l, w, h, fw, fh, in, #)`) closing over everything but the one open `number[]`
        // position - fillHoles interleaves `pointArg` into whichever slot that is, rather than
        // assuming `fn`'s own full parameter list is just `[pointArg]` (it was, for a plain
        // reference, but not in general once `cond` can be a closure).
        const cond = condVal as { name: string; boundArgs: (ClegValue | null)[] };
        const fn = funcs[cond.name];

        const strides = new Array<number>(k);
        strides[0] = 1;
        for (let i = 1; i < k; i++) strides[i] = strides[i - 1] * dims[i - 1];
        const fullN = dims.reduce((p, d) => p * d, 1);
        const localCoordsOf = (n: number): number[] => {
            const coords = new Array<number>(k);
            for (let i = 0; i < k; i++) { coords[i] = n % dims[i]; n = Math.floor(n / dims[i]); }
            return coords;
        };

        // Only surviving (cond-kept) nodes get a board index (compacted, in ascending
        // full-lattice-index order) - boardIdxOf maps a full-lattice index to that compacted index,
        // absent for a point cond rejected.
        const boardIdxOf = new Map<number, number>();
        const survivingLocal: number[][] = [];
        const pos: number[][] = [];
        for (let n = 0; n < fullN; n++) {
            const local = localCoordsOf(n);
            const point = local.map((c, i) => c + lo[i]);
            const pointArg: ClegValue = {
                kind: 'array', elem: NUMBER_TYPE, value: point.map(v => ({ kind: 'number', value: v })),
            };
            const keep = (callUserFunction(fn, fillHoles(cond.boundArgs, [pointArg]), funcs) as { value: boolean }).value;
            if (!keep) continue;
            boardIdxOf.set(n, survivingLocal.length);
            survivingLocal.push(local);
            pos.push(point);
        }
        const N = survivingLocal.length;

        const adj = zeroAdj(N);
        for (let bi = 0; bi < N; bi++) {
            const local = survivingLocal[bi];
            for (let i = 0; i < k; i++)
                for (const delta of [1, -1]) {
                    const nc = local[i] + delta;
                    if (nc < 0 || nc >= dims[i]) continue;
                    const nlocal = local.slice();
                    nlocal[i] = nc;
                    const flat = nlocal.reduce((s, c, j) => s + c * strides[j], 0);
                    const nbi = boardIdxOf.get(flat);
                    if (nbi === undefined) continue;
                    adj[bi][nbi] = 1;
                }
        }

        // Re-center: subtract each dimension's own midpoint - (min + max) / 2, computed from the
        // SURVIVING nodes' own coordinates (not `bounds` itself) - so the shape sits roughly around
        // the origin regardless of where within `bounds` `cond` happened to keep it. No-op (and no
        // division-by-zero-shaped issue) when N === 0 - there's nothing to center.
        if (N > 0) {
            const mid = new Array<number>(k);
            for (let i = 0; i < k; i++) {
                let minC = pos[0][i];
                let maxC = pos[0][i];
                for (let j = 1; j < N; j++) {
                    if (pos[j][i] < minC) minC = pos[j][i];
                    if (pos[j][i] > maxC) maxC = pos[j][i];
                }
                mid[i] = (minC + maxC) / 2;
            }
            for (const p of pos) for (let i = 0; i < k; i++) p[i] -= mid[i];
        }
        return { kind: 'egr', value: make(new Embedding(k, pos), adj) };
    },
};

// ── Type checking ──────────────────────────────────────────────────────────────

interface TypeEnv { vars: Map<string, ClegType>; parent: TypeEnv | null; }
function lookupVarType(env: TypeEnv, name: string): ClegType | undefined {
    for (let e: TypeEnv | null = env; e; e = e.parent) { const t = e.vars.get(name); if (t) return t; }
    return undefined;
}

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
 * produce, computable without evaluating anything (see typecheckClegAsBoard/buildBoardFromCleg below,
 * which use this to validate a program's result type before ever running it).
 *
 * Simplification: does not check that every path through a function actually reaches a `return` -
 * a function whose body falls off the end without one is only caught at evaluation time (see
 * callUserFunction below), not here.
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
                    `number/string/bool/edge/tri/quad may be set elements`);
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

// ── Evaluation ───────────────────────────────────────────────────────────────

interface ValueEnv { vars: Map<string, ClegValue>; parent: ValueEnv | null; }
// Mirrors lookupVarType's own `| undefined` convention - evalExpr's own Identifier/CallExpr cases
// need to tell "not a local variable" apart from "found" (falling back to a top-level function
// reference/call in the former case), unlike lookupValue/setValue below, which are always used where
// typecheckCleg already guarantees the variable exists.
function lookupValueOptional(env: ValueEnv, name: string): ClegValue | undefined {
    for (let e: ValueEnv | null = env; e; e = e.parent) { const v = e.vars.get(name); if (v) return v; }
    return undefined;
}
function lookupValue(env: ValueEnv, name: string): ClegValue {
    const v = lookupValueOptional(env, name);
    // Unreachable in a program that has passed typecheckCleg - every Identifier/AssignStmt there
    // already resolved to a declared variable.
    if (!v) throw new Error(`cleg: undeclared variable '${name}'`);
    return v;
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

/** Deep-clones an array value's own array structure (recursively, for a nested `T[][]`) so indexed
 * mutation (`arr[i] = x;`, see evalStmt's own AssignStmt case) can never be observed through another
 * variable that was previously assigned `= arr` or received it as a function argument - this
 * language's arrays are value types, not shared references (see this file's own top comment).
 * Every other ClegValue kind either can't be mutated in place at all, or (an array ELEMENT that
 * isn't itself an array - a number/egr/sel/set/... ) can only ever be wholesale REPLACED via an
 * indexed assignment, never mutated internally - so only the array *structure* itself needs a fresh
 * copy, not every value reachable from it; called at every site a value is bound to a (potentially
 * long-lived, aliasable) variable - VarDecl's init, a whole-value AssignStmt, and a function
 * argument's own param binding - a no-op passthrough for anything that isn't (or doesn't contain) an
 * array. */
function cloneArrayValue(v: ClegValue): ClegValue {
    if (v.kind !== 'array') return v;
    return { kind: 'array', elem: v.elem, value: v.value.map(cloneArrayValue) };
}

/** Validates that `idx` is a usable index into an array of `length` elements, returning it - shared
 * by evalExpr's own IndexExpr (read) case and evalStmt's own indexed AssignStmt (write) case, so
 * both report an out-of-bounds index the same way. */
function validateArrayIndex(idx: number, length: number): number {
    if (!Number.isInteger(idx) || idx < 0 || idx >= length)
        throw new Error(`cleg: array index ${idx} out of bounds for array of length ${length}`);
    return idx;
}

/** Interleaves `suppliedArgs` (in order) into `boundArgs`' own `null` ("still uninstantiated")
 * slots, producing the full argument list the original function actually needs - used by evalExpr's
 * own CallExpr case whenever it calls through a `func` value, whether that value is a plain
 * function-pointer reference (every slot `null`, so this is just `suppliedArgs` unchanged) or a
 * partial application (see ClegValue's own 'func' doc comment) - one shared interleaving rule for
 * both, rather than treating them as two different cases. */
function fillHoles(boundArgs: (ClegValue | null)[], suppliedArgs: ClegValue[]): ClegValue[] {
    let i = 0;
    return boundArgs.map(b => (b === null ? suppliedArgs[i++] : b));
}

/** Merges a partial-application CallExpr's own `args` (at least one is a HoleExpr) into `boundArgs`'
 * own currently-open (`null`) slots, in order - evaluating each non-hole argument now (once,
 * eagerly) and leaving each hole slot open, producing the NEW boundArgs for the resulting (possibly
 * still-partial) closure. Mirrors checkPartialApplication's own "same merge either way" reasoning:
 * starting from a fresh all-`null` boundArgs (a bare top-level function name, nothing bound yet) or
 * an existing value's own boundArgs (a plain pointer, still all-`null`, or an already-partial
 * closure) is the exact same operation, just a different starting point. */
function mergeBoundArgs(boundArgs: (ClegValue | null)[], args: Expr[], env: ValueEnv, funcs: UserFuncTable): (ClegValue | null)[] {
    let j = 0;
    return boundArgs.map(b => {
        if (b !== null) return b;
        const a = args[j++];
        return a.kind === 'HoleExpr' ? null : cloneArrayValue(evalExpr(a, env, funcs));
    });
}

type UserFuncTable = Record<string, FunctionDecl>;

// Thrown to unwind out of nested blocks/if-statements on `return` - always caught by
// callUserFunction below, never escapes runCleg itself.
class ReturnSignal { constructor(public value: ClegValue) {} }
// Thrown by BreakStmt/ContinueStmt to unwind out of nested blocks/if-statements up to the innermost
// enclosing ForStmt/WhileStmt's own try/catch (see evalStmt's own ForStmt/WhileStmt cases) - never
// escapes past there in a program that has passed typecheckCleg (checkStmt's own `inLoop` check
// already rejected either one outside a loop). A ReturnSignal thrown from inside a loop body is
// neither of these, so it passes straight through both catches unchanged, all the way up to
// callUserFunction, exactly as if the loop weren't there.
class BreakSignal {}
class ContinueSignal {}

function evalBlock(block: Block, parent: ValueEnv, funcs: UserFuncTable): void {
    const env: ValueEnv = { vars: new Map(), parent };
    for (const stmt of block.stmts) evalStmt(stmt, env, funcs);
}

function evalStmt(stmt: Stmt, env: ValueEnv, funcs: UserFuncTable): void {
    switch (stmt.kind) {
        case 'VarDecl':
            env.vars.set(stmt.name, cloneArrayValue(evalExpr(stmt.init, env, funcs)));
            return;
        case 'AssignStmt': {
            const value = cloneArrayValue(evalExpr(stmt.value, env, funcs));
            if (stmt.indices.length === 0) {
                setValue(env, stmt.name, value);
                return;
            }
            // Walk down into the array named `stmt.name`, following every index but the last (each
            // of which - checkStmt's own AssignStmt case already guarantees - lands on another
            // array), then mutate the final slot in place. `target` is the SAME object stored in
            // `env` (lookupValue never copies), which is safe to mutate directly precisely because
            // cloneArrayValue already guarantees nothing else aliases it (see that function's own
            // doc comment).
            let target = lookupValue(env, stmt.name) as { kind: 'array'; value: ClegValue[] };
            for (let i = 0; i < stmt.indices.length - 1; i++) {
                const idxValue = (evalExpr(stmt.indices[i], env, funcs) as { kind: 'number'; value: number }).value;
                const idx = validateArrayIndex(idxValue, target.value.length);
                target = target.value[idx] as { kind: 'array'; value: ClegValue[] };
            }
            const lastIdxValue =
                (evalExpr(stmt.indices[stmt.indices.length - 1], env, funcs) as { kind: 'number'; value: number }).value;
            const lastIdx = validateArrayIndex(lastIdxValue, target.value.length);
            target.value[lastIdx] = value;
            return;
        }
        case 'IfStmt': {
            const cond = evalExpr(stmt.cond, env, funcs) as { kind: 'bool'; value: boolean };
            if (cond.value) evalBlock(stmt.then, env, funcs);
            else if (stmt.else_) stmt.else_.kind === 'Block' ? evalBlock(stmt.else_, env, funcs) : evalStmt(stmt.else_, env, funcs);
            return;
        }
        case 'ForStmt': {
            // One scope for the whole loop (init's own variable, if any, persists across every
            // iteration) - body gets its own further-nested scope each iteration via evalBlock,
            // same as any other BLOCK - see ForStmt's own doc comment. The try/catch around
            // evalBlock is BreakStmt/ContinueStmt's own unwind target (see BreakSignal/
            // ContinueSignal's own doc comment) - `continue` still runs `update` below before the
            // next `cond` check, exactly like real C++; `break` skips straight past the loop
            // entirely, never running `update` again.
            const loopEnv: ValueEnv = { vars: new Map(), parent: env };
            if (stmt.init) evalStmt(stmt.init, loopEnv, funcs);
            while (!stmt.cond || (evalExpr(stmt.cond, loopEnv, funcs) as { kind: 'bool'; value: boolean }).value) {
                try {
                    evalBlock(stmt.body, loopEnv, funcs);
                } catch (e) {
                    if (e instanceof BreakSignal) break;
                    if (!(e instanceof ContinueSignal)) throw e;
                }
                if (stmt.update) evalStmt(stmt.update, loopEnv, funcs);
            }
            return;
        }
        case 'WhileStmt': {
            // Same BreakStmt/ContinueStmt unwind target as ForStmt above - see its own comment.
            while ((evalExpr(stmt.cond, env, funcs) as { kind: 'bool'; value: boolean }).value) {
                try {
                    evalBlock(stmt.body, env, funcs);
                } catch (e) {
                    if (e instanceof BreakSignal) break;
                    if (!(e instanceof ContinueSignal)) throw e;
                }
            }
            return;
        }
        case 'BreakStmt':
            throw new BreakSignal();
        case 'ContinueStmt':
            throw new ContinueSignal();
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
        case 'Identifier': {
            const v = lookupValueOptional(env, expr.name);
            if (v) return v;
            // Not a variable - a bare reference to one of program's own top-level functions, used as
            // a function-pointer value (checkExpr already confirmed this resolves and is func-typed).
            const fn = funcs[expr.name];
            return {
                kind: 'func', params: fn.params.map(p => p.type), returnType: fn.returnType, name: expr.name,
                boundArgs: fn.params.map(() => null),
            };
        }
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
            if (expr.args.some(a => a.kind === 'HoleExpr')) {
                // Partial application - checkExpr already confirmed expr.callee names either a
                // top-level function or a local func-typed variable (never a builtin) whenever any
                // arg is '#'. mergeBoundArgs evaluates each non-hole argument now (once, eagerly),
                // exactly like an ordinary call's own arguments - cloneArrayValue matters here for
                // the same reason it does at any other site a value is bound into a (potentially
                // long-lived) slot: `boundArgs` itself, held inside the resulting closure, is exactly
                // such a slot. Starting from a variable's own boundArgs (rather than a fresh all-null
                // one) is what lets this further-apply an already-partial closure.
                const varValue = lookupValueOptional(env, expr.callee);
                const fv = varValue as { kind: 'func'; name: string; boundArgs: (ClegValue | null)[] } | undefined;
                const name = fv ? fv.name : expr.callee;
                const fn = funcs[name];
                const startingBoundArgs = fv ? fv.boundArgs : fn.params.map(() => null);
                const boundArgs = mergeBoundArgs(startingBoundArgs, expr.args, env, funcs);
                const holeParams = fn.params.filter((_, i) => boundArgs[i] === null).map(p => p.type);
                return { kind: 'func', params: holeParams, returnType: fn.returnType, name, boundArgs };
            }
            const args = expr.args.map(a => evalExpr(a, env, funcs));
            const builtin = BUILTIN_FUNCTIONS[expr.callee];
            if (builtin) return builtin.call(args, funcs);
            // A local variable of func type shadows a same-named top-level function - see checkExpr's
            // own CallExpr case, which already required this to resolve the same way. fillHoles
            // handles a plain (never-partially-applied) function value transparently, since its own
            // `boundArgs` is all `null`.
            const varValue = lookupValueOptional(env, expr.callee);
            if (varValue) {
                const fv = varValue as { kind: 'func'; name: string; boundArgs: (ClegValue | null)[] };
                return callUserFunction(funcs[fv.name], fillHoles(fv.boundArgs, args), funcs);
            }
            return callUserFunction(funcs[expr.callee], args, funcs);
        }
        case 'BinaryExpr': {
            // `&&`/`||` short-circuit here, before ever reaching BINARY_OPERATOR_OVERLOADS below -
            // see BinOp's own doc comment. checkExpr already required both operands to be bool, so
            // the short-circuited result is always just whichever side actually determines it: `&&`
            // returns false without evaluating `right` at all if `left` is already false, otherwise
            // `right`'s own value; `||` is the mirror image.
            if (expr.op === '&&' || expr.op === '||') {
                const l = (evalExpr(expr.left, env, funcs) as { kind: 'bool'; value: boolean }).value;
                if (expr.op === '&&') return l ? evalExpr(expr.right, env, funcs) : { kind: 'bool', value: false };
                return l ? { kind: 'bool', value: true } : evalExpr(expr.right, env, funcs);
            }
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
            if (expr.op === '-') {
                const v = (evalExpr(expr.operand, env, funcs) as { kind: 'number'; value: number }).value;
                return { kind: 'number', value: -v };
            }
            const v = (evalExpr(expr.operand, env, funcs) as { kind: 'bool'; value: boolean }).value;
            return { kind: 'bool', value: !v };
        }
        case 'NilExpr': return { kind: 'array', elem: expr.type, value: [] };
        case 'IndexExpr': {
            const arr = evalExpr(expr.array, env, funcs) as { kind: 'array'; elem: ClegType; value: ClegValue[] };
            const idxValue = (evalExpr(expr.index, env, funcs) as { kind: 'number'; value: number }).value;
            const idx = validateArrayIndex(idxValue, arr.value.length);
            return arr.value[idx];
        }
        // Unreachable - CallExpr's own case above always filters HoleExpr args out before recursing
        // into evalExpr for the rest; the parser never produces one anywhere else.
        case 'HoleExpr':
            throw new Error(`cleg: '#' is only valid as an argument in a partial-application call`);
    }
}

function callUserFunction(fn: FunctionDecl, args: ClegValue[], funcs: UserFuncTable): ClegValue {
    // cloneArrayValue here (not just at the call site's own VarDecl/AssignStmt) is what makes an
    // array argument a genuine value-copy rather than a reference to the caller's own array, exactly
    // like passing one to another variable - see cloneArrayValue's own doc comment.
    const env: ValueEnv = { vars: new Map(fn.params.map((p, i) => [p.name, cloneArrayValue(args[i])])), parent: null };
    try {
        evalBlock(fn.body, env, funcs);
    } catch (e) {
        if (e instanceof ReturnSignal) return e.value;
        throw e;
    }
    throw new Error(`cleg: function '${fn.name}' fell off its own end without a 'return'`);
}

/**
 * Type-checks, then runs an already-parsed `program`: every top-level TopStmt runs in turn, left to
 * right, in one scope shared across all of them (see ClegProgram's own doc comment - entirely
 * separate from any function's own env, so none of this is visible inside a function body) - not
 * just the last one, since an earlier statement can still throw before the last one is ever reached,
 * the usual "run for effect" statement-sequencing semantics. There is no `main` and no other
 * designated entry-point function; the program's own value is whatever its last top-level statement
 * (an ExprStmt - typecheckCleg already required it) evaluated to. Always re-typechecks even if the
 * caller already did (e.g. GameConfig.boardDescr may have been validated once already at edit time,
 * but could also have arrived as untrusted deserialized JSON) - cheap relative to actually
 * evaluating, and a program's `program.functions`/`program.stmts` AST could in principle have been
 * hand-built or tampered with since it was last checked.
 */
export function runClegProgram(program: ClegProgram): ClegValue {
    typecheckCleg(program);
    const funcs: UserFuncTable = {};
    for (const fn of program.functions) funcs[fn.name] = fn;

    // One env shared across every top-level TopStmt - see typecheckCleg's own matching env.
    const env: ValueEnv = { vars: new Map(), parent: null };
    let result: ClegValue | undefined;
    for (const stmt of program.stmts) {
        if (stmt.kind === 'ExprStmt') result = evalExpr(stmt.expr, env, funcs);
        else evalStmt(stmt, env, funcs);
    }
    return result!; // typecheckCleg already required the last top-level statement to be an ExprStmt
}

/** Parses `source`, then runs it via runClegProgram - see that function's own doc comment. */
export function runCleg(source: string): ClegValue {
    return runClegProgram(parseCleg(source));
}

/** Type-checks `program` (see typecheckCleg) and additionally requires its own result type to be
 * `egr` - the shape every GameConfig.boardDescr must satisfy. Throws (with a message naming the
 * actual result type) if `program` type-checks but doesn't produce an `egr`. */
export function typecheckClegAsBoard(program: ClegProgram): void {
    const t = typecheckCleg(program);
    if (t.kind !== 'egr') throw new Error(`cleg: a board description must produce an egr, got ${typeToString(t)}`);
}

/**
 * Type-checks `program` as a board description (typecheckClegAsBoard), runs it (runClegProgram),
 * and unwraps the resulting `egr`'s own BoardConfig - the one entry point every GameConfig ->
 * BoardConfig call site (src/renderer.ts, src/main.ts, server/src/onlineGameManager.ts) uses
 * instead of the old boardType/boardArgs + applyModifiers two-step.
 */
export function buildBoardFromCleg(program: ClegProgram): BoardConfig {
    typecheckClegAsBoard(program);
    return (runClegProgram(program) as { kind: 'egr'; value: BoardConfig }).value;
}
