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
 *
 * This language is split across four files, in dependency order:
 *   - shared/clegBase.ts (this file) - the data-structure declarations shared by all three below:
 *     ClegType/ClegValue and their small set of pure operations, the binary-operator overload
 *     table, and the AST node types.
 *   - shared/clegParser.ts - the lexer, recursive-descent parser (source text -> AST), and the
 *     unparser (AST -> source text, the inverse).
 *   - shared/clegCheck.ts - the static type checker (typecheckCleg and friends).
 *   - shared/clegEval.ts - the builtin-function table (BUILTIN_FUNCTIONS) and the tree-walking
 *     evaluator, plus the runCleg/buildBoardFromCleg-style entry points.
 */

import {
    type BoardConfig, type BoardEdge, type BoardTriangle, type BoardQuad,
    type Selector, type SelectorType, type FormSelector, type BoardModifier,
} from './types.js';

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
    /** Wraps a MultiSelector (defined below, near ClegValue's own 'msel' variant) - a cleg-internal-
     * only concept, unlike sel/formSel/mod, none of which are exposed via shared/types.ts either but
     * wrap something a consumer OUTSIDE cleg's own files also builds/uses (a real
     * Selector/FormSelector/BoardModifier) - nothing outside shared/clegEval.ts ever builds or
     * consumes a MultiSelector directly. */
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
export const SET_ELEM_KINDS = new Set(['number', 'string', 'bool', 'edge', 'tri', 'quad']);

export function typeEquals(a: ClegType, b: ClegType): boolean {
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

export function clegValueType(v: ClegValue): ClegType {
    if (v.kind === 'array') return { kind: 'array', elem: v.elem };
    if (v.kind === 'set') return { kind: 'set', elem: v.elem };
    if (v.kind === 'func') return { kind: 'func', params: v.params, returnType: v.returnType };
    return { kind: v.kind };
}

// Denotes a subset of the FULL N-ary Cartesian product's own node space (fixed once per multiProd
// call, from `boards`' own ORIGINAL, unrestricted sizes - see shared/clegEval.ts's own
// FullProductIndex) - cleg-internal only (see ClegType's own 'msel' doc comment above), built by
// msAll/msBase/msUnion/msInter/msDiff and consumed only by multiProd's own evalMultiSelector (both
// in shared/clegEval.ts) - declared here, alongside ClegValue's own 'msel' variant that carries it,
// rather than in clegEval.ts itself, purely so ClegValue doesn't need a back-reference into a file
// that itself depends on this one. `base`'s own `sel` may syntactically be any Selector - msBase
// itself doesn't check its SelectorType at all, only multiProd's own evaluation does, once it
// actually needs to restrict a specific board (see restrictBoardBySelector) - so a tri/quad selector
// parses/builds fine as an msBase argument and only fails later, when actually evaluated. `all` is
// every original index, unrestricted - the same universal set `msInter(nil(msel))` already denotes
// (the usual absorbing-element identity for an empty intersection fold), just spelled directly
// rather than via that idiom.
export type MultiSelector =
    | { op: 'all' }
    | { op: 'base'; number: number; sel: Selector }
    | { op: 'union' | 'inter'; items: MultiSelector[] }
    | { op: 'diff'; a: MultiSelector; b: MultiSelector };

// ── Binary operators ─────────────────────────────────────────────────────────

export type BinOp = '+' | '-' | '*' | '/' | '%' | '==' | '!=' | '<' | '>' | '<=' | '>=' | '&&' | '||';

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
export interface BinaryOverload {
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
export function clegSetKey(v: ClegValue): string {
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
export function makeClegSet(elem: ClegType, values: ClegValue[]): ClegValue {
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

export const BINARY_OPERATOR_OVERLOADS: Record<BinOp, BinaryOverload[]> = {
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
