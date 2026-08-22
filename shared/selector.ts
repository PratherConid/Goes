import {
    type BoardEdge, makeBoardEdge, type BoardTriangle, makeBoardTriangle, type BoardSquare, makeBoardSquare,
} from './types.js';
import { findTriangles, findSquares } from './topology.js';

// A tiny S-expression language for selecting a subset of a board's nodes, edges, triangles, or
// squares (a "triangle"/"square" here is exactly what shared/topology.ts's findTriangles()/
// findSquares() finds - see BoardTriangle/BoardSquare in shared/types.ts). Grammar (SEL):
//
//   (union SEL SEL)         -- set union
//   (inter SEL SEL)         -- set intersection
//   (diff SEL SEL)          -- set difference (left minus right)
//   (compl SEL)             -- complement, within all objects of whichever kind SEL selects from
//   (more SEL)              -- node/edge only: expands SEL's own result to also include everything
//                               adjacent to it: for a node selector, every node reachable via one
//                               edge from a selected node; for an edge selector, every edge sharing
//                               a node with a selected edge - either way, SEL's own result stays
//                               included too
//   (all)                   -- every object of whichever kind (node/edge/triangle/square) this SEL
//                               is being parsed/evaluated as
//   (none)                  -- no objects of that kind
//   (deg <eq|gt|lt> <num>)  -- node selector only: nodes whose degree is =/>/< a given nonnegative
//                               integer
//   (conva <node|edge|tri|sq> SEL) -- converts SEL (of the given "from" kind) into whichever kind
//                               this selector itself is (the "to" kind - inferred from parsing
//                               context, same as all/none): a "to" object is selected iff ALL of its
//                               associated "from" objects are selected. Two objects (of possibly
//                               different kinds) are associated iff one's own node set is completely
//                               contained in the other's - always well-defined for two differing
//                               kinds, since node/edge/triangle/square have strictly increasing
//                               arity (1/2/3/4), so containment can only run from the smaller-arity
//                               one into the larger. Converting a kind to itself is a no-op (SEL is
//                               returned as-is); triangle <-> square has no meaningful association
//                               and is rejected.
//   (conve <node|edge|tri|sq> SEL) -- same as conva, but a "to" object is selected iff AT LEAST ONE
//                               of its associated "from" objects is selected
//   (rrmn <num> SEL)        -- randomly removes exactly num (a nonnegative integer) items from SEL's
//                               own result, uniformly at random
//   (rrmp <num> SEL)        -- randomly removes a fixed portion of SEL's own result: num (a
//                               nonnegative float) times SEL's own result size, rounded down
//
// `union`/`inter`/`diff`/`compl`/`all`/`none`/`rrmn`/`rrmp` are polymorphic across all four kinds;
// `more` is polymorphic across node/edge only (no adjacency notion is defined here for triangles/
// squares); `conva`/`conve` convert between any two kinds, naming the source kind via their own
// leading node/edge/tri/sq token (unlike `all`/`none`, there's more than one *other* kind it could
// mean, so it can't be inferred purely from parsing context) - except triangle <-> square, which is
// rejected, and a kind converted to itself, which is a no-op (SEL passes through unchanged, not
// wrapped in a conva/conve node at all).
//
// Every Selector node (one monolithic type, below) carries its own `type` (which kind of set it
// denotes) - but rather than parse a type-less tree and infer/validate `type` bottom-up afterward,
// parsing itself is done by four mutually recursive functions, parseNodeSelExpr/parseEdgeSelExpr/
// parseTriangleSelExpr/parseSquareSelExpr (mirroring selectNode()/selectEdge()/selectTriangle()/
// selectSquare()'s own mutual recursion below): each always produces Selectors of its own `type`,
// recursing into whichever of the four conva/conve's own explicit source-kind token names for that
// operand. This is what lets `all`/`none` skip spelling out which kind they mean: `type` comes from
// *which parser reached them*, not from anything written in the expression itself. selectNode()/
// selectEdge()/selectTriangle()/selectSquare() (this file's own separate mutually recursive
// evaluators) still re-check `sel.type` themselves rather than trusting it, since a Selector need
// not always come from this file's own parsers (e.g. a hand-built AST, or one round-tripped through
// JSON).
export type SelectorType = 'node' | 'edge' | 'tri' | 'sq';

export type Selector =
    | { op: 'union' | 'inter' | 'diff'; type: SelectorType; a: Selector; b: Selector }
    | { op: 'compl'; type: SelectorType; a: Selector }
    | { op: 'more'; type: 'node' | 'edge'; a: Selector }
    | { op: 'all' | 'none'; type: SelectorType }
    | { op: 'deg'; type: 'node'; cmp: 'eq' | 'gt' | 'lt'; n: number }
    | { op: 'conva' | 'conve'; type: SelectorType; from: SelectorType; a: Selector }
    | { op: 'rrmn'; type: SelectorType; count: number; a: Selector }
    | { op: 'rrmp'; type: SelectorType; frac: number; a: Selector };

// ── parsing ──────────────────────────────────────────────────────────────────

// '(' and ')' are always their own token, even with no surrounding whitespace (e.g. "(deg eq 5)");
// every other maximal run of non-whitespace, non-paren characters is one token.
function tokenize(s: string): string[] {
    const tokens: string[] = [];
    let i = 0;
    while (i < s.length) {
        const c = s[i];
        if (/\s/.test(c)) { i++; continue; }
        if (c === '(' || c === ')') { tokens.push(c); i++; continue; }
        let j = i + 1;
        while (j < s.length && !/\s/.test(s[j]) && s[j] !== '(' && s[j] !== ')') j++;
        tokens.push(s.slice(i, j));
        i = j;
    }
    return tokens;
}

class ParseCursor {
    private pos = 0;
    constructor(private tokens: string[]) {}

    peek(): string | undefined { return this.tokens[this.pos]; }
    atEnd(): boolean { return this.pos >= this.tokens.length; }

    next(): string {
        if (this.atEnd()) throw new Error('selector: unexpected end of input');
        return this.tokens[this.pos++];
    }

    expect(tok: string): void {
        const t = this.next();
        if (t !== tok) throw new Error(`selector: expected '${tok}', got '${t}'`);
    }
}

// Shared numeric-argument validation for (deg .../rrmn ...)'s count and (rrmp ...)'s portion -
// `context` names the argument in the thrown error (e.g. "(deg ...) argument").
function nextNonnegInt(c: ParseCursor, context: string): number {
    const tok = c.next();
    const n = Number(tok);
    if (!Number.isInteger(n) || n < 0)
        throw new Error(`selector: ${context} must be a nonnegative integer, got '${tok}'`);
    return n;
}

function nextNonnegNumber(c: ParseCursor, context: string): number {
    const tok = c.next();
    const n = Number(tok);
    if (!Number.isFinite(n) || n < 0)
        throw new Error(`selector: ${context} must be a nonnegative number, got '${tok}'`);
    return n;
}

// Reads conva/conve's own leading node/edge/tri/sq token (the "from" kind) and parses its operand
// via the matching one of parseNodeSelExpr/parseEdgeSelExpr/parseTriangleSelExpr/
// parseSquareSelExpr - shared by all four of those functions' own conva/conve case below, `toType`
// being whichever of them called this (the "to" kind, from parsing context, same as all/none).
// Throws if the (from, toType) pair is the one with no defined association (triangle <-> square -
// see this file's own top comment); returns the parsed operand directly, unwrapped, for a same-kind
// conversion (a no-op).
function parseConversion(c: ParseCursor, op: 'conva' | 'conve', toType: SelectorType): Selector {
    const fromTok = c.next();
    if (fromTok !== 'node' && fromTok !== 'edge' && fromTok !== 'tri' && fromTok !== 'sq')
        throw new Error(`selector: (${op} ...) source kind must be 'node', 'edge', 'tri', or 'sq', got '${fromTok}'`);
    if ((fromTok === 'tri' && toType === 'sq') || (fromTok === 'sq' && toType === 'tri'))
        throw new Error(`selector: (${op} ...) has no association defined between 'tri' and 'sq'`);
    const parseFrom = fromTok === 'node' ? parseNodeSelExpr : fromTok === 'edge' ? parseEdgeSelExpr
        : fromTok === 'tri' ? parseTriangleSelExpr : parseSquareSelExpr;
    const a = parseFrom(c);
    c.expect(')');
    return fromTok === toType ? a : { op, type: toType, from: fromTok, a };
}

// Parses a node SEL - mutually recursive with parseEdgeSelExpr/parseTriangleSelExpr/
// parseSquareSelExpr via conva/conve's own operand. Every Selector this returns has `type: 'node'`.
function parseNodeSelExpr(c: ParseCursor): Selector {
    c.expect('(');
    const op = c.next();
    switch (op) {
        case 'union': case 'inter': case 'diff': {
            const a = parseNodeSelExpr(c);
            const b = parseNodeSelExpr(c);
            c.expect(')');
            return { op, type: 'node', a, b };
        }
        case 'compl': {
            const a = parseNodeSelExpr(c);
            c.expect(')');
            return { op: 'compl', type: 'node', a };
        }
        case 'more': {
            const a = parseNodeSelExpr(c);
            c.expect(')');
            return { op: 'more', type: 'node', a };
        }
        case 'all':
            c.expect(')');
            return { op: 'all', type: 'node' };
        case 'none':
            c.expect(')');
            return { op: 'none', type: 'node' };
        case 'deg': {
            const cmpTok = c.next();
            if (cmpTok !== 'eq' && cmpTok !== 'gt' && cmpTok !== 'lt')
                throw new Error(`selector: (deg ...) comparator must be 'eq', 'gt', or 'lt', got '${cmpTok}'`);
            const n = nextNonnegInt(c, '(deg ...) argument');
            c.expect(')');
            return { op: 'deg', type: 'node', cmp: cmpTok, n };
        }
        case 'conva': case 'conve':
            return parseConversion(c, op, 'node');
        case 'rrmn': {
            const count = nextNonnegInt(c, '(rrmn ...) count');
            const a = parseNodeSelExpr(c);
            c.expect(')');
            return { op: 'rrmn', type: 'node', count, a };
        }
        case 'rrmp': {
            const frac = nextNonnegNumber(c, '(rrmp ...) portion');
            const a = parseNodeSelExpr(c);
            c.expect(')');
            return { op: 'rrmp', type: 'node', frac, a };
        }
        default:
            throw new Error(`selector: unknown node-selector operator '${op}'`);
    }
}

// Parses an edge SEL - mutually recursive with parseNodeSelExpr/parseTriangleSelExpr/
// parseSquareSelExpr via conva/conve's own operand. Every Selector this returns has `type: 'edge'`.
function parseEdgeSelExpr(c: ParseCursor): Selector {
    c.expect('(');
    const op = c.next();
    switch (op) {
        case 'union': case 'inter': case 'diff': {
            const a = parseEdgeSelExpr(c);
            const b = parseEdgeSelExpr(c);
            c.expect(')');
            return { op, type: 'edge', a, b };
        }
        case 'compl': {
            const a = parseEdgeSelExpr(c);
            c.expect(')');
            return { op: 'compl', type: 'edge', a };
        }
        case 'more': {
            const a = parseEdgeSelExpr(c);
            c.expect(')');
            return { op: 'more', type: 'edge', a };
        }
        case 'all':
            c.expect(')');
            return { op: 'all', type: 'edge' };
        case 'none':
            c.expect(')');
            return { op: 'none', type: 'edge' };
        case 'conva': case 'conve':
            return parseConversion(c, op, 'edge');
        case 'rrmn': {
            const count = nextNonnegInt(c, '(rrmn ...) count');
            const a = parseEdgeSelExpr(c);
            c.expect(')');
            return { op: 'rrmn', type: 'edge', count, a };
        }
        case 'rrmp': {
            const frac = nextNonnegNumber(c, '(rrmp ...) portion');
            const a = parseEdgeSelExpr(c);
            c.expect(')');
            return { op: 'rrmp', type: 'edge', frac, a };
        }
        default:
            throw new Error(`selector: unknown edge-selector operator '${op}'`);
    }
}

// Parses a triangle SEL - mutually recursive with parseNodeSelExpr/parseEdgeSelExpr via conva/
// conve's own operand (never parseSquareSelExpr - see this file's own top comment on why
// triangle <-> square is rejected). Every Selector this returns has `type: 'tri'`. No `deg`/`more`
// here.
function parseTriangleSelExpr(c: ParseCursor): Selector {
    c.expect('(');
    const op = c.next();
    switch (op) {
        case 'union': case 'inter': case 'diff': {
            const a = parseTriangleSelExpr(c);
            const b = parseTriangleSelExpr(c);
            c.expect(')');
            return { op, type: 'tri', a, b };
        }
        case 'compl': {
            const a = parseTriangleSelExpr(c);
            c.expect(')');
            return { op: 'compl', type: 'tri', a };
        }
        case 'all':
            c.expect(')');
            return { op: 'all', type: 'tri' };
        case 'none':
            c.expect(')');
            return { op: 'none', type: 'tri' };
        case 'conva': case 'conve':
            return parseConversion(c, op, 'tri');
        case 'rrmn': {
            const count = nextNonnegInt(c, '(rrmn ...) count');
            const a = parseTriangleSelExpr(c);
            c.expect(')');
            return { op: 'rrmn', type: 'tri', count, a };
        }
        case 'rrmp': {
            const frac = nextNonnegNumber(c, '(rrmp ...) portion');
            const a = parseTriangleSelExpr(c);
            c.expect(')');
            return { op: 'rrmp', type: 'tri', frac, a };
        }
        default:
            throw new Error(`selector: unknown triangle-selector operator '${op}'`);
    }
}

// Parses a square SEL - the square counterpart of parseTriangleSelExpr above (see its own doc
// comment). Every Selector this returns has `type: 'sq'`.
function parseSquareSelExpr(c: ParseCursor): Selector {
    c.expect('(');
    const op = c.next();
    switch (op) {
        case 'union': case 'inter': case 'diff': {
            const a = parseSquareSelExpr(c);
            const b = parseSquareSelExpr(c);
            c.expect(')');
            return { op, type: 'sq', a, b };
        }
        case 'compl': {
            const a = parseSquareSelExpr(c);
            c.expect(')');
            return { op: 'compl', type: 'sq', a };
        }
        case 'all':
            c.expect(')');
            return { op: 'all', type: 'sq' };
        case 'none':
            c.expect(')');
            return { op: 'none', type: 'sq' };
        case 'conva': case 'conve':
            return parseConversion(c, op, 'sq');
        case 'rrmn': {
            const count = nextNonnegInt(c, '(rrmn ...) count');
            const a = parseSquareSelExpr(c);
            c.expect(')');
            return { op: 'rrmn', type: 'sq', count, a };
        }
        case 'rrmp': {
            const frac = nextNonnegNumber(c, '(rrmp ...) portion');
            const a = parseSquareSelExpr(c);
            c.expect(')');
            return { op: 'rrmp', type: 'sq', frac, a };
        }
        default:
            throw new Error(`selector: unknown square-selector operator '${op}'`);
    }
}

// Shared by parseNodeSelector/parseEdgeSelector/parseTriangleSelector/parseSquareSelector:
// tokenizes `s`, runs `parseExpr` over the whole thing, and rejects any leftover trailing input.
function parseTopLevel(s: string, parseExpr: (c: ParseCursor) => Selector): Selector {
    const tokens = tokenize(s);
    if (tokens.length === 0) throw new Error('selector: empty input');
    const c = new ParseCursor(tokens);
    const sel = parseExpr(c);
    if (!c.atEnd()) throw new Error(`selector: unexpected trailing input starting at '${c.peek()}'`);
    return sel;
}

/** Parses `s` as a node selector (see this file's own top comment for the grammar) - throws if `s`
 * doesn't follow the grammar (an operator not valid for nodes is simply not recognized inside a
 * node-selector context - see parseNodeSelExpr). */
export function parseNodeSelector(s: string): Selector {
    return parseTopLevel(s, parseNodeSelExpr);
}

/** Parses `s` as an edge selector (see this file's own top comment for the grammar) - throws if `s`
 * doesn't follow the grammar. */
export function parseEdgeSelector(s: string): Selector {
    return parseTopLevel(s, parseEdgeSelExpr);
}

/** Parses `s` as a triangle selector (see this file's own top comment for the grammar) - throws if
 * `s` doesn't follow the grammar. */
export function parseTriangleSelector(s: string): Selector {
    return parseTopLevel(s, parseTriangleSelExpr);
}

/** Parses `s` as a square selector (see this file's own top comment for the grammar) - throws if `s`
 * doesn't follow the grammar. */
export function parseSquareSelector(s: string): Selector {
    return parseTopLevel(s, parseSquareSelExpr);
}

/** Formats `sel` back into the S-expression syntax parseNodeSelector()/parseEdgeSelector()/
 * parseTriangleSelector()/parseSquareSelector() accept - the inverse of parsing. Used e.g. to
 * round-trip a BoardModifier's own selector back into command-line text for display (see
 * src/sidePanel.ts's fmtModifiers). */
export function formatSelector(sel: Selector): string {
    switch (sel.op) {
        case 'union': case 'inter': case 'diff':
            return `(${sel.op} ${formatSelector(sel.a)} ${formatSelector(sel.b)})`;
        case 'compl':
            return `(compl ${formatSelector(sel.a)})`;
        case 'more':
            return `(more ${formatSelector(sel.a)})`;
        case 'all': case 'none':
            return `(${sel.op})`;
        case 'deg':
            return `(deg ${sel.cmp} ${sel.n})`;
        case 'conva': case 'conve':
            return `(${sel.op} ${sel.from} ${formatSelector(sel.a)})`;
        case 'rrmn':
            return `(rrmn ${sel.count} ${formatSelector(sel.a)})`;
        case 'rrmp':
            return `(rrmp ${sel.frac} ${formatSelector(sel.a)})`;
    }
}

// A tiny extension of the grammar above for shared/boardConfig.ts's `genericForm` - a FormSelector
// names which KIND of object (triangle or square) genericForm should look for, plus an optional
// inner SEL (of that same kind) restricting which ones of that kind qualify (default: every one
// found). Written as its own small S-expression, `(tri [SEL])` / `(sq [SEL])` - distinct from a
// plain Selector, since a FormSelector isn't itself selecting FROM an existing known-kind set; it's
// declaring which kind to look for in the first place.
export type FormSelector =
    | { kind: 'tri'; sel?: Selector }
    | { kind: 'sq'; sel?: Selector };

// Parses one `(tri [SEL])` / `(sq [SEL])` - mirrors parseNodeSelExpr/etc.'s own "consume '(', read a
// leading token, dispatch" shape, but there's no mutual recursion here: a FormSelector's own SEL is
// parsed via the ordinary parseTriangleSelExpr/parseSquareSelExpr, not another parseFormSelExpr.
function parseFormSelExpr(c: ParseCursor): FormSelector {
    c.expect('(');
    const kind = c.next();
    if (kind !== 'tri' && kind !== 'sq')
        throw new Error(`form selector: expected 'tri' or 'sq', got '${kind}'`);
    if (c.peek() === ')') { c.next(); return { kind }; }
    const sel = kind === 'tri' ? parseTriangleSelExpr(c) : parseSquareSelExpr(c);
    c.expect(')');
    return { kind, sel };
}

/**
 * Parses `s` as zero or more back-to-back form selectors (`(tri [SEL])` / `(sq [SEL])`, see
 * FormSelector's own doc comment) - unlike parseNodeSelector/parseEdgeSelector/etc. above, which
 * each parse exactly one SEL and reject any leftover input, this keeps parsing form selectors until
 * the input is exhausted (an empty/whitespace-only `s` yields an empty list). Used by
 * shared/boardConfig.ts's parseModifier for the `form` modifier's own trailing form-selector list.
 */
export function parseFormSelectors(s: string): FormSelector[] {
    const tokens = tokenize(s);
    const c = new ParseCursor(tokens);
    const out: FormSelector[] = [];
    while (!c.atEnd()) out.push(parseFormSelExpr(c));
    return out;
}

/** Formats one FormSelector back into the `(tri [SEL])` / `(sq [SEL])` syntax parseFormSelectors()
 * accepts - the inverse of parsing. */
export function formatFormSelector(fs: FormSelector): string {
    return fs.sel === undefined ? `(${fs.kind})` : `(${fs.kind} ${formatSelector(fs.sel)})`;
}

/** Formats a whole FormSelector[] back into the space-separated syntax parseFormSelectors() accepts -
 * used e.g. to round-trip a BoardModifier's own `sels` back into command-line text for display (see
 * src/sidePanel.ts's fmtModifiers). */
export function formatFormSelectors(fss: FormSelector[]): string {
    return fss.map(formatFormSelector).join(' ');
}

// ── evaluation ───────────────────────────────────────────────────────────────

// "a node"/"an edge"/"a tri"/"a sq" - shared by each evaluator's own wrong-kind error message below.
function describeSelectorType(type: SelectorType): string {
    return `${type === 'edge' ? 'an' : 'a'} ${type}`;
}

function degree(adj: number[][], i: number): number {
    return adj[i].reduce((s, v) => s + (v ? 1 : 0), 0);
}

// Generic Map-based dedupe, keyed by `key(item)` - the last item seen for a given key overwrites the
// value, but the FIRST occurrence's position in iteration order is kept (a Map.set on an existing
// key doesn't move it), matching Set/Map union semantics throughout this file. Shared by every
// object kind's own `union` case below.
function dedupeByKey<T>(items: T[], key: (item: T) => string | number): T[] {
    const byKey = new Map<string | number, T>();
    for (const item of items) byKey.set(key(item), item);
    return [...byKey.values()];
}

// BoardEdge/BoardTriangle/BoardSquare aren't valid Set/Map keys themselves (two structurally-equal
// values are different objects), so union/inter/diff/compl below key them by these canonical string
// ids (n1 < n2 < ... already makes each one unique per object) whenever they need set-like
// membership tests.
function edgeKey(e: BoardEdge): string {
    return `${e.n1},${e.n2}`;
}
function triKey(t: BoardTriangle): string {
    return `${t.n1},${t.n2},${t.n3}`;
}
function sqKey(s: BoardSquare): string {
    return `${s.n1},${s.n2},${s.n3},${s.n4}`;
}

// Returns a NEW array with exactly `removeCount` (clamped to [0, items.length], since removing more
// than exist isn't meaningful) uniformly-randomly-chosen elements dropped, via a partial
// Fisher-Yates shuffle (only the first `removeCount` positions need to be randomized to pick which
// elements to drop). The rest of this file's set operations are pure and order-preserving; rrmn/rrmp
// (the only Selector ops that use this) are neither - two evaluations of the same selector can
// return different results, and the kept elements' relative order isn't preserved either.
function randomlyRemove<T>(items: T[], removeCount: number): T[] {
    const n = items.length;
    const toRemove = Math.min(Math.max(removeCount, 0), n);
    const shuffled = [...items];
    for (let i = 0; i < toRemove; i++) {
        const j = i + Math.floor(Math.random() * (n - i));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled.slice(toRemove);
}

// True iff `a`'s own members are completely contained in `b`'s, or vice versa - the general
// "association" test conva/conve rely on (see this file's own top comment). Every object kind here
// has a fixed arity (node 1, edge 2, triangle 3, square 4) and every object's own members are
// distinct node indices, so containment can only ever run from the smaller-arity list into the
// larger one; this checks whichever direction applies rather than assuming a fixed order.
function isAssociated(a: number[], b: number[]): boolean {
    const [small, large] = a.length <= b.length ? [a, b] : [b, a];
    const largeSet = new Set(large);
    return small.every(x => largeSet.has(x));
}

// Shared by every evaluator's own conva/conve case: `allTo`/`toMembers` enumerate every object of
// THIS evaluator's own kind (the "to" kind) in the whole graph; `allFrom`/`fromMembers`/`fromKey` do
// the same for SEL's own declared source kind, and `selectedFromKeys` is which of those SEL's own
// operand selects. A "to" object is kept iff ALL (mode 'all', conva) or AT LEAST ONE (mode 'some',
// conve) of its associated "from" objects (per isAssociated above) are selected - vacuously
// true/false (respectively) for a "to" object with no associated "from" objects at all, per ordinary
// `.every()`/`.some()` semantics on an empty array.
function convertObjects<F, T>(
    allTo: T[], toMembers: (to: T) => number[],
    allFrom: F[], fromMembers: (from: F) => number[], fromKey: (from: F) => string | number,
    selectedFromKeys: Set<string | number>, mode: 'all' | 'some',
): T[] {
    return allTo.filter(to => {
        const toM = toMembers(to);
        const associated = allFrom.filter(from => isAssociated(toM, fromMembers(from)));
        const isSelected = (from: F) => selectedFromKeys.has(fromKey(from));
        return mode === 'all' ? associated.every(isSelected) : associated.some(isSelected);
    });
}

/**
 * Evaluates a node Selector against a board's adjacency matrix, returning the set of selected node
 * indices. Mutually recursive with selectEdge()/selectTriangle()/selectSquare() via the conva/conve
 * operators. `pos` isn't used by any selector in the current grammar, but is threaded through
 * (matching the other three evaluators' own signatures) for future position-based selectors.
 */
export function selectNode(adj: number[][], pos: number[][], sel: Selector): Set<number> {
    if (sel.type !== 'node')
        throw new Error(`selectNode: expected a node selector, got ${describeSelectorType(sel.type)} selector (op '${sel.op}')`);
    const N = adj.length;
    switch (sel.op) {
        case 'union': {
            const a = selectNode(adj, pos, sel.a), b = selectNode(adj, pos, sel.b);
            return new Set([...a, ...b]);
        }
        case 'inter': {
            const a = selectNode(adj, pos, sel.a), b = selectNode(adj, pos, sel.b);
            return new Set([...a].filter(x => b.has(x)));
        }
        case 'diff': {
            const a = selectNode(adj, pos, sel.a), b = selectNode(adj, pos, sel.b);
            return new Set([...a].filter(x => !b.has(x)));
        }
        case 'compl': {
            const a = selectNode(adj, pos, sel.a);
            const out = new Set<number>();
            for (let i = 0; i < N; i++) if (!a.has(i)) out.add(i);
            return out;
        }
        case 'more': {
            const a = selectNode(adj, pos, sel.a);
            const out = new Set(a);
            for (const i of a)
                for (let j = 0; j < N; j++)
                    if (adj[i][j]) out.add(j);
            return out;
        }
        case 'all': {
            const out = new Set<number>();
            for (let i = 0; i < N; i++) out.add(i);
            return out;
        }
        case 'none':
            return new Set();
        case 'deg': {
            const out = new Set<number>();
            for (let i = 0; i < N; i++) {
                const d = degree(adj, i);
                if ((sel.cmp === 'eq' && d === sel.n) || (sel.cmp === 'gt' && d > sel.n) || (sel.cmp === 'lt' && d < sel.n))
                    out.add(i);
            }
            return out;
        }
        case 'conva': case 'conve': {
            const mode = sel.op === 'conva' ? 'all' : 'some';
            if (sel.from === 'node') return selectNode(adj, pos, sel.a); // same-kind: no-op (defensive)
            const toNodes = Array.from({ length: N }, (_, i) => i);
            if (sel.from === 'edge') {
                const allFrom = selectEdge(adj, pos, { op: 'all', type: 'edge' });
                const selectedKeys = new Set<string | number>(selectEdge(adj, pos, sel.a).map(edgeKey));
                return new Set(convertObjects(toNodes, n => [n], allFrom, e => [e.n1, e.n2], edgeKey, selectedKeys, mode));
            }
            if (sel.from === 'tri') {
                const allFrom = selectTriangle(adj, pos, { op: 'all', type: 'tri' });
                const selectedKeys = new Set<string | number>(selectTriangle(adj, pos, sel.a).map(triKey));
                return new Set(convertObjects(toNodes, n => [n], allFrom, t => [t.n1, t.n2, t.n3], triKey, selectedKeys, mode));
            }
            const allFrom = selectSquare(adj, pos, { op: 'all', type: 'sq' });
            const selectedKeys = new Set<string | number>(selectSquare(adj, pos, sel.a).map(sqKey));
            return new Set(convertObjects(toNodes, n => [n], allFrom, s => [s.n1, s.n2, s.n3, s.n4], sqKey, selectedKeys, mode));
        }
        case 'rrmn': {
            const base = [...selectNode(adj, pos, sel.a)];
            return new Set(randomlyRemove(base, sel.count));
        }
        case 'rrmp': {
            const base = [...selectNode(adj, pos, sel.a)];
            return new Set(randomlyRemove(base, Math.floor(sel.frac * base.length)));
        }
        default:
            throw new Error(`selectNode: unexpected node-selector op '${(sel as Selector).op}'`);
    }
}

/**
 * Evaluates an edge Selector against a board's adjacency matrix, returning the list of selected
 * edges as BoardEdge values (deduplicated). Mutually recursive with selectNode()/selectTriangle()/
 * selectSquare() via the conva/conve operators.
 */
export function selectEdge(adj: number[][], pos: number[][], sel: Selector): BoardEdge[] {
    if (sel.type !== 'edge')
        throw new Error(`selectEdge: expected an edge selector, got ${describeSelectorType(sel.type)} selector (op '${sel.op}')`);
    const N = adj.length;
    switch (sel.op) {
        case 'union':
            return dedupeByKey([...selectEdge(adj, pos, sel.a), ...selectEdge(adj, pos, sel.b)], edgeKey);
        case 'inter': {
            const a = selectEdge(adj, pos, sel.a);
            const bKeys = new Set(selectEdge(adj, pos, sel.b).map(edgeKey));
            return a.filter(e => bKeys.has(edgeKey(e)));
        }
        case 'diff': {
            const a = selectEdge(adj, pos, sel.a);
            const bKeys = new Set(selectEdge(adj, pos, sel.b).map(edgeKey));
            return a.filter(e => !bKeys.has(edgeKey(e)));
        }
        case 'compl': {
            const aKeys = new Set(selectEdge(adj, pos, sel.a).map(edgeKey));
            const out: BoardEdge[] = [];
            for (let i = 0; i < N; i++)
                for (let j = i + 1; j < N; j++) {
                    if (!adj[i][j]) continue;
                    const e = makeBoardEdge(i, j);
                    if (!aKeys.has(edgeKey(e))) out.push(e);
                }
            return out;
        }
        case 'more': {
            const a = selectEdge(adj, pos, sel.a);
            const aNodes = new Set<number>();
            for (const e of a) { aNodes.add(e.n1); aNodes.add(e.n2); }
            const out: BoardEdge[] = [...a];
            for (let i = 0; i < N; i++)
                for (let j = i + 1; j < N; j++)
                    if (adj[i][j] && (aNodes.has(i) || aNodes.has(j))) out.push(makeBoardEdge(i, j));
            return dedupeByKey(out, edgeKey);
        }
        case 'all': {
            const out: BoardEdge[] = [];
            for (let i = 0; i < N; i++)
                for (let j = i + 1; j < N; j++)
                    if (adj[i][j]) out.push(makeBoardEdge(i, j));
            return out;
        }
        case 'none':
            return [];
        case 'conva': case 'conve': {
            const mode = sel.op === 'conva' ? 'all' : 'some';
            if (sel.from === 'edge') return selectEdge(adj, pos, sel.a); // same-kind: no-op (defensive)
            const allEdges = selectEdge(adj, pos, { op: 'all', type: 'edge' });
            if (sel.from === 'node') {
                const selectedKeys = new Set<string | number>(selectNode(adj, pos, sel.a));
                const allNodes = Array.from({ length: N }, (_, i) => i);
                return convertObjects(allEdges, e => [e.n1, e.n2], allNodes, n => [n], n => n, selectedKeys, mode);
            }
            if (sel.from === 'tri') {
                const allFrom = selectTriangle(adj, pos, { op: 'all', type: 'tri' });
                const selectedKeys = new Set<string | number>(selectTriangle(adj, pos, sel.a).map(triKey));
                return convertObjects(allEdges, e => [e.n1, e.n2], allFrom, t => [t.n1, t.n2, t.n3], triKey, selectedKeys, mode);
            }
            const allFrom = selectSquare(adj, pos, { op: 'all', type: 'sq' });
            const selectedKeys = new Set<string | number>(selectSquare(adj, pos, sel.a).map(sqKey));
            return convertObjects(allEdges, e => [e.n1, e.n2], allFrom, s => [s.n1, s.n2, s.n3, s.n4], sqKey, selectedKeys, mode);
        }
        case 'rrmn': {
            const base = selectEdge(adj, pos, sel.a);
            return randomlyRemove(base, sel.count);
        }
        case 'rrmp': {
            const base = selectEdge(adj, pos, sel.a);
            return randomlyRemove(base, Math.floor(sel.frac * base.length));
        }
        default:
            throw new Error(`selectEdge: unexpected edge-selector op '${(sel as Selector).op}'`);
    }
}

/**
 * Evaluates a triangle Selector against a board's adjacency matrix, returning the list of selected
 * triangles as BoardTriangle values (deduplicated) - the triangle counterpart of selectEdge above.
 * `(all)` is every triangle shared/topology.ts's findTriangles() finds. Mutually recursive with
 * selectNode()/selectEdge() via the conva/conve operators.
 */
export function selectTriangle(adj: number[][], pos: number[][], sel: Selector): BoardTriangle[] {
    if (sel.type !== 'tri')
        throw new Error(`selectTriangle: expected a triangle selector, got ${describeSelectorType(sel.type)} selector (op '${sel.op}')`);
    switch (sel.op) {
        case 'union':
            return dedupeByKey([...selectTriangle(adj, pos, sel.a), ...selectTriangle(adj, pos, sel.b)], triKey);
        case 'inter': {
            const a = selectTriangle(adj, pos, sel.a);
            const bKeys = new Set(selectTriangle(adj, pos, sel.b).map(triKey));
            return a.filter(t => bKeys.has(triKey(t)));
        }
        case 'diff': {
            const a = selectTriangle(adj, pos, sel.a);
            const bKeys = new Set(selectTriangle(adj, pos, sel.b).map(triKey));
            return a.filter(t => !bKeys.has(triKey(t)));
        }
        case 'compl': {
            const aKeys = new Set(selectTriangle(adj, pos, sel.a).map(triKey));
            return findTriangles(adj)
                .map(([u, v, w]) => makeBoardTriangle(u, v, w))
                .filter(t => !aKeys.has(triKey(t)));
        }
        case 'all':
            return findTriangles(adj).map(([u, v, w]) => makeBoardTriangle(u, v, w));
        case 'none':
            return [];
        case 'conva': case 'conve': {
            if (sel.from === 'sq')
                throw new Error(`selectTriangle: no association is defined between 'tri' and 'sq'`);
            const mode = sel.op === 'conva' ? 'all' : 'some';
            if (sel.from === 'tri') return selectTriangle(adj, pos, sel.a); // same-kind: no-op (defensive)
            const allTri = selectTriangle(adj, pos, { op: 'all', type: 'tri' });
            if (sel.from === 'node') {
                const selectedKeys = new Set<string | number>(selectNode(adj, pos, sel.a));
                const allNodes = Array.from({ length: adj.length }, (_, i) => i);
                return convertObjects(allTri, t => [t.n1, t.n2, t.n3], allNodes, n => [n], n => n, selectedKeys, mode);
            }
            const allFrom = selectEdge(adj, pos, { op: 'all', type: 'edge' });
            const selectedKeys = new Set<string | number>(selectEdge(adj, pos, sel.a).map(edgeKey));
            return convertObjects(allTri, t => [t.n1, t.n2, t.n3], allFrom, e => [e.n1, e.n2], edgeKey, selectedKeys, mode);
        }
        case 'rrmn': {
            const base = selectTriangle(adj, pos, sel.a);
            return randomlyRemove(base, sel.count);
        }
        case 'rrmp': {
            const base = selectTriangle(adj, pos, sel.a);
            return randomlyRemove(base, Math.floor(sel.frac * base.length));
        }
        default:
            throw new Error(`selectTriangle: unexpected triangle-selector op '${(sel as Selector).op}'`);
    }
}

/**
 * Evaluates a square Selector against a board's adjacency matrix, returning the list of selected
 * squares as BoardSquare values (deduplicated) - the square counterpart of selectTriangle above.
 * `(all)` is every square shared/topology.ts's findSquares() finds. Mutually recursive with
 * selectNode()/selectEdge() via the conva/conve operators.
 */
export function selectSquare(adj: number[][], pos: number[][], sel: Selector): BoardSquare[] {
    if (sel.type !== 'sq')
        throw new Error(`selectSquare: expected a square selector, got ${describeSelectorType(sel.type)} selector (op '${sel.op}')`);
    switch (sel.op) {
        case 'union':
            return dedupeByKey([...selectSquare(adj, pos, sel.a), ...selectSquare(adj, pos, sel.b)], sqKey);
        case 'inter': {
            const a = selectSquare(adj, pos, sel.a);
            const bKeys = new Set(selectSquare(adj, pos, sel.b).map(sqKey));
            return a.filter(s => bKeys.has(sqKey(s)));
        }
        case 'diff': {
            const a = selectSquare(adj, pos, sel.a);
            const bKeys = new Set(selectSquare(adj, pos, sel.b).map(sqKey));
            return a.filter(s => !bKeys.has(sqKey(s)));
        }
        case 'compl': {
            const aKeys = new Set(selectSquare(adj, pos, sel.a).map(sqKey));
            return findSquares(adj)
                .map(([a, b, c, d]) => makeBoardSquare(a, b, c, d))
                .filter(s => !aKeys.has(sqKey(s)));
        }
        case 'all':
            return findSquares(adj).map(([a, b, c, d]) => makeBoardSquare(a, b, c, d));
        case 'none':
            return [];
        case 'conva': case 'conve': {
            if (sel.from === 'tri')
                throw new Error(`selectSquare: no association is defined between 'tri' and 'sq'`);
            const mode = sel.op === 'conva' ? 'all' : 'some';
            if (sel.from === 'sq') return selectSquare(adj, pos, sel.a); // same-kind: no-op (defensive)
            const allSq = selectSquare(adj, pos, { op: 'all', type: 'sq' });
            if (sel.from === 'node') {
                const selectedKeys = new Set<string | number>(selectNode(adj, pos, sel.a));
                const allNodes = Array.from({ length: adj.length }, (_, i) => i);
                return convertObjects(allSq, s => [s.n1, s.n2, s.n3, s.n4], allNodes, n => [n], n => n, selectedKeys, mode);
            }
            const allFrom = selectEdge(adj, pos, { op: 'all', type: 'edge' });
            const selectedKeys = new Set<string | number>(selectEdge(adj, pos, sel.a).map(edgeKey));
            return convertObjects(allSq, s => [s.n1, s.n2, s.n3, s.n4], allFrom, e => [e.n1, e.n2], edgeKey, selectedKeys, mode);
        }
        case 'rrmn': {
            const base = selectSquare(adj, pos, sel.a);
            return randomlyRemove(base, sel.count);
        }
        case 'rrmp': {
            const base = selectSquare(adj, pos, sel.a);
            return randomlyRemove(base, Math.floor(sel.frac * base.length));
        }
        default:
            throw new Error(`selectSquare: unexpected square-selector op '${(sel as Selector).op}'`);
    }
}
