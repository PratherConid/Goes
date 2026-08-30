import {
    type BoardEdge, makeBoardEdge, type BoardTriangle, type BoardQuad,
    type Selector, type FormSelector, type SelectorType,
} from './types.js';
import { findTriangles, findQuads } from './topology.js';

// A tiny S-expression language for selecting a subset of a board's nodes, edges, triangles, or
// quads (a "triangle"/"quad" here is exactly what shared/topology.ts's findTriangles()/
// findQuads() finds - see BoardTriangle/BoardQuad in shared/types.ts). Grammar (SEL):
//
//   (union SEL...)          -- set union of zero or more operands (zero operands is the empty set)
//   (inter SEL...)          -- set intersection of zero or more operands (zero operands is the
//                               universal set - every object of whichever kind this SEL is, same as
//                               (all) - the usual absorbing-element convention for an empty fold)
//   (diff SEL SEL)          -- set difference (left minus right) - always exactly two operands
//   (compl SEL)             -- complement, within all objects of whichever kind SEL selects from
//   (more [<num>] SEL)      -- node/edge only: expands SEL's own result outward by <num> steps (a
//                               nonnegative integer, default 1 if omitted), repeating the one-step
//                               expansion that many times: for a node selector, one step adds every
//                               node reachable via one edge from the current selection; for an edge
//                               selector, one step adds every edge sharing a node with a currently
//                               selected edge - either way, SEL's own result stays included too, and
//                               0 steps is a no-op
//   (all)                   -- every object of whichever kind (node/edge/triangle/quad) this SEL
//                               is being parsed/evaluated as
//   (none)                  -- no objects of that kind
//   (deg <eq|gt|lt> <num>)  -- node selector only: nodes whose degree is =/>/< a given nonnegative
//                               integer
//   (conva <node|edge|tri|quad> SEL) -- converts SEL (of the given "from" kind) into whichever kind
//                               this selector itself is (the "to" kind - inferred from parsing
//                               context, same as all/none): a "to" object is selected iff ALL of its
//                               associated "from" objects are selected. Two objects (of possibly
//                               different kinds) are associated iff one's own node set is completely
//                               contained in the other's - always well-defined for two differing
//                               kinds, since node/edge/triangle/quad have strictly increasing
//                               arity (1/2/3/4), so containment can only run from the smaller-arity
//                               one into the larger. Converting a kind to itself is a no-op (SEL is
//                               returned as-is); triangle <-> quad has no meaningful association
//                               and is rejected.
//   (conve <node|edge|tri|quad> SEL) -- same as conva, but a "to" object is selected iff AT LEAST ONE
//                               of its associated "from" objects is selected
//   (rrmn <num> SEL)        -- randomly removes exactly num (a nonnegative integer) items from SEL's
//                               own result, uniformly at random
//   (rrmp <num> SEL)        -- randomly removes a fixed portion of SEL's own result: num (a
//                               nonnegative float) times SEL's own result size, rounded down
//
// `union`/`inter`/`diff`/`compl`/`all`/`none`/`rrmn`/`rrmp` are polymorphic across all four kinds;
// `more` is polymorphic across node/edge only (no adjacency notion is defined here for triangles/
// quads); `conva`/`conve` convert between any two kinds, naming the source kind via their own
// leading node/edge/tri/quad token (unlike `all`/`none`, there's more than one *other* kind it could
// mean, so it can't be inferred purely from parsing context) - except triangle <-> quad, which is
// rejected, and a kind converted to itself, which is a no-op (SEL passes through unchanged, not
// wrapped in a conva/conve node at all).
//
// Every Selector node (one monolithic type, below) carries its own `type` (which kind of set it
// denotes) - but rather than parse a type-less tree and infer/validate `type` bottom-up afterward,
// parsing itself is done by one function, parseSelExpr(c, type), self-recursive on the SAME `type`
// throughout (conva/conve, via parseConversion, are the only place a parse ever continues into a
// DIFFERENT type - naming it explicitly via their own leading node/edge/tri/quad token). This is
// what lets `all`/`none` skip spelling out which kind they mean: `type` comes from *which type
// parseSelExpr was called with*, not from anything written in the expression itself. `deg` (node
// only) and `more` (node/edge only) are rejected for any other `type` - see parseSelExpr's own
// cases. selectNode()/selectEdge()/selectTriangle()/selectQuad() (this file's own separate mutually
// recursive evaluators, one per kind rather than one parameterized function, since each returns a
// different container type - see their own doc comments) still re-check `sel.type` themselves
// rather than trusting it, since a Selector need not always come from parseSelExpr (e.g. a
// hand-built AST, or one round-tripped through JSON).
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

// Display name for `type` used in parseSelExpr's own "unknown X-selector operator"/rejection
// messages - 'tri' reads as "triangle" there (unlike e.g. describeSelectorType's "a tri", used
// instead by the select*() evaluators' own wrong-kind-selector messages).
const selectorKindName: Record<SelectorType, string> = { node: 'node', edge: 'edge', tri: 'triangle', quad: 'quad' };

// Reads conva/conve's own leading node/edge/tri/quad token (the "from" kind) and parses its operand
// via parseSelExpr(c, fromTok) - shared by parseSelExpr's own conva/conve case, `toType` being
// whichever `type` it was called with (the "to" kind, from parsing context, same as all/none).
// Throws if the (from, toType) pair is the one with no defined association (triangle <-> quad -
// see this file's own top comment); returns the parsed operand directly, unwrapped, for a same-kind
// conversion (a no-op).
function parseConversion(c: ParseCursor, op: 'conva' | 'conve', toType: SelectorType): Selector {
    const fromTok = c.next();
    if (fromTok !== 'node' && fromTok !== 'edge' && fromTok !== 'tri' && fromTok !== 'quad')
        throw new Error(`selector: (${op} ...) source kind must be 'node', 'edge', 'tri', or 'quad', got '${fromTok}'`);
    if ((fromTok === 'tri' && toType === 'quad') || (fromTok === 'quad' && toType === 'tri'))
        throw new Error(`selector: (${op} ...) has no association defined between 'tri' and 'quad'`);
    const a = parseSelExpr(c, fromTok);
    c.expect(')');
    return fromTok === toType ? a : { op, type: toType, from: fromTok, a };
}

// Parses a SEL of the given `type` - self-recursive on the SAME `type` throughout (conva/conve, via
// parseConversion above, are the only place a parse ever continues into a different type). Every
// Selector this returns has this same `type`, except where `type` doesn't actually match what got
// parsed - impossible, since every case below only reaches a return by consuming input that names
// `type` implicitly (all/none/deg/more/etc.) or explicitly rejects a mismatched one (conva/conve).
// `deg` (node only) and `more` (node/edge only) reject every other `type` with the same "unknown
// operator" message parsing an operator this function has no case for at all would produce (e.g.
// `deg` was never a recognized triangle/quad operator to begin with).
function parseSelExpr(c: ParseCursor, type: SelectorType): Selector {
    c.expect('(');
    const op = c.next();
    switch (op) {
        case 'union': case 'inter': {
            const items: Selector[] = [];
            while (c.peek() !== ')') items.push(parseSelExpr(c, type));
            c.expect(')');
            return { op, type, items };
        }
        case 'diff': {
            const a = parseSelExpr(c, type);
            const b = parseSelExpr(c, type);
            c.expect(')');
            return { op: 'diff', type, a, b };
        }
        case 'compl': {
            const a = parseSelExpr(c, type);
            c.expect(')');
            return { op: 'compl', type, a };
        }
        case 'more': {
            if (type !== 'node' && type !== 'edge')
                throw new Error(`selector: unknown ${selectorKindName[type]}-selector operator 'more'`);
            const steps = c.peek() === '(' ? undefined : nextNonnegInt(c, '(more ...) step count');
            const a = parseSelExpr(c, type);
            c.expect(')');
            return steps === undefined ? { op: 'more', type, a } : { op: 'more', type, steps, a };
        }
        case 'all':
            c.expect(')');
            return { op: 'all', type };
        case 'none':
            c.expect(')');
            return { op: 'none', type };
        case 'deg': {
            if (type !== 'node') throw new Error(`selector: unknown ${selectorKindName[type]}-selector operator 'deg'`);
            const cmpTok = c.next();
            if (cmpTok !== 'eq' && cmpTok !== 'gt' && cmpTok !== 'lt')
                throw new Error(`selector: (deg ...) comparator must be 'eq', 'gt', or 'lt', got '${cmpTok}'`);
            const n = nextNonnegInt(c, '(deg ...) argument');
            c.expect(')');
            return { op: 'deg', type, cmp: cmpTok, n };
        }
        case 'conva': case 'conve':
            return parseConversion(c, op, type);
        case 'rrmn': {
            const count = nextNonnegInt(c, '(rrmn ...) count');
            const a = parseSelExpr(c, type);
            c.expect(')');
            return { op: 'rrmn', type, count, a };
        }
        case 'rrmp': {
            const frac = nextNonnegNumber(c, '(rrmp ...) portion');
            const a = parseSelExpr(c, type);
            c.expect(')');
            return { op: 'rrmp', type, frac, a };
        }
        default:
            throw new Error(`selector: unknown ${selectorKindName[type]}-selector operator '${op}'`);
    }
}

// Shared by parseNodeSelector/parseEdgeSelector/parseTriangleSelector/parseQuadSelector:
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
 * node-selector context - see parseSelExpr). */
export function parseNodeSelector(s: string): Selector {
    return parseTopLevel(s, c => parseSelExpr(c, 'node'));
}

/** Parses `s` as an edge selector (see this file's own top comment for the grammar) - throws if `s`
 * doesn't follow the grammar. */
export function parseEdgeSelector(s: string): Selector {
    return parseTopLevel(s, c => parseSelExpr(c, 'edge'));
}

/** Parses `s` as a triangle selector (see this file's own top comment for the grammar) - throws if
 * `s` doesn't follow the grammar. */
export function parseTriangleSelector(s: string): Selector {
    return parseTopLevel(s, c => parseSelExpr(c, 'tri'));
}

/** Parses `s` as a quad selector (see this file's own top comment for the grammar) - throws if `s`
 * doesn't follow the grammar. */
export function parseQuadSelector(s: string): Selector {
    return parseTopLevel(s, c => parseSelExpr(c, 'quad'));
}

/** Formats `sel` back into the S-expression syntax parseNodeSelector()/parseEdgeSelector()/
 * parseTriangleSelector()/parseQuadSelector() accept - the inverse of parsing. Used e.g. to
 * round-trip a BoardModifier's own selector back into command-line text for display (see
 * src/sidePanel.ts's fmtModifiers). */
export function formatSelector(sel: Selector): string {
    switch (sel.op) {
        case 'union': case 'inter': {
            const inner = sel.items.map(formatSelector).join(' ');
            return inner ? `(${sel.op} ${inner})` : `(${sel.op})`;
        }
        case 'diff':
            return `(diff ${formatSelector(sel.a)} ${formatSelector(sel.b)})`;
        case 'compl':
            return `(compl ${formatSelector(sel.a)})`;
        case 'more':
            return sel.steps === undefined
                ? `(more ${formatSelector(sel.a)})` : `(more ${sel.steps} ${formatSelector(sel.a)})`;
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
        case 'raw':
            // No grammar production exists for embedding an already-materialized SelectedVals as
            // text (unlike every other op, which is built from other Selectors/literals this
            // grammar can already express) - a 'raw' Selector can only be built/consumed
            // programmatically, never round-tripped through source text.
            throw new Error(`selector: 'raw' has no text representation`);
    }
}

// Parses one `(tri [SEL])` / `(quad [SEL])` - mirrors parseSelExpr's own "consume '(', read a
// leading token, dispatch" shape, but there's no mutual recursion here: a FormSelector's own SEL
// is parsed via the ordinary parseSelExpr, not another parseFormSelExpr.
function parseFormSelExpr(c: ParseCursor): FormSelector {
    c.expect('(');
    const kind = c.next();
    if (kind !== 'tri' && kind !== 'quad')
        throw new Error(`form selector: expected 'tri' or 'quad', got '${kind}'`);
    if (c.peek() === ')') { c.next(); return { kind }; }
    const sel = parseSelExpr(c, kind);
    c.expect(')');
    return { kind, sel };
}

/**
 * Parses `s` as zero or more back-to-back form selectors (`(tri [SEL])` / `(quad [SEL])`, see
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

/** Formats one FormSelector back into the `(tri [SEL])` / `(quad [SEL])` syntax parseFormSelectors()
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

// "a node"/"an edge"/"a tri"/"a quad" - shared by each evaluator's own wrong-kind error message below.
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

// BoardEdge/BoardTriangle/BoardQuad aren't valid Set/Map keys themselves (two structurally-equal
// values are different objects), so union/inter/diff/compl below key them by these canonical string
// ids whenever they need set-like membership tests - each type's own canonical-construction
// invariant (BoardEdge/BoardTriangle: n1 < n2 < ...; BoardQuad: the lexicographically-least of its
// own cycle's 8 rotation/reflection relabelings - see makeBoardQuad, shared/types.ts) already makes
// this unique per object, regardless of which vertex/direction it was found from.
function edgeKey(e: BoardEdge): string {
    return `${e.n1},${e.n2}`;
}
function triKey(t: BoardTriangle): string {
    return `${t.n1},${t.n2},${t.n3}`;
}
function quadKey(s: BoardQuad): string {
    return `${s.n1},${s.n2},${s.n3},${s.n4}`;
}

// Returns a NEW array with exactly `removeCount` (clamped to [0, items.length], since removing more
// than exist isn't meaningful) uniformly-randomly-chosen elements dropped, via a partial
// Fisher-Yates shuffle (only the first `removeCount` positions need to be randomized to pick which
// elements to drop). The rest of this file's set operations are pure and order-preserving; rrmn/rrmp
// (the only Selector ops that use this) are neither - two evaluations of the same selector can
// return different results, and the kept elements' relative order isn't preserved either. Exported
// for shared/cleg.ts's own `randRmN` builtin, which performs this exact operation on a cleg set.
export function randomlyRemove<T>(items: T[], removeCount: number): T[] {
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
// has a fixed arity (node 1, edge 2, triangle 3, quad 4) and every object's own members are
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
 * indices. Mutually recursive with selectEdge()/selectTriangle()/selectQuad() via the conva/conve
 * operators. `pos` isn't used by any selector in the current grammar, but is threaded through
 * (matching the other three evaluators' own signatures) for future position-based selectors.
 */
export function selectNode(adj: number[][], pos: number[][], sel: Selector): Set<number> {
    if (sel.type !== 'node')
        throw new Error(`selectNode: expected a node selector, got ${describeSelectorType(sel.type)} selector (op '${sel.op}')`);
    const N = adj.length;
    switch (sel.op) {
        case 'union': {
            const out = new Set<number>();
            for (const item of sel.items) for (const x of selectNode(adj, pos, item)) out.add(x);
            return out;
        }
        case 'inter': {
            // Zero operands is the universal set, matching (all) - see Selector's own doc comment
            // (shared/types.ts) on why an empty inter is the identity for intersection's fold.
            if (sel.items.length === 0) return selectNode(adj, pos, { op: 'all', type: 'node' });
            let acc = selectNode(adj, pos, sel.items[0]);
            for (let i = 1; i < sel.items.length; i++) {
                const next = selectNode(adj, pos, sel.items[i]);
                acc = new Set([...acc].filter(x => next.has(x)));
            }
            return acc;
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
            // Repeats the one-step expansion `steps` times (default 1) - each step only walks
            // `frontier` (the nodes newly added by the PREVIOUS step, not the whole accumulated `out`
            // again), since a node's own neighbors were already fully explored the one time it itself
            // became part of the frontier.
            let frontier = [...a];
            const steps = sel.steps ?? 1;
            for (let s = 0; s < steps && frontier.length > 0; s++) {
                const nextFrontier: number[] = [];
                for (const i of frontier)
                    for (let j = 0; j < N; j++)
                        if (adj[i][j] && !out.has(j)) { out.add(j); nextFrontier.push(j); }
                frontier = nextFrontier;
            }
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
            const allFrom = selectQuad(adj, pos, { op: 'all', type: 'quad' });
            const selectedKeys = new Set<string | number>(selectQuad(adj, pos, sel.a).map(quadKey));
            return new Set(convertObjects(toNodes, n => [n], allFrom, s => [s.n1, s.n2, s.n3, s.n4], quadKey, selectedKeys, mode));
        }
        case 'rrmn': {
            const base = [...selectNode(adj, pos, sel.a)];
            return new Set(randomlyRemove(base, sel.count));
        }
        case 'rrmp': {
            const base = [...selectNode(adj, pos, sel.a)];
            return new Set(randomlyRemove(base, Math.floor(sel.frac * base.length)));
        }
        case 'raw':
            // sel.type !== 'node' was already rejected above, but that doesn't by itself guarantee
            // sel.items (a separately-tagged SelectedVals) agrees - a hand-built Selector could still
            // have the two fields out of sync, so this is checked for real, not just asserted.
            if (sel.items.kind !== 'node')
                throw new Error(`selectNode: 'raw' selector's own items must be node-kind, got '${sel.items.kind}'`);
            return new Set(sel.items.value);
        default:
            throw new Error(`selectNode: unexpected node-selector op '${(sel as Selector).op}'`);
    }
}

/**
 * Evaluates an edge Selector against a board's adjacency matrix, returning the list of selected
 * edges as BoardEdge values (deduplicated). Mutually recursive with selectNode()/selectTriangle()/
 * selectQuad() via the conva/conve operators.
 */
export function selectEdge(adj: number[][], pos: number[][], sel: Selector): BoardEdge[] {
    if (sel.type !== 'edge')
        throw new Error(`selectEdge: expected an edge selector, got ${describeSelectorType(sel.type)} selector (op '${sel.op}')`);
    const N = adj.length;
    switch (sel.op) {
        case 'union':
            return dedupeByKey(sel.items.flatMap(item => selectEdge(adj, pos, item)), edgeKey);
        case 'inter': {
            if (sel.items.length === 0) return selectEdge(adj, pos, { op: 'all', type: 'edge' });
            let acc = selectEdge(adj, pos, sel.items[0]);
            for (let i = 1; i < sel.items.length; i++) {
                const nextKeys = new Set(selectEdge(adj, pos, sel.items[i]).map(edgeKey));
                acc = acc.filter(e => nextKeys.has(edgeKey(e)));
            }
            return acc;
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
            const outByKey = new Map<string | number, BoardEdge>(a.map(e => [edgeKey(e), e]));
            const touchedNodes = new Set<number>();
            for (const e of a) { touchedNodes.add(e.n1); touchedNodes.add(e.n2); }
            // Repeats the one-step expansion `steps` times (default 1). Mirrors the node case's own
            // frontier trick: `frontier` is only the nodes newly touched by the PREVIOUS step, since a
            // node's own incident edges are all added to `outByKey` the one time it itself becomes
            // part of the frontier - a later step never needs to re-scan an already-touched node.
            let frontier = [...touchedNodes];
            const steps = sel.steps ?? 1;
            for (let s = 0; s < steps && frontier.length > 0; s++) {
                const nextFrontier: number[] = [];
                for (const i of frontier)
                    for (let j = 0; j < N; j++) {
                        if (!adj[i][j]) continue;
                        outByKey.set(edgeKey(makeBoardEdge(i, j)), makeBoardEdge(i, j));
                        if (!touchedNodes.has(j)) { touchedNodes.add(j); nextFrontier.push(j); }
                    }
                frontier = nextFrontier;
            }
            return [...outByKey.values()];
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
            const allFrom = selectQuad(adj, pos, { op: 'all', type: 'quad' });
            const selectedKeys = new Set<string | number>(selectQuad(adj, pos, sel.a).map(quadKey));
            return convertObjects(allEdges, e => [e.n1, e.n2], allFrom, s => [s.n1, s.n2, s.n3, s.n4], quadKey, selectedKeys, mode);
        }
        case 'rrmn': {
            const base = selectEdge(adj, pos, sel.a);
            return randomlyRemove(base, sel.count);
        }
        case 'rrmp': {
            const base = selectEdge(adj, pos, sel.a);
            return randomlyRemove(base, Math.floor(sel.frac * base.length));
        }
        case 'raw':
            if (sel.items.kind !== 'edge')
                throw new Error(`selectEdge: 'raw' selector's own items must be edge-kind, got '${sel.items.kind}'`);
            return [...sel.items.value];
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
            return dedupeByKey(sel.items.flatMap(item => selectTriangle(adj, pos, item)), triKey);
        case 'inter': {
            if (sel.items.length === 0) return selectTriangle(adj, pos, { op: 'all', type: 'tri' });
            let acc = selectTriangle(adj, pos, sel.items[0]);
            for (let i = 1; i < sel.items.length; i++) {
                const nextKeys = new Set(selectTriangle(adj, pos, sel.items[i]).map(triKey));
                acc = acc.filter(t => nextKeys.has(triKey(t)));
            }
            return acc;
        }
        case 'diff': {
            const a = selectTriangle(adj, pos, sel.a);
            const bKeys = new Set(selectTriangle(adj, pos, sel.b).map(triKey));
            return a.filter(t => !bKeys.has(triKey(t)));
        }
        case 'compl': {
            const aKeys = new Set(selectTriangle(adj, pos, sel.a).map(triKey));
            return findTriangles(adj).filter(t => !aKeys.has(triKey(t)));
        }
        case 'all':
            return findTriangles(adj);
        case 'none':
            return [];
        case 'conva': case 'conve': {
            if (sel.from === 'quad')
                throw new Error(`selectTriangle: no association is defined between 'tri' and 'quad'`);
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
        case 'raw':
            if (sel.items.kind !== 'tri')
                throw new Error(`selectTriangle: 'raw' selector's own items must be tri-kind, got '${sel.items.kind}'`);
            return [...sel.items.value];
        default:
            throw new Error(`selectTriangle: unexpected triangle-selector op '${(sel as Selector).op}'`);
    }
}

/**
 * Evaluates a quad Selector against a board's adjacency matrix, returning the list of selected
 * quads as BoardQuad values (deduplicated) - the quad counterpart of selectTriangle above.
 * `(all)` is every quad shared/topology.ts's findQuads() finds. Mutually recursive with
 * selectNode()/selectEdge() via the conva/conve operators.
 */
export function selectQuad(adj: number[][], pos: number[][], sel: Selector): BoardQuad[] {
    if (sel.type !== 'quad')
        throw new Error(`selectQuad: expected a quad selector, got ${describeSelectorType(sel.type)} selector (op '${sel.op}')`);
    switch (sel.op) {
        case 'union':
            return dedupeByKey(sel.items.flatMap(item => selectQuad(adj, pos, item)), quadKey);
        case 'inter': {
            if (sel.items.length === 0) return selectQuad(adj, pos, { op: 'all', type: 'quad' });
            let acc = selectQuad(adj, pos, sel.items[0]);
            for (let i = 1; i < sel.items.length; i++) {
                const nextKeys = new Set(selectQuad(adj, pos, sel.items[i]).map(quadKey));
                acc = acc.filter(s => nextKeys.has(quadKey(s)));
            }
            return acc;
        }
        case 'diff': {
            const a = selectQuad(adj, pos, sel.a);
            const bKeys = new Set(selectQuad(adj, pos, sel.b).map(quadKey));
            return a.filter(s => !bKeys.has(quadKey(s)));
        }
        case 'compl': {
            const aKeys = new Set(selectQuad(adj, pos, sel.a).map(quadKey));
            return findQuads(adj).filter(s => !aKeys.has(quadKey(s)));
        }
        case 'all':
            return findQuads(adj);
        case 'none':
            return [];
        case 'conva': case 'conve': {
            if (sel.from === 'tri')
                throw new Error(`selectQuad: no association is defined between 'tri' and 'quad'`);
            const mode = sel.op === 'conva' ? 'all' : 'some';
            if (sel.from === 'quad') return selectQuad(adj, pos, sel.a); // same-kind: no-op (defensive)
            const allQuad = selectQuad(adj, pos, { op: 'all', type: 'quad' });
            if (sel.from === 'node') {
                const selectedKeys = new Set<string | number>(selectNode(adj, pos, sel.a));
                const allNodes = Array.from({ length: adj.length }, (_, i) => i);
                return convertObjects(allQuad, s => [s.n1, s.n2, s.n3, s.n4], allNodes, n => [n], n => n, selectedKeys, mode);
            }
            const allFrom = selectEdge(adj, pos, { op: 'all', type: 'edge' });
            const selectedKeys = new Set<string | number>(selectEdge(adj, pos, sel.a).map(edgeKey));
            return convertObjects(allQuad, s => [s.n1, s.n2, s.n3, s.n4], allFrom, e => [e.n1, e.n2], edgeKey, selectedKeys, mode);
        }
        case 'rrmn': {
            const base = selectQuad(adj, pos, sel.a);
            return randomlyRemove(base, sel.count);
        }
        case 'rrmp': {
            const base = selectQuad(adj, pos, sel.a);
            return randomlyRemove(base, Math.floor(sel.frac * base.length));
        }
        case 'raw':
            if (sel.items.kind !== 'quad')
                throw new Error(`selectQuad: 'raw' selector's own items must be quad-kind, got '${sel.items.kind}'`);
            return [...sel.items.value];
        default:
            throw new Error(`selectQuad: unexpected quad-selector op '${(sel as Selector).op}'`);
    }
}
