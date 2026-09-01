import {
    type BoardEdge, makeBoardEdge, type BoardSimplex, type BoardQuad,
    type Selector, type SelectorType, simpType, simpN,
} from './types.js';
import { findSimplices, findQuads } from './topology.js';

// A tiny S-expression language for selecting a subset of a board's nodes, edges, simplices, or
// quads (a "simplex"/"quad" here is exactly what shared/topology.ts's findSimplices()/
// findQuads() finds - see BoardSimplex/BoardQuad in shared/types.ts). A "simp N" selector denotes
// N+1-node simplices (cliques) - "tri" is accepted everywhere as sugar for "simp 2" (parses to the
// identical Selector, evaluates identically). Grammar (SEL):
//
//   (union SEL...)          -- set union of one or more operands, all the same kind
//   (inter SEL...)          -- set intersection of one or more operands, all the same kind
//   (diff SEL SEL)          -- set difference (left minus right) - always exactly two operands,
//                               both the same kind
//   (compl SEL)             -- complement, within all objects of whichever kind SEL selects from
//   (more [<num>] SEL)      -- node/edge only: expands SEL's own result outward by <num> steps (a
//                               nonnegative integer, default 1 if omitted), repeating the one-step
//                               expansion that many times: for a node selector, one step adds every
//                               node reachable via one edge from the current selection; for an edge
//                               selector, one step adds every edge sharing a node with a currently
//                               selected edge - either way, SEL's own result stays included too, and
//                               0 steps is a no-op
//   (all <node|edge|simp N|tri|quad>)  -- every object of the given kind ("tri" is sugar for "simp 2")
//   (none <node|edge|simp N|tri|quad>) -- no objects of that kind
//   (deg <eq|gt|lt> <num>)  -- node selector only: nodes whose degree is =/>/< a given nonnegative
//                               integer
//   (conva <node|edge|simp N|tri|quad> SEL) -- converts SEL (of whichever kind SEL itself turns out
//                               to be - its "from" kind) into the given kind (its "to" kind, named
//                               explicitly since nothing else determines it): a "to" object is
//                               selected iff ALL of its associated "from" objects are selected. Two
//                               objects (of possibly different kinds) are associated iff one's own
//                               node set is completely contained in the other's - always well-defined
//                               for two differing kinds, since node/edge/simp-N/quad have strictly
//                               increasing-or-unrelated arity (1/2/N+1/4), so containment can only
//                               run from the smaller-arity one into the larger. Converting a kind to
//                               itself (including simp M -> simp M) is a no-op (SEL is returned
//                               as-is); simp <-> quad (of any N, including tri) has no meaningful
//                               association and is rejected; simp M <-> simp N for M != N is allowed,
//                               via the same general containment rule.
//   (conve <node|edge|simp N|tri|quad> SEL) -- same as conva, but a "to" object is selected iff AT
//                               LEAST ONE of its associated "from" objects is selected
//   (rrmn <num> SEL)        -- randomly removes exactly num (a nonnegative integer) items from SEL's
//                               own result, uniformly at random
//   (rrmp <num> SEL)        -- randomly removes a fixed portion of SEL's own result: num (a
//                               nonnegative float) times SEL's own result size, rounded down
//
// `union`/`inter`/`diff`/`compl`/`all`/`none`/`rrmn`/`rrmp` are polymorphic across every kind;
// `more` is polymorphic across node/edge only (no adjacency notion is defined here for simplices/
// quads); `conva`/`conve` convert between any two kinds, naming the RESULT (the "to") kind via
// their own leading node/edge/simp-N/tri/quad token (the "from" kind is instead read off of SEL
// itself, once SEL has been parsed - see below) - except simp <-> quad, which is rejected, and a
// kind converted to itself, which is a no-op (SEL passes through unchanged, not wrapped in a
// conva/conve node at all).
//
// Every Selector node (one monolithic type, below) carries its own `type` (which kind of set it
// denotes). Type inference is bottom-up: parseSelExpr(c) takes no expected-kind parameter at all -
// each case determines its own `type` from what it just parsed (an operand's own already-parsed
// `type`, propagated up unchanged for union/inter/diff/compl/more/rrmn/rrmp; the explicit leading
// token for conva/conve's "to" kind; hardcoded 'node' for deg, since that's the only kind it's ever
// valid for) rather than being told what `type` to parse as by its caller. Two consequences of this
// follow directly from there being no operand (and therefore nothing to infer from) to fall back on:
// `all`/`none` must name their own kind explicitly (there's no longer a parsing context to infer it
// from), and `union`/`inter` require at least one operand (a zero-operand union/inter has nothing to
// infer ITS kind from either - `(all <kind>)`/`(none <kind>)` already cover those identity cases
// directly, so a bare `(union)`/`(inter)` is simply redundant, not just unparseable). `deg` (node
// only) and `more` (node/edge only) still reject any other kind, just checked AFTER parsing their
// own operand (via its own inferred `type`) rather than before, since there's no longer a `type`
// context to check against up front. parseNodeSelector()/parseEdgeSelector()/
// parseTriangleSelector()/parseQuadSelector() below parse via the one context-free parseSelExpr(c)
// and then check the RESULT's own `type` against what each of them promises to return - the mirror
// image of the old top-down scheme, where that same check was made impossible to fail by construction.
// selectNode()/selectEdge()/selectSimp()/selectQuad() (this file's own separate mutually
// recursive evaluators, one per kind - selectSimp parameterized over N, the other three fixed -
// since each returns a different container type - see their own doc comments; selectTriangle() is
// selectSimp()'s own N=2 special case, kept as a thin sugar wrapper) still re-check `sel.type`
// themselves
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

// Display name for `type` used in parseSelExpr's own rejection messages and describeSelectorType
// below - 'simpN' reads as "simp N" (N=2 still reads as "triangle", since that's the name every
// existing triangle-specific error message already uses and callers still recognize).
function selectorKindName(type: SelectorType): string {
    if (type === 'node' || type === 'edge' || type === 'quad') return type;
    const n = simpN(type)!;
    return n === 2 ? 'triangle' : `simp ${n}`;
}

// "a node"/"an edge"/"a simp 3"/"a quad" - shared by parseSelectorAsType's own wrong-result-kind
// message below, and by each evaluator's own wrong-kind error message further down.
function describeSelectorType(type: SelectorType): string {
    const name = selectorKindName(type);
    return `${type === 'edge' ? 'an' : 'a'} ${name}`;
}

// Reads a node/edge/quad/tri/"simp N" kind token (the shape (all ...)/(none ...)/conva/conve's own
// leading token all share) - 'tri' is sugar for simpType(2); 'simp' consumes one more token, an
// integer >= 2, and builds simpType(n). `context` names the production in the thrown error.
function parseSelectorTypeToken(c: ParseCursor, context: string): SelectorType {
    const tok = c.next();
    if (tok === 'node' || tok === 'edge' || tok === 'quad') return tok;
    if (tok === 'tri') return simpType(2);
    if (tok === 'simp') {
        const nTok = c.next();
        const n = Number(nTok);
        if (!Number.isInteger(n) || n < 2)
            throw new Error(`selector: ${context} 'simp' arity must be an integer >= 2, got '${nTok}'`);
        return simpType(n);
    }
    throw new Error(`selector: ${context} kind must be 'node', 'edge', 'simp <n>', 'tri', or 'quad', got '${tok}'`);
}

// True iff `type` is some simp N (of any N) - the general stand-in every place that used to check
// `=== 'tri'` specifically now needs, since a bare 'tri' string no longer exists internally.
function isSimpType(type: SelectorType): boolean {
    return simpN(type) !== null;
}

// Reads conva/conve's own leading kind token (now the "to"/result kind - see this file's own top
// comment) and parses its operand via the ordinary context-free parseSelExpr(c); the operand's own
// bottom-up-inferred `type` is the "from" kind. Throws if the (from, toTok) pair is the one with no
// defined association (simp <-> quad, of any simp arity); returns the parsed operand directly,
// unwrapped, for a same-kind conversion (a no-op).
function parseConversion(c: ParseCursor, op: 'conva' | 'conve'): Selector {
    const toTok = parseSelectorTypeToken(c, `(${op} ...) result`);
    const a = parseSelExpr(c);
    c.expect(')');
    if ((isSimpType(a.type) && toTok === 'quad') || (a.type === 'quad' && isSimpType(toTok)))
        throw new Error(`selector: (${op} ...) has no association defined between 'simp' and 'quad'`);
    return a.type === toTok ? a : { op, type: toTok, from: a.type, a };
}

// Parses one SEL, inferring its own `type` bottom-up rather than being told what to expect (see this
// file's own top comment) - context-free, unlike the old parseSelExpr(c, type). Every case below
// either propagates an operand's own already-parsed `type` unchanged (union/inter/diff/compl/rrmn/
// rrmp - diff/union/inter also check their operands agree with each other), reads an explicit
// leading token because nothing else could supply the kind (all/none's own kind; conva/conve's own
// "to" kind, via parseConversion above), or is hardcoded to a single always-valid kind (deg: always
// 'node'). `deg`/`more` reject a mismatched kind (deg implicitly, by always being 'node' regardless
// of context; more explicitly, by checking its own operand's inferred `type` after the fact) -
// there's no longer a `type` context to check against up front the way the old top-down version did.
function parseSelExpr(c: ParseCursor): Selector {
    c.expect('(');
    const op = c.next();
    switch (op) {
        case 'union': case 'inter': {
            const items: Selector[] = [];
            while (c.peek() !== ')') items.push(parseSelExpr(c));
            c.expect(')');
            if (items.length === 0)
                throw new Error(
                    `selector: (${op} ...) needs at least one operand - its own kind can't be ` +
                    `inferred bottom-up from zero operands; use (all <kind>)/(none <kind>) directly ` +
                    `for the identity case`);
            const type = items[0].type;
            for (let i = 1; i < items.length; i++)
                if (items[i].type !== type)
                    throw new Error(
                        `selector: (${op} ...) operands must all be the same kind - operand 1 is ` +
                        `${selectorKindName(type)}, operand ${i + 1} is ${selectorKindName(items[i].type)}`);
            return { op, type, items };
        }
        case 'diff': {
            const a = parseSelExpr(c);
            const b = parseSelExpr(c);
            c.expect(')');
            if (a.type !== b.type)
                throw new Error(
                    `selector: (diff ...) operands must be the same kind - got ` +
                    `${selectorKindName(a.type)} and ${selectorKindName(b.type)}`);
            return { op: 'diff', type: a.type, a, b };
        }
        case 'compl': {
            const a = parseSelExpr(c);
            c.expect(')');
            return { op: 'compl', type: a.type, a };
        }
        case 'more': {
            const steps = c.peek() === '(' ? undefined : nextNonnegInt(c, '(more ...) step count');
            const a = parseSelExpr(c);
            c.expect(')');
            if (a.type !== 'node' && a.type !== 'edge')
                throw new Error(`selector: (more ...) requires a node or edge selector, got a ${selectorKindName(a.type)} selector`);
            return steps === undefined ? { op: 'more', type: a.type, a } : { op: 'more', type: a.type, steps, a };
        }
        case 'all': case 'none': {
            const type = parseSelectorTypeToken(c, `(${op} ...)`);
            c.expect(')');
            return { op, type };
        }
        case 'deg': {
            const cmpTok = c.next();
            if (cmpTok !== 'eq' && cmpTok !== 'gt' && cmpTok !== 'lt')
                throw new Error(`selector: (deg ...) comparator must be 'eq', 'gt', or 'lt', got '${cmpTok}'`);
            const n = nextNonnegInt(c, '(deg ...) argument');
            c.expect(')');
            return { op: 'deg', type: 'node', cmp: cmpTok, n };
        }
        case 'conva': case 'conve':
            return parseConversion(c, op);
        case 'rrmn': {
            const count = nextNonnegInt(c, '(rrmn ...) count');
            const a = parseSelExpr(c);
            c.expect(')');
            return { op: 'rrmn', type: a.type, count, a };
        }
        case 'rrmp': {
            const frac = nextNonnegNumber(c, '(rrmp ...) portion');
            const a = parseSelExpr(c);
            c.expect(')');
            return { op: 'rrmp', type: a.type, frac, a };
        }
        default:
            throw new Error(`selector: unknown selector operator '${op}'`);
    }
}

/** Parses `s` as a selector of whichever kind it turns out to be, inferred bottom-up from `s` itself
 * (see this file's own top comment) - throws if `s` doesn't follow the grammar. Unlike
 * parseNodeSelector/parseEdgeSelector/parseTriangleSelector/parseQuadSelector below, doesn't check
 * the result's own kind against anything; used wherever a selector's own kind isn't fixed ahead of
 * the call (e.g. cleg's own mkSel builtin, shared/clegEval.ts). */
export function parseSelector(s: string): Selector {
    const tokens = tokenize(s);
    if (tokens.length === 0) throw new Error('selector: empty input');
    const c = new ParseCursor(tokens);
    const sel = parseSelExpr(c);
    if (!c.atEnd()) throw new Error(`selector: unexpected trailing input starting at '${c.peek()}'`);
    return sel;
}

// Shared by parseNodeSelector/parseEdgeSelector/parseTriangleSelector/parseQuadSelector: parses `s`
// via parseSelector above, then checks the result's own bottom-up-inferred `type` against `want` (the
// mirror image of the old top-down scheme, where `want` was threaded in as parsing context and this
// check was unreachable).
function parseSelectorAsType(s: string, want: SelectorType): Selector {
    const sel = parseSelector(s);
    if (sel.type !== want)
        throw new Error(`selector: expected ${describeSelectorType(want)} selector, got ${describeSelectorType(sel.type)} selector (op '${sel.op}')`);
    return sel;
}

/** Parses `s` as a node selector (see this file's own top comment for the grammar) - throws if `s`
 * doesn't follow the grammar, or parses to a selector of a different kind. */
export function parseNodeSelector(s: string): Selector {
    return parseSelectorAsType(s, 'node');
}

/** Parses `s` as an edge selector (see this file's own top comment for the grammar) - throws if `s`
 * doesn't follow the grammar, or parses to a selector of a different kind. */
export function parseEdgeSelector(s: string): Selector {
    return parseSelectorAsType(s, 'edge');
}

/** Parses `s` as a triangle (simp 2) selector (see this file's own top comment for the grammar) -
 * throws if `s` doesn't follow the grammar, or parses to a selector of a different kind. */
export function parseTriangleSelector(s: string): Selector {
    return parseSelectorAsType(s, simpType(2));
}

/** Parses `s` as a quad selector (see this file's own top comment for the grammar) - throws if `s`
 * doesn't follow the grammar, or parses to a selector of a different kind. */
export function parseQuadSelector(s: string): Selector {
    return parseSelectorAsType(s, 'quad');
}

/** Parses `s` as a simp `n` selector (see this file's own top comment for the grammar) - throws if
 * `s` doesn't follow the grammar, or parses to a selector of a different kind/arity. The general,
 * n-parameterized counterpart of parseTriangleSelector (its own n=2 special case). */
export function parseSimpSelector(n: number, s: string): Selector {
    return parseSelectorAsType(s, simpType(n));
}

/** Formats `sel` back into the S-expression syntax parseNodeSelector()/parseEdgeSelector()/
 * parseTriangleSelector()/parseQuadSelector() accept - the inverse of parsing. Used e.g. to
 * round-trip a BoardModifier's own selector back into command-line text for display (see
 * src/sidePanel.ts's fmtModifiers). */
// Renders a SelectorType back into its own grammar token(s) - "node"/"edge"/"quad" as-is, or
// "simp N" (two tokens) for a simp type. Always uses the canonical "simp N" spelling, even for
// simp 2 - round-tripping doesn't preserve "tri" sugar, only meaning (the same simp 2 Selector
// results either way).
function formatSelectorType(type: SelectorType): string {
    const n = simpN(type);
    return n === null ? type : `simp ${n}`;
}

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
            return `(${sel.op} ${formatSelectorType(sel.type)})`;
        case 'deg':
            return `(deg ${sel.cmp} ${sel.n})`;
        case 'conva': case 'conve':
            // sel.type is the "to"/result kind - see this file's own top comment on why that's what
            // the explicit token now names (sel.from, the "to" kind's mirror, is read off sel.a's
            // own type instead, so it doesn't need spelling out here).
            return `(${sel.op} ${formatSelectorType(sel.type)} ${formatSelector(sel.a)})`;
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

// ── evaluation ───────────────────────────────────────────────────────────────

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

// BoardEdge/BoardSimplex/BoardQuad aren't valid Set/Map keys themselves (two structurally-equal
// values are different objects), so union/inter/diff/compl below key them by these canonical string
// ids whenever they need set-like membership tests - each type's own canonical-construction
// invariant (BoardEdge: n1 <= n2; BoardSimplex: nodes ascending; BoardQuad: the
// lexicographically-least of its own cycle's 8 rotation/reflection relabelings - see makeBoardQuad,
// shared/types.ts) already makes this unique per object, regardless of which vertex/direction it
// was found from.
function edgeKey(e: BoardEdge): string {
    return `${e.n1},${e.n2}`;
}
function simpKey(t: BoardSimplex): string {
    return t.nodes.join(',');
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
// for shared/clegEval.ts's own `randRmN` builtin, which performs this exact operation on a cleg set.
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
// has a fixed arity (node 1, edge 2, simp N N+1, quad 4) and every object's own members are
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
 * indices. Mutually recursive with selectEdge()/selectSimp()/selectQuad() via the conva/conve
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
            if (isSimpType(sel.from)) {
                const allFrom = selectSimp(adj, pos, { op: 'all', type: sel.from });
                const selectedKeys = new Set<string | number>(selectSimp(adj, pos, sel.a).map(simpKey));
                return new Set(convertObjects(toNodes, n => [n], allFrom, t => t.nodes, simpKey, selectedKeys, mode));
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
 * edges as BoardEdge values (deduplicated). Mutually recursive with selectNode()/selectSimp()/
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
            if (isSimpType(sel.from)) {
                const allFrom = selectSimp(adj, pos, { op: 'all', type: sel.from });
                const selectedKeys = new Set<string | number>(selectSimp(adj, pos, sel.a).map(simpKey));
                return convertObjects(allEdges, e => [e.n1, e.n2], allFrom, t => t.nodes, simpKey, selectedKeys, mode);
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
 * Evaluates a simp Selector (of any arity N - `sel.type` is some `simp N`) against a board's
 * adjacency matrix, returning the list of selected N-simplices as BoardSimplex values
 * (deduplicated) - the simplex counterpart of selectEdge/selectQuad above. `(all)` is every
 * N-simplex shared/topology.ts's findSimplices() finds. Mutually recursive with selectNode()/
 * selectEdge()/selectQuad() (and itself, for a simp M <-> simp N conversion) via the conva/conve
 * operators.
 */
export function selectSimp(adj: number[][], pos: number[][], sel: Selector): BoardSimplex[] {
    const n = simpN(sel.type);
    if (n === null)
        throw new Error(`selectSimp: expected a simp selector, got ${describeSelectorType(sel.type)} selector (op '${sel.op}')`);
    switch (sel.op) {
        case 'union':
            return dedupeByKey(sel.items.flatMap(item => selectSimp(adj, pos, item)), simpKey);
        case 'inter': {
            if (sel.items.length === 0) return selectSimp(adj, pos, { op: 'all', type: sel.type });
            let acc = selectSimp(adj, pos, sel.items[0]);
            for (let i = 1; i < sel.items.length; i++) {
                const nextKeys = new Set(selectSimp(adj, pos, sel.items[i]).map(simpKey));
                acc = acc.filter(t => nextKeys.has(simpKey(t)));
            }
            return acc;
        }
        case 'diff': {
            const a = selectSimp(adj, pos, sel.a);
            const bKeys = new Set(selectSimp(adj, pos, sel.b).map(simpKey));
            return a.filter(t => !bKeys.has(simpKey(t)));
        }
        case 'compl': {
            const aKeys = new Set(selectSimp(adj, pos, sel.a).map(simpKey));
            return findSimplices(adj, n).filter(t => !aKeys.has(simpKey(t)));
        }
        case 'all':
            return findSimplices(adj, n);
        case 'none':
            return [];
        case 'conva': case 'conve': {
            if (sel.from === 'quad')
                throw new Error(`selectSimp: no association is defined between 'simp' and 'quad'`);
            const mode = sel.op === 'conva' ? 'all' : 'some';
            if (sel.from === sel.type) return selectSimp(adj, pos, sel.a); // same-kind: no-op (defensive)
            const allTo = selectSimp(adj, pos, { op: 'all', type: sel.type });
            if (sel.from === 'node') {
                const selectedKeys = new Set<string | number>(selectNode(adj, pos, sel.a));
                const allNodes = Array.from({ length: adj.length }, (_, i) => i);
                return convertObjects(allTo, t => t.nodes, allNodes, n2 => [n2], n2 => n2, selectedKeys, mode);
            }
            if (sel.from === 'edge') {
                const allFrom = selectEdge(adj, pos, { op: 'all', type: 'edge' });
                const selectedKeys = new Set<string | number>(selectEdge(adj, pos, sel.a).map(edgeKey));
                return convertObjects(allTo, t => t.nodes, allFrom, e => [e.n1, e.n2], edgeKey, selectedKeys, mode);
            }
            // sel.from is some other simp M (M != n, checked above) - the new simp <-> simp
            // conversion, via the same general containment rule as every other pair.
            const allFrom = selectSimp(adj, pos, { op: 'all', type: sel.from });
            const selectedKeys = new Set<string | number>(selectSimp(adj, pos, sel.a).map(simpKey));
            return convertObjects(allTo, t => t.nodes, allFrom, f => f.nodes, simpKey, selectedKeys, mode);
        }
        case 'rrmn': {
            const base = selectSimp(adj, pos, sel.a);
            return randomlyRemove(base, sel.count);
        }
        case 'rrmp': {
            const base = selectSimp(adj, pos, sel.a);
            return randomlyRemove(base, Math.floor(sel.frac * base.length));
        }
        case 'raw':
            if (sel.items.kind !== 'simp')
                throw new Error(`selectSimp: 'raw' selector's own items must be simp-kind, got '${sel.items.kind}'`);
            return [...sel.items.value];
        default:
            throw new Error(`selectSimp: unexpected simp-selector op '${(sel as Selector).op}'`);
    }
}

/** Parses `s` as a triangle (simp 2) selector and evaluates it - selectSimp()'s own N=2 special
 * case, kept as a thin sugar wrapper (see this file's own top comment) for callers that only ever
 * deal in triangles (shared/boardConfig.ts's triangleForm/triCentralize). Throws if `sel` isn't
 * specifically simp 2 (not just any simp N). */
export function selectTriangle(adj: number[][], pos: number[][], sel: Selector): BoardSimplex[] {
    if (sel.type !== simpType(2))
        throw new Error(`selectTriangle: expected a triangle selector, got ${describeSelectorType(sel.type)} selector (op '${sel.op}')`);
    return selectSimp(adj, pos, sel);
}

/**
 * Evaluates a quad Selector against a board's adjacency matrix, returning the list of selected
 * quads as BoardQuad values (deduplicated) - the quad counterpart of selectSimp above.
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
            if (isSimpType(sel.from))
                throw new Error(`selectQuad: no association is defined between 'simp' and 'quad'`);
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
