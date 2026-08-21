import { type BoardEdge, makeBoardEdge } from './types.js';

// A tiny S-expression language for selecting a subset of a board's nodes or edges. Grammar (SEL):
//
//   (union SEL SEL)         -- set union
//   (inter SEL SEL)         -- set intersection
//   (diff SEL SEL)          -- set difference (left minus right)
//   (compl SEL)             -- complement, within all nodes or all edges (whichever kind of
//                               selector this is)
//   (all)                   -- every node, or every edge
//   (none)                  -- no nodes, or no edges
//   (deg <eq|gt|lt> <num>)  -- nodes whose degree is =/>/< a given nonnegative integer
//   (e2n SEL)               -- node selector -> edge selector: an edge is selected iff both its
//                               nodes are selected
//   (n2e SEL)               -- edge selector -> node selector: a node is selected iff it is an
//                               endpoint of some selected edge
//   (rrmn <num> SEL)        -- randomly removes exactly num (a nonnegative integer) items from SEL's
//                               own result, uniformly at random
//   (rrmp <num> SEL)        -- randomly removes a fixed portion of SEL's own result: num (a
//                               nonnegative float) times SEL's own result size, rounded down
//
// `union`/`inter`/`diff`/`compl`/`all`/`none`/`rrmn`/`rrmp` are polymorphic - the exact same syntax
// works over either node-sets or edge-sets. Every Selector node (one monolithic type, below) carries
// its own `type` (which kind of set it denotes) - but rather than parse a type-less tree and infer/
// validate `type` bottom-up afterward, parsing itself is done by two mutually recursive functions,
// parseNodeSelExpr/parseEdgeSelExpr (mirroring selectNode()/selectEdge()'s own mutual recursion
// below): parseNodeSelExpr always produces `type: 'node'` nodes (recursing into parseEdgeSelExpr for
// n2e's own operand), parseEdgeSelExpr always produces `type: 'edge'` nodes (recursing into
// parseNodeSelExpr for e2n's own operand). This is what lets `all`/`none` skip spelling out which
// kind they mean: `type` comes from *which parser reached them*, not from anything written in the
// expression itself. selectNode()/selectEdge() (this file's own separate mutually recursive
// evaluators) still re-check `sel.type` themselves rather than trusting it, since a Selector need
// not always come from this file's own parsers (e.g. a hand-built AST, or one round-tripped through
// JSON).
export type SelectorType = 'node' | 'edge';

export type Selector =
    | { op: 'union' | 'inter' | 'diff'; type: SelectorType; a: Selector; b: Selector }
    | { op: 'compl'; type: SelectorType; a: Selector }
    | { op: 'all' | 'none'; type: SelectorType }
    | { op: 'deg'; type: 'node'; cmp: 'eq' | 'gt' | 'lt'; n: number }
    | { op: 'e2n'; type: 'edge'; a: Selector }
    | { op: 'n2e'; type: 'node'; a: Selector }
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

// Parses a node SEL - mutually recursive with parseEdgeSelExpr via n2e's own operand. Every Selector
// this returns has `type: 'node'`.
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
        case 'n2e': {
            const a = parseEdgeSelExpr(c);
            c.expect(')');
            return { op: 'n2e', type: 'node', a };
        }
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

// Parses an edge SEL - mutually recursive with parseNodeSelExpr via e2n's own operand. Every
// Selector this returns has `type: 'edge'`.
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
        case 'all':
            c.expect(')');
            return { op: 'all', type: 'edge' };
        case 'none':
            c.expect(')');
            return { op: 'none', type: 'edge' };
        case 'e2n': {
            const a = parseNodeSelExpr(c);
            c.expect(')');
            return { op: 'e2n', type: 'edge', a };
        }
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

// Shared by parseNodeSelector/parseEdgeSelector: tokenizes `s`, runs `parseExpr` over the whole
// thing, and rejects any leftover trailing input.
function parseTopLevel(s: string, parseExpr: (c: ParseCursor) => Selector): Selector {
    const tokens = tokenize(s);
    if (tokens.length === 0) throw new Error('selector: empty input');
    const c = new ParseCursor(tokens);
    const sel = parseExpr(c);
    if (!c.atEnd()) throw new Error(`selector: unexpected trailing input starting at '${c.peek()}'`);
    return sel;
}

/** Parses `s` as a node selector (see this file's own top comment for the grammar) - throws if `s`
 * doesn't follow the grammar (an edge-only operator like e2n is simply not a recognized operator
 * inside a node-selector context - see parseNodeSelExpr). */
export function parseNodeSelector(s: string): Selector {
    return parseTopLevel(s, parseNodeSelExpr);
}

/** Parses `s` as an edge selector (see this file's own top comment for the grammar) - throws if `s`
 * doesn't follow the grammar. */
export function parseEdgeSelector(s: string): Selector {
    return parseTopLevel(s, parseEdgeSelExpr);
}

/** Formats `sel` back into the S-expression syntax parseNodeSelector()/parseEdgeSelector() accept -
 * the inverse of parsing. Used e.g. to round-trip a BoardModifier's own selector back into
 * command-line text for display (see src/sidePanel.ts's fmtModifiers). */
export function formatSelector(sel: Selector): string {
    switch (sel.op) {
        case 'union': case 'inter': case 'diff':
            return `(${sel.op} ${formatSelector(sel.a)} ${formatSelector(sel.b)})`;
        case 'compl':
            return `(compl ${formatSelector(sel.a)})`;
        case 'all': case 'none':
            return `(${sel.op})`;
        case 'deg':
            return `(deg ${sel.cmp} ${sel.n})`;
        case 'e2n':
            return `(e2n ${formatSelector(sel.a)})`;
        case 'n2e':
            return `(n2e ${formatSelector(sel.a)})`;
        case 'rrmn':
            return `(rrmn ${sel.count} ${formatSelector(sel.a)})`;
        case 'rrmp':
            return `(rrmp ${sel.frac} ${formatSelector(sel.a)})`;
    }
}

// ── evaluation ───────────────────────────────────────────────────────────────

function degree(adj: number[][], i: number): number {
    return adj[i].reduce((s, v) => s + (v ? 1 : 0), 0);
}

// BoardEdge itself isn't a valid Set/Map key (two structurally-equal BoardEdges are different
// objects), so union/inter/diff/compl below key edges by this canonical numeric id (n1 < n2, so
// unique per edge) whenever they need set-like membership tests.
function edgeKey(N: number, e: BoardEdge): number {
    return e.n1 * N + e.n2;
}

function dedupeEdges(N: number, edges: BoardEdge[]): BoardEdge[] {
    const byKey = new Map<number, BoardEdge>();
    for (const e of edges) byKey.set(edgeKey(N, e), e);
    return [...byKey.values()];
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

/**
 * Evaluates a node Selector against a board's adjacency matrix, returning the set of selected node
 * indices. Mutually recursive with selectEdge() via the n2e operator. `pos` isn't used by any
 * selector in the current grammar, but is threaded through (matching selectEdge()'s own signature)
 * for future position-based selectors.
 */
export function selectNode(adj: number[][], pos: number[][], sel: Selector): Set<number> {
    if (sel.type !== 'node')
        throw new Error(`selectNode: expected a node selector, got an edge selector (op '${sel.op}')`);
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
        case 'n2e': {
            const out = new Set<number>();
            for (const e of selectEdge(adj, pos, sel.a)) { out.add(e.n1); out.add(e.n2); }
            return out;
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
 * edges as BoardEdge values (deduplicated). Mutually recursive with selectNode() via the e2n
 * operator.
 */
export function selectEdge(adj: number[][], pos: number[][], sel: Selector): BoardEdge[] {
    if (sel.type !== 'edge')
        throw new Error(`selectEdge: expected an edge selector, got a node selector (op '${sel.op}')`);
    const N = adj.length;
    switch (sel.op) {
        case 'union':
            return dedupeEdges(N, [...selectEdge(adj, pos, sel.a), ...selectEdge(adj, pos, sel.b)]);
        case 'inter': {
            const a = selectEdge(adj, pos, sel.a);
            const bKeys = new Set(selectEdge(adj, pos, sel.b).map(e => edgeKey(N, e)));
            return a.filter(e => bKeys.has(edgeKey(N, e)));
        }
        case 'diff': {
            const a = selectEdge(adj, pos, sel.a);
            const bKeys = new Set(selectEdge(adj, pos, sel.b).map(e => edgeKey(N, e)));
            return a.filter(e => !bKeys.has(edgeKey(N, e)));
        }
        case 'compl': {
            const aKeys = new Set(selectEdge(adj, pos, sel.a).map(e => edgeKey(N, e)));
            const out: BoardEdge[] = [];
            for (let i = 0; i < N; i++)
                for (let j = i + 1; j < N; j++) {
                    if (!adj[i][j]) continue;
                    const e = makeBoardEdge(i, j);
                    if (!aKeys.has(edgeKey(N, e))) out.push(e);
                }
            return out;
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
        case 'e2n': {
            const nodes = selectNode(adj, pos, sel.a);
            const out: BoardEdge[] = [];
            for (let i = 0; i < N; i++)
                for (let j = i + 1; j < N; j++)
                    if (adj[i][j] && nodes.has(i) && nodes.has(j)) out.push(makeBoardEdge(i, j));
            return out;
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
