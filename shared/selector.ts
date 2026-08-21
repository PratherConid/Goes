import { type BoardEdge, makeBoardEdge } from './types.js';

// A tiny S-expression language for selecting a subset of a board's nodes or edges. Grammar (SEL):
//
//   (union SEL SEL)         -- set union
//   (inter SEL SEL)         -- set intersection
//   (diff SEL SEL)          -- set difference (left minus right)
//   (compl SEL)             -- complement, within all nodes/all edges (whichever SEL denotes)
//   (all <node|edge>)       -- every node, or every edge
//   (none <node|edge>)      -- no nodes, or no edges
//   (deg <eq|gt|lt> <num>)  -- nodes whose degree is =/>/< a given nonnegative integer
//   (e2n SEL)               -- node selector -> edge selector: an edge is selected iff both its
//                               nodes are selected
//   (n2e SEL)               -- edge selector -> node selector: a node is selected iff it is an
//                               endpoint of some selected edge
//
// `union`/`inter`/`diff`/`compl`/`all`/`none` are polymorphic - they work over either node-sets or
// edge-sets, but (for union/inter/diff) both operands must denote the same kind; `all`/`none` take
// no sub-selector to infer their own kind from (unlike compl), so they spell it out explicitly
// instead. `deg` always denotes a node selector; `e2n`/`n2e` each fix both their operand's and their
// own result's kind. Every Selector
// node below carries its own `type` (which kind of set it denotes), computed bottom-up while parsing
// - this lets parseNodeSelector()/parseEdgeSelector() reject a syntactically-valid expression of the
// wrong overall kind, and lets selectNode()/selectEdge() (this file's mutually recursive evaluators)
// re-check `type` themselves rather than trusting a caller-constructed Selector, since a Selector
// need not always come from this file's own parsers (e.g. a hand-built AST, or one round-tripped
// through JSON).
export type SelectorType = 'node' | 'edge';

export type Selector =
    | { op: 'union' | 'inter' | 'diff'; type: SelectorType; a: Selector; b: Selector }
    | { op: 'compl'; type: SelectorType; a: Selector }
    | { op: 'all' | 'none'; type: SelectorType }
    | { op: 'deg'; type: 'node'; cmp: 'eq' | 'gt' | 'lt'; n: number }
    | { op: 'e2n'; type: 'edge'; a: Selector }
    | { op: 'n2e'; type: 'node'; a: Selector };

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

function parseSelExpr(c: ParseCursor): Selector {
    c.expect('(');
    const op = c.next();
    switch (op) {
        case 'union': case 'inter': case 'diff': {
            const a = parseSelExpr(c);
            const b = parseSelExpr(c);
            c.expect(')');
            if (a.type !== b.type)
                throw new Error(`selector: (${op} ...) operands must be the same kind, got '${a.type}' and '${b.type}'`);
            return { op, type: a.type, a, b };
        }
        case 'compl': {
            const a = parseSelExpr(c);
            c.expect(')');
            return { op: 'compl', type: a.type, a };
        }
        case 'all': case 'none': {
            const kindTok = c.next();
            if (kindTok !== 'node' && kindTok !== 'edge')
                throw new Error(`selector: (${op} ...) kind must be 'node' or 'edge', got '${kindTok}'`);
            c.expect(')');
            return { op, type: kindTok };
        }
        case 'deg': {
            const cmpTok = c.next();
            if (cmpTok !== 'eq' && cmpTok !== 'gt' && cmpTok !== 'lt')
                throw new Error(`selector: (deg ...) comparator must be 'eq', 'gt', or 'lt', got '${cmpTok}'`);
            const numTok = c.next();
            const n = Number(numTok);
            if (!Number.isInteger(n) || n < 0)
                throw new Error(`selector: (deg ...) argument must be a nonnegative integer, got '${numTok}'`);
            c.expect(')');
            return { op: 'deg', type: 'node', cmp: cmpTok, n };
        }
        case 'e2n': {
            const a = parseSelExpr(c);
            c.expect(')');
            if (a.type !== 'node')
                throw new Error(`selector: (e2n SEL) requires a node selector, got an edge selector`);
            return { op: 'e2n', type: 'edge', a };
        }
        case 'n2e': {
            const a = parseSelExpr(c);
            c.expect(')');
            if (a.type !== 'edge')
                throw new Error(`selector: (n2e SEL) requires an edge selector, got a node selector`);
            return { op: 'n2e', type: 'node', a };
        }
        default:
            throw new Error(`selector: unknown operator '${op}'`);
    }
}

function parseSelector(s: string): Selector {
    const tokens = tokenize(s);
    if (tokens.length === 0) throw new Error('selector: empty input');
    const c = new ParseCursor(tokens);
    const sel = parseSelExpr(c);
    if (!c.atEnd()) throw new Error(`selector: unexpected trailing input starting at '${c.peek()}'`);
    return sel;
}

/** Parses `s` as a node selector (see this file's own top comment for the grammar) - throws if `s`
 * doesn't follow the grammar, or if it's a syntactically valid selector that denotes edges instead. */
export function parseNodeSelector(s: string): Selector {
    const sel = parseSelector(s);
    if (sel.type !== 'node') throw new Error('selector: expected a node selector, got an edge selector');
    return sel;
}

/** Parses `s` as an edge selector (see this file's own top comment for the grammar) - throws if `s`
 * doesn't follow the grammar, or if it's a syntactically valid selector that denotes nodes instead. */
export function parseEdgeSelector(s: string): Selector {
    const sel = parseSelector(s);
    if (sel.type !== 'edge') throw new Error('selector: expected an edge selector, got a node selector');
    return sel;
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
            return `(${sel.op} ${sel.type})`;
        case 'deg':
            return `(deg ${sel.cmp} ${sel.n})`;
        case 'e2n':
            return `(e2n ${formatSelector(sel.a)})`;
        case 'n2e':
            return `(n2e ${formatSelector(sel.a)})`;
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
        default:
            throw new Error(`selectEdge: unexpected edge-selector op '${(sel as Selector).op}'`);
    }
}
