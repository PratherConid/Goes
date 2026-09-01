import type { BoardArgEntry, BoardConfig, BoardModifier, LocalReplaceSelector, Selector, BoardEdge } from './types.js';
import type { GameConfig } from './gameConfig.js';
// Type-only - see types.ts's own note on why this isn't a real circular runtime import.
import type { ClegProgram } from './clegBase.js';
import { assert, BoardArgType, boardArgNumber, boardArgList, Embedding, simpType } from './types.js';
import { convexHullEdges } from './geometry.js';
import { zeroAdj, mergeBoards } from './topology.js';
import { selectNode, selectEdge, selectTriangle, selectSimp, selectQuad } from './selector.js';
// The FractalDescr/nodeEdgeMergeFlakeRec recursive core, and each "flake" shape's own static
// *FractalDescr() builder, live in fractal.ts (see git history) - the actual BoardConfig-returning
// functions built on them (dodecahedronBoard/dodecahedronFlake/etc., below) stay here alongside
// every other board constructor.
import {
    buildFractal,
    dodecahedronFractalDescr, icosahedronFractalDescr, octahedronFractalDescr,
    regularPolygonFractalDescr, centralPentagonFractalDescr, mengerFractalDescr,
} from './fractal.js';

export function make(posOrEmb: number[][] | Embedding, adj: number[][]): BoardConfig {
    const emb = Array.isArray(posOrEmb) ? new Embedding(2, posOrEmb) : posOrEmb;
    const N = emb.pos.length;
    assert(adj.length === N && (N === 0 || adj[0].length === N), 'adj dimensions must match pos length');
    for (let i = 0; i < N; i++)
        for (let j = 0; j < N; j++)
            assert(adj[i][j] === adj[j][i], `adj must be symmetric: [${i}][${j}]`);
    return { emb, adj, N };
}

/** Glue pairs of nodes in `quot` together. The position of the new node is the average of its predecessors. */
export function quotientBoard(bc: BoardConfig, quot: [number, number][]): BoardConfig {
    const N = bc.N;
    for (const [a, b] of quot)
        assert(a >= 0 && a < N && b >= 0 && b < N, `quot indices [${a}, ${b}] out of bounds for N=${N}`);
    // Union-Find to compute equivalence classes
    const parent = Array.from({ length: N }, (_, i) => i);
    function find(x: number): number {
        while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    }
    for (const [a, b] of quot) {
        const pa = find(a), pb = find(b);
        if (pa !== pb) parent[pa] = pb;
    }
    const roots = Array.from({ length: N }, (_, i) => find(i));
    const uniqueRoots = [...new Set(roots)].sort((a, b) => a - b);
    const rootToNew = new Map(uniqueRoots.map((r, i) => [r, i]));
    const newN = uniqueRoots.length;
    const nodeToNew = roots.map(r => rootToNew.get(r)!);

    // New positions: average of class members, in the natural embedding dimension.
    const embDim = bc.emb.embDim;
    const newPos = Array.from({ length: newN }, () => new Array<number>(embDim).fill(0));
    const cnt = new Array<number>(newN).fill(0);
    for (let i = 0; i < N; i++) {
        const ni = nodeToNew[i];
        for (let k = 0; k < embDim; k++) newPos[ni][k] += bc.emb.pos[i][k];
        cnt[ni]++;
    }
    for (let ni = 0; ni < newN; ni++)
        for (let k = 0; k < embDim; k++) newPos[ni][k] /= cnt[ni];

    // New adjacency: adjacent if any pair across the two classes was adjacent
    const newAdj = zeroAdj(newN);
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            if (!bc.adj[i][j]) continue;
            const ni = nodeToNew[i], nj = nodeToNew[j];
            if (ni !== nj) newAdj[ni][nj] = 1;
        }
    }
    return make(new Embedding(embDim, newPos), newAdj);
}

/**
 * Splits every edge of `bc` into `splitN` unit-length sub-edges, inserting `splitN-1` evenly-spaced
 * new nodes along each original edge. Original node positions are scaled by `splitN` first, so -
 * same as every board-generating function in this file - every sub-edge ends up exactly unit length.
 */
export function edgeSplit(bc: BoardConfig, splitN: number): BoardConfig {
    assert(splitN >= 1, `splitN must be at least 1, got ${splitN}`);
    const N = bc.N;
    const embDim = bc.emb.embDim;
    const pos: number[][] = bc.emb.pos.map(p => p.map(v => v * splitN));
    const edges: [number, number][] = [];
    for (let i = 0; i < N; i++)
        for (let j = i + 1; j < N; j++) {
            if (!bc.adj[i][j]) continue;
            let prev = i;
            for (let k = 1; k < splitN; k++) {
                const t = k / splitN;
                const idx = pos.length;
                pos.push(pos[i].map((v, d) => v + (pos[j][d] - v) * t));
                edges.push([prev, idx]);
                prev = idx;
            }
            edges.push([prev, j]);
        }

    const adj = zeroAdj(pos.length);
    for (const [a, b] of edges) { adj[a][b] = 1; adj[b][a] = 1; }
    return make(new Embedding(embDim, pos), adj);
}

/**
 * Rectifies `bc`: one new node per original edge, at that edge's midpoint (original node positions
 * are doubled first, same trick as `edgeSplit`, so the midpoint `(pos2[i]+pos2[j])/2` is exact).
 *
 * Two new nodes (midpoints of edges incident to a shared original node `v`) are connected iff their
 * edges are angularly adjacent around `v`: take `X`, the set of midpoints of edges incident to `v`;
 * subtract `v`'s own position from each and normalize to unit length; the pair is connected iff its
 * two normalized directions are joined by an edge on the convex hull of all of `v`'s directions (see
 * `shared/geometry.ts`'s `convexHullEdges`). This is what correctly excludes, say, the two diagonal
 * pairs at a degree-4 grid vertex - only the four "consecutive" directions around `v` get connected.
 */
export function rectify(bc: BoardConfig): BoardConfig {
    const N = bc.N;
    const embDim = bc.emb.embDim;
    const pos2 = bc.emb.pos.map(p => p.map(v => v * 2));

    // One new node per original edge (i<j), at the midpoint.
    const edgeIdx = new Map<string, number>(); // "i,j" (i<j) -> new node index
    const pos: number[][] = [];
    for (let i = 0; i < N; i++)
        for (let j = i + 1; j < N; j++) {
            if (!bc.adj[i][j]) continue;
            edgeIdx.set(`${i},${j}`, pos.length);
            pos.push(pos2[i].map((v, k) => (v + pos2[j][k]) / 2));
        }

    // Edges incident to each original node, as [midpoint node index] lists.
    const incident: number[][] = Array.from({ length: N }, () => []);
    for (const [key, idx] of edgeIdx) {
        const [i, j] = key.split(',').map(Number);
        incident[i].push(idx);
        incident[j].push(idx);
    }

    const adj = zeroAdj(pos.length);
    for (let v = 0; v < N; v++) {
        const mids = incident[v];
        if (mids.length < 2) continue;
        const dirs = mids.map(midIdx => {
            // pos[midIdx] is at the doubled scale (see pos2 above), so subtract pos2[v] (not
            // bc.emb.pos[v]) to get a consistently-scaled direction vector - normalizing removes
            // the scale either way, but mixing scales here would give the wrong direction entirely.
            const d = pos[midIdx].map((val, k) => val - pos2[v][k]);
            const len = Math.sqrt(d.reduce((s, x) => s + x * x, 0));
            return d.map(x => x / len);
        });
        for (const [a, b] of convexHullEdges(dirs)) {
            adj[mids[a]][mids[b]] = 1;
            adj[mids[b]][mids[a]] = 1;
        }
    }

    return make(new Embedding(embDim, pos), adj);
}

/**
 * Truncates `bc`: two new nodes per original edge (i<j), one near each endpoint, replacing every
 * original vertex `v` with a small polygon connecting the near-`v` points of `v`'s own incident
 * edges - the same convex-hull-of-directions ring construction `rectify` above uses for its own
 * midpoints (see its own doc comment), just applied to the near-`v` points here instead of a single
 * shared midpoint. The two near-points of an original edge `(i,j)` are themselves connected too,
 * forming that edge's own (shortened) middle segment.
 *
 * Every one of `v`'s own near-points sits at the SAME fraction `t_v` of the way along its own edge
 * (`t_v <= 0.5`, i.e. always within the "near half" closer to `v`) - `t_v` is chosen so that,
 * locally at `v`: the minimal (over `v`'s own edges) distance from a near-`v` point to its own
 * edge's midpoint equals half the minimal (over `v`'s own convex-hull ring edges) distance between
 * two ring-adjacent near-`v` points. Since every near-`v` point is `v + t_v*(u-v)` for its own edge
 * `(v,u)`:
 * - distance to that edge's own midpoint is `(0.5 - t_v) * |u-v|`, minimized over `v`'s edges by
 *   picking `v`'s own shortest incident edge length (`minL`);
 * - distance between two ring-adjacent near-`v` points on edges `(v,u1)`/`(v,u2)` is
 *   `t_v * |(u1-v)-(u2-v)|` (`t_v` factors out identically, since it's the same for every edge at
 *   `v`), minimized over `v`'s own ring edges (`minD`, a fixed geometric quantity independent of
 *   `t_v`).
 *
 * Setting `(0.5-t_v)*minL = 0.5*t_v*minD` and solving gives `t_v = minL / (2*minL + minD)`. A
 * vertex with no ring edges (degree < 2) has no such constraint to solve - `t_v` there defaults to
 * `0.5`, i.e. its near-point(s) sit at the edge midpoint, same as `rectify` would place them.
 *
 * Finally, the whole board is scaled by `1 / (1 - 2*maxPortion)`, `maxPortion` being the largest
 * `t_v` over every vertex with a real (non-fallback) value, i.e. degree >= 2 - a degree-0 or
 * degree-1 vertex's own unused `0.5` fallback is excluded, or a single pendant edge anywhere in
 * the board would force `maxPortion` to exactly `0.5` and the scale factor to a division by zero.
 * The shrunk middle segment of an edge `(i,j)` has length `|i-j| * (1 - t_i - t_j)`, so
 * `1 - 2*maxPortion` is a lower bound on that factor for every edge between two degree->=2
 * vertices (since `t_i, t_j <= maxPortion` there); scaling by its reciprocal keeps every such
 * shrunk segment at least as long as the original edge it came from.
 */
export function truncate(bc: BoardConfig): BoardConfig {
    const N = bc.N;
    const embDim = bc.emb.embDim;
    const pos = bc.emb.pos;

    // Edges incident to each original node v, as {u, vec, len} triples - vec is u's position minus
    // v's (so always points away from v along that edge), len its length.
    const incident: { u: number; vec: number[]; len: number }[][] = Array.from({ length: N }, () => []);
    const edges: [number, number][] = [];
    for (let i = 0; i < N; i++)
        for (let j = i + 1; j < N; j++) {
            if (!bc.adj[i][j]) continue;
            edges.push([i, j]);
            const vecJ = pos[j].map((v, k) => v - pos[i][k]);
            const len = Math.sqrt(vecJ.reduce((s, x) => s + x * x, 0));
            incident[i].push({ u: j, vec: vecJ, len });
            incident[j].push({ u: i, vec: vecJ.map(x => -x), len });
        }

    // Per-node fraction t[v] (see this function's own doc comment for the derivation) and its own
    // convex-hull ring pairs (indices into incident[v], same convention as rectify's own `mids`).
    const t: number[] = new Array(N).fill(0.5);
    const ringPairsOf: [number, number][][] = Array.from({ length: N }, () => []);
    for (let v = 0; v < N; v++) {
        const own = incident[v];
        if (own.length < 2) continue;
        const dirs = own.map(e => e.vec.map(x => x / e.len));
        const ringPairs = convexHullEdges(dirs);
        ringPairsOf[v] = ringPairs;
        if (ringPairs.length === 0) continue;
        const minL = Math.min(...own.map(e => e.len));
        const minD = Math.min(...ringPairs.map(([a, b]) =>
            Math.sqrt(own[a].vec.reduce((s, x, k) => s + (x - own[b].vec[k]) ** 2, 0))));
        t[v] = minL / (2 * minL + minD);
    }

    // Two new nodes per original edge (i<j): nearI (near i, at fraction t[i]) and nearJ (near j).
    const nearIdx = new Map<string, [number, number]>(); // "i,j" (i<j) -> [nearI idx, nearJ idx]
    const posOut: number[][] = [];
    for (const [i, j] of edges) {
        const nearI = pos[i].map((v, k) => v + t[i] * (pos[j][k] - v));
        const nearJ = pos[j].map((v, k) => v + t[j] * (pos[i][k] - v));
        nearIdx.set(`${i},${j}`, [posOut.length, posOut.length + 1]);
        posOut.push(nearI, nearJ);
    }

    const adj = zeroAdj(posOut.length);
    // Shrunk original edges: connect the two near-points of each original edge.
    for (const [i, j] of edges) {
        const [idxI, idxJ] = nearIdx.get(`${i},${j}`)!;
        adj[idxI][idxJ] = 1;
        adj[idxJ][idxI] = 1;
    }
    // Vertex rings: connect hull-adjacent near-v points around each original vertex.
    for (let v = 0; v < N; v++) {
        const own = incident[v];
        const nearAt = own.map(e => {
            const key = v < e.u ? `${v},${e.u}` : `${e.u},${v}`;
            const [idxI, idxJ] = nearIdx.get(key)!;
            return v < e.u ? idxI : idxJ;
        });
        for (const [a, b] of ringPairsOf[v]) {
            adj[nearAt[a]][nearAt[b]] = 1;
            adj[nearAt[b]][nearAt[a]] = 1;
        }
    }

    // Scale so every shrunk edge segment ends up at least as long as its original edge (see this
    // function's own doc comment) - maxPortion only considers vertices with a real (non-fallback)
    // t[v], i.e. degree >= 2; a degree-0 or degree-1 vertex's own unused 0.5 fallback must not
    // count, or a single pendant edge anywhere in the board would force maxPortion to exactly 0.5
    // and blow the scale factor up to a division by zero.
    let maxPortion = -Infinity;
    for (let v = 0; v < N; v++) if (incident[v].length >= 2) maxPortion = Math.max(maxPortion, t[v]);
    const scale = 1 / (1 - 2 * maxPortion);
    const posScaled = posOut.map(p => p.map(x => x * scale));

    return make(new Embedding(embDim, posScaled), adj);
}

/**
 * Merges every pair of nodes whose Euclidean distance (in the natural embedding dimension) is
 * strictly less than `dist` into a single node, via quotientBoard. Closeness is transitive under
 * quotientBoard's union-find, so a chain of nodes each within `dist` of the next all collapse into
 * one node, not just each individual close pair.
 */
export function mergeClose(bc: BoardConfig, dist: number): BoardConfig {
    assert(dist > 0, `dist must be positive, got ${dist}`);
    const dist2 = dist * dist;
    const quot: [number, number][] = [];
    for (let i = 0; i < bc.N; i++)
        for (let j = i + 1; j < bc.N; j++) {
            const d2 = bc.emb.pos[i].reduce((s, v, k) => s + (v - bc.emb.pos[j][k]) ** 2, 0);
            if (d2 < dist2) quot.push([i, j]);
        }
    return quotientBoard(bc, quot);
}

/**
 * The subgraph induced by `nodes`: keeps only the given nodes - compacted to a fresh 0..k-1 index
 * range, in ascending original-index order, positions/embDim otherwise untouched - with two
 * surviving nodes adjacent iff they were already adjacent in `bc`. Unlike quotientBoard/mergeClose,
 * nothing is merged or repositioned; a non-kept node's own incident edges are simply dropped along
 * with it. `nodes` is typically `selectNode(bc.adj, bc.emb.pos, sel)`'s own result (see
 * applyModifier's own NodeInducedSubgraph case) but is taken directly here - a plain Set<number>,
 * not a Selector - so any already-computed node set can be used, not just one selector's own result.
 */
export function nodeInducedSubgraph(bc: BoardConfig, nodes: Set<number>): BoardConfig {
    const kept: number[] = [];
    for (let i = 0; i < bc.N; i++) if (nodes.has(i)) kept.push(i);

    const pos = kept.map(i => bc.emb.pos[i]);
    const adj = zeroAdj(kept.length);
    for (let a = 0; a < kept.length; a++)
        for (let b = a + 1; b < kept.length; b++)
            if (bc.adj[kept[a]][kept[b]]) { adj[a][b] = 1; adj[b][a] = 1; }

    return make(new Embedding(bc.emb.embDim, pos), adj);
}

/**
 * The subgraph induced by `edges`: keeps only the given edges, and only the nodes touched by at
 * least one of them - compacted to a fresh 0..k-1 index range, in ascending original-index order,
 * positions/embDim otherwise untouched. Unlike nodeInducedSubgraph (which keeps every
 * original edge between two surviving nodes, since it starts from a node selection), this keeps
 * exactly the given edges themselves - the standard graph-theory distinction between a node-induced
 * and an edge-induced subgraph - so a node with no kept incident edge doesn't survive at all, even
 * if it's adjacent to other surviving nodes via a non-kept edge. `edges` is typically
 * `selectEdge(bc.adj, bc.emb.pos, sel)`'s own result (see applyModifier's own EdgeInducedSubgraph
 * case) but is taken directly here - a plain BoardEdge[], not a Selector - so any already-computed
 * edge list can be used, not just one selector's own result.
 */
export function edgeInducedSubgraph(bc: BoardConfig, edges: BoardEdge[]): BoardConfig {
    const touched = new Set<number>();
    for (const e of edges) { touched.add(e.n1); touched.add(e.n2); }
    const kept: number[] = [];
    for (let i = 0; i < bc.N; i++) if (touched.has(i)) kept.push(i);
    const newIdx = new Map<number, number>(kept.map((orig, idx) => [orig, idx]));

    const pos = kept.map(i => bc.emb.pos[i]);
    const adj = zeroAdj(kept.length);
    for (const e of edges) {
        const a = newIdx.get(e.n1)!, b = newIdx.get(e.n2)!;
        adj[a][b] = 1;
        adj[b][a] = 1;
    }

    return make(new Embedding(bc.emb.embDim, pos), adj);
}

/**
 * Replaces every selected triangle and/or quad (see topology.ts's findSimplices(adj, 2)/findQuads)
 * in `bc` with its own w-sided lattice - a `triangularBoard(w)`-shaped lattice for a triangle, a
 * `w`-by-`w` grid for a quad - gluing new corners back to the original vertices and gluing every
 * original edge's own new boundary points together across every lattice that consumes that edge as
 * one of its own sides, regardless of whether that lattice came from a triangle- or quad-typed
 * selector - this is what makes a mixed `sels` list meaningful: a triangle and a quad sharing an
 * edge still glue seamlessly, since gluing is driven by shared ORIGINAL edges, not by matching
 * kinds. Each element of `sels` is itself a Selector naming which faces to look for AND restricting
 * which ones of that kind qualify in one go (its own bottom-up-inferred `type` already says simp 2
 * or quad) - pass `(all tri)`/`(all quad)` for "every one found, no restriction". Every element must
 * be a triangle (simp 2) or quad selector, checked at runtime (any other `type` throws) since
 * nothing else constrains it structurally. An unselected/not-looked-for triangle or quad is left untouched, as if
 * it didn't exist. `w` is shared by every selector in `sels`, since two lattices sharing an edge can only glue
 * node-for-node if their own boundary sequences are the same length. triangleForm/quadForm below are
 * the single-kind special cases, each just calling this with one `(all tri)`/`(all quad)`-or-`sel`
 * selector.
 */
export function genericForm(bc: BoardConfig, w: number, sels: Selector[]): BoardConfig {
    assert(w >= 1, `w must be at least 1, got ${w}`);
    const N = bc.N;
    const embDim = bc.emb.embDim;
    const scale = Math.max(w - 1, 1);
    const scaledPos = bc.emb.pos.map(p => p.map(v => v * scale));

    const isFaceSide = new Set<string>(); // "p,q" (p < q) - an original edge consumed by some face
    const cornerQuot: [number, number][] = []; // [original vertex, its own new lattice corner]
    // canonical "p,q" (p < q) -> one boundary-index getter per face that has this original edge as a
    // side, each already reoriented (see addSide) to run from p (k=0) to q (k=w-1) regardless of
    // that face's own corner order - the single generalization of the old triangleForm's
    // edgeToTriangles and quadForm's edgeToSeqs, now spanning every face of every kind at once.
    const edgeToSeqs = new Map<string, ((k: number) => number)[]>();

    function addSide(p: number, q: number, atK: (k: number) => number) {
        const key = `${Math.min(p, q)},${Math.max(p, q)}`;
        isFaceSide.add(key);
        const oriented = p < q ? atK : (k: number) => atK(w - 1 - k);
        if (!edgeToSeqs.has(key)) edgeToSeqs.set(key, []);
        edgeToSeqs.get(key)!.push(oriented);
    }

    // New nodes' own positions/internal edges, collected face by face (a face's own global index
    // range isn't known ahead of time, since it depends on how many triangles/quads each selector
    // in `sels` selects) - merged into one pos/adj array only once every face has been processed.
    const extraPos: number[][] = [];
    const extraEdges: [number, number][] = [];
    let nextIdx = N;

    for (const sel of sels) {
        if (sel.type === simpType(2)) {
            const triangles = selectTriangle(bc.adj, bc.emb.pos, sel);
            const nFace = w * (w + 1) / 2;
            const localIdx = (i: number, j: number) => i * (i + 1) / 2 + j;
            const dirs: [number, number][] = [[1, 0], [1, 1], [0, 1], [-1, 0], [-1, -1], [0, -1]];
            for (const { nodes: [A, B, C] } of triangles) {
                const offset = nextIdx;
                nextIdx += nFace;
                const globalIdx = (i: number, j: number) => offset + localIdx(i, j);
                const cornerA = scaledPos[A], cornerB = scaledPos[B], cornerC = scaledPos[C];
                for (let i = 0; i < w; i++)
                    for (let j = 0; j <= i; j++) {
                        const a = w - 1 - i, b = i - j, c = j;
                        extraPos[globalIdx(i, j) - N] = w === 1
                            ? cornerA.map((_, k) => (cornerA[k] + cornerB[k] + cornerC[k]) / 3)
                            : cornerA.map((_, k) => (a * cornerA[k] + b * cornerB[k] + c * cornerC[k]) / (w - 1));
                    }
                for (let i = 0; i < w; i++)
                    for (let j = 0; j <= i; j++)
                        for (const [di, dj] of dirs) {
                            const ni = i + di, nj = j + dj;
                            if (ni < 0 || ni >= w || nj < 0 || nj > ni) continue;
                            extraEdges.push([globalIdx(i, j), globalIdx(ni, nj)]);
                        }
                cornerQuot.push([A, globalIdx(0, 0)], [B, globalIdx(w - 1, 0)], [C, globalIdx(w - 1, w - 1)]);
                addSide(A, B, k => globalIdx(k, 0));
                addSide(A, C, k => globalIdx(k, k));
                addSide(B, C, k => globalIdx(w - 1, k));
            }
        } else if (sel.type === 'quad') {
            const quads = selectQuad(bc.adj, bc.emb.pos, sel);
            const nFace = w * w;
            const localIdx = (i: number, j: number) => i * w + j;
            const dirs: [number, number][] = [[0, 1], [1, 0], [0, -1], [-1, 0]];
            const denom = (w - 1) * (w - 1);
            for (const { n1: A, n2: B, n3: C, n4: D } of quads) {
                const offset = nextIdx;
                nextIdx += nFace;
                const globalIdx = (i: number, j: number) => offset + localIdx(i, j);
                const cornerA = scaledPos[A], cornerB = scaledPos[B], cornerC = scaledPos[C], cornerD = scaledPos[D];
                for (let i = 0; i < w; i++)
                    for (let j = 0; j < w; j++) {
                        const wA = (w - 1 - i) * (w - 1 - j), wB = (w - 1 - i) * j;
                        const wC = i * j, wD = i * (w - 1 - j);
                        extraPos[globalIdx(i, j) - N] = w === 1
                            ? cornerA.map((_, k) => (cornerA[k] + cornerB[k] + cornerC[k] + cornerD[k]) / 4)
                            : cornerA.map((_, k) =>
                                (wA * cornerA[k] + wB * cornerB[k] + wC * cornerC[k] + wD * cornerD[k]) / denom);
                    }
                for (let i = 0; i < w; i++)
                    for (let j = 0; j < w; j++)
                        for (const [di, dj] of dirs) {
                            const ni = i + di, nj = j + dj;
                            if (ni < 0 || ni >= w || nj < 0 || nj >= w) continue;
                            extraEdges.push([globalIdx(i, j), globalIdx(ni, nj)]);
                        }
                cornerQuot.push(
                    [A, globalIdx(0, 0)], [B, globalIdx(0, w - 1)],
                    [C, globalIdx(w - 1, w - 1)], [D, globalIdx(w - 1, 0)],
                );
                // Same top/right/bottom/left convention as the old quadForm's own naturalSeq -
                // addSide itself handles the min/max reorientation, so these are always declared
                // running from each side's first-listed corner to its second.
                addSide(A, B, k => globalIdx(0, k));
                addSide(B, C, k => globalIdx(k, w - 1));
                addSide(C, D, k => globalIdx(w - 1, w - 1 - k));
                addSide(D, A, k => globalIdx(w - 1 - k, 0));
            }
        } else {
            throw new Error(`genericForm: each selector in sels must be a triangle (simp 2) or quad selector, got a '${sel.type}' selector`);
        }
    }

    const totalN = nextIdx;
    const pos: number[][] = new Array(totalN);
    for (let i = 0; i < N; i++) pos[i] = scaledPos[i];
    for (let i = N; i < totalN; i++) pos[i] = extraPos[i - N];

    const adj = zeroAdj(totalN);
    for (let i = 0; i < N; i++)
        for (let j = i + 1; j < N; j++) {
            if (!bc.adj[i][j] || isFaceSide.has(`${i},${j}`)) continue;
            adj[i][j] = 1;
            adj[j][i] = 1;
        }
    for (const [a, b] of extraEdges) {
        adj[a][b] = 1;
        adj[b][a] = 1;
    }

    const quot: [number, number][] = [...cornerQuot];
    for (const seqs of edgeToSeqs.values()) {
        if (seqs.length < 2) continue;
        for (let s = 1; s < seqs.length; s++)
            for (let k = 0; k < w; k++) quot.push([seqs[0](k), seqs[s](k)]);
    }

    const combined = make(new Embedding(embDim, pos), adj);
    return quotientBoard(combined, quot);
}

/**
 * Replaces every triangle (3 mutually-adjacent, distinct vertices - see topology.ts's
 * findSimplices(adj, 2)) in `bc` with a `triangularBoard(w)`-shaped lattice - the single-kind
 * special case of genericForm (see its own doc comment). `sel`, if given, restricts this to only
 * the triangles it selects (evaluated against `bc`'s own adj/pos) - every other triangle is left
 * untouched, as if it didn't exist (its own sides stay plain edges, even where they'd otherwise
 * have been consumed by/glued to a selected triangle's new subdivided boundary).
 */
export function triangleForm(bc: BoardConfig, w: number, sel?: Selector): BoardConfig {
    return genericForm(bc, w, [sel ?? { op: 'all', type: simpType(2) }]);
}

/**
 * Replaces every quad (4 distinct vertices forming a cycle with no diagonal edges - see
 * topology.ts's findQuads) in `bc` with a `w`-by-`w` grid - the single-kind special case of
 * genericForm (see its own doc comment), the same way triangleForm is. `sel`, if given, restricts
 * this to only the quads it selects (evaluated against `bc`'s own adj/pos) - every other quad is
 * left untouched, as if it didn't exist (its own sides stay plain edges, even where they'd otherwise
 * have been consumed by/glued to a selected quad's new subdivided boundary).
 */
export function quadForm(bc: BoardConfig, w: number, sel?: Selector): BoardConfig {
    return genericForm(bc, w, [sel ?? { op: 'all', type: 'quad' }]);
}

/**
 * Replaces every selected n-simplex and/or quad in `bc` (see topology.ts's findSimplices/findQuads)
 * with its own small local shape: an n-simplex's "pyramid" (one new hub node, connected to all of
 * that face's own corners - SimpCentralize, e.g. a triangle -> tetrahedron), a quad's own pyramid
 * (QuadCentralize), a quad's own octahedron (two new antipodal apex nodes, each connected to all 4
 * corners - QuadOctarize), or a "centering" variant of either hub-and-spoke shape (SimpCentering/
 * QuadCentering - same new hub, but the face's own original edges are dropped rather than kept, so
 * its corners end up connected only through the hub). Unlike genericForm, nothing is subdivided/glued
 * - only that face's own new node(s) and their own edges are added (or, for the Centering branches,
 * added in place of the face's own original edges). Each element of `selectors` names both which
 * faces to look for (via its own `sel`, defaulting the same way each single-kind thin wrapper below
 * documents its own `sel?` parameter) AND which of these local shapes to build there - this is what
 * lets a single `quad` selection mean a pyramid (QuadCentralize), an octahedron (QuadOctarize), or a
 * bare star (QuadCentering), which a bare Selector's own `type` alone could never disambiguate (see
 * LocalReplaceSelector's own doc comment, shared/types.ts) - there's no separate TriCentralize/
 * TriCentering branch, since a triangle is just SimpCentralize's/SimpCentering's own n=2 case
 * (triCentralize below is a thin wrapper over simpCentralize(bc, 2, ...), same as
 * shared/clegEval.ts's own cleg-level triCentralize builtin).
 *
 * Every selected face's own ORIGINAL edges (a simplex's own C(n+1,2) clique edges, or a quad's own
 * 4-cycle) are excluded from `bc.adj`'s straight copy - for every branch except SimpCentering/
 * QuadCentering, they're then re-added explicitly, alongside whichever new edges its own local shape
 * needs: a triangle's shape is a tetrahedron (3 original + 3 hub edges), a quad-pyramid's is 4
 * original + 4 hub edges, a quad-octahedron's is 4 original (the equatorial ring) + 8 apex edges (12
 * total, matching a real octahedron). For every one of THOSE branches this nets out to the same edges
 * as simply keeping the originals and adding the new ones would - expressing it as "the target
 * shape's own full edge set" (rather than "originals, kept, plus new") is what lets SimpCentering/
 * QuadCentering fit the exact same framework, just by skipping that one re-add step: their own target
 * shape is the hub edges ALONE, so the face's own original edges, once excluded from the straight
 * copy, simply stay gone. Two selectors whose own faces share an original edge each independently
 * decide whether to re-add it - if any one of them does, it survives (a shared edge consumed by only
 * Centering-kind selectors, and never re-added by any of them, is genuinely dropped) - harmless
 * either way, since `adj[][]` is a plain 0/1 matrix (setting an edge twice is a no-op), the same way
 * genericForm's own edgeToSeqs accumulates across selectors.
 *
 * QuadOctarize's own two new apex nodes need one MORE embedding dimension than every other new node
 * here (see quadOctarize's own doc comment on why) - so if ANY selector is a QuadOctarize, every
 * node's own position (original and new alike) gets one extra trailing 0 coordinate, keeping one
 * shared Embedding for the whole result; otherwise the embedding dimension is untouched.
 */
export function genericLocalReplace(bc: BoardConfig, selectors: LocalReplaceSelector[]): BoardConfig {
    const N = bc.N;
    const embDim = bc.emb.embDim;
    const needsExtraDim = selectors.some(s => s.kind === 'QuadOctarize');
    const outEmbDim = embDim + (needsExtraDim ? 1 : 0);
    const pad = (p: number[]): number[] => needsExtraDim ? [...p, 0] : p;

    // "i,j" (i<j) - a selected face's own original edge, excluded from bc.adj's straight copy below
    // (and re-added explicitly, alongside every new-node edge, as extraEdges - see this function's
    // own doc comment on why).
    const consumed = new Set<string>();
    const markConsumed = (a: number, b: number) => consumed.add(a < b ? `${a},${b}` : `${b},${a}`);

    const extraPos: number[][] = [];
    const extraEdges: [number, number][] = [];
    let nextIdx = N;

    for (const s of selectors) {
        if (s.kind === 'QuadCentralize' || s.kind === 'QuadOctarize' || s.kind === 'QuadCentering') {
            const sel = s.sel ?? { op: 'all' as const, type: 'quad' as const };
            const quads = selectQuad(bc.adj, bc.emb.pos, sel);
            for (const { n1: A, n2: B, n3: C, n4: D } of quads) {
                markConsumed(A, B); markConsumed(B, C); markConsumed(C, D); markConsumed(D, A);
                // QuadCentering is the one branch that does NOT re-add the quad's own 4-cycle edges -
                // see LocalReplaceSelector's own doc comment (shared/types.ts).
                if (s.kind !== 'QuadCentering') extraEdges.push([A, B], [B, C], [C, D], [D, A]);
                const barycenter = bc.emb.pos[A].map((_, k) =>
                    (bc.emb.pos[A][k] + bc.emb.pos[B][k] + bc.emb.pos[C][k] + bc.emb.pos[D][k]) / 4);
                if (s.kind === 'QuadOctarize') {
                    // Same barycenter/dist derivation this function's own predecessor used to compute
                    // inline (see this function's own doc comment).
                    let dist = 0;
                    for (const idx of [A, B, C, D]) {
                        const diff = bc.emb.pos[idx].map((v, k) => v - barycenter[k]);
                        dist += Math.hypot(...diff) / 4;
                    }
                    const top = nextIdx++, bottom = nextIdx++;
                    extraPos[top - N] = [...barycenter, dist];
                    extraPos[bottom - N] = [...barycenter, -dist];
                    extraEdges.push([top, A], [top, B], [top, C], [top, D]);
                    extraEdges.push([bottom, A], [bottom, B], [bottom, C], [bottom, D]);
                } else {
                    // QuadCentralize or QuadCentering - a single hub either way, differing only in
                    // whether the quad's own 4-cycle edges survive (see above).
                    const hub = nextIdx++;
                    extraPos[hub - N] = pad(barycenter);
                    extraEdges.push([hub, A], [hub, B], [hub, C], [hub, D]);
                }
            }
        } else {
            const n = s.n;
            assert(Number.isInteger(n) && n >= 2, `genericLocalReplace: n must be an integer >= 2, got ${n}`);
            const sel = s.sel ?? { op: 'all' as const, type: simpType(n) };
            const simplices = selectSimp(bc.adj, bc.emb.pos, sel);
            for (const { nodes } of simplices) {
                for (let i = 0; i < nodes.length; i++)
                    for (let j = i + 1; j < nodes.length; j++) {
                        markConsumed(nodes[i], nodes[j]);
                        // SimpCentering is the one branch that does NOT re-add the simplex's own
                        // clique edges - see LocalReplaceSelector's own doc comment (shared/types.ts).
                        if (s.kind !== 'SimpCentering') extraEdges.push([nodes[i], nodes[j]]);
                    }
                const hub = nextIdx++;
                extraPos[hub - N] = pad(bc.emb.pos[nodes[0]].map((_, k) =>
                    nodes.reduce((sum, v) => sum + bc.emb.pos[v][k], 0) / nodes.length));
                extraEdges.push(...nodes.map((v): [number, number] => [hub, v]));
            }
        }
    }

    const totalN = nextIdx;
    const pos: number[][] = new Array(totalN);
    for (let i = 0; i < N; i++) pos[i] = pad(bc.emb.pos[i]);
    for (let i = N; i < totalN; i++) pos[i] = extraPos[i - N];

    const adj = zeroAdj(totalN);
    for (let i = 0; i < N; i++)
        for (let j = i + 1; j < N; j++) {
            if (!bc.adj[i][j] || consumed.has(`${i},${j}`)) continue;
            adj[i][j] = 1;
            adj[j][i] = 1;
        }
    for (const [a, b] of extraEdges) {
        adj[a][b] = 1;
        adj[b][a] = 1;
    }

    return make(new Embedding(outEmbDim, pos), adj);
}

/**
 * Adds one new node ("centralizes") for every n-simplex in `bc`, connected to all n+1 of its own
 * corners - the single-arity special case of genericLocalReplace (see its own doc comment), just with
 * `n` given directly instead of folded into `sel`'s own type. `sel`, if given, restricts this to
 * only the n-simplices it selects (evaluated against `bc`'s own adj/pos, and must itself already be
 * a simp `n` selector) - every other n-simplex is left untouched, as if it didn't exist.
 */
export function simpCentralize(bc: BoardConfig, n: number, sel?: Selector): BoardConfig {
    assert(Number.isInteger(n) && n >= 2, `simpCentralize: n must be an integer >= 2, got ${n}`);
    return genericLocalReplace(bc, [{ kind: 'SimpCentralize', n, sel }]);
}

/**
 * Adds one new node for every n-simplex in `bc`, connected to all n+1 of its own corners - same as
 * simpCentralize, except the simplex's own C(n+1,2) original edges are DROPPED rather than kept, so
 * its corners end up connected only through the new hub, not to each other directly (SimpCentering,
 * the single-arity special case of genericLocalReplace - see its own doc comment). `sel`, if given,
 * restricts this to only the n-simplices it selects - every other n-simplex is left untouched.
 */
export function simpCentering(bc: BoardConfig, n: number, sel?: Selector): BoardConfig {
    assert(Number.isInteger(n) && n >= 2, `simpCentering: n must be an integer >= 2, got ${n}`);
    return genericLocalReplace(bc, [{ kind: 'SimpCentering', n, sel }]);
}

/**
 * Adds one new node ("centralizes") for every triangle in `bc`, connected to all 3 of its own
 * corners - simpCentralize's own n=2 special case. `sel`, if given, restricts this to only the
 * triangles it selects (evaluated against `bc`'s own adj/pos) - every other triangle is left
 * untouched, as if it didn't exist.
 */
export function triCentralize(bc: BoardConfig, sel?: Selector): BoardConfig {
    return simpCentralize(bc, 2, sel);
}

/**
 * Adds one new node for every triangle in `bc`, connected to all 3 of its own corners - same as
 * triCentralize, except the triangle's own 3 original edges are DROPPED rather than kept, so its
 * corners end up connected only through the new hub, not to each other directly - simpCentering's own
 * n=2 special case, the same way triCentralize is simpCentralize's.
 */
export function triCentering(bc: BoardConfig, sel?: Selector): BoardConfig {
    return simpCentering(bc, 2, sel);
}

/**
 * Adds one new node ("centralizes") for every quad in `bc`, connected to all 4 of its own corners -
 * the single-kind special case of genericLocalReplace (see its own doc comment), the same way
 * triCentralize is. `sel`, if given, restricts this to only the quads it selects (evaluated against
 * `bc`'s own adj/pos) - every other quad is left untouched, as if it didn't exist.
 */
export function quadCentralize(bc: BoardConfig, sel?: Selector): BoardConfig {
    return genericLocalReplace(bc, [{ kind: 'QuadCentralize', sel }]);
}

/**
 * Adds one new node for every quad in `bc`, connected to all 4 of its own corners - same as
 * quadCentralize, except the quad's own 4-cycle original edges are DROPPED rather than kept, so its
 * corners end up connected only through the new hub, not to each other directly (QuadCentering, the
 * single-kind special case of genericLocalReplace - see its own doc comment). `sel`, if given,
 * restricts this to only the quads it selects - every other quad is left untouched.
 */
export function quadCentering(bc: BoardConfig, sel?: Selector): BoardConfig {
    return genericLocalReplace(bc, [{ kind: 'QuadCentering', sel }]);
}

/**
 * Adds one new node at `bc`'s barycenter (the component-wise average of every existing node's
 * natural-dimension position), connected to every existing node - a single hub adjacent to the
 * whole board at once, unlike quadForm/triangleForm's per-face subdivision. Existing nodes/edges
 * are otherwise untouched.
 */
export function globalCentralize(bc: BoardConfig): BoardConfig {
    const N = bc.N;
    const embDim = bc.emb.embDim;
    const barycenter = new Array(embDim).fill(0);
    for (const p of bc.emb.pos)
        for (let k = 0; k < embDim; k++) barycenter[k] += p[k] / N;

    const pos = [...bc.emb.pos, barycenter];
    const adj = zeroAdj(N + 1);
    for (let i = 0; i < N; i++)
        for (let j = i + 1; j < N; j++) {
            if (!bc.adj[i][j]) continue;
            adj[i][j] = 1;
            adj[j][i] = 1;
        }
    for (let i = 0; i < N; i++) {
        adj[i][N] = 1;
        adj[N][i] = 1;
    }

    return make(new Embedding(embDim, pos), adj);
}

/**
 * Replaces every selected quad (4-cycle with no diagonal edges - see topology.ts's `findQuads`, same
 * quads `quadForm`/`quadCentralize` work with) with an octahedron - the QuadOctarize single-kind
 * special case of genericLocalReplace (see its own doc comment): two new "apex" nodes per quad, one
 * on each side along a new embedding dimension (see genericLocalReplace's own doc comment on why this
 * needs one), each connected to all 4 of that quad's corners - the quad's own 4-cycle edges become
 * the octahedron's equatorial ring, and the two apexes are NOT connected to each other (antipodal,
 * same as `octahedronBoard()`'s own apex pairs - a plain quad graph plus two such apex nodes is
 * exactly an octahedron's edge set, see that function's doc comment).
 *
 * Each apex sits, in the original `embDim` dimensions, at its quad's barycenter (the component-wise
 * average of its 4 corners), and at +-`dist` along the new dimension, where `dist` is the average
 * distance from each of the quad's 4 corners to that same barycenter (the exact circumradius for a
 * geometrically regular quad, since all 4 corners are then equidistant from it - averaging just keeps
 * this well-defined for a quad whose corners aren't quite equidistant from their own barycenter).
 * `sel`, if given, restricts this to only the quads it selects - every other quad is left untouched,
 * as if it didn't exist.
 */
export function quadOctarize(bc: BoardConfig, sel?: Selector): BoardConfig {
    return genericLocalReplace(bc, [{ kind: 'QuadOctarize', sel }]);
}

/** Multiplies every node's natural-dimension position by `factor` - adjacency/embDim untouched. */
export function scaleBoard(bc: BoardConfig, factor: number): BoardConfig {
    const pos = bc.emb.pos.map(p => p.map(v => v * factor));
    return make(new Embedding(bc.emb.embDim, pos), bc.adj);
}

/**
 * The Cartesian (box) product of two board configs: N = `bc1.N * bc2.N`, one new node per pair
 * `(i, j)` (`i` from `bc1`, `j` from `bc2`), at the concatenated natural position
 * `[...bc1.emb.pos[i], ...bc2.emb.pos[j]]` (embDim = `bc1.emb.embDim + bc2.emb.embDim`). `(i, j)` is
 * adjacent to `(i2, j2)` iff exactly one of:
 *   - `i === i2` and `j` is adjacent to `j2` in `bc2`
 *   - `j === j2` and `i` is adjacent to `i2` in `bc1`
 * (the standard graph Cartesian product - e.g. `cubeLatticeBoard(w, h, d)` is, up to embedding, the
 * product of three path graphs).
 */
export function product(bc1: BoardConfig, bc2: BoardConfig): BoardConfig {
    const N1 = bc1.N, N2 = bc2.N;
    const embDim = bc1.emb.embDim + bc2.emb.embDim;
    const idx = (i: number, j: number) => i * N2 + j;

    const pos: number[][] = [];
    for (let i = 0; i < N1; i++)
        for (let j = 0; j < N2; j++)
            pos.push([...bc1.emb.pos[i], ...bc2.emb.pos[j]]);

    const adj = zeroAdj(N1 * N2);
    for (let i = 0; i < N1; i++)
        for (let j = 0; j < N2; j++) {
            for (let i2 = 0; i2 < N1; i2++)
                if (bc1.adj[i][i2]) adj[idx(i, j)][idx(i2, j)] = 1;
            for (let j2 = 0; j2 < N2; j2++)
                if (bc2.adj[j][j2]) adj[idx(i, j)][idx(i, j2)] = 1;
        }

    return make(new Embedding(embDim, pos), adj);
}

/** A board with `w` nodes forming a simple line: node `i` is connected to node `i + 1`. */
export function linearBoard(w: number): BoardConfig {
    assert(w > 0, `w must be positive, got w=${w}`);
    const pos: number[][] = [];
    for (let i = 0; i < w; i++) pos.push([i - (w - 1) / 2]);
    const adj = zeroAdj(w);
    for (let i = 0; i < w - 1; i++) {
        adj[i][i + 1] = 1;
        adj[i + 1][i] = 1;
    }
    return make(new Embedding(1, pos), adj);
}

/** A rectangular board with width `w` and height `h`. Each node is identified by (col, row) where 0 ≤ col < w, 0 ≤ row < h. */
export function rectangularBoard(w: number, h: number): BoardConfig {
    return hypercuboidBoard(2, [w, h]);
}

// Recognizes `config.boardDescr` as being SYNTACTICALLY exactly one bare `rectB(<NumberLit>,
// <NumberLit>);` statement (no functions declared, nothing else) - the one shape computeStarPoints
// below knows how to derive Go star points from. Deliberately a purely structural AST check, not an
// evaluation of the program (which could be an arbitrarily complex construction, e.g. modify(...)
// calls, a for loop, a helper function) - anything other than this one exact shape means "not
// recognizable as a plain rectangular board" and disables star points, same as the old
// boardType!=='rect' fallback did.
function tryExtractPlainRectDims(program: ClegProgram): [number, number] | null {
    if (program.functions.length !== 0 || program.stmts.length !== 1) return null;
    const stmt = program.stmts[0];
    if (stmt.kind !== 'ExprStmt') return null;
    const expr = stmt.expr;
    if (expr.kind !== 'CallExpr' || expr.callee !== 'rectB' || expr.args.length !== 2) return null;
    const [a, b] = expr.args;
    if (a.kind !== 'NumberLit' || b.kind !== 'NumberLit') return null;
    return [a.value, b.value];
}

/**
 * Traditional Go board star points ("hoshi") for a rectangular board, derived from
 * `config.boardDescr` when it's syntactically just `rectB(w, h);` (see tryExtractPlainRectDims
 * above) - [] for anything else (a different board type, or any modifier/loop/helper-function
 * construction, however it would actually evaluate). Corner points sit at the 3-3 point (boards
 * whose smaller edge is 9 or 11) or the 4-4 point (smaller edge > 11), edge points sit at the
 * midpoint of an odd, >=19-length edge whose cross edge is >=9, and a single center point appears
 * when both edges are odd and >=5 - together reproducing the real 9x9 (4 corners + center), 13x13
 * (4 corners + center), and 19x19 (4 corner + 4 edge + center) star-point layouts. Returned as
 * [x, y] pairs in the same board-coordinate space as BoardConfig.pos (see rectangularBoard()
 * above), ready for the same originX + x*cell / originY - y*cell screen transform.
 */
export function computeStarPoints(config: GameConfig): number[][] {
    const dims = tryExtractPlainRectDims(config.boardDescr);
    if (!dims) return [];
    const [w, h] = dims;
    const toBoard = (c: number, r: number): number[] => [c - (w - 1) / 2, r - (h - 1) / 2];
    const points: number[][] = [];
    // Same inset (distance from an edge) for every star point, corner or edge-midpoint alike -
    // keeps e.g. a 19x9 board's edge-midpoint points on the same row as its corner points instead
    // of assuming the 19x19 board's 4-4 line regardless of h.
    const inset = Math.min(w, h) <= 11 ? 3 : 4;

    if (w >= 9 && h >= 9) {
        for (const c of [inset - 1, w - inset])
            for (const r of [inset - 1, h - inset])
                points.push(toBoard(c, r));
    }
    if (w % 2 === 1 && w >= 19 && h >= 9) {
        const c = (w - 1) / 2;
        points.push(toBoard(c, inset - 1), toBoard(c, h - inset));
    }
    if (h % 2 === 1 && h >= 19 && w >= 9) {
        const r = (h - 1) / 2;
        points.push(toBoard(inset - 1, r), toBoard(w - inset, r));
    }
    if (w % 2 === 1 && h % 2 === 1 && w >= 5 && h >= 5)
        points.push(toBoard((w - 1) / 2, (h - 1) / 2));

    return points;
}

/**
 * A rectangular board with width `w` and height `h` where diagonally adjacent nodes are also
 * connected, but only at every `m`-th square.
 */
export function rectangularDiagonalBoard(w: number, h: number, m: number): BoardConfig {
    assert(w > 0 && h > 0 && m > 0, `w, h, and m must be positive, got w=${w} h=${h} m=${m}`);
    const pos: number[][] = [];
    for (let r = 0; r < h; r++)
        for (let c = 0; c < w; c++)
            pos.push([c - (w - 1) / 2, r - (h - 1) / 2]);
    const adj = zeroAdj(w * h);
    const dirs: [number, number][] = [[0,1],[1,0],[0,-1],[-1,0],[1,1],[-1,1]];
    for (let r = 0; r < h; r++) {
        for (let c = 0; c < w; c++) {
            for (const [dr, dc] of dirs) {
                const nr = r + dr, nc = c + dc;
                if (nr < 0 || nr >= h || nc < 0 || nc >= w) continue;
                if (Math.abs(dr) === 1 && Math.abs(dc) === 1 && m > 1 &&
                  ((r + Math.floor((dr - 1) / 2)) % m !== 0 || c % m !== 0)) continue;
                adj[r*w+c][nr*w+nc] = 1;
                adj[nr*w+nc][r*w+c] = 1;
            }
        }
    }
    return make(pos, adj);
}

/**
 * A cubical board with width `w`, height `h` and depth `d`. Each node is identified by
 * (col, row, slice) where 0 ≤ col < w, 0 ≤ row < h, 0 ≤ slice < d.
 */
export function cubeLatticeBoard(w: number, h: number, d: number): BoardConfig {
    return hypercuboidBoard(3, [w, h, d]);
}

/**
 * The `meshdim`-skeleton of a `dims.length`-dimensional hypercuboid with `dims[i]` points along
 * axis `i`: a node survives (occurs on the board) iff at most `meshdim` of its coordinates are
 * strictly interior to their own axis (i.e. neither 0 nor `dims[i] - 1`) - equivalently, iff it
 * lies on some axis-aligned face of dimension <= `meshdim`. Surviving nodes keep the plain grid
 * adjacency (connected iff they differ by exactly 1 in exactly one coordinate); `rectangularBoard`
 * (below)/`cubeLatticeBoard` (above) are now just this function called with `meshdim` equal to
 * their own full dimension count (so nothing is ever excluded) and 2/3 dims.
 *
 * `meshdim = dims.length` (or higher) keeps everything - the full solid hypercuboid, same as
 * before this parameter existed. `meshdim = 0` keeps only the `2**dims.length` corners (each
 * coordinate pinned to an extreme), though at that point they're never adjacent to each other -
 * corners are only ever unit-step apart from an interior point, and interior points are exactly
 * what a small meshdim excludes. `meshdim = k - 1` (for `dims.length = k`) is the hollow surface
 * of the hypercuboid: e.g. a 3D box's own 6 rectangular faces, glued at their shared edges, with
 * the solid interior excluded - a 4D hypercuboid's `meshdim = 3` gives 8 solid cuboid facets glued
 * at their shared square faces, and so on.
 */
export function hypercuboidBoard(meshdim: number, dims: number[]): BoardConfig {
    assert(dims.length >= 1, `dims must have at least 1 entry, got ${dims.length}`);
    assert(dims.every(d => d > 0), `every dimension must be positive, got [${dims.join(', ')}]`);
    assert(Number.isInteger(meshdim) && meshdim >= 0, `meshdim must be a non-negative integer, got ${meshdim}`);
    const k = dims.length;
    const fullN = dims.reduce((p, d) => p * d, 1);

    const strides = new Array(k);
    strides[0] = 1;
    for (let i = 1; i < k; i++) strides[i] = strides[i - 1] * dims[i - 1];
    const fullIdx = (coords: number[]) => coords.reduce((s, c, i) => s + c * strides[i], 0);
    const coordsOf = (n: number) => {
        const coords = new Array(k);
        for (let i = 0; i < k; i++) { coords[i] = n % dims[i]; n = Math.floor(n / dims[i]); }
        return coords;
    };
    const isInterior = (c: number, dim: number) => c > 0 && c < dim - 1;

    // Only surviving nodes get a board index (compacted, in ascending full-lattice-index order) -
    // boardIdxOf maps a full-lattice index to that compacted index, absent for a culled node.
    const boardIdxOf = new Map<number, number>();
    const survivingCoords: number[][] = [];
    const pos: number[][] = [];
    for (let n = 0; n < fullN; n++) {
        const coords = coordsOf(n);
        const interiorCount = coords.reduce((cnt, c, i) => cnt + (isInterior(c, dims[i]) ? 1 : 0), 0);
        if (interiorCount > meshdim) continue;
        boardIdxOf.set(n, survivingCoords.length);
        survivingCoords.push(coords);
        pos.push(coords.map((c, i) => c - (dims[i] - 1) / 2));
    }
    const N = survivingCoords.length;

    const adj = zeroAdj(N);
    for (let bi = 0; bi < N; bi++) {
        const coords = survivingCoords[bi];
        for (let i = 0; i < k; i++)
            for (const delta of [1, -1]) {
                const nc = coords[i] + delta;
                if (nc < 0 || nc >= dims[i]) continue;
                const ncoords = coords.slice();
                ncoords[i] = nc;
                const nbi = boardIdxOf.get(fullIdx(ncoords));
                if (nbi === undefined) continue; // that neighbor didn't survive the meshdim filter
                adj[bi][nbi] = 1;
            }
    }
    return make(new Embedding(k, pos), adj);
}

/** A triangular board with side length `w`. */
export function triangularBoard(w: number): BoardConfig {
    assert(w > 0, `w must be positive, got w=${w}`);
    const rowDist = Math.sqrt(3) / 2;
    const pos: number[][] = [];
    for (let i = 0; i < w; i++) {
        for (let j = 0; j <= i; j++) {
            pos.push([j - i/2, rowDist * (i - (w-1)/2)]);
        }
    }
    const N = w * (w + 1) / 2;
    const adj = zeroAdj(N);
    const idx = (i: number, j: number) => i * (i + 1) / 2 + j;
    for (let i = 0; i < w; i++)
        for (let j = 0; j <= i; j++)
            for (const [di, dj] of [[1,0],[1,1],[0,1],[-1,0],[-1,-1],[0,-1]]) {
                const ni = i+di, nj = j+dj;
                if (ni < 0 || ni >= w || nj < 0 || nj > ni) continue;
                adj[idx(i,j)][idx(ni,nj)] = 1;
            }
    return make(pos, adj);
}

/**
 * A regular polygon with `n` edges, each of length 1: `n` nodes on a circle of radius
 * `1 / (2*sin(pi/n))` (the standard circumradius of a unit-side regular n-gon), at angle
 * `2*pi*k/n` for node `k`, connected in a cycle (node `k` to node `(k+1) mod n`) - same
 * floating-point-coordinate convention already used by triangularBoard's rowDist above.
 */
export function regularPolygonBoard(n: number): BoardConfig {
    assert(n >= 3, `n must be at least 3, got ${n}`);
    const r = 1 / (2 * Math.sin(Math.PI / n));
    const pos: number[][] = [];
    for (let k = 0; k < n; k++) {
        const theta = (2 * Math.PI * k) / n;
        pos.push([r * Math.cos(theta), r * Math.sin(theta)]);
    }
    const adj = zeroAdj(n);
    for (let k = 0; k < n; k++) {
        const next = (k + 1) % n;
        adj[k][next] = 1;
        adj[next][k] = 1;
    }
    return make(pos, adj);
}

/**
 * A star graph: 1 center node (index 0) connected to `n` outer nodes (indices 1..n), each unit
 * distance from the center at angle `2*pi*k/n` - outer nodes are not connected to each other.
 */
export function starBoard(n: number): BoardConfig {
    assert(n >= 1, `n must be at least 1, got ${n}`);
    const pos: number[][] = [[0, 0]];
    for (let k = 0; k < n; k++) {
        const theta = (2 * Math.PI * k) / n;
        pos.push([Math.cos(theta), Math.sin(theta)]);
    }
    const adj = zeroAdj(n + 1);
    for (let k = 1; k <= n; k++) {
        adj[0][k] = 1;
        adj[k][0] = 1;
    }
    return make(pos, adj);
}

/**
 * Recursive core of sierpinskiSimplex(): builds the order-`n` gasket (n >= 1) directly on the given
 * `corners` (length dim+1, one per simplex vertex). n=1 is the literal (dim+1)-node complete graph
 * (every vertex mutually adjacent); n>1 recurses into dim+1 order-(n-1) copies, one "near" each
 * vertex k - copy k's own (dim+1) corners are: corners[k] itself at position k, and the midpoint of
 * corners[k]/corners[j] at every other position j - and glues every pair of copies (a, b) at their
 * shared edge-midpoint corner (copy a's position b == copy b's position a, both mid(a,b)) - the
 * "hole" (the sub-simplex that a *full* subdivision would place at the centroid-ward positions) is
 * simply never built. Returns the built board plus the (possibly-merged) node index of each of its
 * own dim+1 corners, in the same order as the input `corners`, so an outer recursive call can glue
 * onto them in turn.
 *
 * Generalizes what was 3-corner-special-cased code (P0/P1/P2, sub0/sub1/sub2, two explicit merges)
 * to dim+1 corners, dim+1 copies, and dim+1 incremental merges - checked to produce the exact same
 * per-step computation as that special-cased version when corners.length === 3.
 */
function sierpinskiRec(n: number, corners: number[][]): { pos: number[][]; adj: number[][]; corners: number[] } {
    const k = corners.length; // dim + 1
    if (n === 1) {
        const adj = zeroAdj(k);
        for (let i = 0; i < k; i++)
            for (let j = i + 1; j < k; j++) { adj[i][j] = 1; adj[j][i] = 1; }
        return { pos: corners.map(p => [...p]), adj, corners: corners.map((_, i) => i) };
    }
    const mid = (a: number[], b: number[]) => a.map((v, d) => (v + b[d]) / 2);
    const subs = corners.map((_, k_) =>
        sierpinskiRec(n - 1, corners.map((c, p) => (p === k_ ? corners[k_] : mid(corners[k_], c)))));

    const merges: [[number, number], [number, number]][] = [];
    for (let a = 0; a < k; a++)
        for (let b = a + 1; b < k; b++) merges.push([[a, subs[a].corners[b]], [b, subs[b].corners[a]]]);

    const m = mergeBoards(subs, merges);
    const cornersOut = subs.map((sub, k_) => m.maps[k_][sub.corners[k_]]);

    return { pos: m.pos, adj: m.adj, corners: cornersOut };
}

/**
 * Coordinates of a regular dim-simplex (dim+1 vertices), unit edge length, centroid at the origin,
 * in R^dim. Recursive: a (dim-1)-simplex (by induction, already unit-edge/centroid-at-origin) forms
 * the "base" at height 0 on a fresh axis; an apex sits directly above the base's own centroid, at
 * the height that puts it exactly unit distance from every base vertex (h = sqrt(1 - R^2), where R
 * is a unit-edge (dim-1)-simplex's own circumradius sqrt((dim-1)/(2*dim)) - so h simplifies to
 * sqrt((dim+1)/(2*dim))); the whole (dim+1)-point set is then shifted by -h/(dim+1) on that axis so
 * the overall centroid returns to the origin (every vertex, base or apex, ends up exactly
 * sqrt(dim/(2*(dim+1))) from it - the standard unit-edge dim-simplex circumradius, confirming the
 * construction).
 */
function regularSimplexCoords(dim: number): number[][] {
    if (dim === 1) return [[-0.5], [0.5]];
    const base = regularSimplexCoords(dim - 1);
    const h = Math.sqrt((dim + 1) / (2 * dim));
    const shift = h / (dim + 1);
    const points = base.map(p => [...p, -shift]);
    points.push([...new Array(dim - 1).fill(0), h - shift]);
    return points;
}

/**
 * The Sierpinski dim-simplex (gasket) of order `n` (n >= 1): n=1 is a unit-edge regular dim-simplex
 * (dim+1 mutually-adjacent nodes, see regularSimplexCoords); for n>1, dim+1 copies of order n-1
 * (each half the linear size) sit at the corners of the outer simplex and are glued at their
 * touching edge-midpoints - see sierpinskiRec for the recursive construction. For dim=1 there is no
 * "hole" to remove (a 1-simplex's only subdivision is its own 2 halves), so this degenerates to a
 * plain, fully-subdivided line of 2^(n-1)+1 nodes - not a fractal at all.
 *
 * The outer simplex is placed symmetric about the origin (regularSimplexCoords' own convention),
 * and since the recursive rule above is itself symmetric under the outer simplex's own (dim+1)!
 * vertex-relabeling symmetry group (relabeling corners and rebuilding reproduces the exact same
 * node-position set), the resulting node set's own centroid always lands exactly on the origin too,
 * for every n - not just the n=1 base case.
 */
export function sierpinskiSimplex(dim: number, n: number): BoardConfig {
    assert(Number.isInteger(dim) && dim >= 1, `dim must be a positive integer, got ${dim}`);
    assert(Number.isInteger(n) && n >= 1, `n must be a positive integer, got ${n}`);

    const scale = 2 ** (n - 1); // unit-edge at n=1, doubling per level
    const corners = regularSimplexCoords(dim).map(p => p.map(v => v * scale));

    const built = sierpinskiRec(n, corners);
    return make(new Embedding(dim, built.pos), built.adj);
}

/**
 * The `meshdim`-skeleton of a regular `dim`-simplex subdivided into a lattice of side length `w`:
 * lattice points are identified by barycentric coordinates `(c_0, ..., c_dim)`, non-negative
 * integers summing to `w - 1`, one per simplex vertex; a point survives (occurs on the board) iff
 * at most `meshdim + 1` of its coordinates are nonzero, i.e. it lies on some face of dimension
 * <= `meshdim` (the face spanned by exactly its nonzero-coordinate vertices) - the same "faces of
 * dimension <= meshdim survive" rule as `hypercuboidBoard`, just expressed in barycentric rather
 * than axis-aligned terms. Two surviving points are adjacent iff their coordinates differ by
 * transferring one unit from one coordinate to another (`c_i -= 1, c_j += 1`) AND the smallest
 * face containing both of them (spanned by the union of their two nonzero-coordinate sets, which
 * is always just `c`'s own nonzero set plus the incoming coordinate `j`) still has dimension
 * <= `meshdim` - both endpoints individually surviving is not enough on its own, since a single
 * transfer can otherwise jump between two different low-dimension faces straight through a
 * higher-dimension one they don't share (e.g. `(1,1,1,0) -> (1,1,0,1)`, two different triangular
 * faces of a subdivided tetrahedron, would wrongly connect directly through the solid interior).
 * Each point embeds as the sum of its coordinates against `regularSimplexCoords(dim)`'s own
 * unit-edge vertex positions, which keeps every lattice edge exactly unit length.
 *
 * `meshdim = dim` (or higher) keeps everything - the full solid simplex, `dim = 2` then matching
 * `triangularBoard(w)` node-for-node. `meshdim = 0` keeps only the `dim + 1` corners (each with a
 * single nonzero coordinate equal to `w - 1`), not adjacent to each other for `w > 1`, same as
 * `hypercuboidBoard`'s own `meshdim = 0` case. `meshdim = dim - 1` is the hollow boundary - every
 * proper face glued at its shared sub-faces, solid interior excluded.
 */
export function simplexBoard(meshdim: number, dim: number, w: number): BoardConfig {
    assert(Number.isInteger(dim) && dim >= 1, `dim must be a positive integer, got ${dim}`);
    assert(Number.isInteger(w) && w >= 1, `w must be a positive integer, got ${w}`);
    assert(Number.isInteger(meshdim) && meshdim >= 0, `meshdim must be a non-negative integer, got ${meshdim}`);
    const m = dim + 1, n = w - 1;

    const allCoords: number[][] = [];
    const build = (prefix: number[], remaining: number) => {
        if (prefix.length === m - 1) { allCoords.push([...prefix, remaining]); return; }
        for (let c = 0; c <= remaining; c++) build([...prefix, c], remaining - c);
    };
    build([], n);

    const corners = regularSimplexCoords(dim);
    const boardIdxOf = new Map<string, number>();
    const survivingCoords: number[][] = [];
    const nonzeroCounts: number[] = [];
    const pos: number[][] = [];
    for (const c of allCoords) {
        const nonzeroCount = c.filter(x => x > 0).length;
        if (nonzeroCount > meshdim + 1) continue;
        boardIdxOf.set(c.join(','), survivingCoords.length);
        survivingCoords.push(c);
        nonzeroCounts.push(nonzeroCount);
        const p = new Array(dim).fill(0);
        for (let i = 0; i < m; i++)
            for (let d = 0; d < dim; d++) p[d] += c[i] * corners[i][d];
        pos.push(p);
    }
    const N = survivingCoords.length;

    const adj = zeroAdj(N);
    for (let bi = 0; bi < N; bi++) {
        const c = survivingCoords[bi];
        for (let i = 0; i < m; i++) {
            if (c[i] === 0) continue;
            for (let j = 0; j < m; j++) {
                if (j === i) continue;
                const extra = c[j] === 0 ? 1 : 0;
                if (nonzeroCounts[bi] + extra > meshdim + 1) continue;
                const nc = c.slice();
                nc[i]--; nc[j]++;
                const nbi = boardIdxOf.get(nc.join(','));
                if (nbi === undefined) continue;
                adj[bi][nbi] = 1;
            }
        }
    }
    return make(new Embedding(dim, pos), adj);
}

/**
 * A regular tetrahedron: 4 vertices, all mutually adjacent (K4), 6 unit-length edges. A
 * side-length-w subdivision of its 4 triangular faces is no longer built in here directly - apply
 * the `triangleForm(w)` modifier afterward instead (findSimplices(adj, 2) finds exactly its 4
 * faces on this board, since every 3-subset of K4's vertices is a triangle).
 */
export function tetrahedronBoard(): BoardConfig {
    return simplexBoard(3, 3, 2);
}

/**
 * A regular octahedron: the `n=3` case of `orthoplexBoard()` (see its own doc comment for the
 * general construction) - 6 vertices, 12 unit-length edges, 8 triangular faces. A side-length-w
 * subdivision of its 8 triangular faces can be applied via the `triangleForm(w)` modifier
 * afterward (findSimplices(adj, 2) finds exactly its 8 faces).
 */
export function octahedronBoard(): BoardConfig {
    return orthoplexBoard(3);
}

/**
 * The n-dimensional orthoplex (cross-polytope): 2n vertices at `+-scale` along each axis
 * (pre-scaled so edges come out exactly 1 - the raw distance between two non-antipodal vertices is
 * `sqrt(2)`). Each vertex connects to every other vertex except its own antipode (the one
 * differing only by a sign flip) - vertex `2k` and `2k+1` (the `+-scale` points on axis `k`) are
 * always antipodal pairs, by construction. `n=1` is the degenerate case of 2 isolated nodes: its
 * only other vertex is its own antipode, so nothing is left to connect it to. `n=3` is the regular
 * octahedron (see `octahedronBoard()` above).
 */
export function orthoplexBoard(n: number): BoardConfig {
    assert(n >= 1, `n must be at least 1, got ${n}`);
    const edgeScale = 1 / Math.sqrt(2);
    const pos: number[][] = [];
    for (let k = 0; k < n; k++) {
        const plus = new Array(n).fill(0), minus = new Array(n).fill(0);
        plus[k] = edgeScale;
        minus[k] = -edgeScale;
        pos.push(plus, minus);
    }

    const N = 2 * n;
    const adj = zeroAdj(N);
    const antipode = (i: number) => i % 2 === 0 ? i + 1 : i - 1;
    for (let i = 0; i < N; i++)
        for (let j = 0; j < N; j++)
            if (i !== j && j !== antipode(i)) adj[i][j] = 1;

    return make(new Embedding(n, pos), adj);
}

/**
 * The 24-cell (icositetrachoron): the regular 4-dimensional polytope with 24 vertices, 96
 * unit-length edges, 96 triangular faces and 24 octahedral cells - the unique self-dual regular
 * 4-polytope, with no 3-dimensional analog. Vertices are every point in R^4 with exactly two
 * coordinates equal to `+-1` (any of the `C(4,2) = 6` axis pairs, either of the 4 sign
 * combinations - the D4 root system, 24 points total), pre-scaled by `edgeScale = 1/sqrt(2)` so
 * adjacent vertices come out exactly unit distance apart. Two vertices are adjacent iff their raw
 * (pre-scale) dot product is 1 (raw squared distance 2, matching the unit-scaled edge length) -
 * every vertex works out to exactly 8 such neighbors (`24*8/2 = 96` edges, matching the known edge
 * count).
 */
export function reg24CellBoard(): BoardConfig {
    const edgeScale = 1 / Math.sqrt(2);
    const raw: number[][] = [];
    for (let i = 0; i < 4; i++)
        for (let j = i + 1; j < 4; j++)
            for (const si of [1, -1])
                for (const sj of [1, -1]) {
                    const v = [0, 0, 0, 0];
                    v[i] = si;
                    v[j] = sj;
                    raw.push(v);
                }

    const N = raw.length;
    const pos = raw.map(v => v.map(x => x * edgeScale));

    const adj = zeroAdj(N);
    for (let a = 0; a < N; a++)
        for (let b = a + 1; b < N; b++) {
            const dot = raw[a].reduce((s, x, k) => s + x * raw[b][k], 0);
            if (dot === 1) { adj[a][b] = 1; adj[b][a] = 1; }
        }

    return make(new Embedding(4, pos), adj);
}

/**
 * The 120-cell (hecatonicosachoron): the regular 4-dimensional polytope with 600 vertices, 1200
 * unit-length edges, 720 pentagonal faces and 120 dodecahedral cells - dual to the 600-cell. Raw
 * (pre-scale) vertex coordinates, all at distance `sqrt(8)` from the origin, come from 7 families
 * (`phi` = the golden ratio; `1/phi = phi-1`, `1/phi^2 = 2-phi`, `phi^2 = phi+1`) - the first 4 are
 * every signed arrangement of a 4-value multiset (duplicate-magnitude entries collapse a full
 * permutation set down to fewer distinct position-arrangements), the last 3 only the *even* signed
 * permutations (half of every arrangement, selected by permutation parity on distinct positions,
 * since all 4 values in those 3 families are themselves distinct):
 * - `(0, 0, 2, 2)` - 24 signed arrangements
 * - `(phi, phi, phi, 1/phi^2)` - 64
 * - `(1, 1, 1, sqrt(5))` - 64
 * - `(1/phi, 1/phi, 1/phi, phi^2)` - 64
 * - even permutations of `(0, 1/phi, phi, sqrt(5))` - 96
 * - even permutations of `(0, 1/phi^2, 1, phi^2)` - 96
 * - even permutations of `(1/phi, 1, phi, 2)` - 192
 *
 * (24+64+64+64+96+96+192 = 600, matching the known vertex count.) Two vertices are adjacent iff
 * their raw squared distance is `(3 - sqrt(5))^2` - verified numerically (not just asserted) as the
 * correct edge threshold: it's the minimum nonzero pairwise distance among these 600 points, and
 * cutting there gives exactly 1200 edges with every vertex at degree 4, matching the 120-cell's
 * known edge count and vertex figure. Final positions are scaled by `edgeScale = 1/(3-sqrt(5))` so
 * edges come out exactly unit length.
 */
export function reg120CellBoard(): BoardConfig {
    const PHI = (1 + Math.sqrt(5)) / 2;
    const PHI2 = PHI * PHI;
    const IPHI = 1 / PHI;
    const IPHI2 = 1 / (PHI * PHI);
    const SQRT5 = Math.sqrt(5);
    const edgeScale = 1 / (3 - Math.sqrt(5));

    const permsOf = (arr: number[]): number[][] => {
        if (arr.length <= 1) return [arr.slice()];
        const out: number[][] = [];
        for (let i = 0; i < arr.length; i++) {
            const rest = arr.slice(0, i).concat(arr.slice(i + 1));
            for (const p of permsOf(rest)) out.push([arr[i], ...p]);
        }
        return out;
    };
    const evenPermsOf = (arr: number[]): number[][] => {
        // Permute an index array (rather than arr's own values directly) so parity is well-defined
        // even when arr has repeated magnitudes - not needed by the 3 families this is actually
        // called with (all 4 values distinct there), but keeps the helper correct in general.
        const idxPerms = permsOf(arr.map((_, i) => i));
        return idxPerms
            .filter(p => {
                let inversions = 0;
                for (let i = 0; i < p.length; i++)
                    for (let j = i + 1; j < p.length; j++)
                        if (p[i] > p[j]) inversions++;
                return inversions % 2 === 0;
            })
            .map(p => p.map(i => arr[i]));
    };
    const signVariants = (v: number[]): number[][] => {
        const nonzero: number[] = [];
        v.forEach((x, i) => { if (x !== 0) nonzero.push(i); });
        const k = nonzero.length;
        const out: number[][] = [];
        for (let mask = 0; mask < (1 << k); mask++) {
            const w = v.slice();
            for (let b = 0; b < k; b++) if (mask & (1 << b)) w[nonzero[b]] = -w[nonzero[b]];
            out.push(w);
        }
        return out;
    };

    const raw: number[][] = [];
    const seen = new Set<string>();
    const addPerms = (perms: number[][]) => {
        for (const p of perms)
            for (const s of signVariants(p)) {
                const key = s.map(x => x.toFixed(2)).join(',');
                if (!seen.has(key)) { seen.add(key); raw.push(s); }
            }
    };
    addPerms(permsOf([0, 0, 2, 2]));
    addPerms(permsOf([PHI, PHI, PHI, IPHI2]));
    addPerms(permsOf([1, 1, 1, SQRT5]));
    addPerms(permsOf([IPHI, IPHI, IPHI, PHI2]));
    addPerms(evenPermsOf([0, IPHI, PHI, SQRT5]));
    addPerms(evenPermsOf([0, IPHI2, 1, PHI2]));
    addPerms(evenPermsOf([IPHI, 1, PHI, 2]));

    const N = raw.length;
    const pos = raw.map(v => v.map(x => x * edgeScale));

    const adj = zeroAdj(N);
    const edgeDist2 = (3 - Math.sqrt(5)) ** 2;
    const EPS = 1e-6;
    for (let a = 0; a < N; a++)
        for (let b = a + 1; b < N; b++) {
            const d2 = raw[a].reduce((s, x, k) => s + (x - raw[b][k]) ** 2, 0);
            if (Math.abs(d2 - edgeDist2) < EPS) { adj[a][b] = 1; adj[b][a] = 1; }
        }

    return make(new Embedding(4, pos), adj);
}

/**
 * The 600-cell: the regular 4-dimensional polytope with 120 vertices, 720 unit-length edges, 1200
 * triangular faces and 600 tetrahedral cells - dual to the 120-cell. Raw (pre-scale) vertex
 * coordinates, all at distance 2 from the origin, come from 3 families (`phi` = the golden ratio,
 * `1/phi = phi-1`):
 * - `(0, 0, 0, 2)` - 8 signed arrangements (the first 8+16=24 vertices, combined with the next
 *   family, form an inscribed `reg24CellBoard()`)
 * - `(1, 1, 1, 1)` - 16 signed arrangements
 * - even permutations of `(phi, 1, 1/phi, 0)` - 96 (these form an inscribed "snub 24-cell")
 *
 * (8+16+96 = 120, matching the known vertex count.) Two vertices are adjacent iff their raw squared
 * distance is `(2/phi)^2` - verified numerically (not just asserted) as the correct edge threshold:
 * it's the minimum nonzero pairwise distance among these 120 points, and cutting there gives
 * exactly 720 edges with every vertex at degree 12 (a regular icosahedron's own vertex count,
 * matching the 600-cell's icosahedral vertex figure). Final positions are scaled by
 * `edgeScale = phi/2` so edges come out exactly unit length.
 */
export function reg600CellBoard(): BoardConfig {
    const PHI = (1 + Math.sqrt(5)) / 2;
    const IPHI = 1 / PHI;
    const edgeScale = PHI / 2;

    const permsOf = (arr: number[]): number[][] => {
        if (arr.length <= 1) return [arr.slice()];
        const out: number[][] = [];
        for (let i = 0; i < arr.length; i++) {
            const rest = arr.slice(0, i).concat(arr.slice(i + 1));
            for (const p of permsOf(rest)) out.push([arr[i], ...p]);
        }
        return out;
    };
    const evenPermsOf = (arr: number[]): number[][] => {
        const idxPerms = permsOf(arr.map((_, i) => i));
        return idxPerms
            .filter(p => {
                let inversions = 0;
                for (let i = 0; i < p.length; i++)
                    for (let j = i + 1; j < p.length; j++)
                        if (p[i] > p[j]) inversions++;
                return inversions % 2 === 0;
            })
            .map(p => p.map(i => arr[i]));
    };
    const signVariants = (v: number[]): number[][] => {
        const nonzero: number[] = [];
        v.forEach((x, i) => { if (x !== 0) nonzero.push(i); });
        const k = nonzero.length;
        const out: number[][] = [];
        for (let mask = 0; mask < (1 << k); mask++) {
            const w = v.slice();
            for (let b = 0; b < k; b++) if (mask & (1 << b)) w[nonzero[b]] = -w[nonzero[b]];
            out.push(w);
        }
        return out;
    };

    const raw: number[][] = [];
    const seen = new Set<string>();
    const addPerms = (perms: number[][]) => {
        for (const p of perms)
            for (const s of signVariants(p)) {
                const key = s.map(x => x.toFixed(2)).join(',');
                if (!seen.has(key)) { seen.add(key); raw.push(s); }
            }
    };
    addPerms(permsOf([0, 0, 0, 2]));
    addPerms(permsOf([1, 1, 1, 1]));
    addPerms(evenPermsOf([PHI, 1, IPHI, 0]));

    const N = raw.length;
    const pos = raw.map(v => v.map(x => x * edgeScale));

    const adj = zeroAdj(N);
    const edgeDist2 = (2 / PHI) ** 2;
    const EPS = 1e-6;
    for (let a = 0; a < N; a++)
        for (let b = a + 1; b < N; b++) {
            const d2 = raw[a].reduce((s, x, k) => s + (x - raw[b][k]) ** 2, 0);
            if (Math.abs(d2 - edgeDist2) < EPS) { adj[a][b] = 1; adj[b][a] = 1; }
        }

    return make(new Embedding(4, pos), adj);
}

/**
 * A uniform n-gonal antiprism: 2n vertices - a "top" n-gon (`k = 0..n-1`) at height `+h`, angle
 * `2*pi*k/n`, and a "bottom" n-gon at height `-h`, angle `2*pi*k/n + pi/n` (rotated by half a step
 * relative to the top) - joined by 2n unit-length "slant" edges (top vertex `k` to bottom vertices
 * `k` and `k-1 mod n`, its two nearest bottom neighbors), forming 2n triangles around the belt, in
 * addition to the top/bottom n-gon rings themselves.
 *
 * Both `R` (the n-gon circumradius) and `h` are chosen so every edge - n-gon and slant alike - comes
 * out exactly unit length: `R = 1/(2*sin(pi/n))` makes the n-gon's own edges (a chord subtending
 * angle `2*pi/n`) unit length, the same formula and reasoning as regularPolygonBoard()'s own. A
 * slant edge's own squared length is `(2*R*sin(pi/(2n)))^2 + (2h)^2` (a chord subtending the smaller
 * angle `pi/n` between a top vertex and its nearest bottom neighbor, combined with their height
 * difference `2h`, via the standard "chord + height" distance decomposition for two points on
 * parallel circles) - setting that equal to 1 and solving for `h` gives
 * `h = 0.5*sqrt(1 - 4*R^2*sin^2(pi/(2n)))`. Verified numerically (every edge exactly unit length) for
 * n=3..12.
 *
 * `n=3` is geometrically the regular octahedron (see octahedronBoard()) - every face, including the
 * two "triangular" top/bottom rings, ends up unit-edge - though this function always keeps the two
 * n-gon rings as their own dedicated faces rather than special-casing `n=3` to match
 * orthoplexBoard()'s own vertex layout/numbering.
 */
export function antiprismBoard(n: number): BoardConfig {
    assert(n >= 3, `n must be at least 3, got ${n}`);
    const R = 1 / (2 * Math.sin(Math.PI / n));
    const h = 0.5 * Math.sqrt(1 - 4 * R * R * Math.sin(Math.PI / (2 * n)) ** 2);

    const top = (k: number) => k;
    const bot = (k: number) => n + k;
    const pos: number[][] = new Array(2 * n);
    for (let k = 0; k < n; k++) {
        const topTheta = (2 * Math.PI * k) / n;
        pos[top(k)] = [R * Math.cos(topTheta), R * Math.sin(topTheta), h];
        const botTheta = topTheta + Math.PI / n;
        pos[bot(k)] = [R * Math.cos(botTheta), R * Math.sin(botTheta), -h];
    }

    const adj = zeroAdj(2 * n);
    const connect = (i: number, j: number) => { adj[i][j] = 1; adj[j][i] = 1; };
    for (let k = 0; k < n; k++) {
        connect(top(k), top((k + 1) % n));
        connect(bot(k), bot((k + 1) % n));
        connect(top(k), bot(k));
        connect(top(k), bot((k - 1 + n) % n));
    }

    return make(new Embedding(3, pos), adj);
}

/**
 * A regular dodecahedron: 20 vertices, 12 pentagonal faces, 30 unit-length edges, centered at the
 * origin - dodecahedronFractalDescr()'s own `leafPos`/`leafConn` (see that function's own doc comment
 * for the construction), simply assembled into a BoardConfig.
 */
export function dodecahedronBoard(): BoardConfig {
    const { leafPos, leafConn } = dodecahedronFractalDescr();
    const adj = zeroAdj(leafPos.length);
    for (const [a, b] of leafConn) { adj[a][b] = 1; adj[b][a] = 1; }
    return make(new Embedding(3, leafPos), adj);
}

/**
 * The "flake" fractal generalization of a regular dodecahedron - n=1 is the plain dodecahedron itself
 * (dodecahedronFractalDescr()'s own `leafPos`/`leafConn`, unit edge length); n>1 recurses into 20
 * order-(n-1) copies, one attached at each of the 20 vertices (see nodeEdgeMergeFlakeRec()'s own doc
 * comment for the recursive construction, and dodecahedronFractalDescr()'s own doc comment for how
 * its `subDescr`/`edgeGlueMap` are derived). Unlike a simplex-style flake (sierpinskiSimplex/
 * sierpinskiRec - every pair of adjacent copies sharing a single point), copies here share a full
 * EDGE, growing with recursion depth.
 */
export function dodecahedronFlake(n: number): BoardConfig {
    assert(Number.isInteger(n) && n >= 1, `n must be a positive integer, got ${n}`);
    const built = buildFractal(n, dodecahedronFractalDescr());
    return make(new Embedding(3, built.pos), built.adj);
}

/**
 * A regular icosahedron: 12 vertices, 20 triangular faces, 30 unit-length edges, centered at the
 * origin - icosahedronFractalDescr()'s own `leafPos`/`leafConn` (see that function's own doc comment
 * for the construction), simply assembled into a BoardConfig.
 */
export function icosahedronBoard(): BoardConfig {
    const { leafPos, leafConn } = icosahedronFractalDescr();
    const adj = zeroAdj(leafPos.length);
    for (const [a, b] of leafConn) { adj[a][b] = 1; adj[b][a] = 1; }
    return make(new Embedding(3, leafPos), adj);
}

/**
 * The "flake" fractal generalization of a regular icosahedron - n=1 is the plain icosahedron itself
 * (icosahedronFractalDescr()'s own `leafPos`/`leafConn`, unit edge length); n>1 recurses into 12
 * order-(n-1) copies, one attached at each of the 12 vertices - exactly the same overall scheme as
 * dodecahedronFlake() (see nodeEdgeMergeFlakeRec()'s own doc comment for the recursive construction,
 * and icosahedronFractalDescr()'s own doc comment for how its `subDescr`/`edgeGlueMap` are derived),
 * just with 12 vertices/30 edges instead of 20 vertices/30 edges.
 */
export function icosahedronFlake(n: number): BoardConfig {
    assert(Number.isInteger(n) && n >= 1, `n must be a positive integer, got ${n}`);
    const built = buildFractal(n, icosahedronFractalDescr());
    return make(new Embedding(3, built.pos), built.adj);
}

/**
 * The "flake" fractal generalization of a regular octahedron - n=1 is the plain octahedron itself
 * (octahedronFractalDescr()'s own `leafPos`/`leafConn`, unit edge length); n>1 recurses into 6
 * order-(n-1) copies, one attached at each of the 6 vertices - the same overall scheme as
 * dodecahedronFlake()/icosahedronFlake() (see nodeEdgeMergeFlakeRec()'s own doc comment for the
 * recursive construction, and octahedronFractalDescr()'s own doc comment for how its `subDescr`/
 * `edgeGlueMap` are derived, including why the 3 antipodal non-edges need no glue entry of their own).
 */
export function octahedronFlake(n: number): BoardConfig {
    assert(Number.isInteger(n) && n >= 1, `n must be a positive integer, got ${n}`);
    const built = buildFractal(n, octahedronFractalDescr());
    return make(new Embedding(3, built.pos), built.adj);
}

/**
 * The "flake" fractal generalization of a regular polygon - order=1 is the plain nSides-gon itself
 * (regularPolygonFractalDescr()'s own `leafPos`/`leafConn`, unit edge length, same as
 * regularPolygonBoard()); order>1 recurses into nSides order-(order-1) copies, one attached at each
 * of the nSides vertices - the same overall scheme as dodecahedronFlake()/icosahedronFlake()/
 * octahedronFlake() (see nodeEdgeMergeFlakeRec()'s own doc comment for the recursive construction).
 * Unlike those three (always an edge merge, or - octahedron - always transitively equivalent to one),
 * a regular-polygon flake's base edges merge by a whole growing EDGE when nSides is a multiple of 4,
 * and by a single, non-growing NODE otherwise - see regularPolygonFractalDescr()'s own doc comment
 * for exactly which point(s) and why, and regularPolygonFractalDescr()'s own fractal.ts sibling
 * regularPolygonFlakeRC()'s own doc comment for how `r`, `c` follow from that.
 */
export function regularPolygonFlake(nSides: number, order: number): BoardConfig {
    assert(Number.isInteger(nSides) && nSides >= 3, `nSides must be an integer at least 3, got ${nSides}`);
    assert(Number.isInteger(order) && order >= 1, `order must be a positive integer, got ${order}`);
    const built = buildFractal(order, regularPolygonFractalDescr(nSides, false));
    return make(built.pos, built.adj);
}

/**
 * The "flake" fractal generalization of a regular polygon, with an extra copy sitting at the exact
 * center of every recursion level, glued to all nSides regular copies at once - see
 * regularPolygonFractalDescr()'s own doc comment for the full derivation (why the central copy's own
 * `shift` must be zero and its `scale` must be `c - r`). Otherwise identical to regularPolygonFlake()
 * (see its own doc comment). Only actually adds a central copy when `nSides` is even and greater than
 * 4 (exactly when the central copy's own derived scale comes out positive and non-degenerate - see
 * regularPolygonFractalDescr()'s own doc comment); for any other `nSides`, silently falls back to
 * plain regularPolygonFlake()'s own construction (regularPolygonFractalDescr()'s own `center`
 * argument already no-ops in that case) rather than rejecting the input.
 */
export function centralRegularPolygonFlake(nSides: number, order: number): BoardConfig {
    assert(Number.isInteger(nSides) && nSides >= 3, `nSides must be an integer at least 3, got ${nSides}`);
    assert(Number.isInteger(order) && order >= 1, `order must be a positive integer, got ${order}`);
    const built = buildFractal(order, regularPolygonFractalDescr(nSides, true));
    return make(built.pos, built.adj);
}

/**
 * The special-case pentagon flake with a central copy at every recursion level - see
 * centralPentagonFractalDescr()'s own doc comment for why pentagon needs its own dedicated
 * construction rather than centralRegularPolygonFlake()'s general even-`nSides` one (odd `nSides` has
 * no same-orientation fixed point for a central copy to sit at; pentagon's own central copy instead
 * sits at the opposite orientation, `scale = -r`). Otherwise the same overall scheme as
 * regularPolygonFlake()/centralRegularPolygonFlake() (see nodeEdgeMergeFlakeRec()'s own doc comment
 * for the recursive construction).
 */
export function centralPentagonFlake(order: number): BoardConfig {
    assert(Number.isInteger(order) && order >= 1, `order must be a positive integer, got ${order}`);
    const built = buildFractal(order, centralPentagonFractalDescr());
    return make(built.pos, built.adj);
}

/**
 * The `dim`-dimensional Menger-sponge-family "flake" fractal (dim=1 is the Cantor set, dim=2 the
 * Sierpinski carpet, dim=3 the classical Menger sponge, and so on - see mengerFractalDescr()'s own
 * doc comment for the full derivation `order`/`dim`/`indicator` feed into): `order`=1 is the plain
 * unit `dim`-cube itself; `order`>1 recurses into one order-(order-1) copy per surviving sub-cube of
 * `indicator`'s own `3^dim`-minus-removed-classes subdivision, each sharing a whole growing sub-face
 * - not just a point - with every touching copy (mengerFractalDescr()'s own "hyperface" glue
 * object). `indicator` must have exactly `dim + 1` entries (`mengerFractalDescr()`'s own
 * requirement) - `[0, 0, 1, 1]` at `dim=3` is the classical Menger sponge (the center and 6
 * face-centers removed, 12 edge-mid and 8 corner sub-cubes kept, 20 of 27 total).
 */
export function mengerSpongeFlake(order: number, dim: number, indicator: number[]): BoardConfig {
    assert(Number.isInteger(order) && order >= 1, `order must be a positive integer, got ${order}`);
    assert(Number.isInteger(dim) && dim >= 1, `dim must be a positive integer, got ${dim}`);
    assert(indicator.length === dim + 1,
        `indicator must be a length-${dim + 1} list of 0/1 entries, got [${indicator}]`);
    const built = buildFractal(order, mengerFractalDescr(dim, indicator));
    return make(new Embedding(dim, built.pos), built.adj);
}

/**
 * A triangular-lattice board arranged in a hexagon shape, with `d` layers of triangles surrounding
 * the central point (side length d+1, in hex terms) - the shape used by boards like Havannah/Y.
 * Not tiled by hexagons - see hexagonalBoard (TODO) for that. Cells are indexed by axial
 * coordinates (q, r) with max(|q|, |r|, |q+r|) <= d (hex distance from the center), laid out on the
 * same triangular lattice as `triangularBoard` (unit edge length, rowDist row spacing); each cell
 * connects to its up to six axial neighbors.
 */
export function triangularHexBoard(d: number): BoardConfig {
    assert(d >= 0, `d must be non-negative, got d=${d}`);
    const rowDist = Math.sqrt(3) / 2;
    const coords: [number, number][] = [];
    for (let q = -d; q <= d; q++)
        for (let r = Math.max(-d, -d - q); r <= Math.min(d, d - q); r++)
            coords.push([q, r]);
    const N = coords.length;
    const idx = new Map<string, number>();
    coords.forEach(([q, r], i) => idx.set(`${q},${r}`, i));
    const pos = coords.map(([q, r]) => [q + r / 2, rowDist * r]);
    const adj = zeroAdj(N);
    const dirs: [number, number][] = [[1,0],[1,-1],[0,-1],[-1,0],[-1,1],[0,1]];
    for (let i = 0; i < N; i++) {
        const [q, r] = coords[i];
        for (const [dq, dr] of dirs) {
            const ni = idx.get(`${q+dq},${r+dr}`);
            if (ni !== undefined) adj[i][ni] = 1;
        }
    }
    return make(pos, adj);
}

/**
 * A board actually tiled by regular hexagons: a central hexagonal cell surrounded by `d` further
 * layers of hexagonal cells (honeycomb topology - degree 3 in the interior, degree 2 on the
 * boundary). Built by carving the honeycomb lattice out of the same triangular lattice
 * `triangularHexBoard` uses, rather than laying out hexagons directly: a triangular lattice is
 * 3-colorable (color(q, r) = (q - r) mod 3) into three interlocking triangular sublattices, and the
 * points of any one color class are exactly the *centers* of the hexagonal faces formed by the
 * other two colors' points - so "erasing" one color class turns the triangular lattice into a
 * honeycomb lattice, with the erased points marking where each hexagonal face used to be.
 * `centers` enumerates that color-0 sublattice - itself a triangular lattice, spanned by (q, r) =
 * a*(1,1) + b*(2,-1) (both color 0, at a 60° angle, so (a, b) is a fresh axial coordinate system
 * for it) - restricted to hex-distance <= d in (a, b), i.e. the center cell plus d surrounding
 * rings of cells. Each kept center then contributes its six triangular-lattice neighbors as that
 * hexagon's six corners; two corners are connected iff they're triangular-lattice-adjacent, which
 * reproduces exactly the honeycomb edges (every triangular-lattice edge between two non-color-0
 * points borders exactly two color-0 points, i.e. two hexagonal faces) without needing a separate
 * erase/dedupe pass over edges.
 */
export function hexBoard(d: number): BoardConfig {
    assert(d >= 0, `d must be non-negative, got d=${d}`);
    const rowDist = Math.sqrt(3) / 2;
    const dirs: [number, number][] = [[1,0],[1,-1],[0,-1],[-1,0],[-1,1],[0,1]];

    const centers: [number, number][] = [];
    for (let a = -d; a <= d; a++)
        for (let b = Math.max(-d, -d - a); b <= Math.min(d, d - a); b++)
            centers.push([a + 2 * b, a - b]);

    const vertices = new Map<string, [number, number]>();
    for (const [q, r] of centers)
        for (const [dq, dr] of dirs)
            vertices.set(`${q+dq},${r+dr}`, [q + dq, r + dr]);
    const coords = [...vertices.values()];
    const N = coords.length;
    const idx = new Map<string, number>();
    coords.forEach(([q, r], i) => idx.set(`${q},${r}`, i));
    const pos = coords.map(([q, r]) => [q + r / 2, rowDist * r]);
    const adj = zeroAdj(N);
    for (let i = 0; i < N; i++) {
        const [q, r] = coords[i];
        for (const [dq, dr] of dirs) {
            const ni = idx.get(`${q+dq},${r+dr}`);
            if (ni !== undefined) adj[i][ni] = 1;
        }
    }
    return make(pos, adj);
}

/**
 * A trihexagonal ("hexdel") tiling: hexagons and triangles alternate, 2 of each around every
 * vertex (degree 4 in the interior) - `d` layers of hexagons, connected by triangles, surrounding
 * a central hexagon. Built the same way as `hexBoard` - as the triangular lattice with certain
 * nodes removed - but with a coarser removed sublattice: instead of `hexBoard`'s 1-of-3 coloring
 * (which erases every triangular face along with 2/3 of the nodes, leaving pure honeycomb), this
 * removes the 1-of-4 sublattice where both axial coordinates are even (`centers`, spanned by
 * (2, 0)/(0, 2) - double the original lattice spacing - restricted to hex-distance <= d in its own
 * halved (a, b) = (q/2, r/2) coordinates). Erasing only 1/4 of the nodes leaves each surviving node
 * with 4 of its 6 original neighbors (2 got erased), which works out to exactly 2 hexagon-bordering
 * and 2 triangle-bordering edges per node - so, just like `hexBoard`, simply connecting
 * triangular-lattice-adjacent survivors reproduces both the hexagons (surrounding each erased node)
 * and the triangles (the untouched elementary triangles of the original lattice) with no separate
 * pass needed for either.
 */
export function trihexBoard(d: number): BoardConfig {
    assert(d >= 0, `d must be non-negative, got d=${d}`);
    const rowDist = Math.sqrt(3) / 2;
    const dirs: [number, number][] = [[1,0],[1,-1],[0,-1],[-1,0],[-1,1],[0,1]];

    const centers: [number, number][] = [];
    for (let a = -d; a <= d; a++)
        for (let b = Math.max(-d, -d - a); b <= Math.min(d, d - a); b++)
            centers.push([2 * a, 2 * b]);

    const vertices = new Map<string, [number, number]>();
    for (const [q, r] of centers)
        for (const [dq, dr] of dirs)
            vertices.set(`${q+dq},${r+dr}`, [q + dq, r + dr]);
    const coords = [...vertices.values()];
    const N = coords.length;
    const idx = new Map<string, number>();
    coords.forEach(([q, r], i) => idx.set(`${q},${r}`, i));
    const pos = coords.map(([q, r]) => [q + r / 2, rowDist * r]);
    const adj = zeroAdj(N);
    for (let i = 0; i < N; i++) {
        const [q, r] = coords[i];
        for (const [dq, dr] of dirs) {
            const ni = idx.get(`${q+dq},${r+dr}`);
            if (ni !== undefined) adj[i][ni] = 1;
        }
    }
    return make(pos, adj);
}

/**
 * A `w × h` grid of `g × g` squares, each rotated 45° and internally connected as an ordinary
 * grid, but with NO connections between squares yet - `gap` is the extra distance (beyond the
 * squares' own natural closest-corner distance) separating adjacent squares. Returns `pos`/`adj`
 * for the disconnected squares plus `interConn`: for every pair of horizontally/vertically
 * adjacent squares, the pair of their own closest corner node indices, left for a caller to glue
 * or connect as it sees fit (see `glueTwistedSquareBoard`/`twistedSquareBoard`, gap=0/1 respectively).
 */
function tiltedDisconnectedSquareBoard(w: number, h: number, g: number, gap: number) {
    const rm = Math.SQRT2 / 2;
    const sqWidth = (g - 1) * Math.SQRT2 + gap;
    const pos: number[][] = [];
    for (let rb = 0; rb < h; rb++)
        for (let cb = 0; cb < w; cb++) {
            const bx = (cb - (w-1)/2) * sqWidth;
            const by = (rb - (h-1)/2) * sqWidth;
            for (let r = 0; r < g; r++)
                for (let c = 0; c < g; c++) {
                    const lx = c - (g-1)/2, ly = r - (g-1)/2;
                    pos.push([bx + rm*lx - rm*ly, by + rm*lx + rm*ly]);
                }
        }
    const N = w * h * g * g;
    const adj = zeroAdj(N);
    const bIdx = (rb: number, cb: number) => (rb * w + cb) * g * g;
    for (let rb = 0; rb < h; rb++)
        for (let cb = 0; cb < w; cb++) {
            const b = bIdx(rb, cb);
            // Edges within the squares
        for (let r = 0; r < g; r++)
                for (let c = 0; c < g; c++)
                    for (const [dr, dc] of [[0,1],[1,0],[0,-1],[-1,0]]) {
                        const nr = r+dr, nc = c+dc;
                        if (nr<0||nr>=g||nc<0||nc>=g) continue;
                        adj[b+r*g+c][b+nr*g+nc] = 1;
                    }
        }
    // Connections between the squares
    const interConn: [number, number][] = [];
    for (let rb = 0; rb < h; rb++)
        for (let cb = 0; cb < w; cb++) {
            const b = bIdx(rb, cb);
            for (const [dr, dc] of [[0,1],[1,0],[0,-1],[-1,0]]) {
                const nrb = rb+dr, ncb = cb+dc;
                if (nrb<0||nrb>=h||ncb<0||ncb>=w) continue;
                const nb = bIdx(nrb, ncb);
                const selfIdx  = ((dr - dc + 1) >> 1) * g * (g-1) + ((dr + dc + 1) >> 1) * (g-1);
                const otherIdx = g*g - 1 - selfIdx;
                interConn.push([b + selfIdx, nb + otherIdx]);
            }
        }
    return { pos, adj, interConn, N };
}

/**
 * A board of `w × h` squares each rotated 45°, arranged in a rectangle. The squares have
 * the usual square topology. The closest nodes of two adjacent squares are glued together.
 */
export function glueTwistedSquareBoard(w: number, h: number, g: number): BoardConfig {
    assert(w > 0 && h > 0 && g > 0, `w, h, and g must be positive, got w=${w} h=${h} g=${g}`);
    const { pos, adj, interConn } = tiltedDisconnectedSquareBoard(w, h, g, 0.0);
    const bc = make(pos, adj);
    return quotientBoard(bc, interConn);
}

/**
 * A board of `w × h` squares each rotated 45°, arranged in a rectangle. The squares have
 * the usual square topology. The closest nodes of two adjacent squares are connected.
 */
export function twistedSquareBoard(w: number, h: number, g: number): BoardConfig {
    assert(w > 0 && h > 0 && g > 0, `w, h, and g must be positive, got w=${w} h=${h} g=${g}`);
    const { pos, adj, interConn } = tiltedDisconnectedSquareBoard(w, h, g, 1.0);
    for (const [i, j] of interConn) { adj[i][j] = 1; adj[j][i] = 1; }
    return make(pos, adj);
}

/**
 * A `w × h` grid of unit squares, each rotated ±30° in a checkerboard pattern, arranged as a snub
 * square tiling. Each pair of adjacent squares is connected two ways: their nearest corners are
 * glued into a single shared node, and their next-nearest corners are joined by one new edge - that
 * new edge, together with each square's own two boundary edges reaching its own glued/joined
 * corners, closes into a genuine 3-node triangular gap face (see topology.ts's findSimplices(adj, 2)) the
 * same way each square's own 4 corners already form a genuine quad face (findQuads).
 */
export function snubSquareBoard(w: number, h: number): BoardConfig {
    assert(w > 0 && h > 0, `w and h must be positive, got w=${w} h=${h}`);

    const spacing = 0.5 + Math.sqrt(3) / 2;
    const pos: number[][] = [];
    for (let rb = 0; rb < h; rb++)
        for (let cb = 0; cb < w; cb++) {
            const bx = (cb - (w-1)/2) * spacing;
            const by = (rb - (h-1)/2) * spacing;
            const angle = ((rb + cb) % 2 === 0 ? -1 : 1) * Math.PI / 6;
            const ca = Math.cos(angle), sa = Math.sin(angle);
            for (const [r, c] of [[0,0],[0,1],[1,0],[1,1]] as [number, number][]) {
                const lx = c - 0.5, ly = r - 0.5;
                pos.push([bx + ca*lx - sa*ly, by + sa*lx + ca*ly]);
            }
        }
    const N = w * h * 4;
    const adj = zeroAdj(N);
    const bIdx = (rb: number, cb: number) => (rb * w + cb) * 4;
    // NW/NE/SW/SE index into each cell's own 4 corners, in the same order pos was built above.
    const cornerIdx: Record<'NW'|'NE'|'SW'|'SE', number> = { NW: 0, NE: 1, SW: 2, SE: 3 };

    // Each cell's own 4-cycle (a genuine quad face - no diagonal edges).
    const sides: ['NW'|'NE'|'SW'|'SE', 'NW'|'NE'|'SW'|'SE'][] = [['NW','NE'], ['NW','SW'], ['NE','SE'], ['SW','SE']];
    for (let rb = 0; rb < h; rb++)
        for (let cb = 0; cb < w; cb++) {
            const b = bIdx(rb, cb);
            for (const [a, c] of sides) {
                adj[b+cornerIdx[a]][b+cornerIdx[c]] = 1;
                adj[b+cornerIdx[c]][b+cornerIdx[a]] = 1;
            }
        }

    // Inter-cell connections, keyed by self cell's checkerboard parity then by "dr,dc" (only forward
    // directions, so each neighboring pair of cells is handled once): glue is the single coincident
    // corner pair (merged via quotientBoard below); tri is the one new same-distance edge that closes
    // the triangular gap face (see this function's own doc comment above).
    type Corner = 'NW'|'NE'|'SW'|'SE';
    const CONN: Record<number, Record<string, { glue: [Corner,Corner], tri: [Corner,Corner] }>> = {
        0: {
            '0,1':  { glue: ['SE','SW'], tri: ['NE', 'NW'] },
            '1,0':  { glue: ['SW','NW'], tri: ['SE', 'NE'] },
        },
        1: {
            '0,1':  { glue: ['NE','NW'], tri: ['SE', 'SW'] },
            '1,0':  { glue: ['SE','NE'], tri: ['SW', 'NW'] },
        },
    };
    const interConn: [number, number][] = [];
    for (let rb = 0; rb < h; rb++)
        for (let cb = 0; cb < w; cb++) {
            const b = bIdx(rb, cb);
            const entry = CONN[(rb + cb) % 2];
            for (const [dr, dc] of [[0,1],[1,0]] as [number, number][]) {
                const nrb = rb+dr, ncb = cb+dc;
                if (nrb<0||nrb>=h||ncb<0||ncb>=w) continue;
                const nb = bIdx(nrb, ncb);
                const conn = entry[`${dr},${dc}`];
                interConn.push([b + cornerIdx[conn.glue[0]], nb + cornerIdx[conn.glue[1]]]);
                const i = b + cornerIdx[conn.tri[0]], j = nb + cornerIdx[conn.tri[1]];
                adj[i][j] = 1; adj[j][i] = 1;
            }
        }

    const bc = make(pos, adj);
    return quotientBoard(bc, interConn);
}


export enum PrescribedBoard {
    linearBoard,
    rectangularBoard,
    rectangularDiagonalBoard,
    cubeLatticeBoard,
    hypercuboidBoard,
    triangularBoard,
    regularPolygonBoard,
    tetrahedronBoard,
    dodecahedronBoard,
    icosahedronBoard,
    triangularHexBoard,
    hexBoard,
    trihexBoard,
    snubSquareBoard,
    twistedSquareBoard,
    glueTwistedSquareBoard,
    starBoard,
    octahedronBoard,
    sierpinskiSimplex,
    simplexBoard,
    orthoplexBoard,
    reg24CellBoard,
    reg120CellBoard,
    reg600CellBoard,
    dodecahedronFlake,
    icosahedronFlake,
    octahedronFlake,
    regularPolygonFlake,
    centralRegularPolygonFlake,
    centralPentagonFlake,
    mengerSpongeFlake,
    antiprismBoard
}

// k Number-typed args in a row - shorthand for PrescribedBoardMap's common case below (every board
// type except hypercuboidBoard, whose own trailing arg is CommaSeparatedNumbers - see its entry).
const nums = (k: number): BoardArgType[] => new Array(k).fill(BoardArgType.Number);

// Tuple shape: [argTypes, cleg name (the actual BUILTIN_FUNCTIONS key - includes cleg.ts's own "B"
// suffix already, so no name-mangling is needed at the registration site there), a human-readable
// bracketed argument list (already parenthesized, exactly as it'd read in a real cleg call - no
// further processing needed to display it, e.g. "(w, h)"), description]. Used directly by
// src/renderer.ts's "Prescribed Boards" command-reference table (no separate lookup structure of its
// own), and by cleg.ts's board-constructor BUILTIN_FUNCTIONS registration loop.
export const PrescribedBoardMap: Record<PrescribedBoard, [BoardArgType[], string, string, string]> = {
    [PrescribedBoard.linearBoard]:
        [nums(1), "lineB", "(w)", "A simple line of w nodes"],
    [PrescribedBoard.rectangularBoard]:
        [nums(2), "rectB", "(w, h)", "Rectangular board"],
    [PrescribedBoard.rectangularDiagonalBoard]:
        [nums(3), "rectdB", "(w, h, m)", "Rectangular + diagonal connections every m squares"],
    [PrescribedBoard.cubeLatticeBoard]:
        [nums(3), "cublatB", "(w, h, d)", "Cubical board"],
    [PrescribedBoard.hypercuboidBoard]:
        [[BoardArgType.Number, BoardArgType.CommaSeparatedNumbers], "hcubB", "(meshdim, [w, h, ...])",
            "Hypercuboidal board (meshdim-skeleton: max interior coords a surviving node may have, "
            + "then a comma-separated list of dimension sizes)"],
    [PrescribedBoard.triangularBoard]:
        [nums(1), "triB", "(w)", "Triangular board of side w"],
    [PrescribedBoard.regularPolygonBoard]:
        [nums(1), "regpolyB", "(n)", "Regular polygon with n unit-length edges"],
    [PrescribedBoard.tetrahedronBoard]:
        [nums(0), "tetraB", "()", "Regular tetrahedron (4 vertices, all mutually adjacent, unit-length edges)"],
    [PrescribedBoard.dodecahedronBoard]:
        [nums(0), "dodecaB", "()", "Regular dodecahedron (20 vertices, 12 pentagonal faces, unit-length edges)"],
    [PrescribedBoard.icosahedronBoard]:
        [nums(0), "icosaB", "()", "Regular icosahedron (12 vertices, 20 triangular faces, unit-length edges)"],
    [PrescribedBoard.triangularHexBoard]:
        [nums(1), "trihexB", "(d)",
            "Triangular-lattice board in a hexagon shape, with d layers of triangles around the center"],
    [PrescribedBoard.hexBoard]:
        [nums(1), "hexB", "(d)", "Hexagon-tiled board with d layers of hexagons around a center hexagon"],
    [PrescribedBoard.trihexBoard]:
        [nums(1), "hexdelB", "(d)",
            "Trihexagonal (hexdel) board, d layers of hexagons connected by triangles around a center hexagon"],
    [PrescribedBoard.snubSquareBoard]:
        [nums(2), "snubsqB", "(w, h)", "Snub square board"],
    [PrescribedBoard.twistedSquareBoard]:
        [nums(3), "twsqB", "(w, h, g)", "Twisted-square board (g\xD7g squares)"],
    [PrescribedBoard.glueTwistedSquareBoard]:
        [nums(3), "gtsqB", "(w, h, g)", "Glued-twisted-square board (g\xD7g squares)"],
    [PrescribedBoard.starBoard]:
        [nums(1), "starB", "(n)", "Star graph: 1 center node connected to n outer nodes"],
    [PrescribedBoard.octahedronBoard]:
        [nums(0), "octaB", "()", "Regular octahedron (6 vertices, 8 triangular faces, unit-length edges)"],
    [PrescribedBoard.sierpinskiSimplex]:
        [nums(2), "sierB", "(dim, n)", "Sierpinski dim-simplex (gasket) of order n"],
    [PrescribedBoard.simplexBoard]:
        [nums(3), "simplexB", "(meshdim, dim, w)",
            "Simplex board (meshdim-skeleton of a regular dim-simplex, side length w)"],
    [PrescribedBoard.orthoplexBoard]:
        [nums(1), "orthoB", "(n)", "n-dimensional orthoplex (cross-polytope), unit-length edges"],
    [PrescribedBoard.reg24CellBoard]:
        [nums(0), "reg24CellB", "()",
            "Regular 24-cell (24 vertices, 96 triangular faces, unit-length edges)"],
    [PrescribedBoard.reg120CellBoard]:
        [nums(0), "reg120CellB", "()",
            "Regular 120-cell (600 vertices, 720 pentagonal faces, unit-length edges)"],
    [PrescribedBoard.reg600CellBoard]:
        [nums(0), "reg600CellB", "()",
            "Regular 600-cell (120 vertices, 1200 triangular faces, unit-length edges)"],
    [PrescribedBoard.dodecahedronFlake]:
        [nums(1), "dodflakeB", "(n)", "Dodecahedron flake fractal of order n (n=1 is the plain dodecahedron)"],
    [PrescribedBoard.icosahedronFlake]:
        [nums(1), "icoflakeB", "(n)", "Icosahedron flake fractal of order n (n=1 is the plain icosahedron)"],
    [PrescribedBoard.octahedronFlake]:
        [nums(1), "octaflakeB", "(n)", "Octahedron flake fractal of order n (n=1 is the plain octahedron)"],
    [PrescribedBoard.regularPolygonFlake]:
        [nums(2), "polyflakeB", "(sides, n)",
            "Regular polygon flake fractal of order n (n=1 is the plain sides-gon)"],
    [PrescribedBoard.centralRegularPolygonFlake]:
        [nums(2), "cpolyflakeB", "(sides, n)",
            "Regular polygon flake with a central copy at every level (sides must be even, &gt; 4)"],
    [PrescribedBoard.centralPentagonFlake]:
        [nums(1), "cpentflakeB", "(n)",
            "Pentagon flake with an opposite-orientation central copy at every level (n=1 is the plain pentagon)"],
    [PrescribedBoard.mengerSpongeFlake]:
        [[BoardArgType.Number, BoardArgType.Number, BoardArgType.ZeroOneList], "mengerB",
            '(order, dim, "indicator")',
            "Menger-sponge-family flake fractal (dim=1 Cantor set, dim=2 Sierpinski carpet, dim=3 Menger "
            + "sponge, ...) of the given order; indicator is a dim+1-length 0/1 string, e.g. \"0011\" at "
            + "dim=3 for the classical Menger sponge"],
    [PrescribedBoard.antiprismBoard]:
        [nums(1), "apB", "(n)", "Uniform n-gonal antiprism (2 n-gons + 2n triangles), unit-length edges"],
};

// Shorthand for PrescribedBoardFns below: `num`/`list` pull a positional BoardArgEntry's own
// number/number[] back out (see boardArgNumber()/boardArgList()'s own doc comments) - every entry
// here reads exactly one BoardArgEntry per positional arg, never a flattened/resliced array.
const num = boardArgNumber, list = boardArgList;

export const PrescribedBoardFns: Record<PrescribedBoard, (...args: BoardArgEntry[]) => BoardConfig> = {
    [PrescribedBoard.linearBoard]:               (...a) => linearBoard(num(a[0])),
    [PrescribedBoard.rectangularBoard]:         (...a) => rectangularBoard(num(a[0]), num(a[1])),
    [PrescribedBoard.rectangularDiagonalBoard]: (...a) => rectangularDiagonalBoard(num(a[0]), num(a[1]), num(a[2])),
    [PrescribedBoard.cubeLatticeBoard]:         (...a) => cubeLatticeBoard(num(a[0]), num(a[1]), num(a[2])),
    [PrescribedBoard.hypercuboidBoard]:         (...a) => hypercuboidBoard(num(a[0]), list(a[1])),
    [PrescribedBoard.triangularBoard]:          (...a) => triangularBoard(num(a[0])),
    [PrescribedBoard.regularPolygonBoard]:      (...a) => regularPolygonBoard(num(a[0])),
    [PrescribedBoard.tetrahedronBoard]:         () => tetrahedronBoard(),
    [PrescribedBoard.dodecahedronBoard]:        () => dodecahedronBoard(),
    [PrescribedBoard.icosahedronBoard]:         () => icosahedronBoard(),
    [PrescribedBoard.triangularHexBoard]:       (...a) => triangularHexBoard(num(a[0])),
    [PrescribedBoard.hexBoard]:                 (...a) => hexBoard(num(a[0])),
    [PrescribedBoard.trihexBoard]:               (...a) => trihexBoard(num(a[0])),
    [PrescribedBoard.snubSquareBoard]:          (...a) => snubSquareBoard(num(a[0]), num(a[1])),
    [PrescribedBoard.twistedSquareBoard]:       (...a) => twistedSquareBoard(num(a[0]), num(a[1]), num(a[2])),
    [PrescribedBoard.glueTwistedSquareBoard]:   (...a) => glueTwistedSquareBoard(num(a[0]), num(a[1]), num(a[2])),
    [PrescribedBoard.starBoard]:                 (...a) => starBoard(num(a[0])),
    [PrescribedBoard.octahedronBoard]:          () => octahedronBoard(),
    [PrescribedBoard.sierpinskiSimplex]:        (...a) => sierpinskiSimplex(num(a[0]), num(a[1])),
    [PrescribedBoard.simplexBoard]:             (...a) => simplexBoard(num(a[0]), num(a[1]), num(a[2])),
    [PrescribedBoard.orthoplexBoard]:           (...a) => orthoplexBoard(num(a[0])),
    [PrescribedBoard.reg24CellBoard]:           () => reg24CellBoard(),
    [PrescribedBoard.reg120CellBoard]:          () => reg120CellBoard(),
    [PrescribedBoard.reg600CellBoard]:          () => reg600CellBoard(),
    [PrescribedBoard.dodecahedronFlake]:        (...a) => dodecahedronFlake(num(a[0])),
    [PrescribedBoard.icosahedronFlake]:         (...a) => icosahedronFlake(num(a[0])),
    [PrescribedBoard.octahedronFlake]:          (...a) => octahedronFlake(num(a[0])),
    [PrescribedBoard.regularPolygonFlake]:      (...a) => regularPolygonFlake(num(a[0]), num(a[1])),
    [PrescribedBoard.centralRegularPolygonFlake]: (...a) => centralRegularPolygonFlake(num(a[0]), num(a[1])),
    [PrescribedBoard.centralPentagonFlake]:     (...a) => centralPentagonFlake(num(a[0])),
    [PrescribedBoard.mengerSpongeFlake]:        (...a) => mengerSpongeFlake(num(a[0]), num(a[1]), list(a[2])),
    [PrescribedBoard.antiprismBoard]:           (...a) => antiprismBoard(num(a[0])),
};

export function applyModifier(bc: BoardConfig, modifier: BoardModifier): BoardConfig {
    switch (modifier.kind) {
        case 'Rectify': return rectify(bc);
        case 'Truncate': return truncate(bc);
        case 'EdgeSplit': return edgeSplit(bc, modifier.splitN);
        case 'MergeClose': return mergeClose(bc, modifier.dist);
        case 'TriangleForm': return triangleForm(bc, modifier.w, modifier.sel);
        case 'QuadForm': return quadForm(bc, modifier.w, modifier.sel);
        case 'Form': return genericForm(bc, modifier.w, modifier.sels);
        case 'LocalReplace': return genericLocalReplace(bc, modifier.selectors);
        case 'GlobalCentralize': return globalCentralize(bc);
        case 'Scale': return scaleBoard(bc, modifier.factor);
        case 'NodeInducedSubgraph': return nodeInducedSubgraph(bc, selectNode(bc.adj, bc.emb.pos, modifier.sel));
        case 'EdgeInducedSubgraph': return edgeInducedSubgraph(bc, selectEdge(bc.adj, bc.emb.pos, modifier.sel));
    }
}

/** Applies every modifier in `modifiers`, in order, to `bc`. */
export function applyModifiers(bc: BoardConfig, modifiers: BoardModifier[]): BoardConfig {
    return modifiers.reduce((current, m) => applyModifier(current, m), bc);
}