import type { GameConfig } from './types.js';
import { convexHullEdges } from './geometry.js';
import { findTriangles, findSquares } from './topology.js';

/**
 * A board's node positions in their natural embedding dimension (embDim - 2 for most boards, 3 for
 * cubeLatticeBoard, 4 for hypercubeBoard), plus the linear map (projMat, 2 x embDim) that projects them
 * down to a 2D render position. Kept separate from the 2D render position so that geometric
 * operations that care about real dimensionality (e.g. a convex-hull-based rectify()) can operate on
 * `pos` directly instead of an already-flattened 2D approximation.
 */
/** Applies a 2 x embDim projMat to a single embDim-length point, returning its 2D projection. */
export function projectPoint(projMat: number[][], p: number[]): number[] {
    return [
        p.reduce((s, v, k) => s + projMat[0][k] * v, 0),
        p.reduce((s, v, k) => s + projMat[1][k] * v, 0),
    ];
}

export class Embedding {
    embDim: number;
    pos: number[][];       // N x embDim
    projMat: number[][];   // 2 x embDim - projects natural coords to a 2D render position

    constructor(embDim: number, pos: number[][], projMat: number[][]) {
        assert(pos.every(p => p.length === embDim), 'Embedding: pos row length must equal embDim');
        assert(projMat.length === 2 && projMat.every(r => r.length === embDim),
            'Embedding: projMat must be 2 x embDim');
        this.embDim = embDim;
        this.pos = pos;
        this.projMat = projMat;
    }

    /** The 2D render position: projMat applied to each row of pos. */
    project(): number[][] {
        return this.pos.map(p => projectPoint(this.projMat, p));
    }
}

export interface BoardConfig {
    emb: Embedding;    // natural-dimension node positions + their 2D projection
    adj: number[][];  // N×N symmetric adjacency matrix, entries 0/1
    N: number;
}

function assert(cond: boolean, msg: string): asserts cond {
    if (!cond) throw new Error(`Assertion failed: ${msg}`);
}

const IDENTITY_2X2 = [[1, 0], [0, 1]];

function make(posOrEmb: number[][] | Embedding, adj: number[][]): BoardConfig {
    const emb = Array.isArray(posOrEmb) ? new Embedding(2, posOrEmb, IDENTITY_2X2) : posOrEmb;
    const N = emb.pos.length;
    assert(adj.length === N && (N === 0 || adj[0].length === N), 'adj dimensions must match pos length');
    for (let i = 0; i < N; i++)
        for (let j = 0; j < N; j++)
            assert(adj[i][j] === adj[j][i], `adj must be symmetric: [${i}][${j}]`);
    return { emb, adj, N };
}

function zeroAdj(N: number): number[][] {
    return Array.from({ length: N }, () => new Array<number>(N).fill(0));
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

    // New positions: average of class members (in the natural embedding dimension - projMat is
    // linear, so its projected 2D average equals the average of the already-projected positions).
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
    return make(new Embedding(embDim, newPos, bc.emb.projMat), newAdj);
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
    return make(new Embedding(embDim, pos, bc.emb.projMat), adj);
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

    return make(new Embedding(embDim, pos, bc.emb.projMat), adj);
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
 * Replaces every triangle (3 mutually-adjacent, distinct vertices - see topology.ts's
 * findTriangles) in `bc` with a `triangularBoard(w)`-shaped lattice.
 */
export function triangleForm(bc: BoardConfig, w: number): BoardConfig {
    assert(w >= 1, `w must be at least 1, got ${w}`);
    const N = bc.N;
    const embDim = bc.emb.embDim;
    const scale = Math.max(w - 1, 1);
    const scaledPos = bc.emb.pos.map(p => p.map(v => v * scale));

    const triangles = findTriangles(bc.adj); // each [A, B, C] with A < B < C
    const nFace = w * (w + 1) / 2;
    const localIdx = (i: number, j: number) => i * (i + 1) / 2 + j;
    const globalIdx = (t: number, i: number, j: number) => N + t * nFace + localIdx(i, j);

    const isTriangleSide = new Set<string>(); // "p,q" (p < q)
    for (const [A, B, C] of triangles) {
        isTriangleSide.add(`${A},${B}`);
        isTriangleSide.add(`${A},${C}`);
        isTriangleSide.add(`${B},${C}`);
    }

    const totalN = N + triangles.length * nFace;
    const pos: number[][] = new Array(totalN);
    for (let i = 0; i < N; i++) pos[i] = scaledPos[i];

    const adj = zeroAdj(totalN);
    for (let i = 0; i < N; i++)
        for (let j = i + 1; j < N; j++) {
            if (!bc.adj[i][j] || isTriangleSide.has(`${i},${j}`)) continue;
            adj[i][j] = 1;
            adj[j][i] = 1;
        }

    const dirs: [number, number][] = [[1,0],[1,1],[0,1],[-1,0],[-1,-1],[0,-1]];
    for (let t = 0; t < triangles.length; t++) {
        const [A, B, C] = triangles[t];
        const cornerA = scaledPos[A], cornerB = scaledPos[B], cornerC = scaledPos[C];
        for (let i = 0; i < w; i++)
            for (let j = 0; j <= i; j++) {
                const a = w - 1 - i, b = i - j, c = j;
                pos[globalIdx(t, i, j)] = w === 1
                    ? cornerA.map((_, k) => (cornerA[k] + cornerB[k] + cornerC[k]) / 3)
                    : cornerA.map((_, k) => (a * cornerA[k] + b * cornerB[k] + c * cornerC[k]) / (w - 1));
            }
        for (let i = 0; i < w; i++)
            for (let j = 0; j <= i; j++)
                for (const [di, dj] of dirs) {
                    const ni = i + di, nj = j + dj;
                    if (ni < 0 || ni >= w || nj < 0 || nj > ni) continue;
                    adj[globalIdx(t, i, j)][globalIdx(t, ni, nj)] = 1;
                }
    }

    // Boundary node sequence (as (i,j) pairs) for triangle t's edge between vertex-pair (p, q) -
    // same left/right/bottom convention as triangularBoard's own row/col indexing.
    function boundarySeq(t: number, p: number, q: number): [number, number][] {
        const [A, B, C] = triangles[t];
        if (p === A && q === B) return Array.from({ length: w }, (_, i): [number, number] => [i, 0]);
        if (p === A && q === C) return Array.from({ length: w }, (_, i): [number, number] => [i, i]);
        if (p === B && q === C) return Array.from({ length: w }, (_, j): [number, number] => [w - 1, j]);
        throw new Error(`triangleForm: triangle ${t} does not have edge (${p},${q})`);
    }

    const quot: [number, number][] = [];
    for (let t = 0; t < triangles.length; t++) {
        const [A, B, C] = triangles[t];
        quot.push([A, globalIdx(t, 0, 0)], [B, globalIdx(t, w - 1, 0)], [C, globalIdx(t, w - 1, w - 1)]);
    }
    const edgeToTriangles = new Map<string, number[]>();
    for (let t = 0; t < triangles.length; t++) {
        const [A, B, C] = triangles[t];
        for (const [p, q] of [[A, B], [A, C], [B, C]] as [number, number][]) {
            const key = `${p},${q}`;
            if (!edgeToTriangles.has(key)) edgeToTriangles.set(key, []);
            edgeToTriangles.get(key)!.push(t);
        }
    }
    for (const [key, ts] of edgeToTriangles) {
        if (ts.length < 2) continue;
        const [p, q] = key.split(',').map(Number);
        const canonical = boundarySeq(ts[0], p, q);
        for (let k = 1; k < ts.length; k++) {
            const seq = boundarySeq(ts[k], p, q);
            for (let idx = 0; idx < w; idx++)
                quot.push([globalIdx(ts[0], canonical[idx][0], canonical[idx][1]),
                    globalIdx(ts[k], seq[idx][0], seq[idx][1])]);
        }
    }

    const combined = make(new Embedding(embDim, pos, bc.emb.projMat), adj);
    return quotientBoard(combined, quot);
}

/**
 * Replaces every square (4 distinct vertices forming a cycle with no diagonal edges - see
 * topology.ts's findSquares) in `bc` with a `w`-by-`w` grid, the same way `triangleForm` replaces
 * triangles with `triangularBoard(w)`-shaped lattices.
 */
export function squareForm(bc: BoardConfig, w: number): BoardConfig {
    assert(w >= 1, `w must be at least 1, got ${w}`);
    const N = bc.N;
    const embDim = bc.emb.embDim;
    const scale = Math.max(w - 1, 1);
    const scaledPos = bc.emb.pos.map(p => p.map(v => v * scale));

    const squares = findSquares(bc.adj); // each [A, B, C, D] in cycle order
    const nFace = w * w;
    const localIdx = (i: number, j: number) => i * w + j;
    const globalIdx = (t: number, i: number, j: number) => N + t * nFace + localIdx(i, j);

    const isSquareSide = new Set<string>(); // "p,q" (p < q)
    for (const [A, B, C, D] of squares)
        for (const [p, q] of [[A, B], [B, C], [C, D], [D, A]] as [number, number][])
            isSquareSide.add(`${Math.min(p, q)},${Math.max(p, q)}`);

    const totalN = N + squares.length * nFace;
    const pos: number[][] = new Array(totalN);
    for (let i = 0; i < N; i++) pos[i] = scaledPos[i];

    const adj = zeroAdj(totalN);
    for (let i = 0; i < N; i++)
        for (let j = i + 1; j < N; j++) {
            if (!bc.adj[i][j] || isSquareSide.has(`${i},${j}`)) continue;
            adj[i][j] = 1;
            adj[j][i] = 1;
        }

    const dirs: [number, number][] = [[0, 1], [1, 0], [0, -1], [-1, 0]];
    for (let t = 0; t < squares.length; t++) {
        const [A, B, C, D] = squares[t];
        const cornerA = scaledPos[A], cornerB = scaledPos[B], cornerC = scaledPos[C], cornerD = scaledPos[D];
        const denom = (w - 1) * (w - 1);
        for (let i = 0; i < w; i++)
            for (let j = 0; j < w; j++) {
                const wA = (w - 1 - i) * (w - 1 - j), wB = (w - 1 - i) * j;
                const wC = i * j, wD = i * (w - 1 - j);
                pos[globalIdx(t, i, j)] = w === 1
                    ? cornerA.map((_, k) => (cornerA[k] + cornerB[k] + cornerC[k] + cornerD[k]) / 4)
                    : cornerA.map((_, k) =>
                        (wA * cornerA[k] + wB * cornerB[k] + wC * cornerC[k] + wD * cornerD[k]) / denom);
            }
        for (let i = 0; i < w; i++)
            for (let j = 0; j < w; j++)
                for (const [di, dj] of dirs) {
                    const ni = i + di, nj = j + dj;
                    if (ni < 0 || ni >= w || nj < 0 || nj >= w) continue;
                    adj[globalIdx(t, i, j)][globalIdx(t, ni, nj)] = 1;
                }
    }

    // The "natural" boundary sequence (local (i,j) pairs, k=0..w-1) for square t's side `side`
    // (0=A-B top row, 1=B-C right col, 2=C-D bottom row, 3=D-A left col), running from that side's
    // first-listed corner (k=0) to its second (k=w-1), matching the block's own A/B/C/D corner
    // assignment: (0,0)=A, (0,w-1)=B, (w-1,w-1)=C, (w-1,0)=D.
    function naturalSeq(side: 0 | 1 | 2 | 3): [number, number][] {
        if (side === 0) return Array.from({ length: w }, (_, j): [number, number] => [0, j]);
        if (side === 1) return Array.from({ length: w }, (_, i): [number, number] => [i, w - 1]);
        if (side === 2) return Array.from({ length: w }, (_, k): [number, number] => [w - 1, w - 1 - k]);
        return Array.from({ length: w }, (_, k): [number, number] => [w - 1 - k, 0]);
    }

    const quot: [number, number][] = [];
    for (let t = 0; t < squares.length; t++) {
        const [A, B, C, D] = squares[t];
        quot.push(
            [A, globalIdx(t, 0, 0)], [B, globalIdx(t, 0, w - 1)],
            [C, globalIdx(t, w - 1, w - 1)], [D, globalIdx(t, w - 1, 0)],
        );
    }
    // Unlike triangleForm's A < B < C corner convention, a square's cycle order isn't globally
    // monotonic in vertex index, so each side's natural sequence is explicitly re-oriented here to
    // always run from min(endpoint) to max(endpoint) - the shared canonical direction every square
    // touching that original edge agrees on, regardless of its own cycle orientation.
    const edgeToSeqs = new Map<string, { t: number; seq: [number, number][] }[]>();
    for (let t = 0; t < squares.length; t++) {
        const [A, B, C, D] = squares[t];
        const sides: [number, number, 0 | 1 | 2 | 3][] = [[A, B, 0], [B, C, 1], [C, D, 2], [D, A, 3]];
        for (const [ep1, ep2, side] of sides) {
            const key = `${Math.min(ep1, ep2)},${Math.max(ep1, ep2)}`;
            const seq = ep1 < ep2 ? naturalSeq(side) : [...naturalSeq(side)].reverse();
            if (!edgeToSeqs.has(key)) edgeToSeqs.set(key, []);
            edgeToSeqs.get(key)!.push({ t, seq });
        }
    }
    for (const entries of edgeToSeqs.values()) {
        if (entries.length < 2) continue;
        const canonical = entries[0].seq;
        for (let k = 1; k < entries.length; k++) {
            const seq = entries[k].seq;
            for (let idx = 0; idx < w; idx++)
                quot.push([
                    globalIdx(entries[0].t, canonical[idx][0], canonical[idx][1]),
                    globalIdx(entries[k].t, seq[idx][0], seq[idx][1]),
                ]);
        }
    }

    const combined = make(new Embedding(embDim, pos, bc.emb.projMat), adj);
    return quotientBoard(combined, quot);
}

/**
 * The default projMat assigned to a freshly-`product()`-ed board (see below): dims 0 and 1 map
 * straight to x/y (identity - the whole matrix is exactly IDENTITY_2X2 when embDim <= 2), and every
 * pair of dims beyond that alternates contributing a halved-again magnitude to x then y, e.g.
 * embDim=8 gives `[[1, 0, 1/2, 0, 1/4, 0, 1/8, 0], [0, 1, 0, 1/2, 0, 1/4, 0, 1/8]]`. Dim `d`
 * contributes magnitude `2^-floor(d/2)` to x if `d` is even, to y if `d` is odd (which also
 * reproduces the d=0/d=1 identity part with no separate case needed, since `2^-floor(0/2)` and
 * `2^-floor(1/2)` both equal 1).
 */
function defaultProductProjMat(embDim: number): number[][] {
    const row0 = new Array<number>(embDim).fill(0);
    const row1 = new Array<number>(embDim).fill(0);
    for (let d = 0; d < embDim; d++) {
        const mag = 2 ** -Math.floor(d / 2);
        if (d % 2 === 0) row0[d] = mag; else row1[d] = mag;
    }
    return [row0, row1];
}

/**
 * The Cartesian (box) product of two board configs: N = `bc1.N * bc2.N`, one new node per pair
 * `(i, j)` (`i` from `bc1`, `j` from `bc2`), at the concatenated natural position
 * `[...bc1.emb.pos[i], ...bc2.emb.pos[j]]` (embDim = `bc1.emb.embDim + bc2.emb.embDim`). `(i, j)` is
 * adjacent to `(i2, j2)` iff exactly one of:
 *   - `i === i2` and `j` is adjacent to `j2` in `bc2`
 *   - `j === j2` and `i` is adjacent to `i2` in `bc1`
 * (the standard graph Cartesian product - e.g. `cubeLatticeBoard(w, h, d)` is, up to embedding, the
 * product of three path graphs). Uses a fresh default projMat (see defaultProductProjMat) rather than
 * either factor's own projMat, since neither one alone is meaningful at the combined dimension.
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

    return make(new Embedding(embDim, pos, defaultProductProjMat(embDim)), adj);
}

/** A rectangular board with width `w` and height `h`. Each node is identified by (col, row) where 0 ≤ col < w, 0 ≤ row < h. */
export function rectangularBoard(w: number, h: number): BoardConfig {
    assert(w > 0 && h > 0, `w and h must be positive, got w=${w} h=${h}`);
    const pos: number[][] = [];
    for (let r = 0; r < h; r++)
        for (let c = 0; c < w; c++)
            pos.push([c - (w - 1) / 2, r - (h - 1) / 2]);
    const adj = zeroAdj(w * h);
    for (let r = 0; r < h; r++)
        for (let c = 0; c < w; c++)
            for (const [dr, dc] of [[0,1],[1,0],[0,-1],[-1,0]]) {
                const nr = r + dr, nc = c + dc;
                if (nr < 0 || nr >= h || nc < 0 || nc >= w) continue;
                adj[r*w+c][nr*w+nc] = 1;
            }
    return make(pos, adj);
}

/**
 * Traditional Go board star points ("hoshi") for a rectangular board,
 * derived from `config.boardType`/`boardArgs` - [] for any non-'rect'
 * board. Corner points sit at the 3-3 point (boards whose smaller edge is
 * 9 or 11) or the 4-4 point (smaller edge > 11), edge points sit at the
 * midpoint of an odd, >=19-length edge whose cross edge is >=9, and a
 * single center point appears when both edges are odd and >=5 - together
 * reproducing the real 9x9 (4 corners + center), 13x13 (4 corners +
 * center), and 19x19 (4 corner + 4 edge + center) star-point layouts.
 * Returned as [x, y] pairs in the same board-coordinate space as
 * BoardConfig.pos (see rectangularBoard() above), ready for the same
 * originX + x*cell / originY - y*cell screen transform.
 */
export function computeStarPoints(config: GameConfig): number[][] {
    if (config.boardType !== 'rect') return [];
    if (config.boardModifiers.length > 0) return [];
    const [w, h] = config.boardArgs;
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
    assert(w > 0 && h > 0 && d > 0, `w, h, and d must be positive, got w=${w} h=${h} d=${d}`);
    // Natural 3D coords (col, row, slice), centered. Rendered via projMat below - chosen so that
    // projMat . [c', r', s] == the old direct 2D formula [(d*c'+0.8*s)*scale, (d*r'+0.8*s)*scale]
    // exactly (each slice's w x h grid scaled up by d, then diagonally offset by the slice index).
    const pos: number[][] = [];
    for (let s = 0; s < d; s++)
        for (let r = 0; r < h; r++)
            for (let c = 0; c < w; c++)
                pos.push([c - (w-1)/2, r - (h-1)/2, s]);
    const scale = d > 1 ? 1 / 1.2 : 1;
    const projMat = [[d * scale, 0, 0.8 * scale], [0, d * scale, 0.8 * scale]];
    const N = w * h * d;
    const adj = zeroAdj(N);
    const idx = (r: number, c: number, s: number) => s * h * w + r * w + c;
    for (let s = 0; s < d; s++)
        for (let r = 0; r < h; r++)
            for (let c = 0; c < w; c++)
                for (const [dr, dc, ds] of [[0,1,0],[1,0,0],[0,-1,0],[-1,0,0],[0,0,1],[0,0,-1]]) {
                    const nr = r+dr, nc = c+dc, ns = s+ds;
                    if (nr<0||nr>=h||nc<0||nc>=w||ns<0||ns>=d) continue;
                    adj[idx(r,c,s)][idx(nr,nc,ns)] = 1;
                }
    return make(new Embedding(3, pos, projMat), adj);
}

/**
 * A hypercubical board with width `w`, height `h`, depth `d` and hyperdepth `t`. Each node is
 * identified by (col, row, slice, hyperslice) where 0 ≤ col < w, 0 ≤ row < h, 0 ≤ slice < d,
 * 0 ≤ hyperslice < t.
 */
export function hypercubeBoard(w: number, h: number, d: number, t: number): BoardConfig {
    assert(w > 0 && h > 0 && d > 0 && t > 0, `w, h, d, and t must be positive, got w=${w} h=${h} d=${d} t=${t}`);
    // Natural 4D coords (col, row, slice, hyperslice), all centered. projMat below reproduces the
    // old direct 2D "grid-of-grids" formula exactly: d copies of the w x h grid tiled horizontally
    // (spacing w+1), t rows of those tiled vertically (spacing h+1).
    const pos: number[][] = [];
    for (let s = 0; s < t; s++)
        for (let u = 0; u < d; u++)
            for (let r = 0; r < h; r++)
                for (let c = 0; c < w; c++)
                    pos.push([c - (w-1)/2, r - (h-1)/2, u - (d-1)/2, s - (t-1)/2]);
    const projMat = [[1, 0, w+1, 0], [0, 1, 0, h+1]];
    const N = w * h * d * t;
    const adj = zeroAdj(N);
    const idx = (r: number, c: number, u: number, s: number) =>
        ((s * d + u) * h + r) * w + c;
    const dirs4: [number, number, number, number][] = [
        [0,1,0,0],[1,0,0,0],[0,-1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,-1,0],[0,0,0,1],[0,0,0,-1],
    ];
    for (let s = 0; s < t; s++)
        for (let u = 0; u < d; u++)
            for (let r = 0; r < h; r++)
                for (let c = 0; c < w; c++)
                    for (const [dr,dc,du,ds] of dirs4) {
                        const nr=r+dr, nc=c+dc, nu=u+du, ns=s+ds;
                        if (nr<0||nr>=h||nc<0||nc>=w||nu<0||nu>=d||ns<0||ns>=t) continue;
                        adj[idx(r,c,u,s)][idx(nr,nc,nu,ns)] = 1;
                    }
    return make(new Embedding(4, pos, projMat), adj);
}

/** A triangular board with side length `w`. */
export function triangularBoard(w: number): BoardConfig {
    assert(w > 0, `w must be positive, got w=${w}`);
    const rowDist = Math.sqrt(3) / 2;
    const pos: number[][] = [];
    for (let i = 0; i < w; i++) {
        for (let j = 0; j <= i; j++) {
            pos.push([j - i/2, rowDist * (i + 1 - w / 3)]);
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
 * A regular tetrahedron: 4 vertices, all mutually adjacent (K4), 6 unit-length edges. A
 * side-length-w subdivision of its 4 triangular faces is no longer built in here directly - apply
 * the `triangleForm(w)` modifier afterward instead (findTriangles finds exactly its 4 faces on this
 * board, since every 3-subset of K4's vertices is a triangle).
 */
export function tetrahedronBoard(): BoardConfig {
    const edgeScale = 1 / (2 * Math.sqrt(2));
    const pos: number[][] = [
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1],
    ].map(v => v.map(x => x * edgeScale));

    const adj = zeroAdj(4);
    for (let i = 0; i < 4; i++)
        for (let j = 0; j < 4; j++)
            if (i !== j) adj[i][j] = 1;

    // Simple axonometric-style projection (matches cubeLatticeBoard/dodecahedronBoard/
    // icosahedronBoard's own hand-tuned approach for a 3D shape): x/y project straight through, z
    // nudges diagonally so depth stays visible.
    const projMat = [[1, 0, 0.4], [0, 1, 0.4]];
    return make(new Embedding(3, pos, projMat), adj);
}

/**
 * A regular dodecahedron: 20 vertices, 12 pentagonal faces, 30 unit-length edges, centered at the
 * origin. Vertices form 4 groups of the classic "three mutually orthogonal golden rectangles"
 * construction (`phi` = golden ratio): 8 cube corners `(sa, sb, sc)`, plus 4+4+4 more at
 * `(0, sb/phi, sc*phi)`, `(sa/phi, sb*phi, 0)`, `(sa*phi, 0, sc/phi)` - each coordinate
 * independently `+-1`. At that raw scale, edge length is `2/phi`; every coordinate below is
 * pre-multiplied by `phi/2` so edges come out exactly 1.
 *
 * Connectivity (worked out by checking which vertex pairs land exactly 1 apart, then cross-checked
 * against the dodecahedron's known 30-edge, degree-3-per-vertex structure): each cube vertex
 * `(sa, sb, sc)` connects to exactly one vertex in each of the other 3 groups - `(0, sb/phi,
 * sc*phi)`, `(sa/phi, sb*phi, 0)`, `(sa*phi, 0, sc/phi)` - and each of those 12 non-cube vertices'
 * third edge (beyond its 2 cube-vertex edges) goes to its own sign-flipped partner within the same
 * group, e.g. `(0, sb/phi, sc*phi)` - `(0, -sb/phi, sc*phi)`.
 */
export function dodecahedronBoard(): BoardConfig {
    const phi = (1 + Math.sqrt(5)) / 2;
    const scale = phi / 2; // normalizes edge length (2/phi at the raw scale above) to exactly 1
    const s = (bit: number) => (bit === 0 ? 1 : -1); // 0/1 sign-bit -> +-1

    const xIdx = (sa: number, sb: number, sc: number) => sa * 4 + sb * 2 + sc;
    const yIdx = (sb: number, sc: number) => 8 + sb * 2 + sc;
    const zIdx = (sa: number, sb: number) => 12 + sa * 2 + sb;
    const wIdx = (sa: number, sc: number) => 16 + sa * 2 + sc;

    const pos: number[][] = new Array(20);
    for (let sa = 0; sa < 2; sa++)
        for (let sb = 0; sb < 2; sb++)
            for (let sc = 0; sc < 2; sc++)
                pos[xIdx(sa, sb, sc)] = [s(sa) * scale, s(sb) * scale, s(sc) * scale];
    for (let sb = 0; sb < 2; sb++)
        for (let sc = 0; sc < 2; sc++)
            pos[yIdx(sb, sc)] = [0, (s(sb) / phi) * scale, s(sc) * phi * scale];
    for (let sa = 0; sa < 2; sa++)
        for (let sb = 0; sb < 2; sb++)
            pos[zIdx(sa, sb)] = [(s(sa) / phi) * scale, s(sb) * phi * scale, 0];
    for (let sa = 0; sa < 2; sa++)
        for (let sc = 0; sc < 2; sc++)
            pos[wIdx(sa, sc)] = [s(sa) * phi * scale, 0, (s(sc) / phi) * scale];

    const adj = zeroAdj(20);
    const connect = (i: number, j: number) => { adj[i][j] = 1; adj[j][i] = 1; };
    for (let sa = 0; sa < 2; sa++)
        for (let sb = 0; sb < 2; sb++)
            for (let sc = 0; sc < 2; sc++) {
                const x = xIdx(sa, sb, sc);
                connect(x, yIdx(sb, sc));
                connect(x, zIdx(sa, sb));
                connect(x, wIdx(sa, sc));
            }
    for (let sc = 0; sc < 2; sc++) connect(yIdx(0, sc), yIdx(1, sc));
    for (let sb = 0; sb < 2; sb++) connect(zIdx(0, sb), zIdx(1, sb));
    for (let sa = 0; sa < 2; sa++) connect(wIdx(sa, 0), wIdx(sa, 1));

    // Simple axonometric-style projection (matches cubeLatticeBoard's own hand-tuned approach for a
    // 3D shape): x/y project straight through, z nudges diagonally so depth stays visible.
    const projMat = [[1, 0, 0.4], [0, 1, 0.4]];
    return make(new Embedding(3, pos, projMat), adj);
}

/**
 * A regular icosahedron: 12 vertices, 20 triangular faces, 30 unit-length edges, centered at the
 * origin. Vertices form 3 groups of 4, each the set of cyclic-coordinate permutations of
 * `(0, +-1, +-phi)` (`phi` = golden ratio) sharing one fixed-zero axis: `A(sp, sq) = (0, sp,
 * sq*phi)`, `B(sp, sq) = (sp, sq*phi, 0)`, `C(sp, sq) = (sq*phi, 0, sp)`, each coordinate
 * independently `+-1`. At that raw scale, edge length is 2; every coordinate below is
 * pre-multiplied by 1/2 so edges come out exactly 1.
 *
 * Connectivity (worked out the same way as dodecahedronBoard: checking which vertex pairs land
 * exactly the minimum distance apart, then cross-checked against the icosahedron's known 30-edge,
 * degree-5-per-vertex structure - this one is easy to get backwards by hand, so every relation
 * below was independently re-derived algebraically, not just pattern-matched from a couple of
 * examples): within each group, `(sp, sq)` connects to its own sign-flipped-`sp` partner
 * `(-sp, sq)`. Across groups, the three relations cycle A -> B -> C -> A, each keyed off the
 * *sending* group's own `sp`: `A(sp, sq)` connects to both `B(+-1, sp)`; `B(sp, sq)` connects to
 * both `C(+-1, sp)`; `C(sp, sq)` connects to both `A(+-1, sp)`.
 */
export function icosahedronBoard(): BoardConfig {
    const phi = (1 + Math.sqrt(5)) / 2;
    const scale = 0.5; // normalizes edge length (2 at the raw scale above) to exactly 1
    const s = (bit: number) => (bit === 0 ? 1 : -1); // 0/1 sign-bit -> +-1

    const aIdx = (sp: number, sq: number) => sp * 2 + sq;
    const bIdx = (sp: number, sq: number) => 4 + sp * 2 + sq;
    const cIdx = (sp: number, sq: number) => 8 + sp * 2 + sq;

    const pos: number[][] = new Array(12);
    for (let sp = 0; sp < 2; sp++)
        for (let sq = 0; sq < 2; sq++) {
            pos[aIdx(sp, sq)] = [0, s(sp) * scale, s(sq) * phi * scale];
            pos[bIdx(sp, sq)] = [s(sp) * scale, s(sq) * phi * scale, 0];
            pos[cIdx(sp, sq)] = [s(sq) * phi * scale, 0, s(sp) * scale];
        }

    const adj = zeroAdj(12);
    const connect = (i: number, j: number) => { adj[i][j] = 1; adj[j][i] = 1; };

    // Same-group edges: flip sp, keep sq.
    for (let sq = 0; sq < 2; sq++) {
        connect(aIdx(0, sq), aIdx(1, sq));
        connect(bIdx(0, sq), bIdx(1, sq));
        connect(cIdx(0, sq), cIdx(1, sq));
    }
    // Cross-group edges: A(sp,sq) ~ B(*,sp); B(sp,sq) ~ C(*,sp); C(sp,sq) ~ A(*,sp).
    for (let sp = 0; sp < 2; sp++)
        for (let sq = 0; sq < 2; sq++)
            for (let free = 0; free < 2; free++) {
                connect(aIdx(sp, sq), bIdx(free, sp));
                connect(bIdx(sp, sq), cIdx(free, sp));
                connect(cIdx(sp, sq), aIdx(free, sp));
            }

    // Simple axonometric-style projection (matches cubeLatticeBoard/dodecahedronBoard's own
    // hand-tuned approach for a 3D shape): x/y project straight through, z nudges diagonally so
    // depth stays visible.
    const projMat = [[1, 0, 0.4], [0, 1, 0.4]];
    return make(new Embedding(3, pos, projMat), adj);
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

/** Auxiliary function for `twistedSquareBoard` and `glueTwistedSquareBoard`. Not used by the renderer directly. */
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
 * A `w × h` grid of `g × g` squares, each rotated ±30° in a checkerboard pattern, arranged as a
 * snub square tiling. Adjacent squares are glued or triangle-connected at their nearest corners.
 */
export function snubSquareBoard(w: number, h: number, g: number): BoardConfig {
    assert(w > 0 && h > 0 && g > 0, `w, h, and g must be positive, got w=${w} h=${h} g=${g}`);

    const spacing = (g - 1) * (0.5 + Math.sqrt(3) / 2);
    const pos: number[][] = [];
    for (let rb = 0; rb < h; rb++)
        for (let cb = 0; cb < w; cb++) {
            const bx = (cb - (w-1)/2) * spacing;
            const by = (rb - (h-1)/2) * spacing;
            const angle = ((rb + cb) % 2 === 0 ? -1 : 1) * Math.PI / 6;
            const ca = Math.cos(angle), sa = Math.sin(angle);
            for (let r = 0; r < g; r++)
                for (let c = 0; c < g; c++) {
                    const lx = c - (g-1)/2, ly = r - (g-1)/2;
                    pos.push([bx + ca*lx - sa*ly, by + sa*lx + ca*ly]);
                }
        }
    const N = w * h * g * g;
    const adj = zeroAdj(N);
    const bIdx = (rb: number, cb: number) => (rb * w + cb) * g * g;
    const cornerRC: Record<'NW'|'NE'|'SW'|'SE', [number, number]> =
        { NW: [0,0], NE: [0,g-1], SW: [g-1,0], SE: [g-1,g-1] };
    const corner = (name: 'NW'|'NE'|'SW'|'SE'): [number, number] => cornerRC[name];

    // Edges within each big cell (ordinary rectangular grid)
    for (let rb = 0; rb < h; rb++)
        for (let cb = 0; cb < w; cb++) {
            const b = bIdx(rb, cb);
            for (let r = 0; r < g; r++)
                for (let c = 0; c < g; c++)
                    for (const [dr, dc] of [[0,1],[1,0],[0,-1],[-1,0]]) {
                        const nr = r+dr, nc = c+dc;
                        if (nr<0||nr>=g||nc<0||nc>=g) continue;
                        adj[b+r*g+c][b+nr*g+nc] = 1;
                    }
        }

    // Inter-cell connections, keyed by self cell's checkerboard parity then by "dr,dc" (only
    // forward directions, so each neighboring pair of big cells is handled once): glue is a single
    // coincident corner pair (merged via quotientBoard below); tri is three equal-distance pairs
    // (wired as new edges into adj instead).
    type Corner = 'NW'|'NE'|'SW'|'SE';
    const CONN: Record<number, Record<string, { glue?: [Corner,Corner], tri?: [Corner,Corner] }>> = {
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
                if (conn.glue) {
                    const [sr,sc] = corner(conn.glue[0]), [or_,oc] = corner(conn.glue[1]);
                    interConn.push([b + sr*g + sc, nb + or_*g + oc]);
                }
                if (conn.tri) {
                    const [sr,sc] = corner(conn.tri[0]), [or_,oc] = corner(conn.tri[1]);
                    const i = b + sr*g + sc, j = nb + or_*g + oc;
                    adj[i][j] = 1; adj[j][i] = 1;
                }
            }
        }

    const bc = make(pos, adj);
    return quotientBoard(bc, interConn);
}

/**
 * Triangle-inflated variant of snubSquareBoard: same w×h grid of g×g squares (rotated ±30° in the
 * same checkerboard pattern), but every square-to-square gap is filled by an actual side-length-g
 * triangular sub-board (same construction as triangularBoard(g)) instead of a single glued corner
 * plus a bare edge - squares only ever touch via triangles, and every connection is a whole-edge glue.
 */
export function snubSquareTriBoard(w: number, h: number, g: number): BoardConfig {
    assert(w > 0 && h > 0 && g > 0, `w, h, and g must be positive, got w=${w} h=${h} g=${g}`);

    const spacing = (g - 1) * (0.5 + Math.sqrt(3) / 2);
    const nTri = g * (g + 1) / 2;
    const triIdx = (i: number, j: number) => i * (i + 1) / 2 + j;

    const sqIdx = (x: number, y: number) => (y * w + x) * g * g;
    const sqN = w * h * g * g;
    const hCount = (w - 1) * h;
    const vCount = w * (h - 1);
    const hBase = (x: number, y: number) => sqN + (y * (w - 1) + x) * nTri;
    const vBase = (x: number, y: number) => sqN + hCount * nTri + (y * w + x) * nTri;
    const N = sqN + (hCount + vCount) * nTri;

    // Positions: squares laid out exactly as in snubSquareBoard first, then h-triangles, then
    // v-triangles. Each triangle's (i,0)/(i,i) nodes are placed at the exact real position of the
    // square corner they glue to (already pushed above, read back via sqIdx - never
    // recomputed independently), and every other node (i,j), 0<j<i, is placed by linear
    // interpolation between them at fraction j/i: triangularBoard's own template has row i's nodes
    // collinear and evenly spaced ((i,0)=[-i/2,Y], (i,j)=[j-i/2,Y], (i,i)=[i/2,Y], same Y), and the
    // square-to-triangle glue is an exact rigid (distance-preserving) fit - the gap between a
    // triangle's (i,0)/(i,i) glue targets is exactly i, matching the template's own (i,0)-(i,i)
    // distance for every i, not just the endpoints - so this interpolation reproduces the triangle's
    // true real position exactly, unlike a naive unrotated placeholder.
    const pos: number[][] = [];
    for (let y = 0; y < h; y++)
        for (let x = 0; x < w; x++) {
            const bx = (x - (w-1)/2) * spacing, by = (y - (h-1)/2) * spacing;
            const angle = ((x + y) % 2 === 0 ? -1 : 1) * Math.PI / 6;
            const ca = Math.cos(angle), sa = Math.sin(angle);
            for (let r = 0; r < g; r++)
                for (let c = 0; c < g; c++) {
                    const lx = c - (g-1)/2, ly = r - (g-1)/2;
                    pos.push([bx + ca*lx - sa*ly, by + sa*lx + ca*ly]);
                }
        }
    for (let y = 0; y < h; y++)
        for (let x = 0; x < w - 1; x++) {
            const p = (x + y) % 2;
            for (let i = 0; i < g; i++) {
                const selfR = p === 0 ? g-1-i : i;
                const left = pos[sqIdx(x, y) + selfR*g + (g-1)];
                const right = pos[sqIdx(x+1, y) + selfR*g + 0];
                for (let j = 0; j <= i; j++) {
                    const t = i === 0 ? 0 : j / i;
                    pos.push([left[0] + (right[0]-left[0])*t, left[1] + (right[1]-left[1])*t]);
                }
            }
        }
    for (let y = 0; y < h - 1; y++)
        for (let x = 0; x < w; x++) {
            const p = (x + y) % 2;
            for (let i = 0; i < g; i++) {
                const selfC = p === 0 ? i : g-1-i;
                const left = pos[sqIdx(x, y) + (g-1)*g + selfC];
                const right = pos[sqIdx(x, y+1) + 0*g + selfC];
                for (let j = 0; j <= i; j++) {
                    const t = i === 0 ? 0 : j / i;
                    pos.push([left[0] + (right[0]-left[0])*t, left[1] + (right[1]-left[1])*t]);
                }
            }
        }

    const adj = zeroAdj(N);

    // Intra-square edges (ordinary rectangular grid; no direct square-to-square connections here).
    for (let y = 0; y < h; y++)
        for (let x = 0; x < w; x++) {
            const b = sqIdx(x, y);
            for (let r = 0; r < g; r++)
                for (let c = 0; c < g; c++)
                    for (const [dr, dc] of [[0,1],[1,0],[0,-1],[-1,0]]) {
                        const nr = r+dr, nc = c+dc;
                        if (nr<0||nr>=g||nc<0||nc>=g) continue;
                        adj[b+r*g+c][b+nr*g+nc] = 1;
                    }
        }

    // Intra-triangle edges (mirrors triangularBoard's own edge loop).
    const triDirs: [number, number][] = [[1,0],[1,1],[0,1],[-1,0],[-1,-1],[0,-1]];
    const addTriEdges = (base: number) => {
        for (let i = 0; i < g; i++)
            for (let j = 0; j <= i; j++)
                for (const [di, dj] of triDirs) {
                    const ni = i+di, nj = j+dj;
                    if (ni < 0 || ni >= g || nj < 0 || nj > ni) continue;
                    adj[base+triIdx(i,j)][base+triIdx(ni,nj)] = 1;
                }
    };
    for (let y = 0; y < h; y++)
        for (let x = 0; x < w - 1; x++) addTriEdges(hBase(x, y));
    for (let y = 0; y < h - 1; y++)
        for (let x = 0; x < w; x++) addTriEdges(vBase(x, y));

    // Gluing: every square-triangle and triangle-triangle edge, merged via a single quotientBoard call.
    const interConn: [number, number][] = [];
    for (let y = 0; y < h; y++)
        for (let x = 0; x < w - 1; x++) {
            const p = (x + y) % 2;
            const base = hBase(x, y);
            for (let i = 0; i < g; i++) {
                const selfR = p === 0 ? g-1-i : i;
                interConn.push([base + triIdx(i,0), sqIdx(x, y) + selfR*g + (g-1)]);
                interConn.push([base + triIdx(i,i), sqIdx(x+1, y) + selfR*g + 0]);
            }
            if (p === 1 && y + 1 <= h - 1) {
                const nbase = hBase(x, y+1);
                for (let j = 0; j < g; j++)
                    interConn.push([base + triIdx(g-1,j), nbase + triIdx(g-1,j)]);
            }
        }
    for (let y = 0; y < h - 1; y++)
        for (let x = 0; x < w; x++) {
            const p = (x + y) % 2;
            const base = vBase(x, y);
            for (let i = 0; i < g; i++) {
                const selfC = p === 0 ? i : g-1-i;
                interConn.push([base + triIdx(i,0), sqIdx(x, y) + (g-1)*g + selfC]);
                interConn.push([base + triIdx(i,i), sqIdx(x, y+1) + 0*g + selfC]);
            }
            if (p === 0 && x + 1 <= w - 1) {
                const nbase = vBase(x+1, y);
                for (let j = 0; j < g; j++)
                    interConn.push([base + triIdx(g-1,j), nbase + triIdx(g-1,j)]);
            }
        }

    const bc = make(pos, adj);
    return quotientBoard(bc, interConn);
}


export enum PrescribedBoard {
    rectangularBoard,
    rectangularDiagonalBoard,
    cubeLatticeBoard,
    hypercubeBoard,
    triangularBoard,
    regularPolygonBoard,
    tetrahedronBoard,
    dodecahedronBoard,
    icosahedronBoard,
    triangularHexBoard,
    hexBoard,
    trihexBoard,
    snubSquareBoard,
    snubSquareTriBoard,
    twistedSquareBoard,
    glueTwistedSquareBoard
}

export const PrescribedBoardMap: Record<PrescribedBoard, [number, string, string, string]> = {
    [PrescribedBoard.rectangularBoard]:
        [2, "rect", "&lt;w&gt; &lt;h&gt;", "Rectangular board"],
    [PrescribedBoard.rectangularDiagonalBoard]:
        [3, "rectd", "&lt;w&gt; &lt;h&gt; &lt;m&gt;", "Rectangular + diagonal connections every m squares"],
    [PrescribedBoard.cubeLatticeBoard]:         [3, "cublat", "&lt;w&gt; &lt;h&gt; &lt;d&gt;", "Cubical board"],
    [PrescribedBoard.hypercubeBoard]:
        [4, "hcub", "&lt;w&gt; &lt;h&gt; &lt;d&gt; &lt;t&gt;", "Hypercubical board"],
    [PrescribedBoard.triangularBoard]:
        [1, "tri", "&lt;w&gt;", "Triangular board of side w"],
    [PrescribedBoard.regularPolygonBoard]:
        [1, "regpoly", "&lt;n&gt;", "Regular polygon with n unit-length edges"],
    [PrescribedBoard.tetrahedronBoard]:
        [0, "tetra", "", "Regular tetrahedron (4 vertices, all mutually adjacent, unit-length edges)"],
    [PrescribedBoard.dodecahedronBoard]:
        [0, "dodeca", "", "Regular dodecahedron (20 vertices, 12 pentagonal faces, unit-length edges)"],
    [PrescribedBoard.icosahedronBoard]:
        [0, "icosa", "", "Regular icosahedron (12 vertices, 20 triangular faces, unit-length edges)"],
    [PrescribedBoard.triangularHexBoard]:
        [1, "trihex", "&lt;d&gt;",
            "Triangular-lattice board in a hexagon shape, with d layers of triangles around the center"],
    [PrescribedBoard.hexBoard]:
        [1, "hex", "&lt;d&gt;", "Hexagon-tiled board with d layers of hexagons around a center hexagon"],
    [PrescribedBoard.trihexBoard]:
        [1, "hexdel", "&lt;d&gt;",
            "Trihexagonal (hexdel) board, d layers of hexagons connected by triangles around a center hexagon"],
    [PrescribedBoard.snubSquareBoard]:
        [3, "snubsq", "&lt;w&gt; &lt;h&gt; &lt;g&gt;", "Snub square board (g\xD7g squares)"],
    [PrescribedBoard.snubSquareTriBoard]:
        [3, "snubsqtri", "&lt;w&gt; &lt;h&gt; &lt;g&gt;",
            "Snub square board with the connecting triangles as g\xD7g triangular boards too"],
    [PrescribedBoard.twistedSquareBoard]:
        [3, "twsq", "&lt;w&gt; &lt;h&gt; &lt;g&gt;", "Twisted-square board (g\xD7g squares)"],
    [PrescribedBoard.glueTwistedSquareBoard]:
        [3, "gtsq", "&lt;w&gt; &lt;h&gt; &lt;g&gt;", "Glued-twisted-square board (g\xD7g squares)"],
};

export const PrescribedBoardFns: Record<PrescribedBoard, (...args: number[]) => BoardConfig> = {
    [PrescribedBoard.rectangularBoard]:         (...a) => rectangularBoard(a[0], a[1]),
    [PrescribedBoard.rectangularDiagonalBoard]: (...a) => rectangularDiagonalBoard(a[0], a[1], a[2]),
    [PrescribedBoard.cubeLatticeBoard]:         (...a) => cubeLatticeBoard(a[0], a[1], a[2]),
    [PrescribedBoard.hypercubeBoard]:           (...a) => hypercubeBoard(a[0], a[1], a[2], a[3]),
    [PrescribedBoard.triangularBoard]:          (...a) => triangularBoard(a[0]),
    [PrescribedBoard.regularPolygonBoard]:      (...a) => regularPolygonBoard(a[0]),
    [PrescribedBoard.tetrahedronBoard]:         () => tetrahedronBoard(),
    [PrescribedBoard.dodecahedronBoard]:        () => dodecahedronBoard(),
    [PrescribedBoard.icosahedronBoard]:         () => icosahedronBoard(),
    [PrescribedBoard.triangularHexBoard]:       (...a) => triangularHexBoard(a[0]),
    [PrescribedBoard.hexBoard]:                 (...a) => hexBoard(a[0]),
    [PrescribedBoard.trihexBoard]:               (...a) => trihexBoard(a[0]),
    [PrescribedBoard.snubSquareBoard]:          (...a) => snubSquareBoard(a[0], a[1], a[2]),
    [PrescribedBoard.snubSquareTriBoard]:       (...a) => snubSquareTriBoard(a[0], a[1], a[2]),
    [PrescribedBoard.twistedSquareBoard]:       (...a) => twistedSquareBoard(a[0], a[1], a[2]),
    [PrescribedBoard.glueTwistedSquareBoard]:   (...a) => glueTwistedSquareBoard(a[0], a[1], a[2]),
};

/**
 * Command name (PrescribedBoardMap[pb][1], e.g. 'rect', 'cublat') -> PrescribedBoard enum value -
 * shared by buildPrescribedBoard below and parseModifier's beginprod validation.
 */
const PRESCRIBED_BOARD_BY_NAME = new Map<string, PrescribedBoard>(
    (Object.entries(PrescribedBoardMap) as [string, [number, string, string, string]][])
        .map(([k, [, cmd]]) => [cmd, Number(k) as PrescribedBoard]),
);

/**
 * Builds a board from its command-name kind (e.g. 'rect', 'cublat' - see PrescribedBoardMap) and
 * positional args - the same string-keyed dispatch renderer.ts's `_cmdToBoard` builds from the same
 * PrescribedBoardMap/PrescribedBoardFns pairing. Used by applyModifiers's BeginProd handling below.
 * Throws for an unrecognized kind.
 */
function buildPrescribedBoard(kind: string, args: number[]): BoardConfig {
    const pb = PRESCRIBED_BOARD_BY_NAME.get(kind);
    if (pb === undefined) throw new Error(`Unknown board type: ${kind}`);
    return PrescribedBoardFns[pb](...args);
}

export type BoardModifier =
    | { kind: 'Rectify' }
    | { kind: 'EdgeSplit'; splitN: number }
    | { kind: 'MergeClose'; dist: number }
    | { kind: 'TriangleForm'; w: number }
    | { kind: 'SquareForm'; w: number }
    | { kind: 'Prod'; boardType: string; boardArgs: number[] }
    | { kind: 'BeginProd'; boardType: string; boardArgs: number[] }
    | { kind: 'EndProd' };

/** mc's default `dist` when called with no argument - see parseModifier and renderer.ts's command reference panel. */
export const MC_DEFAULT_DIST = 0.01;

/**
 * Parses a board-type command name (e.g. 'rect', 'cublat' - see PrescribedBoardMap) plus its
 * positional dimension args, shared by `prod`/`beginprod`'s parseModifier branches below: the
 * board type is validated eagerly via PRESCRIBED_BOARD_BY_NAME, and its args must number at least
 * PrescribedBoardMap's required count for that type or this throws; extras beyond that count are
 * silently truncated (so e.g. a leftover product-context arg doesn't need to be stripped by the
 * caller).
 */
function parseBoardTypeArgs(cmdName: string, args: string[]): { boardType: string; boardArgs: number[] } {
    assert(args.length >= 1, `${cmdName} takes at least 1 argument (board type), got ${args.length}`);
    const [boardType, ...argStrs] = args;
    const pb = PRESCRIBED_BOARD_BY_NAME.get(boardType);
    if (pb === undefined) throw new Error(`${cmdName}: unknown board type "${boardType}"`);
    const requiredArgs = PrescribedBoardMap[pb][0];
    assert(argStrs.length >= requiredArgs,
        `${cmdName}: board type "${boardType}" requires ${requiredArgs} argument(s), got ${argStrs.length}`);
    const boardArgs = argStrs.slice(0, requiredArgs).map(Number);
    assert(boardArgs.every(n => Number.isInteger(n)),
        `${cmdName}: board args must be integers, got "${argStrs.join(' ')}"`);
    return { boardType, boardArgs };
}

/**
 * Parses a BoardModifier from its command name ('rect', 'es', 'mc', 'triform', 'sqform', 'prod',
 * 'beginprod', 'endprod') and string args - see applyModifier/applyModifiers. mc's arg is optional: with none,
 * `dist` defaults to MC_DEFAULT_DIST. prod/beginprod's first arg is a board-type command name and
 * the rest are that type's own positional dimension args - see parseBoardTypeArgs.
 */
export function parseModifier(name: string, args: string[]): BoardModifier {
    if (name === 'rect') {
        assert(args.length === 0, `rect takes no arguments, got ${args.length}`);
        return { kind: 'Rectify' };
    }
    if (name === 'es') {
        assert(args.length === 1, `es takes exactly 1 argument (splitN), got ${args.length}`);
        const splitN = Number(args[0]);
        assert(Number.isInteger(splitN) && splitN >= 1, `es: splitN must be a positive integer, got "${args[0]}"`);
        return { kind: 'EdgeSplit', splitN };
    }
    if (name === 'mc') {
        assert(args.length <= 1, `mc takes at most 1 argument (dist), got ${args.length}`);
        const dist = args.length === 0 ? MC_DEFAULT_DIST : Number(args[0]);
        assert(Number.isFinite(dist) && dist > 0, `mc: dist must be a positive number, got "${args[0]}"`);
        return { kind: 'MergeClose', dist };
    }
    if (name === 'triform') {
        assert(args.length === 1, `triform takes exactly 1 argument (w), got ${args.length}`);
        const w = Number(args[0]);
        assert(Number.isInteger(w) && w >= 1, `triform: w must be a positive integer, got "${args[0]}"`);
        return { kind: 'TriangleForm', w };
    }
    if (name === 'sqform') {
        assert(args.length === 1, `sqform takes exactly 1 argument (w), got ${args.length}`);
        const w = Number(args[0]);
        assert(Number.isInteger(w) && w >= 1, `sqform: w must be a positive integer, got "${args[0]}"`);
        return { kind: 'SquareForm', w };
    }
    if (name === 'prod') {
        const { boardType, boardArgs } = parseBoardTypeArgs('prod', args);
        return { kind: 'Prod', boardType, boardArgs };
    }
    if (name === 'beginprod') {
        const { boardType, boardArgs } = parseBoardTypeArgs('beginprod', args);
        return { kind: 'BeginProd', boardType, boardArgs };
    }
    if (name === 'endprod') {
        assert(args.length === 0, `endprod takes no arguments, got ${args.length}`);
        return { kind: 'EndProd' };
    }
    throw new Error(`Unknown board modifier: ${name}`);
}

/**
 * Applies `modifier` to `bc`, dispatching to `rectify` / `edgeSplit` / `mergeClose` /
 * `triangleForm` / `squareForm` / `product` (Prod builds a fresh board from its own boardType/boardArgs via
 * buildPrescribedBoard, then multiplies it into `bc`). Does NOT accept BeginProd/EndProd - those have no meaning applied to a
 * single board in isolation (BeginProd starts a whole new board for applyModifiers to build up
 * separately - potentially with further modifiers of its own before the product happens, unlike
 * Prod's one-shot immediate product - and EndProd's `product()` needs that suspended outer board
 * back too) - see applyModifiers, which handles both specially and is the only valid way to apply a
 * modifier list containing them.
 */
export function applyModifier(bc: BoardConfig, modifier: BoardModifier): BoardConfig {
    switch (modifier.kind) {
        case 'Rectify': return rectify(bc);
        case 'EdgeSplit': return edgeSplit(bc, modifier.splitN);
        case 'MergeClose': return mergeClose(bc, modifier.dist);
        case 'TriangleForm': return triangleForm(bc, modifier.w);
        case 'SquareForm': return squareForm(bc, modifier.w);
        case 'Prod': return product(bc, buildPrescribedBoard(modifier.boardType, modifier.boardArgs));
        case 'BeginProd':
        case 'EndProd':
            throw new Error(`applyModifier: ${modifier.kind} must be applied via applyModifiers, not directly`);
    }
}

/**
 * Applies every modifier in `modifiers`, in order, to `bc`. Most modifiers just transform the
 * "current" board via applyModifier, but BeginProd/EndProd (rejected by applyModifier itself - see
 * its doc comment) are handled specially here, via a stack of boards suspended to be multiplied back
 * in later:
 *   - BeginProd pushes the current board onto the stack and starts a fresh "current" board (built via
 *     buildPrescribedBoard from its boardType/boardArgs), so that modifiers up to the matching EndProd
 *     transform this new board instead of the outer one.
 *   - EndProd pops the suspended outer board and replaces "current" with `product(outer, current)` -
 *     the two multiplied together.
 * BeginProd/EndProd pairs may nest (a BeginProd inside another BeginProd...EndProd span just pushes a
 * second stack entry). Throws on an EndProd with no matching BeginProd (empty stack), or if the
 * modifier list ends with an unmatched BeginProd (non-empty stack).
 */
export function applyModifiers(bc: BoardConfig, modifiers: BoardModifier[]): BoardConfig {
    let current = bc;
    const stack: BoardConfig[] = [];
    for (const m of modifiers) {
        if (m.kind === 'BeginProd') {
            stack.push(current);
            current = buildPrescribedBoard(m.boardType, m.boardArgs);
        } else if (m.kind === 'EndProd') {
            assert(stack.length > 0, 'applyModifiers: endprod with no matching beginprod');
            const outer = stack.pop()!;
            current = product(outer, current);
        } else {
            current = applyModifier(current, m);
        }
    }
    assert(stack.length === 0, `applyModifiers: ${stack.length} unmatched beginprod(s)`);
    return current;
}