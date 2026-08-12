/**
 * Graph-topology utilities operating on plain N×N adjacency matrices (the same representation as
 * `BoardConfig.adj`), independent of any board-specific geometry.
 */

/** An all-zero N×N adjacency matrix - the usual starting point before filling in edges. */
export function zeroAdj(N: number): number[][] {
    return Array.from({ length: N }, () => new Array<number>(N).fill(0));
}

/** An adjacency-list view of a graph: `list[i]` is the set of `i`'s neighbors. */
export type AdjacencyList = Set<number>[];

/** Converts an N×N adjacency matrix into an adjacency list, each node's neighbors stored as a
 * `Set` (not an array) so membership checks - the hot path for both findTriangles/findSquares
 * below - are O(1) instead of O(degree). */
export function toAdjacencyList(adj: number[][]): AdjacencyList {
    const N = adj.length;
    const list: AdjacencyList = Array.from({ length: N }, () => new Set<number>());
    for (let i = 0; i < N; i++)
        for (let j = 0; j < N; j++)
            if (adj[i][j]) list[i].add(j);
    return list;
}

/**
 * Finds every triangle (3 distinct, pairwise-adjacent vertices) in `adj`, each reported exactly
 * once as `[u, v, w]` with `u < v < w`. Converts to an adjacency list first (see toAdjacencyList),
 * then for each vertex `u` only looks at neighbors `v > u`, and for each such `v` only looks at
 * neighbors `w > v` of `v`, checking whether `w` is also a neighbor of `u` via an O(1) set lookup -
 * fixing this increasing order is what guarantees each triangle is found exactly once (via its
 * unique u < v < w labeling), with no separate deduplication pass needed.
 */
export function findTriangles(adj: number[][]): [number, number, number][] {
    const N = adj.length;
    const adjList = toAdjacencyList(adj);
    const triangles: [number, number, number][] = [];
    for (let u = 0; u < N; u++)
        for (const v of adjList[u]) {
            if (v <= u) continue;
            for (const w of adjList[v]) {
                if (w <= v) continue;
                if (adjList[u].has(w)) triangles.push([u, v, w]);
            }
        }
    return triangles;
}

/**
 * Finds every "square" - 4 distinct vertices `a, b, c, d` forming a cycle `a-b-c-d-a` (all 4 cycle
 * edges present) whose two diagonals `a-c` and `b-d` are BOTH absent (a proper induced 4-cycle, not
 * merely 4 vertices of a denser subgraph that happens to contain one) - each reported exactly once
 * as `[a, b, c, d]` in that cycle order.
 *
 * Converts to an adjacency list first (see toAdjacencyList), then for every non-adjacent pair
 * `(p, q)` with `p < q` (a candidate diagonal), finds their common neighbors and, for every pair of
 * common neighbors `(r, s)` that are themselves non-adjacent (the other candidate diagonal), reports
 * the square `p-r-q-s-p`. A square has exactly two diagonals, so this raw scan finds each one twice
 * - once starting from each diagonal - which is resolved by only emitting when `(p, q)` is the
 * lexicographically smaller of the two (`p < min(r, s)`; the two diagonals can never share a vertex,
 * since all 4 square vertices are distinct, so this comparison is never ambiguous).
 */
export function findSquares(adj: number[][]): [number, number, number, number][] {
    const N = adj.length;
    const adjList = toAdjacencyList(adj);
    const squares: [number, number, number, number][] = [];
    for (let p = 0; p < N; p++)
        for (let q = p + 1; q < N; q++) {
            if (adjList[p].has(q)) continue; // p-q would be an edge, not a diagonal
            const common: number[] = [];
            for (const x of adjList[p]) if (adjList[q].has(x)) common.push(x);
            for (let i = 0; i < common.length; i++)
                for (let j = i + 1; j < common.length; j++) {
                    const r = Math.min(common[i], common[j]);
                    const s = Math.max(common[i], common[j]);
                    if (adjList[r].has(s)) continue; // r-s would be an edge, not a diagonal
                    if (p < r) squares.push([p, r, q, s]);
                }
        }
    return squares;
}

/**
 * Union-Find: given `N` nodes and a list of pairs to merge, returns each node's equivalence-class
 * index, compressed to a dense `0..M-1` range (`M` = number of distinct classes) in ascending order
 * of each class's lowest original member. Internal to mergeBoards() (below) - resolves every merge
 * instruction at once, so a chain like `(0,3)~(1,5)` and `(1,5)~(2,7)` correctly collapses `(0,3)`
 * and `(2,7)` into the same node too, even though no single instruction names both directly.
 */
function unionFindClasses(N: number, pairs: [number, number][]): number[] {
    const parent = Array.from({ length: N }, (_, i) => i);
    function find(x: number): number {
        while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    }
    for (const [a, b] of pairs) {
        const pa = find(a), pb = find(b);
        if (pa !== pb) parent[pa] = pb;
    }
    const roots = Array.from({ length: N }, (_, i) => find(i));
    const uniqueRoots = [...new Set(roots)].sort((a, b) => a - b);
    const rootToNew = new Map(uniqueRoots.map((r, i) => [r, i]));
    return roots.map(r => rootToNew.get(r)!);
}

/**
 * Combines a list of boards into one, additionally identifying every `([b1, i1], [b2, i2])` pair in
 * `merges` (board index, that board's own local node index) as the same node - every merge is
 * resolved in one batch via unionFindClasses(), not board-by-board, so callers never need to fold
 * boards in one at a time just to keep every merge target "already placed": a merge between two
 * boards that haven't been introduced to each other by any other merge is handled exactly like any
 * other. The merged node keeps whichever input position is encountered first (callers are expected
 * to only merge pairs whose positions already coincide, e.g. two recursive `sierpinskiSimplex`
 * sub-copies sharing a corner). Returns the combined board plus, for each input board (in the same
 * order as `boards`), a map from that board's own local indices to its final index in the combined
 * board - needed by callers that must keep tracking specific nodes (like sierpinskiSimplex's own
 * outer corners) across further merges. `pos` here is opaque per-node data (typically a real
 * position, but never inspected as one) simply carried along through the same merge as `adj`.
 */
export function mergeBoards(
    boards: { pos: number[][]; adj: number[][] }[], merges: [[number, number], [number, number]][],
): { pos: number[][]; adj: number[][]; maps: number[][] } {
    const offset: number[] = new Array(boards.length).fill(0);
    for (let i = 1; i < boards.length; i++) offset[i] = offset[i - 1] + boards[i - 1].pos.length;
    const total = boards.reduce((s, b) => s + b.pos.length, 0);
    const g = (b: number, local: number) => offset[b] + local;

    const nodeToNew = unionFindClasses(total, merges.map(([[b1, i1], [b2, i2]]) => [g(b1, i1), g(b2, i2)]));
    const newN = total === 0 ? 0 : Math.max(...nodeToNew) + 1;

    const pos: number[][] = new Array(newN);
    for (let b = 0; b < boards.length; b++)
        for (let local = 0; local < boards[b].pos.length; local++)
            pos[nodeToNew[g(b, local)]] = boards[b].pos[local];

    const adj = zeroAdj(newN);
    for (let b = 0; b < boards.length; b++) {
        const board = boards[b];
        for (let i = 0; i < board.pos.length; i++)
            for (let j = 0; j < board.pos.length; j++)
                if (board.adj[i][j]) adj[nodeToNew[g(b, i)]][nodeToNew[g(b, j)]] = 1;
    }

    const maps = boards.map((board, b) => board.pos.map((_, local) => nodeToNew[g(b, local)]));
    return { pos, adj, maps };
}
