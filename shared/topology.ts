/**
 * Graph-topology utilities operating on plain N×N adjacency matrices (the same representation as
 * `BoardConfig.adj`), independent of any board-specific geometry.
 */

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
