// Shared graph assertions/measurements for the board-construction tests (not a test file itself).
// Everything here works on a plain adjacency matrix, so it applies equally to a BoardConfig's own
// `adj` and to the hand-built matrices some tests construct directly.
import assert from 'node:assert/strict';
import { Embedding, type BoardConfig } from '../shared/types.ts';

export function edgeCount(adj: number[][]): number {
    return adj.flat().reduce((s, v) => s + v, 0) / 2;
}

export function degrees(adj: number[][]): number[] {
    return adj.map(row => row.reduce((s, v) => s + v, 0));
}

export function degree(adj: number[][], i: number): number {
    return adj[i].reduce((s, v) => s + v, 0);
}

// Degrees sorted ascending - an isomorphism-tolerant fingerprint, for comparing two boards whose
// node numbering isn't guaranteed to line up (e.g. anything that went through quotientBoard).
export function degreeSequence(adj: number[][]): number[] {
    return degrees(adj).sort((a, b) => a - b);
}

export function assertSymmetricNoSelfLoops(adj: number[][]) {
    for (let i = 0; i < adj.length; i++) {
        assert.equal(adj[i][i], 0, `self-loop at ${i}`);
        for (let j = 0; j < adj.length; j++) assert.equal(adj[i][j], adj[j][i], `asymmetric at ${i},${j}`);
    }
}

export function assertConnected(adj: number[][]) {
    const N = adj.length;
    const seen = new Set([0]);
    const stack = [0];
    while (stack.length) {
        const v = stack.pop()!;
        for (let j = 0; j < N; j++) if (adj[v][j] && !seen.has(j)) { seen.add(j); stack.push(j); }
    }
    assert.equal(seen.size, N, 'graph should stay fully connected');
}

// A BoardConfig with `N` nodes and exactly the given undirected edges. `pos` defaults to a
// degenerate line embedding, which is enough for any modifier whose result is judged purely on
// graph shape; pass a real one where the test also cares about where new nodes land.
export function boardFromEdges(N: number, edges: [number, number][], pos?: number[][]): BoardConfig {
    const adj: number[][] = Array.from({ length: N }, () => new Array(N).fill(0));
    for (const [i, j] of edges) { adj[i][j] = 1; adj[j][i] = 1; }
    return { N, adj, emb: new Embedding(2, pos ?? adj.map((_, i) => [i, 0])) };
}
