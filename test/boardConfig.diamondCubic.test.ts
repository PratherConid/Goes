// Regression tests for diamondCubicBoard: a diamond cubic lattice (uniform tetrahedral
// 4-coordination) shaped like a regular tetrahedron of side length w, built directly from
// barycentric coordinates (one hub per "up" unit tetrahedron, connected to its own 4 corners - see
// its own doc comment, shared/boardConfig.ts, for why "down" tetrahedra need no separate handling:
// every original edge already belongs to some up-tetrahedron, so none survive either way).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { diamondCubicBoard } from '../shared/boardConfig.ts';

function edgeCount(adj: number[][]): number {
    return adj.flat().reduce((s, v) => s + v, 0) / 2;
}

function degrees(adj: number[][]): number[] {
    return adj.map(row => row.reduce((s, v) => s + v, 0));
}

// C(w+1, 3) = (w-1)*w*(w+1)/6 - the number of up-tetrahedra (barycentric points summing to
// (w - 1) - 1 = w - 2, each with 4 coordinates), each contributing exactly one hub and 4 edges.
function expectedHubCount(w: number): number {
    return ((w - 1) * w * (w + 1)) / 6;
}

// C(n+3, 3) with n = w - 1 - the number of barycentric-coordinate lattice points summing to n.
function latticePointCount(w: number): number {
    const n = w - 1;
    if (n < 0) return 0;
    return ((n + 1) * (n + 2) * (n + 3)) / 6;
}

test('w=1 is a single isolated point', () => {
    const bc = diamondCubicBoard(1);
    assert.equal(bc.N, 1);
    assert.equal(edgeCount(bc.adj), 0);
});

test('w=2 is a bare 4-armed star: the whole tetrahedron centered with no surviving original edges', () => {
    const bc = diamondCubicBoard(2);
    assert.equal(bc.N, 5); // 4 original corners + 1 hub
    assert.equal(edgeCount(bc.adj), 4);
    const deg = degrees(bc.adj);
    assert.deepEqual([...deg].sort((a, b) => a - b), [1, 1, 1, 1, 4]);
});

test('every node has degree <= 4 (true diamond coordination, never more) across a range of w', () => {
    for (let w = 1; w <= 12; w++) {
        const bc = diamondCubicBoard(w);
        for (const d of degrees(bc.adj)) assert.ok(d <= 4, `w=${w}: found degree ${d} > 4`);
    }
});

test('no original lattice edge survives - every edge is a hub-to-corner bond, so edge count is ' +
    'exactly 4 times the hub count', () => {
    for (let w = 1; w <= 12; w++) {
        const bc = diamondCubicBoard(w);
        const hubs = expectedHubCount(w);
        assert.equal(bc.N, hubs + latticePointCount(w), `w=${w}: node count`);
        assert.equal(edgeCount(bc.adj), hubs * 4, `w=${w}: edge count`);
    }
});

test('every hub has degree exactly 4', () => {
    const w = 9;
    const bc = diamondCubicBoard(w);
    const hubs = expectedHubCount(w);
    const origN = bc.N - hubs;
    const deg = degrees(bc.adj);
    for (let i = origN; i < bc.N; i++) assert.equal(deg[i], 4, `hub ${i} has degree ${deg[i]}`);
});

test('the board is connected', () => {
    const bc = diamondCubicBoard(9);
    const seen = new Set<number>([0]);
    const queue = [0];
    while (queue.length > 0) {
        const cur = queue.pop()!;
        for (let j = 0; j < bc.N; j++)
            if (bc.adj[cur][j] && !seen.has(j)) { seen.add(j); queue.push(j); }
    }
    assert.equal(seen.size, bc.N);
});

test('rejects a non-positive/non-integer w', () => {
    assert.throws(() => diamondCubicBoard(0), /w must be a positive integer/);
    assert.throws(() => diamondCubicBoard(-1), /w must be a positive integer/);
    assert.throws(() => diamondCubicBoard(2.5), /w must be a positive integer/);
});
