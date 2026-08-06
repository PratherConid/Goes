// Regression tests for the squareForm board modifier: replaces every found square with a w-by-w
// grid, gluing new corners back to the original vertices and gluing shared square edges together.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    squareForm, rectangularBoard, cubeLatticeBoard, triangularBoard, parseModifier, applyModifier,
    Embedding, type BoardConfig,
} from '../shared/boardConfig.ts';

function degreeSequence(adj: number[][]): number[] {
    return adj.map(row => row.reduce((s, v) => s + v, 0)).sort((a, b) => a - b);
}

function edgeCount(adj: number[][]): number {
    return adj.flat().reduce((s: number, v: number) => s + v, 0) / 2;
}

function assertSymmetricNoSelfLoops(adj: number[][]) {
    for (let i = 0; i < adj.length; i++) {
        assert.equal(adj[i][i], 0, `self-loop at ${i}`);
        for (let j = 0; j < adj.length; j++) assert.equal(adj[i][j], adj[j][i], `asymmetric at ${i},${j}`);
    }
}

function assertConnected(adj: number[][]) {
    const N = adj.length;
    const seen = new Set([0]);
    const stack = [0];
    while (stack.length) {
        const v = stack.pop()!;
        for (let j = 0; j < N; j++) if (adj[v][j] && !seen.has(j)) { seen.add(j); stack.push(j); }
    }
    assert.equal(seen.size, N, 'graph should stay fully connected');
}

test('w=2 is a topological no-op on any board (cube: 6 square faces, every edge shared by 2)', () => {
    // quotientBoard's node numbering isn't guaranteed to preserve the original node order/labels,
    // so this checks node and edge counts (and the degree sequence) rather than exact adjacency -
    // still a strong structural check, just isomorphism-tolerant rather than identity-tolerant.
    const bc = cubeLatticeBoard(2, 2, 2);
    const result = squareForm(bc, 2);
    assert.equal(result.N, bc.N);
    assert.equal(edgeCount(result.adj), edgeCount(bc.adj));
    assert.deepEqual(degreeSequence(result.adj), degreeSequence(bc.adj));
});

test('w=1 collapses an isolated square (rectangularBoard(2,2), a bare 4-cycle) into a single point', () => {
    const bc = rectangularBoard(2, 2);
    const result = squareForm(bc, 1);
    assert.equal(result.N, 1);
    assert.deepEqual(result.adj, [[0]]);
});

test('an isolated square at w=3 reproduces rectangularBoard(3,3) exactly (N, degree sequence, edge count)', () => {
    const bc = rectangularBoard(2, 2); // the one square IS the whole board, so no sharing/interference
    const result = squareForm(bc, 3);
    const expected = rectangularBoard(3, 3);
    assert.equal(result.N, expected.N);
    assert.deepEqual(degreeSequence(result.adj), degreeSequence(expected.adj));
    assert.equal(edgeCount(result.adj), edgeCount(expected.adj));
    assertSymmetricNoSelfLoops(result.adj);
    assertConnected(result.adj);
});

test('two squares sharing one edge glue that edge\'s new boundary nodes into shared nodes', () => {
    // a,b,e,f and a,b,g,h: two squares glued along edge a-b, no e-g/f-h/etc edges.
    const adj = [
        [0, 1, 0, 1, 0, 1], // a: b, f, h
        [1, 0, 1, 0, 1, 0], // b: a, e, g
        [0, 1, 0, 1, 0, 0], // e: b, f
        [1, 0, 1, 0, 0, 0], // f: a, e
        [0, 1, 0, 0, 0, 1], // g: b, h
        [1, 0, 0, 0, 1, 0], // h: a, g
    ];
    const pos = [[0, 0], [1, 0], [1.5, 1], [0.5, 1], [1.5, -1], [0.5, -1]];
    const emb = new Embedding(2, pos, [[1, 0], [0, 1], [0, 0]]);
    const bc: BoardConfig = { N: 6, adj, emb };
    const result = squareForm(bc, 3);
    // 6 original + (4 non-shared boundary/interior nodes per square) + (1 shared a-b midpoint) = 15.
    assert.equal(result.N, 15);
    assert.equal(edgeCount(result.adj), 22);
    assertSymmetricNoSelfLoops(result.adj);
    assertConnected(result.adj);
});

test('three squares sharing one edge (a "book" of squares) still glue that edge to shared nodes', () => {
    // a,b,e,f / a,b,g,h / a,b,i,j: three squares all sharing edge a-b, no cross edges between them.
    const N = 8;
    const adj: number[][] = Array.from({ length: N }, () => new Array(N).fill(0));
    const edge = (i: number, j: number) => { adj[i][j] = 1; adj[j][i] = 1; };
    const a = 0, b = 1, e = 2, f = 3, g = 4, h = 5, i = 6, j = 7;
    edge(a, b); edge(b, e); edge(e, f); edge(f, a);
    edge(b, g); edge(g, h); edge(h, a);
    edge(b, i); edge(i, j); edge(j, a);
    const pos = [[0, 0], [1, 0], [1.5, 1], [0.5, 1], [1.5, -1], [0.5, -1], [1.5, 2], [0.5, 2]];
    const emb = new Embedding(2, pos, [[1, 0], [0, 1], [0, 0]]);
    const bc: BoardConfig = { N, adj, emb };
    const result = squareForm(bc, 3);
    // 8 original + (4 non-shared nodes per square x 3 squares) + (1 shared a-b midpoint) = 21.
    assert.equal(result.N, 21);
    assert.equal(edgeCount(result.adj), 32);
    assertSymmetricNoSelfLoops(result.adj);
    assertConnected(result.adj);
    const degrees = result.adj.map(row => row.reduce((s, v) => s + v, 0));
    // The shared a-b midpoint connects to a, b, and all 3 squares' centers - degree 5 - and is the
    // unique node with that degree (every square's own center only reaches degree 4).
    assert.equal(degrees.filter(d => d === 5).length, 1, 'exactly one node (the shared midpoint) has degree 5');
});

test('runs cleanly (no crash, stays connected/symmetric) on a triangle board, which has no squares at all', () => {
    const bc = triangularBoard(2); // only 3 nodes - too few for any 4-cycle to exist
    const result = squareForm(bc, 3);
    assert.equal(result.N, bc.N, 'no squares found, so nothing should change');
    assert.deepEqual(result.adj, bc.adj);
});

test('parseModifier("sqform", ...) parses w and rejects malformed input', () => {
    assert.deepEqual(parseModifier('sqform', ['3']), { kind: 'SquareForm', w: 3 });
    assert.throws(() => parseModifier('sqform', []));
    assert.throws(() => parseModifier('sqform', ['0']));
    assert.throws(() => parseModifier('sqform', ['abc']));
});

test('applyModifier("SquareForm", ...) round-trips through the same result as calling squareForm directly', () => {
    const bc = rectangularBoard(2, 2);
    const modifier = parseModifier('sqform', ['3']);
    assert.deepEqual(applyModifier(bc, modifier), squareForm(bc, 3));
});
