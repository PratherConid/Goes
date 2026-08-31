// Regression tests for genericForm and the "form" board modifier: the generalization of
// triangleForm/quadForm to an arbitrary list of selectors (each a tri- or quad-typed Selector,
// checked at runtime - see genericForm's own doc comment), gluing shared ORIGINAL edges across
// every selected face regardless of kind.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    genericForm, triangleForm, quadForm, triangularBoard, rectangularBoard, applyModifier,
} from '../shared/boardConfig.ts';
import { Embedding, type BoardConfig } from '../shared/types.ts';
import { parseTriangleSelector, parseQuadSelector } from '../shared/selector.ts';

function edgeCount(adj: number[][]): number {
    return adj.flat().reduce((s, v) => s + v, 0) / 2;
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

test('genericForm with a single (all tri) selector is identical to triangleForm', () => {
    const bc = triangularBoard(2);
    assert.deepEqual(genericForm(bc, 3, [{ op: 'all', type: 'tri' }]), triangleForm(bc, 3));
});

test('genericForm with a single (all quad) selector is identical to quadForm', () => {
    const bc = rectangularBoard(2, 2);
    assert.deepEqual(genericForm(bc, 3, [{ op: 'all', type: 'quad' }]), quadForm(bc, 3));
});

test('genericForm rejects a node/edge selector in sels at runtime', () => {
    const bc = triangularBoard(2);
    assert.throws(
        () => genericForm(bc, 3, [{ op: 'all', type: 'node' }]),
        /must be a triangle or quad selector, got a node selector/);
});

test('a triangle and a quad sharing an edge glue seamlessly across kinds', () => {
    // Triangle 0-1-2 and quad (cycle) 1-3-4-2 share edge (1,2).
    const N = 5;
    const adj: number[][] = Array.from({ length: N }, () => new Array(N).fill(0));
    const edge = (i: number, j: number) => { adj[i][j] = 1; adj[j][i] = 1; };
    edge(0, 1); edge(1, 2); edge(2, 0);
    edge(1, 3); edge(3, 4); edge(4, 2);
    const emb = new Embedding(2, adj.map((_, i): [number, number] => [i, 0]));
    const bc: BoardConfig = { N, adj, emb };

    const triOnly = genericForm(bc, 3, [parseTriangleSelector('(all tri)')]);
    const quadOnly = genericForm(bc, 3, [parseQuadSelector('(all quad)')]);
    const both = genericForm(bc, 3, [parseTriangleSelector('(all tri)'), parseQuadSelector('(all quad)')]);
    assertSymmetricNoSelfLoops(both.adj);
    assertConnected(both.adj);
    // If the shared edge (1,2) weren't glued across kinds, `both` would just be the disjoint union
    // of triOnly and quadOnly's own new nodes (minus the N originals double-counted) - one node fewer
    // than that proves the shared edge's own midpoint (w=3 -> 1 interior boundary point) was merged
    // into a single node instead of being duplicated once per face.
    assert.equal(both.N, triOnly.N + quadOnly.N - N - 1);
});

test('an empty sels list is a total no-op', () => {
    const bc = triangularBoard(2);
    const result = genericForm(bc, 3, []);
    assert.deepEqual(result.adj, bc.adj);
});

test('applyModifier("Form", ...) matches calling genericForm directly', () => {
    const bc = triangularBoard(2);
    const sels = [{ op: 'all' as const, type: 'tri' as const }];
    assert.deepEqual(applyModifier(bc, { kind: 'Form', w: 3, sels }), genericForm(bc, 3, sels));
});
