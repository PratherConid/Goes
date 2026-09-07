// Regression tests for genericForm and the "form" board modifier: the generalization of
// triangleForm/quadForm/quadDiagForm to an arbitrary list of FormSelectors (each a TriForm-,
// QuadForm-, or QuadDiagForm-kind selected face - see genericForm's own doc comment and
// FormSelector's own, shared/types.ts), gluing shared ORIGINAL edges across every selected face
// regardless of kind.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    genericForm, triangleForm, quadForm, quadDiagForm, triangularBoard, rectangularBoard, applyModifier,
} from '../shared/boardConfig.ts';
import { Embedding, type BoardConfig, type FormSelector, type Selector } from '../shared/types.ts';
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

test('genericForm with a single TriForm selector is identical to triangleForm', () => {
    const bc = triangularBoard(2);
    assert.deepEqual(genericForm(bc, 3, [{ kind: 'TriForm' }]), triangleForm(bc, 3));
});

test('genericForm with a single QuadForm selector is identical to quadForm', () => {
    const bc = rectangularBoard(2, 2);
    assert.deepEqual(genericForm(bc, 3, [{ kind: 'QuadForm' }]), quadForm(bc, 3));
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

    const triOnly = genericForm(bc, 3, [{ kind: 'TriForm', sel: parseTriangleSelector('(all tri)') }]);
    const quadOnly = genericForm(bc, 3, [{ kind: 'QuadForm', sel: parseQuadSelector('(all quad)') }]);
    const both = genericForm(bc, 3, [
        { kind: 'TriForm', sel: parseTriangleSelector('(all tri)') },
        { kind: 'QuadForm', sel: parseQuadSelector('(all quad)') },
    ]);
    assertSymmetricNoSelfLoops(both.adj);
    assertConnected(both.adj);
    // If the shared edge (1,2) weren't glued across kinds, `both` would just be the disjoint union
    // of triOnly and quadOnly's own new nodes (minus the N originals double-counted) - one node fewer
    // than that proves the shared edge's own midpoint (w=3 -> 1 interior boundary point) was merged
    // into a single node instead of being duplicated once per face.
    assert.equal(both.N, triOnly.N + quadOnly.N - N - 1);
});

test('quadDiagForm has w*w + (w-1)*(w-1) nodes per quad, only diagonal edges', () => {
    const bc = rectangularBoard(2, 2);
    for (const w of [1, 2, 3, 4]) {
        const result = quadDiagForm(bc, w);
        assert.equal(result.N, w * w + (w - 1) * (w - 1));
        assertSymmetricNoSelfLoops(result.adj);
        assertConnected(result.adj);
        // Every new node beyond bc.N is either a "primary" (w*w of them, degree 4 in the interior,
        // fewer on the boundary since some primaries are original quad corners with only 1 diagonal
        // neighbor) or "center" node (degree exactly 4, since a center is never on the boundary) -
        // edgeCount below should match 4*(w-1)*(w-1) (each center contributes exactly 4 edges, and no
        // primary-primary or center-center edge exists to double-count).
        assert.equal(edgeCount(result.adj), 4 * (w - 1) * (w - 1));
    }
});

test('quadDiagForm w=1 collapses the quad to a single point', () => {
    const bc = rectangularBoard(2, 2);
    const result = quadDiagForm(bc, 1);
    assert.equal(result.N, 1);
});

test('applyModifier("QuadDiagForm", ...) matches calling quadDiagForm directly', () => {
    const bc = rectangularBoard(2, 2);
    assert.deepEqual(applyModifier(bc, { kind: 'QuadDiagForm', w: 3 }), quadDiagForm(bc, 3));
});

test('two quadDiagForm quads sharing an edge glue seamlessly', () => {
    // Two quads (0-1-2-3 and 1-4-5-2) sharing edge (1,2).
    const N = 6;
    const adj: number[][] = Array.from({ length: N }, () => new Array(N).fill(0));
    const edge = (i: number, j: number) => { adj[i][j] = 1; adj[j][i] = 1; };
    edge(0, 1); edge(1, 2); edge(2, 3); edge(3, 0);
    edge(1, 4); edge(4, 5); edge(5, 2);
    const emb = new Embedding(2, adj.map((_, i): [number, number] => [i, 0]));
    const bc: BoardConfig = { N, adj, emb };

    const quad1: Selector =
        { op: 'raw', type: 'quad', items: { kind: 'quad', value: [{ n1: 0, n2: 1, n3: 2, n4: 3 }] } };
    const quad2: Selector =
        { op: 'raw', type: 'quad', items: { kind: 'quad', value: [{ n1: 1, n2: 4, n3: 5, n4: 2 }] } };
    for (const w of [2, 3, 4]) {
        const only1 = genericForm(bc, w, [{ kind: 'QuadDiagForm', sel: quad1 }]);
        const only2 = genericForm(bc, w, [{ kind: 'QuadDiagForm', sel: quad2 }]);
        const both = genericForm(bc, w, [{ kind: 'QuadDiagForm', sel: parseQuadSelector('(all quad)') }]);
        assertSymmetricNoSelfLoops(both.adj);
        assertConnected(both.adj);
        // Corner quotienting alone accounts for only1.N + only2.N - N; the shared edge's own w-2
        // interior boundary points (k=1..w-2 - the endpoints k=0/k=w-1 are corners, already merged)
        // are what's left to glue - if they weren't, `both` would have one extra node per interior
        // point instead of sharing it.
        assert.equal(both.N, only1.N + only2.N - N - Math.max(w - 2, 0));
    }
});

test('an empty sels list is a total no-op', () => {
    const bc = triangularBoard(2);
    const result = genericForm(bc, 3, []);
    assert.deepEqual(result.adj, bc.adj);
});

test('applyModifier("Form", ...) matches calling genericForm directly', () => {
    const bc = triangularBoard(2);
    const sels: FormSelector[] = [{ kind: 'TriForm' }];
    assert.deepEqual(applyModifier(bc, { kind: 'Form', w: 3, sels }), genericForm(bc, 3, sels));
});
