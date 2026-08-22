// Regression tests for genericForm and the "form" board modifier: the generalization of
// triangleForm/squareForm to an arbitrary list of FormSelectors (each naming a kind - 'tri' or 'sq'
// - plus an optional inner selector), gluing shared ORIGINAL edges across every selected face
// regardless of kind.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    genericForm, triangleForm, squareForm, triangularBoard, rectangularBoard, parseModifier,
    applyModifier, Embedding, type BoardConfig,
} from '../shared/boardConfig.ts';
import { parseFormSelectors, formatFormSelectors, parseTriangleSelector } from '../shared/selector.ts';

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

test('genericForm with a single (tri) selector is identical to triangleForm', () => {
    const bc = triangularBoard(2);
    assert.deepEqual(genericForm(bc, 3, [{ kind: 'tri' }]), triangleForm(bc, 3));
});

test('genericForm with a single (sq) selector is identical to squareForm', () => {
    const bc = rectangularBoard(2, 2);
    assert.deepEqual(genericForm(bc, 3, [{ kind: 'sq' }]), squareForm(bc, 3));
});

test('a triangle and a square sharing an edge glue seamlessly across kinds', () => {
    // Triangle 0-1-2 and square (cycle) 1-3-4-2 share edge (1,2).
    const N = 5;
    const adj: number[][] = Array.from({ length: N }, () => new Array(N).fill(0));
    const edge = (i: number, j: number) => { adj[i][j] = 1; adj[j][i] = 1; };
    edge(0, 1); edge(1, 2); edge(2, 0);
    edge(1, 3); edge(3, 4); edge(4, 2);
    const emb = new Embedding(2, adj.map((_, i): [number, number] => [i, 0]), [[1, 0], [0, 1], [0, 0]]);
    const bc: BoardConfig = { N, adj, emb };

    const triOnly = genericForm(bc, 3, parseFormSelectors('(tri)'));
    const sqOnly = genericForm(bc, 3, parseFormSelectors('(sq)'));
    const both = genericForm(bc, 3, parseFormSelectors('(tri) (sq)'));
    assertSymmetricNoSelfLoops(both.adj);
    assertConnected(both.adj);
    // If the shared edge (1,2) weren't glued across kinds, `both` would just be the disjoint union
    // of triOnly and sqOnly's own new nodes (minus the N originals double-counted) - one node fewer
    // than that proves the shared edge's own midpoint (w=3 -> 1 interior boundary point) was merged
    // into a single node instead of being duplicated once per face.
    assert.equal(both.N, triOnly.N + sqOnly.N - N - 1);
});

test('an empty sels list is a total no-op', () => {
    const bc = triangularBoard(2);
    const result = genericForm(bc, 3, []);
    assert.deepEqual(result.adj, bc.adj);
});

test('parseModifier("form", ...) parses w and one or more form selectors, rejecting malformed input', () => {
    assert.deepEqual(parseModifier('form', ['3', '(tri)']), { kind: 'Form', w: 3, sels: [{ kind: 'tri' }] });
    assert.deepEqual(
        parseModifier('form', ['4', '(tri)', '(sq)']),
        { kind: 'Form', w: 4, sels: [{ kind: 'tri' }, { kind: 'sq' }] },
    );
    assert.deepEqual(
        parseModifier('form', ['4', '(tri', '(conve', 'node', '(deg', 'gt', '1)))']),
        { kind: 'Form', w: 4, sels: [{ kind: 'tri', sel: parseTriangleSelector('(conve node (deg gt 1))') }] },
    );
    assert.throws(() => parseModifier('form', []));
    assert.throws(() => parseModifier('form', ['0', '(tri)']));
    assert.throws(() => parseModifier('form', ['abc', '(tri)']));
    assert.throws(() => parseModifier('form', ['3']), /at least 1 form selector/);
    assert.throws(() => parseModifier('form', ['3', '(rect)']), /expected 'tri' or 'sq'/);
});

test('applyModifier("Form", ...) round-trips through the same result as calling genericForm directly', () => {
    const bc = triangularBoard(2);
    const modifier = parseModifier('form', ['3', '(tri)']);
    assert.deepEqual(applyModifier(bc, modifier), genericForm(bc, 3, [{ kind: 'tri' }]));
});

test('formatFormSelectors round-trips parseFormSelectors', () => {
    const text = '(tri (conve node (deg gt 1))) (sq)';
    assert.equal(formatFormSelectors(parseFormSelectors(text)), text);
    assert.equal(formatFormSelectors(parseFormSelectors('')), '');
});
