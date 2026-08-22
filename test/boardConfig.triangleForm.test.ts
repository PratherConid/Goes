// Regression tests for the triangleForm board modifier: replaces every found triangle with a
// side-length-w triangular board, gluing new corners back to the original vertices and gluing
// shared triangle edges together.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    triangleForm, triangularBoard, icosahedronBoard, dodecahedronBoard, parseModifier, applyModifier,
    Embedding, type BoardConfig,
} from '../shared/boardConfig.ts';
import { parseTriangleSelector } from '../shared/selector.ts';

function degreeSequence(adj: number[][]): number[] {
    return adj.map(row => row.reduce((s, v) => s + v, 0)).sort((a, b) => a - b);
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

test('w=2 is a topological no-op on any board (icosahedron: 20 triangles, every edge shared by 2)', () => {
    // quotientBoard's node numbering isn't guaranteed to preserve the original node order/labels,
    // so this checks node and edge counts (and the degree sequence) rather than exact adjacency -
    // still a strong structural check, just isomorphism-tolerant rather than identity-tolerant.
    const bc = icosahedronBoard();
    const result = triangleForm(bc, 2);
    const edgeCount = (adj: number[][]) => adj.flat().reduce((s: number, v: number) => s + v, 0) / 2;
    assert.equal(result.N, bc.N);
    assert.equal(edgeCount(result.adj), edgeCount(bc.adj));
    assert.deepEqual(degreeSequence(result.adj), degreeSequence(bc.adj));
});

test('w=1 collapses an isolated triangle (triangularBoard(2), a bare K3) into a single point', () => {
    const bc = triangularBoard(2);
    const result = triangleForm(bc, 1);
    assert.equal(result.N, 1);
    assert.deepEqual(result.adj, [[0]]);
});

test('an isolated triangle at w=3 reproduces triangularBoard(3) exactly (N, degree sequence, edge count)', () => {
    const bc = triangularBoard(2); // the one triangle IS the whole board, so no sharing/interference
    const result = triangleForm(bc, 3);
    const expected = triangularBoard(3);
    assert.equal(result.N, expected.N);
    assert.deepEqual(degreeSequence(result.adj), degreeSequence(expected.adj));
    const edgeCount = (adj: number[][]) => adj.flat().reduce((s: number, v: number) => s + v, 0) / 2;
    assert.equal(edgeCount(result.adj), edgeCount(expected.adj));
    assertSymmetricNoSelfLoops(result.adj);
    assertConnected(result.adj);
});

test('two triangles sharing one edge glue that edge\'s new midpoint into a single shared node', () => {
    // a,b,c and a,b,d: two triangles glued along edge a-b (a "diamond"/"kite"), no c-d edge.
    const adj = [
        [0, 1, 1, 1], // a: b, c, d
        [1, 0, 1, 1], // b: a, c, d
        [1, 1, 0, 0], // c: a, b
        [1, 1, 0, 0], // d: a, b
    ];
    const emb = new Embedding(2, [[0, 0], [1, 0], [0.5, 1], [0.5, -1]], [[1, 0], [0, 1], [0, 0]]);
    const bc: BoardConfig = { N: 4, adj, emb };
    const result = triangleForm(bc, 3);
    // 4 original + (2 non-shared midpoints per triangle: a-c, b-c, a-d, b-d) + (1 shared a-b midpoint) = 9.
    assert.equal(result.N, 9);
    assertSymmetricNoSelfLoops(result.adj);
    assertConnected(result.adj);
    // The shared a-b midpoint connects to a, b, and all 4 of the non-shared midpoints - degree 6 -
    // and is the unique node with that degree (every other new node has degree < 6 here).
    const degrees = result.adj.map(row => row.reduce((s, v) => s + v, 0));
    assert.equal(degrees.filter(d => d === 6).length, 1, 'exactly one node (the shared midpoint) has degree 6');
});

test('three triangles sharing one edge (a "book" of triangles) still glue that edge to a single node', () => {
    // a,b,c1 / a,b,c2 / a,b,c3: three triangles all sharing edge a-b, no ci-cj edges.
    const adj = [
        [0, 1, 1, 1, 1],
        [1, 0, 1, 1, 1],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
    ];
    const emb = new Embedding(
        2, [[0, 0], [1, 0], [0.5, 1], [0.5, -1], [0.5, 2]], [[1, 0], [0, 1], [0, 0]],
    );
    const bc: BoardConfig = { N: 5, adj, emb };
    const result = triangleForm(bc, 3);
    // 5 original + (2 non-shared midpoints per triangle x 3 triangles) + (1 shared a-b midpoint) = 12.
    assert.equal(result.N, 12);
    assertSymmetricNoSelfLoops(result.adj);
    assertConnected(result.adj);
    const degrees = result.adj.map(row => row.reduce((s, v) => s + v, 0));
    // The shared a-b midpoint connects to a, b, and all 6 non-shared midpoints (2 per triangle x 3) - degree 8.
    assert.equal(degrees.filter(d => d === 8).length, 1, 'exactly one node (the shared midpoint) has degree 8');
});

test('runs cleanly (no crash, stays connected/symmetric) on the dodecahedron, which has no triangles at all', () => {
    const bc = dodecahedronBoard();
    const result = triangleForm(bc, 3);
    assert.equal(result.N, bc.N, 'no triangles found, so nothing should change');
    assert.deepEqual(result.adj, bc.adj);
});

test('parseModifier("triform", ...) parses w and rejects malformed input', () => {
    assert.deepEqual(parseModifier('triform', ['3']), { kind: 'TriangleForm', w: 3 });
    assert.throws(() => parseModifier('triform', []));
    assert.throws(() => parseModifier('triform', ['0']));
    assert.throws(() => parseModifier('triform', ['abc']));
});

test('applyModifier("TriangleForm", ...) round-trips through the same result as calling triangleForm directly', () => {
    const bc = triangularBoard(2);
    const modifier = parseModifier('triform', ['3']);
    assert.deepEqual(applyModifier(bc, modifier), triangleForm(bc, 3));
});

test('an optional trailing triangle selector restricts triform to only the triangles it selects', () => {
    // _parseCommand (src/renderer.ts) splits the whole command line on whitespace before calling
    // parseModifier, so a selector's own internal parens/spaces arrive pre-split like this.
    const modifier = parseModifier('triform', ['3', '(conve', 'node', '(deg', 'eq', '3))']);
    assert.deepEqual(modifier, {
        kind: 'TriangleForm', w: 3, sel: parseTriangleSelector('(conve node (deg eq 3))'),
    });
    const bc = triangularBoard(2);
    const direct = triangleForm(bc, 3, parseTriangleSelector('(conve node (deg eq 3))'));
    assert.deepEqual(applyModifier(bc, modifier), direct);
});

test('sel restricts triangleForm to only the selected triangles - an unselected one is left ' +
    'untouched, even where it would otherwise have shared a glued corner/edge', () => {
    // Bowtie: triangles {0,1,2} and {2,3,4} sharing only vertex 2 (no shared edge - so an unselected
    // triangle here has none of its own sides consumed by the selected one), plus a pendant node 5
    // on node 0 alone, making node 0 the graph's unique degree-3 node.
    const adj = [
        [0, 1, 1, 0, 0, 1],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 0],
    ];
    const emb = new Embedding(2, adj.map((_, i): [number, number] => [i, 0]), [[1, 0], [0, 1], [0, 0]]);
    const bc: BoardConfig = { N: 6, adj, emb };
    // Selects only the triangle containing the degree-3 node (0) - triangle {0,1,2}, not {2,3,4}.
    const sel = parseTriangleSelector('(conve node (deg eq 3))');
    const result = triangleForm(bc, 3, sel);
    // All 6 original nodes survive (triangle {2,3,4} is untouched) plus triangularBoard(3)'s own 6
    // face nodes for the one selected triangle, minus 3 corners glued back to nodes 0/1/2 = 9.
    assert.equal(result.N, 9);
    assertSymmetricNoSelfLoops(result.adj);
    assertConnected(result.adj);
    const edgeCount = (a: number[][]) => a.flat().reduce((s, v) => s + v, 0) / 2;
    // Triangle {2,3,4}'s own 3 edges plus the 0-5 pendant edge (untouched) = 4, plus every edge of a
    // triangularBoard(3) for the one subdivided triangle (its 3 corners are nodes 0,1,2 themselves).
    assert.equal(edgeCount(result.adj), 4 + edgeCount(triangularBoard(3).adj));
    // Selecting nothing (an empty selector) is a total no-op - no triangle qualifies.
    const none = triangleForm(bc, 3, parseTriangleSelector('(none)'));
    assert.deepEqual(none.adj, bc.adj);
});
