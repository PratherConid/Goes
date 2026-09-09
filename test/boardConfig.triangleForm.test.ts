// Regression tests for the triangleForm board modifier: replaces every found triangle with a
// side-length-w triangular board, gluing new corners back to the original vertices and gluing
// shared triangle edges together.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    triangleForm, triangularBoard, icosahedronBoard, dodecahedronBoard, applyModifier,
} from '../shared/boardConfig.ts';
import { parseTriangleSelector } from '../shared/selector.ts';
import {
    edgeCount, degrees, degreeSequence, assertSymmetricNoSelfLoops, assertConnected, boardFromEdges,
} from './graphHelpers.ts';

test('w=2 is a topological no-op on any board (icosahedron: 20 triangles, every edge shared by 2)', () => {
    // quotientBoard's node numbering isn't guaranteed to preserve the original node order/labels,
    // so this checks node and edge counts (and the degree sequence) rather than exact adjacency -
    // still a strong structural check, just isomorphism-tolerant rather than identity-tolerant.
    const bc = icosahedronBoard();
    const result = triangleForm(bc, 2);
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
    assert.equal(edgeCount(result.adj), edgeCount(expected.adj));
    assertSymmetricNoSelfLoops(result.adj);
    assertConnected(result.adj);
});

test('two triangles sharing one edge glue that edge\'s new midpoint into a single shared node', () => {
    // a,b,c and a,b,d: two triangles glued along edge a-b (a "diamond"/"kite"), no c-d edge.
    const bc = boardFromEdges(
        4, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3]], [[0, 0], [1, 0], [0.5, 1], [0.5, -1]],
    );
    const result = triangleForm(bc, 3);
    // 4 original + (2 non-shared midpoints per triangle: a-c, b-c, a-d, b-d) + (1 shared a-b midpoint) = 9.
    assert.equal(result.N, 9);
    assertSymmetricNoSelfLoops(result.adj);
    assertConnected(result.adj);
    // The shared a-b midpoint connects to a, b, and all 4 of the non-shared midpoints - degree 6 -
    // and is the unique node with that degree (every other new node has degree < 6 here).
    assert.equal(degrees(result.adj).filter(d => d === 6).length, 1,
        'exactly one node (the shared midpoint) has degree 6');
});

test('three triangles sharing one edge (a "book" of triangles) still glue that edge to a single node', () => {
    // a,b,c1 / a,b,c2 / a,b,c3: three triangles all sharing edge a-b, no ci-cj edges.
    const bc = boardFromEdges(
        5, [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4]],
        [[0, 0], [1, 0], [0.5, 1], [0.5, -1], [0.5, 2]],
    );
    const result = triangleForm(bc, 3);
    // 5 original + (2 non-shared midpoints per triangle x 3 triangles) + (1 shared a-b midpoint) = 12.
    assert.equal(result.N, 12);
    assertSymmetricNoSelfLoops(result.adj);
    assertConnected(result.adj);
    // The shared a-b midpoint connects to a, b, and all 6 non-shared midpoints (2 per triangle x 3) - degree 8.
    assert.equal(degrees(result.adj).filter(d => d === 8).length, 1,
        'exactly one node (the shared midpoint) has degree 8');
});

test('runs cleanly (no crash, stays connected/symmetric) on the dodecahedron, which has no triangles at all', () => {
    const bc = dodecahedronBoard();
    const result = triangleForm(bc, 3);
    assert.equal(result.N, bc.N, 'no triangles found, so nothing should change');
    assert.deepEqual(result.adj, bc.adj);
});

test('applyModifier("TriangleForm", ...) matches calling triangleForm directly', () => {
    const bc = triangularBoard(2);
    assert.deepEqual(applyModifier(bc, { kind: 'TriangleForm', w: 3 }), triangleForm(bc, 3));
});

test('an optional trailing triangle selector restricts triform to only the triangles it selects', () => {
    const sel = parseTriangleSelector('(conve tri (deg eq 3))');
    const bc = triangularBoard(2);
    const direct = triangleForm(bc, 3, sel);
    assert.deepEqual(applyModifier(bc, { kind: 'TriangleForm', w: 3, sel }), direct);
});

test('sel restricts triangleForm to only the selected triangles - an unselected one is left ' +
    'untouched, even where it would otherwise have shared a glued corner/edge', () => {
    // Bowtie: triangles {0,1,2} and {2,3,4} sharing only vertex 2 (no shared edge - so an unselected
    // triangle here has none of its own sides consumed by the selected one), plus a pendant node 5
    // on node 0 alone, making node 0 the graph's unique degree-3 node.
    const bc = boardFromEdges(6, [[0, 1], [0, 2], [0, 5], [1, 2], [2, 3], [2, 4], [3, 4]]);
    // Selects only the triangle containing the degree-3 node (0) - triangle {0,1,2}, not {2,3,4}.
    const sel = parseTriangleSelector('(conve tri (deg eq 3))');
    const result = triangleForm(bc, 3, sel);
    // All 6 original nodes survive (triangle {2,3,4} is untouched) plus triangularBoard(3)'s own 6
    // face nodes for the one selected triangle, minus 3 corners glued back to nodes 0/1/2 = 9.
    assert.equal(result.N, 9);
    assertSymmetricNoSelfLoops(result.adj);
    assertConnected(result.adj);
    // Triangle {2,3,4}'s own 3 edges plus the 0-5 pendant edge (untouched) = 4, plus every edge of a
    // triangularBoard(3) for the one subdivided triangle (its 3 corners are nodes 0,1,2 themselves).
    assert.equal(edgeCount(result.adj), 4 + edgeCount(triangularBoard(3).adj));
    // Selecting nothing (an empty selector) is a total no-op - no triangle qualifies.
    const none = triangleForm(bc, 3, parseTriangleSelector('(none tri)'));
    assert.deepEqual(none.adj, bc.adj);
});
