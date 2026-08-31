// Regression tests for genericCentralize and the "centralize"/"triCentralize"/"quadCentralize" board
// modifiers: adding one barycenter-positioned hub node per selected triangle/quad, connected to all
// of that face's own corners - unlike genericForm, nothing is subdivided or glued, so a mixed tri/quad
// `sels` list is just independent per-face hub additions (see genericCentralize's own doc comment).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    genericCentralize, triCentralize, quadCentralize, triangularBoard, rectangularBoard, applyModifier,
} from '../shared/boardConfig.ts';
import { Embedding, type BoardConfig } from '../shared/types.ts';
import { parseTriangleSelector } from '../shared/selector.ts';

function edgeCount(adj: number[][]): number {
    return adj.flat().reduce((s, v) => s + v, 0) / 2;
}

function degree(adj: number[][], i: number): number {
    return adj[i].reduce((s, v) => s + v, 0);
}

test('genericCentralize with a single (all tri) selector is identical to triCentralize', () => {
    const bc = triangularBoard(2);
    assert.deepEqual(genericCentralize(bc, [{ op: 'all', type: 'tri' }]), triCentralize(bc));
});

test('genericCentralize with a single (all quad) selector is identical to quadCentralize', () => {
    const bc = rectangularBoard(2, 2);
    assert.deepEqual(genericCentralize(bc, [{ op: 'all', type: 'quad' }]), quadCentralize(bc));
});

test('genericCentralize rejects a node/edge selector in sels at runtime', () => {
    const bc = triangularBoard(2);
    assert.throws(
        () => genericCentralize(bc, [{ op: 'all', type: 'node' }]),
        /must be a triangle or quad selector, got a node selector/);
});

test('triCentralize adds one hub per triangle, connected to all 3 corners, original edges untouched', () => {
    // triangularBoard(2) is a bare K3: 3 nodes, 1 triangle, 3 edges.
    const bc = triangularBoard(2);
    const result = triCentralize(bc);
    assert.equal(result.N, 4);
    assert.equal(edgeCount(result.adj), 6); // 3 original + 3 hub-to-corner
    // Original 3x3 block is untouched.
    for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++)
            assert.equal(result.adj[i][j], bc.adj[i][j], `original edge [${i}][${j}] changed`);
    // The hub (node 3) is connected to all 3 original nodes, and nothing else.
    assert.equal(degree(result.adj, 3), 3);
    for (let i = 0; i < 3; i++) assert.equal(result.adj[3][i], 1);
});

test('quadCentralize adds one hub per quad, connected to all 4 corners, original edges untouched', () => {
    // rectangularBoard(2, 2) is a single 4-cycle: 4 nodes, 1 quad, 4 edges.
    const bc = rectangularBoard(2, 2);
    const result = quadCentralize(bc);
    assert.equal(result.N, 5);
    assert.equal(edgeCount(result.adj), 8); // 4 original + 4 hub-to-corner
    for (let i = 0; i < 4; i++)
        for (let j = 0; j < 4; j++)
            assert.equal(result.adj[i][j], bc.adj[i][j], `original edge [${i}][${j}] changed`);
    assert.equal(degree(result.adj, 4), 4);
    for (let i = 0; i < 4; i++) assert.equal(result.adj[4][i], 1);
});

test('the new hub sits at its triangle\'s own barycenter', () => {
    const bc = triangularBoard(2);
    const result = triCentralize(bc);
    const [A, B, C] = [0, 1, 2].map(i => bc.emb.pos[i]);
    const expected = A.map((_, k) => (A[k] + B[k] + C[k]) / 3);
    assert.deepEqual(result.emb.pos[3], expected);
});

test('sel restricts triCentralize to only the selected triangles - an unselected one gets no hub', () => {
    // Bowtie: triangles {0,1,2} and {2,3,4} sharing only vertex 2, plus a pendant node 5 on node 0
    // alone, making node 0 the graph's unique degree-3 node.
    const adj = [
        [0, 1, 1, 0, 0, 1],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 0],
    ];
    const emb = new Embedding(2, adj.map((_, i): [number, number] => [i, 0]));
    const bc: BoardConfig = { N: 6, adj, emb };
    // Selects only the triangle containing the degree-3 node (0) - triangle {0,1,2}, not {2,3,4}.
    const sel = parseTriangleSelector('(conve tri (deg eq 3))');
    const result = triCentralize(bc, sel);
    // Exactly one hub added (for {0,1,2}), connected to 0/1/2 only.
    assert.equal(result.N, 7);
    assert.equal(degree(result.adj, 6), 3);
    assert.equal(result.adj[6][0], 1);
    assert.equal(result.adj[6][1], 1);
    assert.equal(result.adj[6][2], 1);
    assert.equal(result.adj[6][3], 0);
    assert.equal(result.adj[6][4], 0);
    assert.equal(result.adj[6][5], 0);
});

test('a mixed tri/quad sels list adds one independent hub per face - unlike genericForm, nothing glues', () => {
    // Triangle 0-1-2 and quad (cycle) 1-3-4-2 share edge (1,2).
    const N = 5;
    const adj: number[][] = Array.from({ length: N }, () => new Array(N).fill(0));
    const edge = (i: number, j: number) => { adj[i][j] = 1; adj[j][i] = 1; };
    edge(0, 1); edge(1, 2); edge(2, 0);
    edge(1, 3); edge(3, 4); edge(4, 2);
    const emb = new Embedding(2, adj.map((_, i): [number, number] => [i, 0]));
    const bc: BoardConfig = { N, adj, emb };

    const both = genericCentralize(bc, [{ op: 'all', type: 'tri' }, { op: 'all', type: 'quad' }]);
    // One hub per face - no gluing, so always exactly N + (number of faces), regardless of shared edges.
    assert.equal(both.N, N + 2);
    const triHub = 5, quadHub = 6;
    assert.equal(degree(both.adj, triHub), 3);
    assert.equal(degree(both.adj, quadHub), 4);
    // The two hubs are never connected to each other.
    assert.equal(both.adj[triHub][quadHub], 0);
});

test('an empty sels list is a total no-op', () => {
    const bc = triangularBoard(2);
    const result = genericCentralize(bc, []);
    assert.deepEqual(result.adj, bc.adj);
});

test('applyModifier matches calling triCentralize/quadCentralize/genericCentralize directly', () => {
    const triBc = triangularBoard(2);
    assert.deepEqual(applyModifier(triBc, { kind: 'TriCentralize' }), triCentralize(triBc));
    const quadBc = rectangularBoard(2, 2);
    assert.deepEqual(applyModifier(quadBc, { kind: 'QuadCentralize' }), quadCentralize(quadBc));
    const sels = [{ op: 'all' as const, type: 'tri' as const }];
    assert.deepEqual(applyModifier(triBc, { kind: 'Centralize', sels }), genericCentralize(triBc, sels));
});
