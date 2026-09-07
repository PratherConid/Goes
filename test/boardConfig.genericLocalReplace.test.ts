// Regression tests for genericLocalReplace and the "localReplace"/"triCentralize"/"quadCentralize"/
// "simpCentralize"/"quadOctarize"/"quadCentering"/"simpCentering" board modifiers: replacing every
// selected face with its own small local shape (a hub-connected pyramid for QuadCentralize/
// SimpCentralize - triCentralize is a thin wrapper over SimpCentralize's own n=2 case, not a separate
// LocalReplaceSelector branch - see that type's own doc comment, shared/types.ts - a two-apex
// octahedron for QuadOctarize, or a bare hub-and-spoke star with the face's own original edges
// dropped for QuadCentering/SimpCentering) - unlike genericForm, nothing is subdivided or glued, so a
// selectors list mixing several branches is just independent per-face replacements (see
// genericLocalReplace's own doc comment).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    genericLocalReplace, simpCentralize, simpCentering, triCentralize, triCentering, quadCentralize,
    quadCentering, quadOctarize, triangularBoard, rectangularBoard, tetrahedronBoard, applyModifier,
} from '../shared/boardConfig.ts';
import { Embedding, type BoardConfig, type LocalReplaceSelector } from '../shared/types.ts';
import { parseTriangleSelector } from '../shared/selector.ts';

function edgeCount(adj: number[][]): number {
    return adj.flat().reduce((s, v) => s + v, 0) / 2;
}

function degree(adj: number[][], i: number): number {
    return adj[i].reduce((s, v) => s + v, 0);
}

test('genericLocalReplace with a single SimpCentralize n=2 selector is identical to triCentralize', () => {
    const bc = triangularBoard(2);
    assert.deepEqual(genericLocalReplace(bc, [{ kind: 'SimpCentralize', n: 2 }]), triCentralize(bc));
});

test('genericLocalReplace with a single QuadCentralize selector is identical to quadCentralize', () => {
    const bc = rectangularBoard(2, 2);
    assert.deepEqual(genericLocalReplace(bc, [{ kind: 'QuadCentralize' }]), quadCentralize(bc));
});

test('genericLocalReplace with a single QuadOctarize selector is identical to quadOctarize', () => {
    const bc = rectangularBoard(2, 2);
    assert.deepEqual(genericLocalReplace(bc, [{ kind: 'QuadOctarize' }]), quadOctarize(bc));
});

test('triCentralize adds one hub per triangle, connected to all 3 corners, original edges unchanged (removed then re-added)', () => {
    // triangularBoard(2) is a bare K3: 3 nodes, 1 triangle, 3 edges.
    const bc = triangularBoard(2);
    const result = triCentralize(bc);
    assert.equal(result.N, 4);
    assert.equal(edgeCount(result.adj), 6); // 3 original + 3 hub-to-corner
    // Original 3x3 block reads the same as before, even though genericLocalReplace excludes these
    // from its straight copy and re-adds them explicitly as part of the tetrahedron's own edge set.
    for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++)
            assert.equal(result.adj[i][j], bc.adj[i][j], `original edge [${i}][${j}] changed`);
    // The hub (node 3) is connected to all 3 original nodes, and nothing else.
    assert.equal(degree(result.adj, 3), 3);
    for (let i = 0; i < 3; i++) assert.equal(result.adj[3][i], 1);
});

test('quadCentralize adds one hub per quad, connected to all 4 corners, original edges unchanged (removed then re-added)', () => {
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

test('quadOctarize replaces a quad with a real octahedron: 12 edges total, two antipodal degree-4 apexes not connected to each other', () => {
    // rectangularBoard(2, 2) is a single 4-cycle: 4 nodes, 1 quad, 4 equatorial edges.
    const bc = rectangularBoard(2, 2);
    const result = quadOctarize(bc);
    assert.equal(result.N, 6); // 4 corners + 2 apexes
    assert.equal(edgeCount(result.adj), 12); // 4 equatorial (re-added) + 4+4 apex-to-corner
    // The equatorial ring reads the same as bc's own 4-cycle, even though it was excluded from the
    // straight copy and re-added explicitly as part of the octahedron's own edge set.
    for (let i = 0; i < 4; i++)
        for (let j = 0; j < 4; j++)
            assert.equal(result.adj[i][j], bc.adj[i][j], `original edge [${i}][${j}] changed`);
    const [top, bottom] = [4, 5];
    assert.equal(degree(result.adj, top), 4);
    assert.equal(degree(result.adj, bottom), 4);
    for (let i = 0; i < 4; i++) {
        assert.equal(result.adj[top][i], 1);
        assert.equal(result.adj[bottom][i], 1);
    }
    // The two apexes are antipodal - never connected to each other.
    assert.equal(result.adj[top][bottom], 0);
    // Every node's own embedding has one extra dimension beyond bc's own.
    assert.equal(result.emb.embDim, bc.emb.embDim + 1);
});

test('quadOctarize accepts a restricting sel, unlike before this refactor', () => {
    // Two quads sharing an edge: 0-1-2-3 and 1-4-5-2 (cycle order), sharing edge (1, 2).
    const N = 6;
    const adj: number[][] = Array.from({ length: N }, () => new Array(N).fill(0));
    const edge = (i: number, j: number) => { adj[i][j] = 1; adj[j][i] = 1; };
    edge(0, 1); edge(1, 2); edge(2, 3); edge(3, 0);
    edge(1, 4); edge(4, 5); edge(5, 2);
    const emb = new Embedding(2, adj.map((_, i): [number, number] => [i, 0]));
    const bc: BoardConfig = { N, adj, emb };

    const restrictedSel = { op: 'all' as const, type: 'quad' as const };
    const result = quadOctarize(bc, restrictedSel);
    // Both quads found -> 6 original + 2*2 apexes = 10.
    assert.equal(result.N, 10);
});

test('genericLocalReplace with a single QuadCentering selector is identical to quadCentering', () => {
    const bc = rectangularBoard(2, 2);
    assert.deepEqual(genericLocalReplace(bc, [{ kind: 'QuadCentering' }]), quadCentering(bc));
});

test('quadCentering adds one hub per quad, connected to all 4 corners, but the quad\'s own 4 ' +
    'original edges are dropped rather than kept', () => {
    // rectangularBoard(2, 2) is a single 4-cycle: 4 nodes, 1 quad, 4 edges.
    const bc = rectangularBoard(2, 2);
    const result = quadCentering(bc);
    assert.equal(result.N, 5);
    assert.equal(edgeCount(result.adj), 4); // only the 4 hub-to-corner edges - no original edges survive
    for (let i = 0; i < 4; i++)
        for (let j = 0; j < 4; j++)
            assert.equal(result.adj[i][j], 0, `original edge [${i}][${j}] should have been dropped`);
    assert.equal(degree(result.adj, 4), 4);
    for (let i = 0; i < 4; i++) assert.equal(result.adj[4][i], 1);
});

test('simpCentering generalizes quadCentering to n-simplices: a triangle loses its own 3 edges, ' +
    'ending up connected only through the new hub', () => {
    // triangularBoard(2) is a bare K3: 3 nodes, 1 triangle, 3 edges.
    const bc = triangularBoard(2);
    const result = simpCentering(bc, 2);
    assert.equal(result.N, 4);
    assert.equal(edgeCount(result.adj), 3); // only the 3 hub-to-corner edges
    for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++)
            assert.equal(result.adj[i][j], 0, `original edge [${i}][${j}] should have been dropped`);
    assert.equal(degree(result.adj, 3), 3);
    for (let i = 0; i < 3; i++) assert.equal(result.adj[3][i], 1);
});

test('simpCentering generalizes beyond triangles: n=3 on a K4 tetrahedron drops all 6 original ' +
    'edges, leaving only the 4 hub-to-corner edges', () => {
    const bc = tetrahedronBoard();
    const result = simpCentering(bc, 3);
    assert.equal(result.N, 5);
    assert.equal(edgeCount(result.adj), 4);
    for (let i = 0; i < 4; i++)
        for (let j = 0; j < 4; j++)
            assert.equal(result.adj[i][j], 0, `original edge [${i}][${j}] should have been dropped`);
    assert.equal(degree(result.adj, 4), 4);
    for (let i = 0; i < 4; i++) assert.equal(result.adj[4][i], 1);
});

test('a shared edge consumed only by Centering-kind selectors is genuinely dropped, even when two faces share it', () => {
    // Two triangles sharing edge (1, 2): {0,1,2} and {1,2,3}.
    const N = 4;
    const adj: number[][] = Array.from({ length: N }, () => new Array(N).fill(0));
    const edge = (i: number, j: number) => { adj[i][j] = 1; adj[j][i] = 1; };
    edge(0, 1); edge(1, 2); edge(2, 0);
    edge(1, 3); edge(3, 2);
    const emb = new Embedding(2, adj.map((_, i): [number, number] => [i, 0]));
    const bc: BoardConfig = { N, adj, emb };

    const result = simpCentering(bc, 2);
    // Both triangles get their own hub; the shared edge (1, 2) is consumed by both but re-added by
    // neither, so it stays gone.
    assert.equal(result.N, 6);
    assert.equal(result.adj[1][2], 0);
});

test('a mixed selectors list adds one independent local replacement per face - unlike genericForm, nothing glues', () => {
    // Triangle 0-1-2 and quad (cycle) 1-3-4-2 share edge (1,2).
    const N = 5;
    const adj: number[][] = Array.from({ length: N }, () => new Array(N).fill(0));
    const edge = (i: number, j: number) => { adj[i][j] = 1; adj[j][i] = 1; };
    edge(0, 1); edge(1, 2); edge(2, 0);
    edge(1, 3); edge(3, 4); edge(4, 2);
    const emb = new Embedding(2, adj.map((_, i): [number, number] => [i, 0]));
    const bc: BoardConfig = { N, adj, emb };

    const selectors: LocalReplaceSelector[] = [{ kind: 'SimpCentralize', n: 2 }, { kind: 'QuadCentralize' }];
    const both = genericLocalReplace(bc, selectors);
    // One hub per face - no gluing, so always exactly N + (number of faces), regardless of shared edges.
    assert.equal(both.N, N + 2);
    const triHub = 5, quadHub = 6;
    assert.equal(degree(both.adj, triHub), 3);
    assert.equal(degree(both.adj, quadHub), 4);
    // The two hubs are never connected to each other.
    assert.equal(both.adj[triHub][quadHub], 0);
    // The shared original edge (1, 2) is still exactly one edge (not doubled/dropped) even though
    // both the triangle's own and the quad's own replacement independently re-add it.
    assert.equal(both.adj[1][2], 1);
});

test('an empty selectors list is a total no-op', () => {
    const bc = triangularBoard(2);
    const result = genericLocalReplace(bc, []);
    assert.deepEqual(result.adj, bc.adj);
});

test('applyModifier matches calling triCentralize/quadCentralize/quadOctarize/quadCentering/' +
    'simpCentering/genericLocalReplace directly', () => {
    const triBc = triangularBoard(2);
    assert.deepEqual(
        applyModifier(triBc, { kind: 'LocalReplace', selectors: [{ kind: 'SimpCentralize', n: 2 }] }),
        triCentralize(triBc));
    const quadBc = rectangularBoard(2, 2);
    assert.deepEqual(
        applyModifier(quadBc, { kind: 'LocalReplace', selectors: [{ kind: 'QuadCentralize' }] }),
        quadCentralize(quadBc));
    assert.deepEqual(
        applyModifier(quadBc, { kind: 'LocalReplace', selectors: [{ kind: 'QuadOctarize' }] }),
        quadOctarize(quadBc));
    assert.deepEqual(
        applyModifier(quadBc, { kind: 'LocalReplace', selectors: [{ kind: 'QuadCentering' }] }),
        quadCentering(quadBc));
    assert.deepEqual(
        applyModifier(triBc, { kind: 'LocalReplace', selectors: [{ kind: 'SimpCentering', n: 2 }] }),
        simpCentering(triBc, 2));
    const selectors: LocalReplaceSelector[] = [{ kind: 'SimpCentralize', n: 2 }, { kind: 'QuadCentralize' }];
    assert.deepEqual(
        applyModifier(triBc, { kind: 'LocalReplace', selectors }),
        genericLocalReplace(triBc, selectors));
});

test('triCentralize is exactly simpCentralize(bc, 2, sel)', () => {
    const bc = triangularBoard(2);
    assert.deepEqual(triCentralize(bc), simpCentralize(bc, 2));
    const sel = parseTriangleSelector('(conve tri (deg eq 3))');
    assert.deepEqual(triCentralize(bc, sel), simpCentralize(bc, 2, sel));
});

test('triCentering is exactly simpCentering(bc, 2, sel)', () => {
    const bc = triangularBoard(2);
    assert.deepEqual(triCentering(bc), simpCentering(bc, 2));
    const sel = parseTriangleSelector('(conve tri (deg eq 3))');
    assert.deepEqual(triCentering(bc, sel), simpCentering(bc, 2, sel));
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

test('the new hub sits at its triangle\'s own barycenter', () => {
    const bc = triangularBoard(2);
    const result = triCentralize(bc);
    const [A, B, C] = [0, 1, 2].map(i => bc.emb.pos[i]);
    const expected = A.map((_, k) => (A[k] + B[k] + C[k]) / 3);
    assert.deepEqual(result.emb.pos[3], expected);
});

test('simpCentralize generalizes beyond triangles: n=3 on a K4 tetrahedron adds one hub connected ' +
    'to all 4 corners, at their exact barycenter', () => {
    // tetrahedronBoard() is K4: 4 nodes, all mutually adjacent - exactly one simp-3 object (the
    // whole K4), so simpCentralize(bc, 3) adds exactly one hub.
    const bc = tetrahedronBoard();
    const result = simpCentralize(bc, 3);
    assert.equal(result.N, 5);
    assert.equal(degree(result.adj, 4), 4);
    for (let i = 0; i < 4; i++) assert.equal(result.adj[4][i], 1);
    // Original 4x4 block untouched.
    for (let i = 0; i < 4; i++)
        for (let j = 0; j < 4; j++)
            assert.equal(result.adj[i][j], bc.adj[i][j], `original edge [${i}][${j}] changed`);
    const expected = bc.emb.pos[0].map((_, k) =>
        (bc.emb.pos[0][k] + bc.emb.pos[1][k] + bc.emb.pos[2][k] + bc.emb.pos[3][k]) / 4);
    assert.deepEqual(result.emb.pos[4], expected);
});

test('genericLocalReplace rejects an out-of-range SimpCentralize n at evaluation time', () => {
    const bc = triangularBoard(2);
    assert.throws(
        () => genericLocalReplace(bc, [{ kind: 'SimpCentralize', n: 1 }]),
        /n must be an integer >= 2, got 1/);
});

test('genericLocalReplace rejects an out-of-range SimpCentering n at evaluation time', () => {
    const bc = triangularBoard(2);
    assert.throws(
        () => genericLocalReplace(bc, [{ kind: 'SimpCentering', n: 1 }]),
        /n must be an integer >= 2, got 1/);
    assert.throws(() => simpCentering(bc, 0), /n must be an integer >= 2, got 0/);
});
