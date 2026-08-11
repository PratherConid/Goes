// Regression tests for regularPolygonFlake: the "flake" fractal generalization of a regular
// polygon - built by the same shared nodeEdgeMergeFlakeRec() as dodecahedronFlake/icosahedronFlake/
// octahedronFlake (see that function's own doc comment in boardConfig.ts for the full derivation).
// Unlike those three (always an edge merge), a polygon's own base edges merge by a whole growing
// EDGE when nSides is a multiple of 4, and by a single non-growing NODE otherwise - both cases are
// covered here, cross-verified (during development) against an independent mergeClose()-based
// reference construction for nSides=3..12, order=1..3.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    regularPolygonBoard, regularPolygonFlake, PrescribedBoard, PrescribedBoardMap, PrescribedBoardFns,
    BoardArgType,
} from '../shared/boardConfig.ts';

const EPS = 1e-9;
function dist(a: number[], b: number[]): number {
    return Math.sqrt(a.reduce((s, v, k) => s + (v - b[k]) ** 2, 0));
}
function circumradius(pos: number[][]): number {
    return Math.max(...pos.map(p => Math.hypot(...p)));
}

test('order=1 is exactly regularPolygonBoard(nSides): same N, adjacency, and positions', () => {
    for (const nSides of [3, 4, 5, 6, 8, 12]) {
        const base = regularPolygonBoard(nSides);
        const flake = regularPolygonFlake(nSides, 1);
        assert.equal(flake.N, base.N, `nSides=${nSides}`);
        assert.deepEqual(flake.adj, base.adj, `nSides=${nSides}`);
        assert.deepEqual(flake.emb.pos, base.emb.pos, `nSides=${nSides}`);
    }
});

test('every edge has length exactly 1, adjacency is symmetric with no self-loops, and no ' +
    'duplicate-position nodes - for both merge-by-node (nSides=5,6) and merge-by-edge (nSides=4,8)', () => {
    for (const nSides of [3, 4, 5, 6, 8]) {
        for (let order = 1; order <= 3; order++) {
            const bc = regularPolygonFlake(nSides, order);
            for (let i = 0; i < bc.N; i++) {
                assert.equal(bc.adj[i][i], 0, `nSides=${nSides} order=${order} self-loop at ${i}`);
                for (let j = 0; j < bc.N; j++)
                    assert.equal(bc.adj[i][j], bc.adj[j][i], `nSides=${nSides} order=${order} [${i}][${j}]`);
                for (let j = i + 1; j < bc.N; j++) {
                    const d = dist(bc.emb.pos[i], bc.emb.pos[j]);
                    if (bc.adj[i][j])
                        assert.ok(Math.abs(d - 1) < EPS, `nSides=${nSides} order=${order} edge ${i}-${j} length ${d}`);
                    assert.ok(d > 1e-6, `nSides=${nSides} order=${order} nodes ${i},${j} coincide`);
                }
            }
        }
    }
});

test('circumradius grows by a constant factor per level, for both merge types', () => {
    for (const nSides of [3, 4, 5, 6, 8]) {
        const radii = [1, 2, 3].map(order => circumradius(regularPolygonFlake(nSides, order).emb.pos));
        const ratio1 = radii[1] / radii[0], ratio2 = radii[2] / radii[1];
        assert.ok(Math.abs(ratio1 - ratio2) < EPS, `nSides=${nSides}: ratios ${ratio1} vs ${ratio2} differ`);
    }
});

test('merge-by-node (nSides not a multiple of 4): exact node/edge counts and degree distributions', () => {
    const cases: [number, number, number, [number, number][]][] = [
        [3, 2, 9, [[2, 3], [4, 3]]],
        [3, 3, 27, [[2, 3], [4, 12]]],
        [5, 2, 25, [[2, 15], [4, 5]]],
        [6, 2, 36, [[2, 24], [4, 6]]],
    ];
    for (const [nSides, order, expectedEdges, expectedDegDist] of cases) {
        const bc = regularPolygonFlake(nSides, order);
        let edgeCount = 0;
        const degrees = bc.adj.map(row => row.reduce((s, v) => s + v, 0));
        for (let i = 0; i < bc.N; i++)
            for (let j = i + 1; j < bc.N; j++)
                if (bc.adj[i][j]) edgeCount++;
        assert.equal(edgeCount, expectedEdges, `nSides=${nSides} order=${order}`);
        for (const [deg, count] of expectedDegDist)
            assert.equal(degrees.filter(d => d === deg).length, count, `nSides=${nSides} order=${order} deg=${deg}`);
    }
});

test('merge-by-edge (nSides a multiple of 4): exact node/edge counts and degree distributions', () => {
    const cases: [number, number, number, number, [number, number][]][] = [
        [4, 2, 9, 12, [[2, 4], [3, 4], [4, 1]]],
        [4, 3, 25, 40, [[2, 4], [3, 12], [4, 9]]],
        [8, 2, 48, 56, [[2, 32], [3, 16]]],
        [12, 2, 120, 132, [[2, 96], [3, 24]]],
    ];
    for (const [nSides, order, expectedN, expectedEdges, expectedDegDist] of cases) {
        const bc = regularPolygonFlake(nSides, order);
        assert.equal(bc.N, expectedN, `nSides=${nSides} order=${order}`);
        let edgeCount = 0;
        const degrees = bc.adj.map(row => row.reduce((s, v) => s + v, 0));
        for (let i = 0; i < bc.N; i++)
            for (let j = i + 1; j < bc.N; j++)
                if (bc.adj[i][j]) edgeCount++;
        assert.equal(edgeCount, expectedEdges, `nSides=${nSides} order=${order}`);
        for (const [deg, count] of expectedDegDist)
            assert.equal(degrees.filter(d => d === deg).length, count, `nSides=${nSides} order=${order} deg=${deg}`);
    }
});

test('rejects a non-positive/non-integer nSides or order', () => {
    assert.throws(() => regularPolygonFlake(2, 1));
    assert.throws(() => regularPolygonFlake(3.5, 1));
    assert.throws(() => regularPolygonFlake(5, 0));
    assert.throws(() => regularPolygonFlake(5, -1));
    assert.throws(() => regularPolygonFlake(5, 1.5));
});

test('is registered as the "polyflake" prescribed board type, taking exactly 2 Number arguments', () => {
    const [argTypes, cmd] = PrescribedBoardMap[PrescribedBoard.regularPolygonFlake];
    assert.equal(cmd, 'polyflake');
    assert.deepEqual(argTypes, [BoardArgType.Number, BoardArgType.Number]);
    assert.deepEqual(PrescribedBoardFns[PrescribedBoard.regularPolygonFlake](6, 2), regularPolygonFlake(6, 2));
});
