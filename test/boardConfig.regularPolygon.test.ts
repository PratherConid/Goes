// Regression tests for regularPolygonBoard: a regular n-gon with every edge exactly length 1.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    regularPolygonBoard, PrescribedBoard, PrescribedBoardMap, PrescribedBoardFns, BoardArgType,
} from '../shared/boardConfig.ts';

const EPS = 1e-9;
function dist(a: number[], b: number[]): number {
    return Math.sqrt(a.reduce((s, v, k) => s + (v - b[k]) ** 2, 0));
}

test('has n nodes arranged in a simple cycle (every node degree exactly 2)', () => {
    for (const n of [3, 4, 5, 6, 8]) {
        const bc = regularPolygonBoard(n);
        assert.equal(bc.N, n);
        for (let k = 0; k < n; k++) {
            const degree = bc.adj[k].reduce((s, v) => s + v, 0);
            assert.equal(degree, 2, `node ${k} of a ${n}-gon should have degree 2`);
            assert.equal(bc.adj[k][(k + 1) % n], 1, `node ${k} should connect to node ${(k + 1) % n}`);
            assert.equal(bc.adj[k][(k - 1 + n) % n], 1, `node ${k} should connect to node ${(k - 1 + n) % n}`);
        }
    }
});

test('every edge has length exactly 1', () => {
    for (const n of [3, 4, 5, 6, 8, 12]) {
        const bc = regularPolygonBoard(n);
        for (let k = 0; k < n; k++) {
            const next = (k + 1) % n;
            assert.ok(
                Math.abs(dist(bc.emb.pos[k], bc.emb.pos[next]) - 1) < EPS,
                `edge ${k}-${next} of a ${n}-gon should have length 1`,
            );
        }
    }
});

test('a square (n=4) reproduces the unit-square vertex positions up to rotation', () => {
    const bc = regularPolygonBoard(4);
    // Circumradius of a unit square is sqrt(2)/2 - every vertex is that far from the origin.
    for (const p of bc.emb.pos) assert.ok(Math.abs(dist(p, [0, 0]) - Math.SQRT1_2) < EPS);
});

test('rejects n < 3', () => {
    assert.throws(() => regularPolygonBoard(2));
    assert.throws(() => regularPolygonBoard(0));
});

test('is registered as the "regpoly" prescribed board type, taking exactly 1 Number argument', () => {
    const [argTypes, cmd] = PrescribedBoardMap[PrescribedBoard.regularPolygonBoard];
    assert.equal(cmd, 'regpoly');
    assert.deepEqual(argTypes, [BoardArgType.Number]);
    assert.deepEqual(PrescribedBoardFns[PrescribedBoard.regularPolygonBoard](6), regularPolygonBoard(6));
});
