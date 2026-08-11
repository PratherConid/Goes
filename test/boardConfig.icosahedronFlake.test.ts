// Regression tests for icosahedronFlake: the "flake" fractal generalization of a regular
// icosahedron - the same construction as dodecahedronFlake (see that function's own doc comment in
// boardConfig.ts for the full derivation), just with 12 vertices/30 edges instead of 20 vertices/30
// edges, and c/r = phi (not phi^2), giving a levels-grow-by-(1+phi) size relation instead of (2+phi).
//
// Built once per n below and reused across assertions, rather than rebuilt per test.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    icosahedronBoard, icosahedronFlake, PrescribedBoard, PrescribedBoardMap, PrescribedBoardFns, BoardArgType,
} from '../shared/boardConfig.ts';

const EPS = 1e-9;
const PHI = (1 + Math.sqrt(5)) / 2;
function dist(a: number[], b: number[]): number {
    return Math.sqrt(a.reduce((s, v, k) => s + (v - b[k]) ** 2, 0));
}
function circumradius(pos: number[][]): number {
    return Math.max(...pos.map(p => Math.hypot(...p)));
}

const flakes = [1, 2, 3].map(n => icosahedronFlake(n));

test('n=1 is exactly icosahedronBoard(): same N, adjacency, and positions', () => {
    const base = icosahedronBoard();
    const flake = flakes[0];
    assert.equal(flake.N, base.N);
    assert.deepEqual(flake.adj, base.adj);
    assert.deepEqual(flake.emb.pos, base.emb.pos);
});

test('every edge has length exactly 1, adjacency is symmetric with no self-loops, and no ' +
    'duplicate-position nodes - for every n', () => {
    for (let n = 1; n <= 3; n++) {
        const bc = flakes[n - 1];
        for (let i = 0; i < bc.N; i++) {
            assert.equal(bc.adj[i][i], 0, `n=${n} self-loop at ${i}`);
            for (let j = 0; j < bc.N; j++) assert.equal(bc.adj[i][j], bc.adj[j][i], `n=${n} [${i}][${j}]`);
            for (let j = i + 1; j < bc.N; j++) {
                const d = dist(bc.emb.pos[i], bc.emb.pos[j]);
                if (bc.adj[i][j]) assert.ok(Math.abs(d - 1) < EPS, `n=${n} edge ${i}-${j} length ${d}`);
                assert.ok(d > 1e-6, `n=${n} nodes ${i},${j} coincide`);
            }
        }
    }
});

test('circumradius grows by exactly (1+phi) per level', () => {
    const [r1, r2, r3] = flakes.map(bc => circumradius(bc.emb.pos));
    assert.ok(Math.abs(r2 / r1 - (1 + PHI)) < EPS, `n=2/n=1 ratio was ${r2 / r1}`);
    assert.ok(Math.abs(r3 / r2 - (1 + PHI)) < EPS, `n=3/n=2 ratio was ${r3 / r2}`);
});

test('n=2 has exactly 104 nodes and 330 edges', () => {
    const bc = flakes[1];
    assert.equal(bc.N, 104);
    let edgeCount = 0;
    for (let i = 0; i < bc.N; i++)
        for (let j = i + 1; j < bc.N; j++)
            if (bc.adj[i][j]) edgeCount++;
    assert.equal(edgeCount, 330);
});

test('n=2: 84 nodes keep degree 5 (unmerged), 20 nodes (2 per shared edge x 30 edges, with some ' +
    'multi-way overlap) become degree 12', () => {
    const bc = flakes[1];
    const degrees = bc.adj.map(row => row.reduce((s, v) => s + v, 0));
    assert.equal(degrees.filter(d => d === 5).length, 84);
    assert.equal(degrees.filter(d => d === 12).length, 20);
});

test('rejects a non-positive/non-integer n', () => {
    assert.throws(() => icosahedronFlake(0));
    assert.throws(() => icosahedronFlake(-1));
    assert.throws(() => icosahedronFlake(1.5));
});

test('is registered as the "icoflake" prescribed board type, taking exactly 1 Number argument', () => {
    const [argTypes, cmd] = PrescribedBoardMap[PrescribedBoard.icosahedronFlake];
    assert.equal(cmd, 'icoflake');
    assert.deepEqual(argTypes, [BoardArgType.Number]);
    assert.deepEqual(PrescribedBoardFns[PrescribedBoard.icosahedronFlake](2), flakes[1]);
});
