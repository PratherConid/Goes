// Regression tests for dodecahedronFlake: the "flake" fractal generalization of a regular
// dodecahedron. n=1 is the plain dodecahedron; n>1 recurses into 20 copies of n-1, one attached at
// each vertex via S_i(x) = r*x + c*verts[i] (same r, c, no rotation, at every level) - unlike a
// simplex-style flake, adjacent copies share a full, growing EDGE (dodecaFlakeRec's own
// edgeChains), not a single point, which is what fixes c/r = phi^2 and reproduces the
// levels-grow-by-(2+phi) size relation (see dodecahedronFlake's own doc comment in boardConfig.ts
// for the full derivation).
//
// dodecahedronFlake(3) is somewhat expensive (6680 nodes) - built once per n below and reused
// across assertions, rather than rebuilt per test, to keep this file's total runtime reasonable.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    dodecahedronBoard, dodecahedronFlake, PrescribedBoard, PrescribedBoardMap, PrescribedBoardFns, BoardArgType,
    numArg,
} from '../shared/boardConfig.ts';

const EPS = 1e-9;
const PHI = (1 + Math.sqrt(5)) / 2;
function dist(a: number[], b: number[]): number {
    return Math.sqrt(a.reduce((s, v, k) => s + (v - b[k]) ** 2, 0));
}
function circumradius(pos: number[][]): number {
    return Math.max(...pos.map(p => Math.hypot(...p)));
}

const flakes = [1, 2, 3].map(n => dodecahedronFlake(n));

test('n=1 is exactly dodecahedronBoard(): same N, adjacency, and positions', () => {
    const base = dodecahedronBoard();
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

test('circumradius grows by exactly (2+phi) per level', () => {
    const [r1, r2, r3] = flakes.map(bc => circumradius(bc.emb.pos));
    assert.ok(Math.abs(r2 / r1 - (2 + PHI)) < EPS, `n=2/n=1 ratio was ${r2 / r1}`);
    assert.ok(Math.abs(r3 / r2 - (2 + PHI)) < EPS, `n=3/n=2 ratio was ${r3 / r2}`);
});

test('n=2 has exactly 340 nodes (20 leaf copies x 20 vertices, minus 2 merged per shared edge x 30 ' +
    'base edges) and 570 edges', () => {
    const bc = flakes[1];
    assert.equal(bc.N, 340);
    let edgeCount = 0;
    for (let i = 0; i < bc.N; i++)
        for (let j = i + 1; j < bc.N; j++)
            if (bc.adj[i][j]) edgeCount++;
    assert.equal(edgeCount, 570);
});

test('n=2: 280 nodes keep degree 3 (unmerged), 60 nodes (2 per shared edge x 30 edges) become degree 5', () => {
    const bc = flakes[1];
    const degrees = bc.adj.map(row => row.reduce((s, v) => s + v, 0));
    assert.equal(degrees.filter(d => d === 3).length, 280);
    assert.equal(degrees.filter(d => d === 5).length, 60);
});

test('rejects a non-positive/non-integer n', () => {
    assert.throws(() => dodecahedronFlake(0));
    assert.throws(() => dodecahedronFlake(-1));
    assert.throws(() => dodecahedronFlake(1.5));
});

test('is registered as the "dodflake" prescribed board type, taking exactly 1 Number argument', () => {
    const [argTypes, cmd] = PrescribedBoardMap[PrescribedBoard.dodecahedronFlake];
    assert.equal(cmd, 'dodflake');
    assert.deepEqual(argTypes, [BoardArgType.Number]);
    assert.deepEqual(PrescribedBoardFns[PrescribedBoard.dodecahedronFlake](numArg(2)), flakes[1]);
});
