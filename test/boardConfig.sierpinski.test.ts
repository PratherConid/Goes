// Regression tests for sierpinskiTriangle: the Sierpinski gasket board type. n=0 is a single node;
// n=1 is a unit-side triangle; n>1 is three copies of n-1 glued at touching corners (the classic
// middle-sub-triangle "hole" is never built).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { sierpinskiTriangle, PrescribedBoard, PrescribedBoardMap, PrescribedBoardFns } from '../shared/boardConfig.ts';

const EPS = 1e-9;
function dist(a: number[], b: number[]): number {
    return Math.sqrt(a.reduce((s, v, k) => s + (v - b[k]) ** 2, 0));
}
function centroid(pos: number[][]): number[] {
    return pos[0].map((_, k) => pos.reduce((s, p) => s + p[k], 0) / pos.length);
}

test('n=0 is a single node at the origin, with no edges', () => {
    const bc = sierpinskiTriangle(0);
    assert.equal(bc.N, 1);
    assert.deepEqual(bc.emb.pos, [[0, 0]]);
    assert.equal(bc.adj[0][0], 0);
});

test('n=1 is a unit-side triangle (3 mutually-adjacent nodes), same circumradius as regularPolygonBoard(3)', () => {
    const bc = sierpinskiTriangle(1);
    assert.equal(bc.N, 3);
    for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++)
            assert.equal(bc.adj[i][j], i === j ? 0 : 1);
    for (const p of bc.emb.pos) assert.ok(Math.abs(dist(p, [0, 0]) - 1 / Math.sqrt(3)) < EPS);
});

test('node count follows N(1)=3, N(n)=3*N(n-1)-3 (three copies glued at 3 shared corners)', () => {
    let expected = 3;
    for (let n = 1; n <= 5; n++) {
        assert.equal(sierpinskiTriangle(n).N, expected, `n=${n}`);
        expected = 3 * expected - 3;
    }
});

test('every edge has length exactly 1, for every n', () => {
    for (const n of [1, 2, 3, 4]) {
        const bc = sierpinskiTriangle(n);
        for (let i = 0; i < bc.N; i++)
            for (let j = i + 1; j < bc.N; j++)
                if (bc.adj[i][j]) assert.ok(Math.abs(dist(bc.emb.pos[i], bc.emb.pos[j]) - 1) < EPS, `n=${n} edge ${i}-${j}`);
    }
});

test('the centroid of every node is exactly the origin, for every n (not just the symmetric base cases)', () => {
    for (const n of [0, 1, 2, 3, 4, 5]) {
        const c = centroid(sierpinskiTriangle(n).emb.pos);
        assert.ok(Math.abs(c[0]) < EPS && Math.abs(c[1]) < EPS, `n=${n}: centroid ${JSON.stringify(c)}`);
    }
});

test('exactly the 3 outer corners have degree 2; every glued interior corner has degree 4', () => {
    for (const n of [2, 3, 4]) {
        const bc = sierpinskiTriangle(n);
        const degrees = bc.adj.map(row => row.reduce((s, v) => s + v, 0));
        const degree2 = degrees.filter(d => d === 2).length;
        const degree4 = degrees.filter(d => d === 4).length;
        assert.equal(degree2, 3, `n=${n}`);
        assert.equal(degree4, bc.N - 3, `n=${n}`);
    }
});

test('rejects negative or non-integer n', () => {
    assert.throws(() => sierpinskiTriangle(-1));
    assert.throws(() => sierpinskiTriangle(1.5));
});

test('is registered as the "sier" prescribed board type, taking exactly 1 argument', () => {
    const [numArgs, cmd] = PrescribedBoardMap[PrescribedBoard.sierpinskiTriangle];
    assert.equal(cmd, 'sier');
    assert.equal(numArgs, 1);
    assert.deepEqual(PrescribedBoardFns[PrescribedBoard.sierpinskiTriangle](3), sierpinskiTriangle(3));
});
