// Regression tests for sierpinskiSimplex: the Sierpinski dim-simplex gasket board type. n=1 is a
// unit-edge regular dim-simplex; n>1 is dim+1 copies of n-1 glued at touching corners (the classic
// "hole" - the sub-simplex a full subdivision would keep at the non-corner positions - is never
// built). dim=1 has no hole to remove at all, so it degenerates to a plain bisected line; dim=2 is
// the classic Sierpinski triangle.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    sierpinskiSimplex, PrescribedBoard, PrescribedBoardMap, PrescribedBoardFns,
} from '../shared/boardConfig.ts';
import { BoardArgType, numArg } from '../shared/types.ts';

const EPS = 1e-9;
function dist(a: number[], b: number[]): number {
    return Math.sqrt(a.reduce((s, v, k) => s + (v - b[k]) ** 2, 0));
}
function centroid(pos: number[][]): number[] {
    return pos[0].map((_, k) => pos.reduce((s, p) => s + p[k], 0) / pos.length);
}

test('n=1 is a unit-edge regular dim-simplex: dim+1 mutually-adjacent nodes, all degree dim', () => {
    for (const dim of [1, 2, 3, 4]) {
        const bc = sierpinskiSimplex(dim, 1);
        assert.equal(bc.N, dim + 1, `dim=${dim}`);
        for (let i = 0; i < bc.N; i++)
            for (let j = 0; j < bc.N; j++)
                assert.equal(bc.adj[i][j], i === j ? 0 : 1, `dim=${dim} [${i}][${j}]`);
        for (const p of bc.emb.pos) assert.equal(p.length, dim, `dim=${dim}`);
    }
});

test('dim=1 has no "hole" to remove, so it is a plain bisected line with 2^(n-1)+1 nodes', () => {
    let expected = 2; // n=1
    for (let n = 1; n <= 6; n++) {
        assert.equal(sierpinskiSimplex(1, n).N, expected, `n=${n}`);
        expected = 2 * expected - 1;
    }
});

test('node count follows N(1)=dim+1, N(n)=(dim+1)*N(n-1) - C(dim+1,2)', () => {
    for (const dim of [2, 3, 4]) {
        const step = (dim * (dim + 1)) / 2; // C(dim+1, 2)
        let expected = dim + 1;
        for (let n = 1; n <= 4; n++) {
            assert.equal(sierpinskiSimplex(dim, n).N, expected, `dim=${dim} n=${n}`);
            expected = (dim + 1) * expected - step;
        }
    }
});

test('dim=2 exactly reproduces the classic Sierpinski triangle node counts (3, 6, 15, 42, 123)', () => {
    const expected = [3, 6, 15, 42, 123];
    for (let n = 1; n <= 5; n++) assert.equal(sierpinskiSimplex(2, n).N, expected[n - 1], `n=${n}`);
});

test('every edge has length exactly 1, for every dim and n', () => {
    for (const dim of [1, 2, 3, 4]) {
        for (const n of [1, 2, 3]) {
            const bc = sierpinskiSimplex(dim, n);
            for (let i = 0; i < bc.N; i++)
                for (let j = i + 1; j < bc.N; j++)
                    if (bc.adj[i][j])
                        assert.ok(
                            Math.abs(dist(bc.emb.pos[i], bc.emb.pos[j]) - 1) < EPS, `dim=${dim} n=${n} edge ${i}-${j}`,
                        );
        }
    }
});

test('the centroid of every node is exactly the origin, for every dim and n', () => {
    for (const dim of [1, 2, 3, 4]) {
        for (const n of [1, 2, 3, 4]) {
            const c = centroid(sierpinskiSimplex(dim, n).emb.pos);
            for (const v of c) assert.ok(Math.abs(v) < EPS, `dim=${dim} n=${n}: centroid ${JSON.stringify(c)}`);
        }
    }
});

test('for n>=2: exactly dim+1 outer-corner nodes have degree dim, every glued node has degree 2*dim', () => {
    for (const dim of [1, 2, 3, 4]) {
        for (const n of [2, 3]) {
            const bc = sierpinskiSimplex(dim, n);
            const degrees = bc.adj.map(row => row.reduce((s, v) => s + v, 0));
            const cornerCount = degrees.filter(d => d === dim).length;
            const gluedCount = degrees.filter(d => d === 2 * dim).length;
            assert.equal(cornerCount, dim + 1, `dim=${dim} n=${n}`);
            assert.equal(gluedCount, bc.N - (dim + 1), `dim=${dim} n=${n}`);
        }
    }
});

test('rejects a non-positive/non-integer dim, or a non-positive/non-integer n', () => {
    assert.throws(() => sierpinskiSimplex(0, 1));
    assert.throws(() => sierpinskiSimplex(1.5, 1));
    assert.throws(() => sierpinskiSimplex(2, 0));
    assert.throws(() => sierpinskiSimplex(2, -1));
    assert.throws(() => sierpinskiSimplex(2, 1.5));
});

test('is registered as the "sierB" prescribed board type, taking exactly 2 Number arguments', () => {
    const [argTypes, cmd] = PrescribedBoardMap[PrescribedBoard.sierpinskiSimplex];
    assert.equal(cmd, 'sierB');
    assert.deepEqual(argTypes, [BoardArgType.Number, BoardArgType.Number]);
    assert.deepEqual(PrescribedBoardFns[PrescribedBoard.sierpinskiSimplex](numArg(2), numArg(3)), sierpinskiSimplex(2, 3));
});
