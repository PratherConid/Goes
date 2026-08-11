// Regression tests for hypercuboidBoard: the meshdim-skeleton of a dims.length-dimensional
// box-lattice board. rectangularBoard(w, h) and cubeLatticeBoard(w, h, d) are now just
// hypercuboidBoard(2, [w, h]) / hypercuboidBoard(3, [w, h, d]) (meshdim = their own full dimension
// count, so nothing is excluded) - see their own one-line bodies in shared/boardConfig.ts.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    hypercuboidBoard, rectangularBoard, cubeLatticeBoard,
    PrescribedBoard, PrescribedBoardMap, PrescribedBoardFns, BoardArgType, parseModifier,
} from '../shared/boardConfig.ts';

function degrees(bc: { adj: number[][] }): number[] {
    return bc.adj.map(row => row.reduce((s, v) => s + v, 0));
}

test('meshdim >= dims.length keeps the full solid lattice - 1D path', () => {
    const bc = hypercuboidBoard(1, [5]);
    assert.equal(bc.N, 5);
    assert.equal(bc.emb.embDim, 1);
    assert.deepEqual(degrees(bc), [1, 2, 2, 2, 1]);
});

test('rectangularBoard(2, 2) is a 4-cycle (a square): every node degree 2', () => {
    const bc = rectangularBoard(2, 2);
    assert.equal(bc.N, 4);
    assert.deepEqual(degrees(bc), [2, 2, 2, 2]);
});

test('cubeLatticeBoard(2, 2, 2) is the standard cube graph: every node degree 3', () => {
    const bc = cubeLatticeBoard(2, 2, 2);
    assert.equal(bc.N, 8);
    assert.deepEqual(degrees(bc), new Array(8).fill(3));
});

test('meshdim >= dims.length: dims=[2,2,2,2,2] is the 5-dimensional hypercube graph, degree 5', () => {
    const bc = hypercuboidBoard(5, [2, 2, 2, 2, 2]);
    assert.equal(bc.N, 32);
    assert.equal(bc.emb.embDim, 5);
    assert.deepEqual(degrees(bc), new Array(32).fill(5));
});

test('meshdim=2 on a 3x3x3 cube is the hollow surface (Rubik\'s-cube shell): 26 of 27 nodes ' +
    '(only the center is excluded)', () => {
    const bc = hypercuboidBoard(2, [3, 3, 3]);
    assert.equal(bc.N, 26);
});

test('meshdim=2 on a 5x5x5 cube: N = 5^3 - 3^3 = 98 (outer shell minus the solid interior), ' +
    'the 8 true corners have degree 3, everything else on the shell has degree 4', () => {
    const bc = hypercuboidBoard(2, [5, 5, 5]);
    assert.equal(bc.N, 98);
    const degs = degrees(bc);
    assert.equal(degs.filter(d => d === 3).length, 8);
    assert.equal(degs.filter(d => d === 4).length, 90);
});

test('meshdim=3 on a 4D 3x3x3x3 hypercuboid keeps 3^4 - 1 = 80 nodes (only the single ' +
    'fully-interior center point is excluded)', () => {
    const bc = hypercuboidBoard(3, [3, 3, 3, 3]);
    assert.equal(bc.N, 80);
});

test('meshdim=0 keeps only the 2^k corners - degenerate: no two corners are ever unit-step ' +
    'apart, so every corner ends up with degree 0', () => {
    const bc = hypercuboidBoard(0, [3, 3, 3]);
    assert.equal(bc.N, 8);
    assert.deepEqual(degrees(bc), new Array(8).fill(0));
});

test('N is the product of dims when meshdim covers the full dimension count, and every position ' +
    'has dims.length coordinates', () => {
    for (const dims of [[3], [4, 5], [3, 3, 3], [2, 3, 4, 5]]) {
        const bc = hypercuboidBoard(dims.length, dims);
        assert.equal(bc.N, dims.reduce((p, d) => p * d, 1), `dims=${dims}`);
        for (const p of bc.emb.pos) assert.equal(p.length, dims.length, `dims=${dims}`);
    }
});

test('rejects an empty dims list, a non-positive dimension, or an invalid meshdim', () => {
    assert.throws(() => hypercuboidBoard(1, []));
    assert.throws(() => hypercuboidBoard(1, [3, 0]));
    assert.throws(() => hypercuboidBoard(1, [3, -1]));
    assert.throws(() => hypercuboidBoard(-1, [3, 3]));
    assert.throws(() => hypercuboidBoard(1.5, [3, 3]));
});

test('is registered as the "hcub" prescribed board type, taking a Number then a ' +
    'CommaSeparatedNumbers argument', () => {
    const [argTypes, cmd] = PrescribedBoardMap[PrescribedBoard.hypercuboidBoard];
    assert.equal(cmd, 'hcub');
    assert.deepEqual(argTypes, [BoardArgType.Number, BoardArgType.CommaSeparatedNumbers]);
    assert.deepEqual(
        PrescribedBoardFns[PrescribedBoard.hypercuboidBoard](4, 5, 5, 2, 2), hypercuboidBoard(4, [5, 5, 2, 2]),
    );
});

test('parseModifier(\'beginprod\', [\'hcub\', ...]) parses meshdim then splits the comma-separated ' +
    'dims token into boardArgs', () => {
    assert.deepEqual(parseModifier('beginprod', ['hcub', '4', '5,5,2,2']),
        { kind: 'BeginProd', boardType: 'hcub', boardArgs: [4, 5, 5, 2, 2] });
});

test('parseModifier(\'beginprod\', [\'hcub\', ...]) still truncates to its 2 required tokens, ' +
    'ignoring anything after them', () => {
    assert.deepEqual(parseModifier('beginprod', ['hcub', '4', '5,5,2,2', '99']),
        { kind: 'BeginProd', boardType: 'hcub', boardArgs: [4, 5, 5, 2, 2] });
});

test('parseModifier(\'beginprod\', ...) is unaffected for an ordinary comma-free board type', () => {
    assert.deepEqual(parseModifier('beginprod', ['rect', '3', '3']),
        { kind: 'BeginProd', boardType: 'rect', boardArgs: [3, 3] });
});
