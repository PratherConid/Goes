// Regression tests for the `product` board-transform function and the `beginprod`/`endprod`
// modifier pair that drives it via applyModifiers's stack (applyModifier itself must reject both).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    rectangularBoard, product, parseModifier, applyModifier, applyModifiers, numArg,
} from '../shared/boardConfig.ts';

test('product against a single-node board reproduces the other factor\'s own adjacency exactly', () => {
    const single = rectangularBoard(1, 1);
    const path3 = rectangularBoard(3, 1); // 3-node path: 0-1-2
    const merged = product(single, path3);
    assert.equal(merged.N, 3);
    assert.deepEqual(merged.adj, path3.adj);
});

test('positions concatenate the two factors\' own natural-dimension positions', () => {
    const a = rectangularBoard(2, 1); // nodes at x=-0.5, 0.5 (y=0)
    const b = rectangularBoard(1, 1); // single node at (0, 0)
    const merged = product(a, b);
    assert.equal(merged.emb.embDim, 4);
    assert.deepEqual(merged.emb.pos, [
        [-0.5, 0, 0, 0],
        [0.5, 0, 0, 0],
    ]);
});

test('the default projMat maps dims 0/1/2 straight to x/y/z, then cycles a halving magnitude', () => {
    const merged = product(rectangularBoard(2, 1), rectangularBoard(2, 1)); // embDim = 2+2 = 4
    assert.deepEqual(merged.emb.projMat, [
        [1, 0, 0, 1 / 2],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ]);
});

test('the cycling magnitude keeps halving every 3 dims (embDim=8)', () => {
    const merged = product(
        product(rectangularBoard(2, 1), rectangularBoard(2, 1)),
        product(rectangularBoard(2, 1), rectangularBoard(2, 1)),
    ); // embDim = 4+4 = 8
    assert.deepEqual(merged.emb.projMat, [
        [1, 0, 0, 1 / 2, 0, 0, 1 / 4, 0],
        [0, 1, 0, 0, 1 / 2, 0, 0, 1 / 4],
        [0, 0, 1, 0, 0, 1 / 2, 0, 0],
    ]);
});

test('product of a 2x2 grid and a 2-path is graph-isomorphic to a 2x2x2 cube (N, edge count, degree)', () => {
    const merged = product(rectangularBoard(2, 2), rectangularBoard(1, 2));
    assert.equal(merged.N, 8);
    const edgeCount = merged.adj.flat().reduce((s: number, v: number) => s + v, 0) / 2;
    assert.equal(edgeCount, 12, '|E(GxH)| = |V(G)|*|E(H)| + |E(G)|*|V(H)| = 4*1 + 4*2 = 12');
    const degrees = merged.adj.map(row => row.reduce((s: number, v: number) => s + v, 0));
    assert.deepEqual(degrees, new Array(8).fill(3), 'every node: deg_G(i) + deg_H(j) = 2 + 1 = 3');
});

test('parseModifier parses beginprod\'s board type and integer args, and bare endprod', () => {
    assert.deepEqual(parseModifier('beginprod', ['rect', '3', '3']),
        { kind: 'BeginProd', boardType: 'rect', boardArgs: [numArg(3), numArg(3)] });
    assert.deepEqual(parseModifier('endprod', []), { kind: 'EndProd' });
});

test('parseModifier rejects malformed beginprod/endprod', () => {
    assert.throws(() => parseModifier('beginprod', []), 'needs at least a board type');
    assert.throws(() => parseModifier('beginprod', ['rect', '3', 'abc']), 'non-integer board arg');
    assert.throws(() => parseModifier('beginprod', ['nope', '1', '2']), 'unknown board type');
    assert.throws(() => parseModifier('endprod', ['x']), 'endprod takes no arguments');
});

test('parseModifier(\'beginprod\', ...) truncates extra board args to the type\'s required count', () => {
    assert.deepEqual(parseModifier('beginprod', ['rect', '3', '3', '99', '100']),
        { kind: 'BeginProd', boardType: 'rect', boardArgs: [numArg(3), numArg(3)] });
});

test('parseModifier(\'beginprod\', ...) throws when fewer board args than required are given', () => {
    assert.throws(() => parseModifier('beginprod', ['rect', '3']),
        /requires 2 argument\(s\), got 1/);
});

test('applyModifier rejects BeginProd/EndProd directly - only applyModifiers may apply them', () => {
    const bc = rectangularBoard(2, 2);
    assert.throws(() => applyModifier(bc, { kind: 'BeginProd', boardType: 'rect', boardArgs: [numArg(1), numArg(1)] }));
    assert.throws(() => applyModifier(bc, { kind: 'EndProd' }));
});

test('applyModifiers: beginprod...endprod multiplies the finished inner board into the outer one', () => {
    const outer = rectangularBoard(2, 1);
    const result = applyModifiers(outer, [
        parseModifier('beginprod', ['rect', '1', '2']),
        parseModifier('endprod', []),
    ]);
    assert.deepEqual(result, product(outer, rectangularBoard(1, 2)));
});

test('applyModifiers: modifiers between beginprod/endprod transform the inner board, not the outer one', () => {
    const outer = rectangularBoard(2, 1);
    const result = applyModifiers(outer, [
        parseModifier('beginprod', ['rect', '2', '2']),
        parseModifier('es', ['2']),
        parseModifier('endprod', []),
    ]);
    const expectedInner = applyModifier(rectangularBoard(2, 2), parseModifier('es', ['2']));
    assert.deepEqual(result, product(outer, expectedInner));
});

test('applyModifiers: beginprod/endprod pairs nest, each popping its own stack entry', () => {
    const outer = rectangularBoard(3, 1);
    const result = applyModifiers(outer, [
        parseModifier('beginprod', ['rect', '1', '2']),
        parseModifier('beginprod', ['rect', '2', '1']),
        parseModifier('endprod', []),
        parseModifier('endprod', []),
    ]);
    const innerProduct = product(rectangularBoard(1, 2), rectangularBoard(2, 1));
    assert.deepEqual(result, product(outer, innerProduct));
});

test('applyModifiers throws on an unmatched endprod or a dangling beginprod', () => {
    const bc = rectangularBoard(2, 2);
    assert.throws(() => applyModifiers(bc, [parseModifier('endprod', [])]), 'endprod with no beginprod');
    assert.throws(
        () => applyModifiers(bc, [parseModifier('beginprod', ['rect', '1', '1'])]),
        'beginprod never closed',
    );
});
