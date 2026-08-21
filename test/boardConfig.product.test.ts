// Regression tests for the `product` board-transform function and the tree-shaped 'Prod'
// BoardModifier that drives it: `prod` parses straight to a one-shot Prod node (empty nested
// `modifiers`), while `beginprod`/`endprod` are handled by parseModifiers - which parses a whole
// modifiers-list text at once, folding a beginprod...endprod span (however many commands long)
// into a single nested Prod node via its own self-recursion - see BoardModifier's/parseModifiers's
// own doc comments in shared/boardConfig.ts.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    rectangularBoard, product, parseModifier, parseModifiers, applyModifier, applyModifiers, numArg,
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

test('parseModifier(\'prod\', ...) parses a one-shot Prod node with empty nested modifiers', () => {
    assert.deepEqual(parseModifier('prod', ['rect', '3', '3']),
        { kind: 'Prod', boardType: 'rect', boardArgs: [numArg(3), numArg(3)], modifiers: [] });
});

test('parseModifiers builds a nested Prod node from a beginprod...endprod span', () => {
    assert.deepEqual(parseModifiers('beginprod rect 3 3; endprod'),
        [{ kind: 'Prod', boardType: 'rect', boardArgs: [numArg(3), numArg(3)], modifiers: [] }]);
});

test('parseModifiers rejects malformed beginprod/endprod', () => {
    assert.throws(() => parseModifiers('beginprod'), 'needs at least a board type');
    assert.throws(() => parseModifiers('beginprod rect 3 abc; endprod'), 'non-integer board arg');
    assert.throws(() => parseModifiers('beginprod nope 1 2; endprod'), 'unknown board type');
    assert.throws(() => parseModifiers('beginprod rect 3 3; endprod x'), 'endprod takes no arguments');
    assert.throws(() => parseModifiers('endprod'), 'endprod with no matching beginprod');
});

test('parseModifiers(\'beginprod ...\') truncates extra board args to the type\'s required count', () => {
    assert.deepEqual(parseModifiers('beginprod rect 3 3 99 100; endprod'),
        [{ kind: 'Prod', boardType: 'rect', boardArgs: [numArg(3), numArg(3)], modifiers: [] }]);
});

test('parseModifiers(\'beginprod ...\') throws when fewer board args than required are given', () => {
    assert.throws(() => parseModifiers('beginprod rect 3; endprod'),
        /requires 2 argument\(s\), got 1/);
});

test('applyModifier applies a Prod node: builds boardType/boardArgs, applies its own nested ' +
    'modifiers, then multiplies the result into bc', () => {
    const outer = rectangularBoard(2, 1);
    const result = applyModifier(
        outer, { kind: 'Prod', boardType: 'rect', boardArgs: [numArg(1), numArg(2)], modifiers: [] });
    assert.deepEqual(result, product(outer, rectangularBoard(1, 2)));
});

test('a Prod node\'s own nested modifiers transform its inner board, not the outer one', () => {
    const outer = rectangularBoard(2, 1);
    const modifiers = parseModifiers('beginprod rect 2 2; es 2; endprod');
    const result = applyModifiers(outer, modifiers);
    const expectedInner = applyModifier(rectangularBoard(2, 2), parseModifier('es', ['2']));
    assert.deepEqual(result, product(outer, expectedInner));
});

test('beginprod/endprod pairs nest into a nested Prod node inside a nested Prod node', () => {
    const outer = rectangularBoard(3, 1);
    const modifiers = parseModifiers('beginprod rect 1 2; beginprod rect 2 1; endprod; endprod');
    assert.deepEqual(modifiers, [{
        kind: 'Prod', boardType: 'rect', boardArgs: [numArg(1), numArg(2)],
        modifiers: [{ kind: 'Prod', boardType: 'rect', boardArgs: [numArg(2), numArg(1)], modifiers: [] }],
    }]);
    const result = applyModifiers(outer, modifiers);
    const innerProduct = product(rectangularBoard(1, 2), rectangularBoard(2, 1));
    assert.deepEqual(result, product(outer, innerProduct));
});

test('a beginprod left unmatched throws (an unclosed beginprod at the end of the text)', () => {
    assert.throws(() => parseModifiers('beginprod rect 1 1'), /missing matching endprod/);
});
