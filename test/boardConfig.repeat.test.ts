// Regression tests for the tree-shaped 'Repeat' BoardModifier ("repeat <num>"..."endrepeat",
// mirroring "beginprod"..."endprod" - see BoardModifier's/parseModifiers's own doc comments in
// shared/boardConfig.ts) plus how it nests against 'Prod' when both appear in the same text.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { rectangularBoard, parseModifiers, applyModifiers, numArg } from '../shared/boardConfig.ts';

test('parseModifiers builds a nested Repeat node from a repeat...endrepeat span', () => {
    assert.deepEqual(parseModifiers('repeat 3; scale 2; endrepeat'),
        [{ kind: 'Repeat', count: 3, modifiers: [{ kind: 'Scale', factor: 2 }] }]);
});

test('applyModifier applies a Repeat node: applies its own nested modifiers count times in a row', () => {
    const bc = rectangularBoard(2, 1); // positions [-0.5, 0], [0.5, 0]
    const result = applyModifiers(bc, parseModifiers('repeat 3; scale 2; endrepeat'));
    assert.deepEqual(result.emb.pos, bc.emb.pos.map(p => p.map(v => v * 2 ** 3)));
});

test('repeat 0 is a no-op regardless of its nested modifiers', () => {
    const bc = rectangularBoard(2, 1);
    const result = applyModifiers(bc, parseModifiers('repeat 0; scale 2; endrepeat'));
    assert.deepEqual(result, bc);
});

test('parseModifiers rejects malformed repeat/endrepeat', () => {
    assert.throws(() => parseModifiers('repeat; endrepeat'), /repeat takes exactly 1 argument/);
    assert.throws(() => parseModifiers('repeat abc; endrepeat'), /count must be a nonnegative integer/);
    assert.throws(() => parseModifiers('repeat -1; endrepeat'), /count must be a nonnegative integer/);
    assert.throws(() => parseModifiers('repeat 1.5; endrepeat'), /count must be a nonnegative integer/);
    assert.throws(() => parseModifiers('repeat 3; endrepeat x'), /endrepeat takes no arguments/);
    assert.throws(() => parseModifiers('endrepeat'), /endrepeat: no matching repeat/);
    assert.throws(() => parseModifiers('repeat 3'), /repeat: missing matching endrepeat/);
});

test('repeat/beginprod nest freely inside each other, each closed by its own closer', () => {
    const modifiers = parseModifiers('beginprod rect 2 2; repeat 2; es 1; endrepeat; endprod');
    assert.deepEqual(modifiers, [{
        kind: 'Prod', boardType: 'rect', boardArgs: [numArg(2), numArg(2)],
        modifiers: [{ kind: 'Repeat', count: 2, modifiers: [{ kind: 'EdgeSplit', splitN: 1 }] }],
    }]);
});

test('a closer that does not match its nearest still-open opener throws', () => {
    assert.throws(() => parseModifiers('beginprod rect 1 1; endrepeat; endprod'),
        /beginprod: expected matching endprod, got 'endrepeat'/);
    assert.throws(() => parseModifiers('repeat 2; endprod; endrepeat'),
        /repeat: expected matching endrepeat, got 'endprod'/);
});
