// Regression tests for shared/avlTree.ts's set semantics: `size` must track the number of distinct
// values actually held (per the tree's comparator) - BoardState's turn index is derived from
// sortedSituations.size, so a drifting count would silently misattribute turns.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { AVLTree } from '../shared/avlTree.ts';

const numCmp = (a: number, b: number) => a - b;

test('inserting a value already in the tree is a no-op, size included', () => {
    const t = new AVLTree(numCmp);
    t.insert(1);
    t.insert(2);
    t.insert(1);
    assert.equal(t.size, 2);
    assert.ok(t.has(1));
});

test('removing a value not in the tree is a no-op, size included', () => {
    const t = new AVLTree(numCmp);
    t.insert(1);
    t.remove(5);
    assert.equal(t.size, 1);
    t.remove(1);
    assert.equal(t.size, 0);
    t.remove(1);
    assert.equal(t.size, 0);
    assert.equal(t.has(1), false);
});
