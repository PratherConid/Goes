// Regression tests for the mergeClose board modifier: merges every pair of nodes
// closer than `dist` into one node, transitively (a chain of close nodes all
// collapse into a single node, not just each individual close pair).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    rectangularBoard, mergeClose, parseModifier, applyModifier, MC_DEFAULT_DIST,
} from '../shared/boardConfig.ts';

test('a chain of collinear nodes each 1 apart all merge into one node when dist > 1', () => {
    const bc = rectangularBoard(3, 1); // nodes at x=-1, 0, 1 - each adjacent pair 1 apart
    const merged = mergeClose(bc, 1.1);
    assert.equal(merged.N, 1, 'node0-node1 (dist 1) and node1-node2 (dist 1) merge directly; ' +
        'node0-node2 (dist 2) then merges transitively via node1, even though it is not itself close');
});

test('distance exactly equal to dist does not merge (strict less-than)', () => {
    const bc = rectangularBoard(2, 1); // two nodes exactly 1 apart
    assert.equal(mergeClose(bc, 1).N, 2, 'dist=1 is not < 1, so the pair stays separate');
    assert.equal(mergeClose(bc, 1.0001).N, 1, 'a dist just above the true distance merges the pair');
});

test('a dist below every pairwise distance leaves the board unchanged', () => {
    const bc = rectangularBoard(2, 2);
    const merged = mergeClose(bc, 0.5);
    assert.equal(merged.N, bc.N);
    assert.deepEqual(merged.adj, bc.adj);
});

test('a 2x2 grid fully collapses into one node once dist exceeds the edge length ' +
    '(diagonal pairs merge transitively via the grid cycle, despite being farther than dist)', () => {
    const bc = rectangularBoard(2, 2); // 4 corners, grid edges length 1, diagonals length sqrt(2)
    const merged = mergeClose(bc, 1.1);
    assert.equal(merged.N, 1);
});

test('parseModifier("mc", ...) round-trips through applyModifier the same as calling mergeClose directly', () => {
    const bc = rectangularBoard(3, 1);
    const modifier = parseModifier('mc', ['1.1']);
    assert.deepEqual(modifier, { kind: 'MergeClose', dist: 1.1 });
    assert.deepEqual(applyModifier(bc, modifier), mergeClose(bc, 1.1));
});

test('parseModifier("mc", ...) with no argument defaults dist to MC_DEFAULT_DIST', () => {
    assert.deepEqual(parseModifier('mc', []), { kind: 'MergeClose', dist: MC_DEFAULT_DIST });
});

test('parseModifier("mc", ...) rejects a non-positive or malformed dist, or too many arguments', () => {
    assert.throws(() => parseModifier('mc', ['0']));
    assert.throws(() => parseModifier('mc', ['-1']));
    assert.throws(() => parseModifier('mc', ['abc']));
    assert.throws(() => parseModifier('mc', ['1', '2']));
});
