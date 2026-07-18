// Regression test: komi must actually affect winner computation.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { BoardState } from '../shared/boardState.ts';
import { rectangularBoard } from '../shared/boardConfig.ts';

// Both players pass immediately (no PLACE moves), so the game ends tied 0-0
// on stones/territory - isolates komi as the only thing that can decide a
// winner, same "both pass" pattern used in boardState.finishedGame.test.ts.
function playToTiedFinish(komi: number[]) {
    const bc = rectangularBoard(1, 1);
    const bs = new BoardState(2, 2, [{ player: 1, stones: [1, 0], protected: [0, 0], friendly: [0, 0] }, { player: 2, stones: [0, 1], protected: [0, 0], friendly: [0, 0] }], [[null, null], [null, null]], [null, null], { 1: new Set([1]), 2: new Set([2]) }, false, 'area', komi, 'situational', false, null, new Array(bc.N).fill(0), bc);
    assert.equal(bs.makeMove(null), true);
    assert.equal(bs.makeMove(null), true);
    assert.equal(bs.gameOver(), true);
    return bs;
}

test('with no komi, a 0-0 tie has both players as winners', () => {
    const bs = playToTiedFinish([0, 0]);
    assert.deepEqual(bs.winners, [1, 2]);
});

test('komi breaks the tie in favor of the player with more komi', () => {
    const bs = playToTiedFinish([0, 1]);
    assert.deepEqual(bs.winners, [2]);
});

test('a resigned player\'s komi does not count', () => {
    const bc = rectangularBoard(3, 3);
    // 3 players so resigning 1 leaves 2 active, matching the pattern used in
    // boardState.pass.test.ts (resigning down to 1 active player would end
    // the game via the separate "<=1 active players" rule instead).
    const turnList = [{ player: 1, stones: [1, 0, 0], protected: [0, 0, 0], friendly: [0, 0, 0] }, { player: 2, stones: [0, 1, 0], protected: [0, 0, 0], friendly: [0, 0, 0] }, { player: 3, stones: [0, 0, 1], protected: [0, 0, 0], friendly: [0, 0, 0] }];
    const bs = new BoardState(3, 3, turnList, [[null, null, null], [null, null, null], [null, null, null]], [null, null, null], { 1: new Set([1]), 2: new Set([2]), 3: new Set([3]) }, false, 'area', [0, 5, 0], 'situational', false, null, new Array(bc.N).fill(0), bc);
    bs.resign(2);
    assert.equal(bs.makeMove(null), true);
    assert.equal(bs.makeMove(null), true);
    assert.equal(bs.makeMove(null), true);
    assert.equal(bs.gameOver(), true);
    // player 2's komi=5 would otherwise make them the sole winner.
    assert.deepEqual(bs.winners, [1, 3]);
});
