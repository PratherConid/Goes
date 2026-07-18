// Regression tests: stoneToPlayerMap maps each stone type to a *set* of
// scoring players (not just one) - a stone can score for multiple players
// (each gets its full point value, not a split share) or for none at all.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { BoardState } from '../shared/boardState.ts';
import { rectangularBoard } from '../shared/boardConfig.ts';

test('a stone mapped to multiple players adds its full point value to each, not split', () => {
    // Each group is separated by an empty buffer cell so it keeps a real
    // liberty and isn't swept by the early-capture cleanup before the passes
    // that end the game are even made.
    const bc = rectangularBoard(8, 1);
    const board = [1, 1, 1, 0, 2, 2, 2, 0];   // 3 stone1, 3 stone2
    const turnList = [
        { player: 1, stones: [1, 0], protected: [0, 0], friendly: [0, 0] },
        { player: 2, stones: [0, 1], protected: [0, 0], friendly: [0, 0] },
    ];
    // stone1 scores only for player 1; stone2 scores for both players 2 and 3.
    const stoneToPlayerMap = { 1: new Set([1]), 2: new Set([2, 3]) };
    const bs = new BoardState(2, 3, turnList, [[null, null, null], [null, null, null]], [null, null], stoneToPlayerMap, false, 'stone', [0, 0, 0], 'situational', false, null, board, bc);

    assert.equal(bs.makeMove(null), true);
    assert.equal(bs.makeMove(null), true);
    assert.equal(bs.gameOver(), true);
    // player 1: 3 (owns stone1 alone). players 2 and 3: 3 each (stone2's full
    // 3 points go to *each* of them - if it were split, they'd get 1.5 and
    // player 1 would be the sole winner instead of a 3-way tie).
    assert.deepEqual(bs.winners, [1, 2, 3]);
});

test('a stone mapped to an empty set scores for no one', () => {
    const bc = rectangularBoard(6, 1);
    const board = [1, 1, 0, 3, 3, 0];   // 2 stone1, 2 stone3
    const turnList = [
        { player: 1, stones: [1, 0, 0], protected: [0, 0, 0], friendly: [0, 0, 0] },
        { player: 2, stones: [0, 1, 0], protected: [0, 0, 0], friendly: [0, 0, 0] },
    ];
    // stone2 never appears on the board; stone3 does, but maps to no players.
    const stoneToPlayerMap = { 1: new Set([1]), 2: new Set([2]), 3: new Set<number>() };
    const bs = new BoardState(3, 2, turnList, [[null, null], [null, null], [null, null]], [null, null, null], stoneToPlayerMap, false, 'stone', [0, 0], 'situational', false, null, board, bc);

    assert.equal(bs.makeMove(null), true);
    assert.equal(bs.makeMove(null), true);
    assert.equal(bs.gameOver(), true);
    // player 1: 2 (owns stone1). player 2: 0 - stone3's 2 points score for no
    // one, so player 1 is the sole winner rather than a tie.
    assert.deepEqual(bs.winners, [1]);
});
