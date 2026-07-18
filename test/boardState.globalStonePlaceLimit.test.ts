// Regression tests for GameConfig/BoardState.globalStonePlaceLimit: a cap on
// how many times a stone color may EVER be placed in total, summed across
// EVERY player (unlike playerStonePlaceLimit, which is per-player). There's
// no separate running-count field for this - BoardState derives the current
// global count on the fly by summing playerStonePlaceCnt[stone-1] across
// every player (see calculateLegalMoves in shared/boardState.ts).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { BoardState } from '../shared/boardState.ts';
import { rectangularBoard } from '../shared/boardConfig.ts';

// 5x5 empty board; players 1 and 2 alternate turns, both offered both stones
// each turn, so captures/liberties never come into play - isolating the
// placement-count-limit mechanisms from the rest of the rules. Node indices
// used below (0, 4) are the board's far corners, so placements never interact
// via adjacency either.
function emptyGame(playerStonePlaceLimit: (number | null)[][], globalStonePlaceLimit: (number | null)[]) {
    const bc = rectangularBoard(5, 5);
    const turnList = [
        { player: 1, stones: [1, 1], protected: [0, 0], friendly: [0, 0] },
        { player: 2, stones: [1, 1], protected: [0, 0], friendly: [0, 0] },
    ];
    return new BoardState(2, 2, turnList, playerStonePlaceLimit, globalStonePlaceLimit, { 1: new Set([1]), 2: new Set([2]) },
        false, 'area', [0, 0], 'situational', false, null, new Array(bc.N).fill(0), bc);
}

test('a stone becomes illegal for EVERY player once the global limit is reached, even one who never placed it themselves', () => {
    // stone1 global limit 1 (across both players combined); stone2 unlimited; no per-player limits.
    const bs = emptyGame([[null, null], [null, null]], [1, null]);

    assert.ok(bs.legalMovesData().legalsForStone[1].size > 0, 'stone1 starts legal (global count 0, limit 1)');
    assert.equal(bs.makeMove(0, 1), true, 'player1 places stone1 (uses up the global budget)');

    // It's now player2's turn - they never placed stone1 themselves, but the global limit is still exhausted.
    assert.equal(bs.legalMovesData().legalsForStone[1].size, 0, 'stone1 is illegal for player2 too - global limit reached');
    assert.equal(bs.makeMove(4, 1), false, 'player2 placing stone1 is rejected');
    assert.ok(bs.legalMovesData().legalsForStone[2].size > 0, 'stone2 (unlimited) remains legal for player2');
});

test('null means unlimited: repeated placements across different players stay legal', () => {
    const bs = emptyGame([[null, null], [null, null]], [null, null]);
    assert.equal(bs.makeMove(0, 1), true);   // player1 places stone1
    assert.equal(bs.makeMove(4, 1), true);   // player2 places stone1
    assert.ok(bs.legalMovesData().legalsForStone[1].size > 0, 'stone1 stays legal - no global limit');
});

test('the global count sums placements across multiple different players', () => {
    const bs = emptyGame([[null, null], [null, null]], [2, null]);
    assert.equal(bs.makeMove(0, 1), true, 'player1 places stone1 (global count -> 1)');
    assert.ok(bs.legalMovesData().legalsForStone[1].size > 0, 'stone1 still legal - global count 1 < limit 2');
    assert.equal(bs.makeMove(4, 1), true, 'player2 places stone1 (global count -> 2)');
    assert.equal(bs.legalMovesData().legalsForStone[1].size, 0, 'stone1 now illegal for everyone - global limit reached');
});

test('per-player and global limits both apply - whichever is hit first blocks the stone', () => {
    // stone1: player1 limited to 1 placement; the global limit is 5, well
    // short of being the binding constraint here.
    const bs = emptyGame([[1, null], [null, null]], [5, null]);
    assert.equal(bs.makeMove(0, 1), true, 'player1 places stone1 once');
    assert.equal(bs.makeMove(4, 2), true, 'player2 places stone2 (turn cycles back to player1)');
    // Player1's own per-player limit for stone1 (1) is now reached, even
    // though the global limit (5) still has plenty of room (only 1 placed so far).
    assert.equal(bs.legalMovesData().legalsForStone[1].size, 0, 'stone1 illegal for player1 - per-player limit reached first');
});
