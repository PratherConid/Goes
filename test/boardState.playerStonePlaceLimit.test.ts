// Regression tests for GameConfig/BoardState.playerStonePlaceLimit: a per-
// player, per-stone-color cap on how many times that stone may ever be
// placed by that player over the whole game (null = unlimited). Enforced in
// calculateLegalMoves - a stone whose limit is reached becomes illegal for
// that player everywhere on the board, exactly as if it were never offered
// that turn (see shared/boardState.ts).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { BoardState } from '../shared/boardState.ts';
import { rectangularBoard } from '../shared/boardConfig.ts';

// 5x5 empty board; player 1 alone moves every turn (turnList has a single
// entry), offered both stones each turn, so captures/liberties never come
// into play - isolating the placement-count-limit mechanism from the rest of
// the rules. Node indices used below (0, 4, 20) are the board's far corners,
// so placements never interact via adjacency either.
function emptyGame(playerStonePlaceLimit: (number | null)[][]) {
    const bc = rectangularBoard(5, 5);
    const turnList = [{ player: 1, stones: [1, 1], protected: [0, 0], friendly: [0, 0] }];
    return new BoardState(2, 2, turnList, playerStonePlaceLimit, [null, null], { 1: new Set([1]), 2: new Set([2]) },
        false, 'area', [0, 0], 'situational', false, null, new Array(bc.N).fill(0), bc);
}

test('a stone becomes illegal for a player once their placement limit is reached, while other stones/players are unaffected', () => {
    // stone1 limited to 1 placement for player 1; stone2 unlimited.
    const bs = emptyGame([[1, null], [null, null]]);

    assert.ok(bs.legalMovesData().legalsForStone[1].size > 0, 'stone1 starts legal (0 placements so far, limit 1)');
    assert.ok(bs.legalMovesData().legalsForStone[2].size > 0, 'stone2 (unlimited) is legal');

    assert.equal(bs.makeMove(0, 1), true, 'first stone1 placement succeeds');
    assert.equal(bs.playerStonePlaceCnt()[0][0], 1, 'stone1/player1 count is now 1');

    assert.equal(bs.legalMovesData().legalsForStone[1].size, 0, 'stone1 is now illegal for player1 everywhere - limit reached');
    assert.equal(bs.makeMove(4, 1), false, 'placing stone1 again is rejected');
    assert.ok(bs.legalMovesData().legalsForStone[2].size > 0, 'stone2 (unlimited) remains legal');
    assert.equal(bs.makeMove(4, 2), true, 'stone2 is still placeable');
});

test('null means unlimited: repeated placements of the same stone stay legal', () => {
    const bs = emptyGame([[null, null], [null, null]]);
    assert.equal(bs.makeMove(0, 1), true);
    assert.equal(bs.makeMove(4, 1), true);
    assert.equal(bs.makeMove(20, 1), true);
    assert.equal(bs.playerStonePlaceCnt()[0][0], 3);
    assert.ok(bs.legalMovesData().legalsForStone[1].size > 0, 'stone1 stays legal - no limit');
});

test('HistoryEntry.playerStonePlaceCnt is cumulative across plies and correctly rewinds on withdrawMove()', () => {
    const bs = emptyGame([[null, null], [null, null]]);
    assert.deepEqual(bs.history[0].playerStonePlaceCnt, [[0, 0], [0, 0]], 'genesis entry starts at all zeros');

    bs.makeMove(0, 1);
    assert.deepEqual(bs.history[bs.history.length - 1].playerStonePlaceCnt, [[1, 0], [0, 0]]);

    bs.makeMove(4, 2);
    assert.deepEqual(bs.history[bs.history.length - 1].playerStonePlaceCnt, [[1, 0], [1, 0]]);

    bs.makeMove(20, 1);
    assert.deepEqual(bs.history[bs.history.length - 1].playerStonePlaceCnt, [[2, 0], [1, 0]]);

    bs.withdrawMove();
    assert.deepEqual(bs.playerStonePlaceCnt(), [[1, 0], [1, 0]], 'live count rewinds to match the previous ply');

    bs.withdrawMove();
    assert.deepEqual(bs.playerStonePlaceCnt(), [[1, 0], [0, 0]]);
});

test('a pass leaves playerStonePlaceCnt unchanged', () => {
    const bc = rectangularBoard(1, 1);
    const turnList = [
        { player: 1, stones: [1, 0], protected: [0, 0], friendly: [0, 0] },
        { player: 2, stones: [0, 1], protected: [0, 0], friendly: [0, 0] },
    ];
    const bs = new BoardState(2, 2, turnList, [[null, null], [null, null]], [null, null], { 1: new Set([1]), 2: new Set([2]) },
        false, 'area', [0, 0], 'situational', false, null, new Array(bc.N).fill(0), bc);
    bs.makeMove(null);
    bs.makeMove(null);
    assert.deepEqual(bs.playerStonePlaceCnt(), [[0, 0], [0, 0]], 'passes never increment the count');
});
