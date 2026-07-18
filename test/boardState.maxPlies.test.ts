// Regression tests for GameConfig/BoardState.maxPlies: a hard cap on the total
// number of plies before the game is automatically ended, regardless of
// whether the move that reaches it was a PLACE or a PASS. This is a separate
// game-over trigger from the pre-existing "everyone passes consecutively"
// mechanism (see boardState.pass.test.ts) - neither uses a distinct GAMEOVER
// move type; instead BoardState.gameOver() checks maxPlies live (against
// history.length), independently of MoveInfo.allPassed, which only reflects
// the all-passed trigger specifically (see shared/boardState.ts's
// makeMove/gameOver/allPassed).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { BoardState } from '../shared/boardState.ts';
import { rectangularBoard } from '../shared/boardConfig.ts';
import { MoveType } from '../shared/types.ts';

// 5x5 empty board. `twoPlayerTurns` picks between a single-player turnList
// (so a lone PLACE move can be isolated from the consecutive-pass mechanism,
// which needs turnList.length plies to trigger) and a 2-player alternating
// turnList (so a single PASS's consecutivePasses=1 stays below
// turnList.length=2, isolating maxPlies as the sole cause of game-over).
function game(maxPlies: number | null, twoPlayerTurns: boolean) {
    const bc = rectangularBoard(5, 5);
    const turnList = twoPlayerTurns
        ? [
            { player: 1, stones: [1, 1], protected: [0, 0], friendly: [0, 0] },
            { player: 2, stones: [1, 1], protected: [0, 0], friendly: [0, 0] },
        ]
        : [{ player: 1, stones: [1, 1], protected: [0, 0], friendly: [0, 0] }];
    return new BoardState(2, 2, turnList, [[null, null], [null, null]], [null, null], { 1: new Set([1]), 2: new Set([2]) },
        false, 'area', [0, 0], 'situational', false, maxPlies, new Array(bc.N).fill(0), bc);
}

test('maxPlies: null means unlimited - the game never auto-ends from ply count alone', () => {
    const bs = game(null, false);
    assert.equal(bs.makeMove(0, 1), true);
    assert.equal(bs.makeMove(4, 1), true);
    assert.equal(bs.makeMove(20, 1), true);
    assert.equal(bs.gameOver(), false);
});

test('reaching maxPlies via a PLACE move ends the game', () => {
    const bs = game(1, false);
    assert.equal(bs.makeMove(0, 1), true, 'the single allowed ply is a placement');

    assert.equal(bs.lastMove().moveType, MoveType.PLACE);
    assert.equal(bs.lastMove().allPassed, false, 'this PLACE move did not complete an all-pass round - maxPlies alone ends the game');
    assert.equal(bs.gameOver(), true);
    assert.notEqual(bs.getView().winners, null);

    assert.equal(bs.makeMove(4, 1), false, 'no further moves once the game is over');
});

test('reaching maxPlies via an ordinary (non-terminal) PASS move ends the game', () => {
    // turnList.length=2, so a single pass's consecutivePasses (1) alone would
    // never trigger the old consecutive-pass mechanism - maxPlies is the only
    // thing ending the game here.
    const bs = game(1, true);
    assert.equal(bs.makeMove(null), true, "player 1's only allowed ply is a pass");

    assert.equal(bs.lastMove().moveType, MoveType.PASS);
    assert.equal(bs.lastMove().consecutivePasses, 1, 'sanity: below turnList.length=2');
    assert.equal(bs.lastMove().allPassed, false, 'consecutivePasses=1 < turnList.length=2 - maxPlies alone ends the game, not an all-pass round');
    assert.equal(bs.gameOver(), true);
});

test('withdrawMove() past a maxPlies-triggered game-over move un-ends the game', () => {
    const bs = game(1, false);
    bs.makeMove(0, 1);
    assert.equal(bs.gameOver(), true);

    bs.withdrawMove();
    assert.equal(bs.gameOver(), false);
    assert.equal(bs.getView().winners, null);
});

test('the pre-existing terminal-consecutive-pass game-over still works, now reported via MoveInfo.allPassed on a PASS move', () => {
    const bs = game(null, true);
    assert.equal(bs.makeMove(null), true, 'player 1 passes');
    assert.equal(bs.gameOver(), false, 'sanity: only 1 consecutive pass so far, turnList.length=2');

    assert.equal(bs.makeMove(null), true, 'player 2 also passes - 2nd consecutive pass ends the round');
    assert.equal(bs.lastMove().moveType, MoveType.PASS, 'regression: no more distinct GAMEOVER move type');
    assert.equal(bs.lastMove().allPassed, true);
    assert.equal(bs.gameOver(), true);
});
