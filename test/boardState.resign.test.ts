// Regression tests for resignation-caused game-over: BoardState.gameOver()
// checks resignedPlayers live (see shared/boardState.ts) rather than having
// resign() stamp a flag onto the current last move, so resign()/withdrawMove()
// never touch moveInfos/legalMoveHistory at all - the resignation is recorded
// purely in `resigns`, and gameOver() re-derives "<=1 non-resigned players"
// from it on every call, including across withdrawMove() (resigns are never
// un-resigned).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { BoardState } from '../shared/boardState.ts';
import { rectangularBoard } from '../shared/boardConfig.ts';

function twoPlayerGame() {
    const bc = rectangularBoard(5, 5);
    const turnList = [
        { player: 1, stones: [1, 1], protected: [0, 0], friendly: [0, 0] },
        { player: 2, stones: [1, 1], protected: [0, 0], friendly: [0, 0] },
    ];
    return new BoardState(2, 2, turnList, [[null, null], [null, null]], [null, null], { 1: new Set([1]), 2: new Set([2]) },
        false, 'area', [0, 0], 'situational', false, null, new Array(bc.N).fill(0), bc);
}

test('resigning before any move is made ends the game via the _noMove sentinel', () => {
    const bs = twoPlayerGame();
    assert.equal(bs.gameOver(), false);

    bs.resign(1);
    assert.equal(bs.gameOver(), true, 'only 1 non-resigned player remains');
    assert.notEqual(bs.getView().winners, null);
    assert.equal(bs.makeMove(0, 1), false, 'no further moves once the game is over');
});

test('resigning after real moves have been made ends the game via the last real move', () => {
    const bs = twoPlayerGame();
    bs.makeMove(0, 1);
    bs.makeMove(4, 1);
    assert.equal(bs.gameOver(), false);

    bs.resign(1);
    assert.equal(bs.gameOver(), true);
    assert.equal(bs.lastMove().allPassed, false, 'resign() never stamps anything onto MoveInfo - gameOver() checks resignedPlayers directly');
});

test('resigning with more than one non-resigned player left does not end the game', () => {
    const bc = rectangularBoard(5, 5);
    const turnList = [
        { player: 1, stones: [1, 0, 0], protected: [0, 0, 0], friendly: [0, 0, 0] },
        { player: 2, stones: [0, 1, 0], protected: [0, 0, 0], friendly: [0, 0, 0] },
        { player: 3, stones: [0, 0, 1], protected: [0, 0, 0], friendly: [0, 0, 0] },
    ];
    const bs = new BoardState(3, 3, turnList, [[null, null, null], [null, null, null], [null, null, null]], [null, null, null],
        { 1: new Set([1]), 2: new Set([2]), 3: new Set([3]) }, false, 'area', [0, 0, 0], 'situational', false, null, new Array(bc.N).fill(0), bc);

    bs.resign(1);
    assert.equal(bs.gameOver(), false, '2 non-resigned players remain');

    bs.resign(2);
    assert.equal(bs.gameOver(), true, 'only player 3 remains');
});

test('withdrawMove() past a resignation-ended game stays game-over (resignedPlayers is untouched by withdrawMove)', () => {
    const bs = twoPlayerGame();
    bs.makeMove(0, 1);
    bs.makeMove(4, 1);
    bs.resign(2);
    assert.equal(bs.gameOver(), true);
    assert.equal(bs.lastMove().pos, 4, 'sanity: this is the most recent real move');

    bs.withdrawMove();
    // The resignation itself is permanent (withdrawMove doesn't un-resign) and
    // gameOver() checks resignedPlayers live, so no re-stamping is needed for
    // the game to still register as over here.
    assert.equal(bs.gameOver(), true, 'resignation-caused game-over survives withdrawing a move played after it');
    assert.equal(bs.lastMove().pos, 0);
});
