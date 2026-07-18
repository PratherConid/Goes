// Regression tests for TurnInfo.friendly: a stone color marked friendly for a
// turn doesn't count as blocking anyone else's liberties that turn - a group
// adjacent to a friendly-colored stone treats that cell as a liberty rather
// than an occupied neighbor (see groupLiberty in shared/boardState.ts).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { BoardState } from '../shared/boardState.ts';
import { rectangularBoard } from '../shared/boardConfig.ts';

// 3x3 board: black (1) is a lone stone at the center (4), completely
// surrounded by white (2) at 1, 3, 5, 7 - normally zero liberties. White
// moves first (turnIdx 0), so black qualifies as an "other" group for
// earlyOppCapture regardless of white's own friendliness.
function surroundedBlackBoard() {
    const bc = rectangularBoard(3, 3);
    const board = new Array(bc.N).fill(0);
    board[4] = 1;
    board[1] = 2; board[3] = 2; board[5] = 2; board[7] = 2;
    return { bc, board };
}

test('a friendly color is not counted as blocking a neighboring group\'s liberties', () => {
    const { bc, board } = surroundedBlackBoard();
    const turnList = [
        { player: 2, stones: [0, 1], protected: [0, 0], friendly: [0, 1] },  // white (stone 2) friendly
        { player: 1, stones: [1, 0], protected: [0, 0], friendly: [0, 0] },
    ];
    const bs = new BoardState(2, 2, turnList, [[null, null], [null, null]], [null, null], { 1: new Set([1]), 2: new Set([2]) }, false, 'area', [0, 0], 'situational', false, null, board, bc);

    assert.deepEqual(bs.legalMovesData().passCapture, new Set(),
        'black has real liberties via its friendly white neighbors, so it is not an early-capture target');

    assert.equal(bs.makeMove(null), true);   // white passes
    assert.equal(bs.board[4], 1, 'black is untouched - it was never actually down to zero liberties');
});

test('without friendly, the same lone stone is captured as soon as its neighbor moves', () => {
    const { bc, board } = surroundedBlackBoard();
    const turnList = [
        { player: 2, stones: [0, 1], protected: [0, 0], friendly: [0, 0] },  // white not friendly
        { player: 1, stones: [1, 0], protected: [0, 0], friendly: [0, 0] },
    ];
    const bs = new BoardState(2, 2, turnList, [[null, null], [null, null]], [null, null], { 1: new Set([1]), 2: new Set([2]) }, false, 'area', [0, 0], 'situational', false, null, board, bc);

    assert.deepEqual(bs.legalMovesData().passCapture, new Set([4]),
        'black is genuinely at zero liberties and is an early-capture target');

    assert.equal(bs.makeMove(null), true);   // white passes
    assert.equal(bs.board[4], 0, 'black is captured, even by a mere pass (earlyOppCapture applies unconditionally)');
});

// 3x3 board: white (2) is a lone stone at the center (4) with exactly one
// liberty (node 1, its only empty neighbor); black (1) occupies 3, 5, 7.
// Black playing at node 1 would normally fill white's last liberty and
// capture it.
function almostCapturedWhiteBoard() {
    const bc = rectangularBoard(3, 3);
    const board = new Array(bc.N).fill(0);
    board[4] = 2;
    board[3] = 1; board[5] = 1; board[7] = 1;
    return { bc, board };
}

test('a friendly mover cannot capture an opponent group by filling its last liberty', () => {
    const { bc, board } = almostCapturedWhiteBoard();
    const turnList = [
        { player: 1, stones: [1, 0], protected: [0, 0], friendly: [1, 0] },  // black (stone 1) friendly
        { player: 2, stones: [0, 1], protected: [0, 0], friendly: [0, 0] },
    ];
    const bs = new BoardState(2, 2, turnList, [[null, null], [null, null]], [null, null], { 1: new Set([1]), 2: new Set([2]) }, false, 'area', [0, 0], 'situational', false, null, board, bc);

    assert.deepEqual(bs.legalMovesData().captures[1][1], new Set(),
        'legal (black still has other empty neighbors to move into), but captures nothing');
    assert.equal(bs.makeMove(1), true);
    assert.equal(bs.board[4], 2, 'white survives - a friendly stone does not take away white\'s last liberty');
    assert.equal(bs.board[1], 1, 'black\'s new stone is placed as normal');
});

test('without friendly, the same move captures the opponent as expected', () => {
    const { bc, board } = almostCapturedWhiteBoard();
    const turnList = [
        { player: 1, stones: [1, 0], protected: [0, 0], friendly: [0, 0] },
        { player: 2, stones: [0, 1], protected: [0, 0], friendly: [0, 0] },
    ];
    const bs = new BoardState(2, 2, turnList, [[null, null], [null, null]], [null, null], { 1: new Set([1]), 2: new Set([2]) }, false, 'area', [0, 0], 'situational', false, null, board, bc);

    assert.deepEqual(bs.legalMovesData().captures[1][1], new Set([4]));
    assert.equal(bs.makeMove(1), true);
    assert.equal(bs.board[4], 0, 'white is captured as normal');
});
