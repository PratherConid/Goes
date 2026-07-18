// Regression tests distinguishing positional from situational superko.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { BoardState } from '../shared/boardState.ts';
import { rectangularBoard } from '../shared/boardConfig.ts';
import type { KoRule } from '../shared/types.ts';

// A minimal ko shape on a 5x5 board (row-major index = row*5+col): Black at
// the ko point (12) is in atari with its only liberty at 13; White flanks 12
// on the other 3 sides, Black flanks 13's other 3 neighbors so that White
// capturing at 13 leaves the new White stone in atari too (liberty = 12).
function koSetupBoard(bc: ReturnType<typeof rectangularBoard>): number[] {
    const board = new Array(bc.N).fill(0);
    board[12] = 1; board[7] = 2; board[17] = 2; board[11] = 2;
    board[8] = 1; board[18] = 1; board[14] = 1;
    return board;
}

// turnList stones=[2,1,2] (length 3, White first): an immediate 2-ply
// capture-then-recapture lands on a different plyCount%turnList.length
// than the initial setup did. This is what actually distinguishes the two
// rules - under a plain period-2 alternation, an immediate recapture always
// preserves parity, so positional and situational agree (both forbid it);
// the difference only shows up when the board recurs after a ply-gap that
// isn't a multiple of the turn period.
function setupKo(koRule: KoRule): BoardState {
    const bc = rectangularBoard(5, 5);
    const turnList = [
        { player: 2, stones: [0, 1], protected: [0, 0], friendly: [0, 0] },
        { player: 1, stones: [1, 0], protected: [0, 0], friendly: [0, 0] },
        { player: 2, stones: [0, 1], protected: [0, 0], friendly: [0, 0] },
    ];
    const bs = new BoardState(2, 2, turnList, [[null, null], [null, null]], [null, null], { 1: new Set([1]), 2: new Set([2]) }, false, 'area', [0, 0], koRule, false, null, koSetupBoard(bc), bc);
    assert.equal(bs.makeMove(13), true, 'White captures the Black ko stone at 12');
    return bs;
}

test('positional superko forbids recapturing the ko point regardless of the mover', () => {
    const bs = setupKo('positional');
    assert.equal(bs.legalMovesData().captures[1][12], null);
    assert.equal(bs.makeMove(12), false);
});

test('situational superko allows the same recapture, since the mover differs from the original occurrence', () => {
    const bs = setupKo('situational');
    assert.ok(bs.legalMovesData().captures[1][12] !== null);
    assert.equal(bs.makeMove(12), true);
});
