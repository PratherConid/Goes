// Regression tests for the FinishedGame / BoardState.fromFinishedGame reconstruction
// pipeline used to persist and resync finished online games.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { BoardState } from '../shared/boardState.ts';
import { rectangularBoard } from '../shared/boardConfig.ts';
import { FinishedGame, GameConfig } from '../shared/types.ts';

// numPlayers=2 (not 1): BoardState.gameOver() treats "<=1 non-resigned players"
// as an automatic win, so a 1-player game would already be "over" at
// construction and every makeMove() call would short-circuit to false.
//
// Both players pass immediately (no PLACE moves at all), so the game reaches
// game over via 2 consecutive passes (turnList.length=2) without ever
// placing a stone - simplest way to reach a deterministic finish without any
// board/liberty considerations.
function playShortGame() {
    const bc = rectangularBoard(1, 1);
    const bs = new BoardState(2, 2, [{ player: 1, stones: [1, 0], protected: [0, 0], friendly: [0, 0] }, { player: 2, stones: [0, 1], protected: [0, 0], friendly: [0, 0] }], [[null, null], [null, null]], [null, null], { 1: new Set([1]), 2: new Set([2]) }, false, 'area', [0, 0], 'situational', false, null, new Array(bc.N).fill(0), bc);
    assert.equal(bs.makeMove(null), true, 'player 1 passes');
    assert.equal(bs.gameOver(), false, 'sanity: only 1 consecutive pass so far, turnList.length=2');
    assert.equal(bs.makeMove(null), true, 'player 2 also passes - 2nd consecutive pass ends the game');
    assert.equal(bs.gameOver(), true);
    return { bc, bs };
}

test('BoardState.fromFinishedGame reconstructs the same final view as the live game', () => {
    const { bc, bs } = playShortGame();
    const config = new GameConfig('rect', [1, 1], 2, 2, [{ player: 1, stones: [1, 0], protected: [0, 0], friendly: [0, 0] }, { player: 2, stones: [0, 1], protected: [0, 0], friendly: [0, 0] }], [[null, null], [null, null]], [null, null], { 1: new Set([1]), 2: new Set([2]) }, false, 'area', [0, 0], 'situational', false, null);
    const fg = new FinishedGame(config, bs.moveInfos().map(m => ({ pos: m.pos, stone: m.stone })), new Map(bs.resigns));

    const reconstructed = BoardState.fromFinishedGame(fg, bc);

    const liveView = bs.getView();
    const reView = reconstructed.getView();
    assert.equal(reView.gameOver, liveView.gameOver);
    assert.deepEqual(reView.winners, liveView.winners);
    assert.deepEqual(reView.situations[reView.situations.length - 1].board, liveView.situations[liveView.situations.length - 1].board);
});

test('FinishedGame.toJSON()/fromJSON() round-trips', () => {
    const { bs } = playShortGame();
    const config = new GameConfig('rect', [1, 1], 2, 2, [{ player: 1, stones: [1, 0], protected: [0, 0], friendly: [0, 0] }, { player: 2, stones: [0, 1], protected: [0, 0], friendly: [0, 0] }], [[null, null], [null, null]], [null, null], { 1: new Set([1]), 2: new Set([2]) }, false, 'area', [0, 0], 'situational', false, null);
    const fg = new FinishedGame(config, bs.moveInfos().map(m => ({ pos: m.pos, stone: m.stone })), new Map(bs.resigns));

    const roundTripped = FinishedGame.fromJSON(JSON.parse(JSON.stringify(fg.toJSON())));

    assert.deepEqual(roundTripped.moves, fg.moves);
    assert.deepEqual([...roundTripped.resigns.entries()], [...fg.resigns.entries()]);
    assert.equal(roundTripped.config.numStones, fg.config.numStones);
    assert.equal(roundTripped.config.boardType, fg.config.boardType);
});
