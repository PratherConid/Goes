// Regression tests for the `forcedPassOnly` pass-legality rule in BoardState.
// (Guards against the class of bug fixed on 2026-07-13: pass being disabled
// regardless of forcedPassOnly, due to unrelated turn-ownership logic.)
//
// All scenarios use numPlayers >= 2: BoardState.gameOver() treats "<= 1
// non-resigned players" as an automatic win, so a numPlayers=1 (or a
// down-to-1-active-player) config is game-over from construction, which
// would make these tests exercise the gameOver() short-circuit instead of
// the pass rule itself.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { BoardState } from '../shared/boardState.ts';
import { rectangularBoard } from '../shared/boardConfig.ts';

function freshState(forcedPassOnly: boolean) {
    const bc = rectangularBoard(3, 3);
    return new BoardState(2, 2, [{ player: 1, stones: [1, 0], protected: [0, 0], friendly: [0, 0] }, { player: 2, stones: [0, 1], protected: [0, 0], friendly: [0, 0] }], [[null, null], [null, null]], [null, null], { 1: new Set([1]), 2: new Set([2]) }, forcedPassOnly, 'area', [0, 0], 'situational', false, null, new Array(bc.N).fill(0), bc);
}

test('forcedPassOnly=false: pass is enabled even when legal PLACE moves exist', () => {
    const bs = freshState(false);
    assert.equal(bs.noTradLegal(), false, 'sanity: board is empty, legal moves should exist');
    assert.equal(bs.getView().passEnabled, true);
    assert.equal(bs.makeMove(null), true);
});

test('forcedPassOnly=true: pass is disabled while legal PLACE moves exist', () => {
    const bs = freshState(true);
    assert.equal(bs.noTradLegal(), false, 'sanity: board is empty, legal moves should exist');
    assert.equal(bs.getView().passEnabled, false);
    assert.equal(bs.makeMove(null), false);
});

test('forcedPassOnly=true: pass is enabled once no legal PLACE moves remain', () => {
    // A single node with no adjacency (rectangularBoard(1,1) has no edges at
    // all) can never legally hold a stone: placing there is always a
    // zero-liberty suicide, which is disallowed. So noTradLegal() is true
    // from the very first ply, without needing to play any moves (which
    // would risk violating the engine's "no group has 0 liberties" invariant
    // if hand-constructed instead of reached via legal play).
    const bc = rectangularBoard(1, 1);
    const bs = new BoardState(2, 2, [{ player: 1, stones: [1, 0], protected: [0, 0], friendly: [0, 0] }, { player: 2, stones: [0, 1], protected: [0, 0], friendly: [0, 0] }], [[null, null], [null, null]], [null, null], { 1: new Set([1]), 2: new Set([2]) }, true, 'area', [0, 0], 'situational', false, null, new Array(bc.N).fill(0), bc);
    assert.equal(bs.gameOver(), false, 'sanity: 2 active players, no moves played yet');
    assert.equal(bs.noTradLegal(), true, 'the lone node cannot legally hold a stone');
    assert.equal(bs.getView().passEnabled, true);
    assert.equal(bs.makeMove(null), true);
});

test('a resigned player may always pass, regardless of forcedPassOnly', () => {
    // 3 players so resigning 1 leaves 2 active - otherwise gameOver()'s
    // "<=1 active players" rule would end the game via the resignation itself,
    // masking whether the pass-permission logic is what actually allowed it.
    const bc = rectangularBoard(3, 3);
    const bs = new BoardState(3, 3, [{ player: 1, stones: [1, 0, 0], protected: [0, 0, 0], friendly: [0, 0, 0] }, { player: 2, stones: [0, 1, 0], protected: [0, 0, 0], friendly: [0, 0, 0] }, { player: 3, stones: [0, 0, 1], protected: [0, 0, 0], friendly: [0, 0, 0] }], [[null, null, null], [null, null, null], [null, null, null]], [null, null, null], { 1: new Set([1]), 2: new Set([2]), 3: new Set([3]) }, true, 'area', [0, 0, 0], 'situational', false, null, new Array(bc.N).fill(0), bc);
    assert.equal(bs.noTradLegal(), false, 'sanity: board is empty, legal moves should exist');
    bs.resign(1);
    assert.equal(bs.gameOver(), false, 'sanity: 2 non-resigned players remain');
    assert.equal(bs.getView().passEnabled, true);
    assert.equal(bs.makeMove(null), true);
});
