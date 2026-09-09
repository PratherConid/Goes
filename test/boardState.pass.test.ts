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
import { rectangularBoard } from '../shared/boardConfig.ts';
import { makeBoardState, soloStoneTurns } from './boardStateHelpers.ts';

function freshState(forcedPassOnly: boolean) {
    return makeBoardState(rectangularBoard(3, 3), soloStoneTurns(2), { forcedPassOnly });
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
    const bs = makeBoardState(rectangularBoard(1, 1), soloStoneTurns(2), { forcedPassOnly: true });
    assert.equal(bs.gameOver(), false, 'sanity: 2 active players, no moves played yet');
    assert.equal(bs.noTradLegal(), true, 'the lone node cannot legally hold a stone');
    assert.equal(bs.getView().passEnabled, true);
    assert.equal(bs.makeMove(null), true);
});

test('a resigned player may always pass, regardless of forcedPassOnly', () => {
    // 3 players so resigning 1 leaves 2 active - otherwise gameOver()'s
    // "<=1 active players" rule would end the game via the resignation itself,
    // masking whether the pass-permission logic is what actually allowed it.
    const bs = makeBoardState(rectangularBoard(3, 3), soloStoneTurns(3), { forcedPassOnly: true });
    assert.equal(bs.noTradLegal(), false, 'sanity: board is empty, legal moves should exist');
    bs.resign(1);
    assert.equal(bs.gameOver(), false, 'sanity: 2 non-resigned players remain');
    assert.equal(bs.getView().passEnabled, true);
    assert.equal(bs.makeMove(null), true);
});
