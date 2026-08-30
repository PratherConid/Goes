// Regression tests for BoardState.withdrawTo() - the bulk-rewind helper backing the online
// consensus withdraw feature (see server/src/onlineGameManager.ts's requestWithdraw()/
// respondToWithdraw()). Unlike withdrawMove() (which leaves resignation strictly alone - see
// its own comment and boardState.resign.test.ts), withdrawTo() re-keys any resignation recorded
// on a withdrawn ply onto the target ply itself, so it survives fromFinishedGame()'s ply-keyed
// replay loop instead of being silently orphaned/lost.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { BoardState, MoveType } from '../shared/boardState.ts';
import { rectangularBoard } from '../shared/boardConfig.ts';
import { FinishedGame, GameConfig } from '../shared/types.ts';
import { parseCleg } from '../shared/cleg.ts';

function twoPlayerGame() {
    const bc = rectangularBoard(5, 5);
    const turnList = [
        { player: 1, stones: [1, 1], protected: [0, 0], friendly: [0, 0] },
        { player: 2, stones: [1, 1], protected: [0, 0], friendly: [0, 0] },
    ];
    return new BoardState(2, 2, turnList, [[null, null], [null, null]], [null, null], { 1: new Set([1]), 2: new Set([2]) },
        false, 'area', [0, 0], 'situational', false, null, new Array(bc.N).fill(0), bc);
}

// 3 players so a single resignation doesn't immediately end the game (leaves 2 active), letting
// tests exercise resignation bookkeeping without the game being over throughout.
function threePlayerGame() {
    const bc = rectangularBoard(5, 5);
    const turnList = [
        { player: 1, stones: [1, 1, 1], protected: [0, 0, 0], friendly: [0, 0, 0] },
        { player: 2, stones: [1, 1, 1], protected: [0, 0, 0], friendly: [0, 0, 0] },
        { player: 3, stones: [1, 1, 1], protected: [0, 0, 0], friendly: [0, 0, 0] },
    ];
    const bs = new BoardState(
        3, 3, turnList,
        [[null, null, null], [null, null, null], [null, null, null]], [null, null, null],
        { 1: new Set([1]), 2: new Set([2]), 3: new Set([3]) }, false, 'area', [0, 0, 0],
        'situational', false, null, new Array(bc.N).fill(0), bc,
    );
    return { bc, bs, turnList };
}

test('withdrawTo() rewinds board/moveInfos/situations to the position right before targetPly', () => {
    const bs = twoPlayerGame();
    bs.makeMove(0, 1);
    bs.makeMove(4, 1);
    bs.makeMove(1, 2);
    assert.equal(bs.moveInfos().length, 3);

    bs.withdrawTo(1);
    assert.equal(bs.moveInfos().length, 1);
    assert.equal(bs.lastMove().pos, 0);
    assert.equal(bs.situations.length, 2);
});

test('withdrawTo() is a no-op when targetPly is already the tip or out of range', () => {
    const bs = twoPlayerGame();
    bs.makeMove(0, 1);
    bs.makeMove(4, 1);
    const before = bs.moveInfos().length;

    bs.withdrawTo(2);   // situations.length - 1 === 2: already the tip
    assert.equal(bs.moveInfos().length, before);
    bs.withdrawTo(-1);
    assert.equal(bs.moveInfos().length, before);
    bs.withdrawTo(100);
    assert.equal(bs.moveInfos().length, before);
});

test('a resignation recorded at/before targetPly is left untouched', () => {
    const { bs } = threePlayerGame();
    bs.makeMove(0, 1);   // ply 0, player 1
    bs.makeMove(1, 1);   // ply 1, player 2
    bs.resign(3);        // key = situations.length - 1 = 2
    bs.advanceResigned(); // player 3's forced pass -> ply 2
    bs.makeMove(2, 1);   // ply 3, player 1
    bs.makeMove(3, 1);   // ply 4, player 2
    assert.deepEqual(bs.resigns.get(2), [3]);

    bs.withdrawTo(3);
    assert.equal(bs.moveInfos().length, 3);
    assert.deepEqual([...bs.resigns.entries()], [[2, [3]]], 'the ply-2 key is untouched, no new key added');
});

test('a resignation recorded after targetPly is re-keyed onto targetPly, and advanceResigned() ' +
    'auto-passes the now-resigned player whose turn falls there - surviving a fromFinishedGame() replay', () => {
    const { bc, bs, turnList } = threePlayerGame();
    bs.makeMove(0, 1);   // ply 0, player 1
    bs.makeMove(1, 1);   // ply 1, player 2
    bs.makeMove(2, 1);   // ply 2, player 3
    bs.makeMove(3, 1);   // ply 3, player 1
    bs.makeMove(5, 1);   // ply 4, player 2
    bs.resign(3);        // key = situations.length - 1 = 5
    assert.equal(bs.gameOver(), false, '2 non-resigned players remain');
    assert.deepEqual(bs.resigns.get(5), [3]);

    // targetPly=2: turnList[2 % 3] is player 3, so rewinding here lands squarely on the
    // resigned player's own turn - exactly the case advanceResigned() must handle afterward.
    bs.withdrawTo(2);
    assert.deepEqual([...bs.resigns.entries()], [[2, [3]]], 'resignation re-keyed from ply 5 onto targetPly 2');
    assert.equal(bs.moveInfos().length, 2);
    assert.equal(bs.nextTurn.player, 3);

    bs.advanceResigned();
    assert.equal(bs.moveInfos().length, 3, 'player 3 was auto-passed');
    assert.equal(bs.lastMove().moveType, MoveType.PASS);
    assert.equal(bs.gameOver(), false);
    assert.equal(bs.nextTurn.player, 1);

    const config = new GameConfig(
        parseCleg('rectB(5, 5);'), 3, 3, turnList,
        [[null, null, null], [null, null, null], [null, null, null]], [null, null, null],
        { 1: new Set([1]), 2: new Set([2]), 3: new Set([3]) }, false, 'area', [0, 0, 0],
        'situational', false, null,
    );
    const fg = new FinishedGame(config, bs.moveInfos().map(m => ({ pos: m.pos, stone: m.stone })), new Map(bs.resigns));
    const reconstructed = BoardState.fromFinishedGame(fg, bc);

    assert.deepEqual(reconstructed.getView().resignedPlayers, bs.getView().resignedPlayers);
    assert.equal(reconstructed.gameOver(), bs.gameOver());
    assert.deepEqual(
        reconstructed.getView().situations[reconstructed.getView().situations.length - 1].board,
        bs.getView().situations[bs.getView().situations.length - 1].board,
    );
});
