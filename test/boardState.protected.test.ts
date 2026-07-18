// Regression test: under a turn/protection schedule that lets each side plant
// a stone inside the opponent's eye, an otherwise-alive two-eyed group can be
// killed - demonstrating that `protected` interacts with normal capture rules
// rather than being a separate, isolated mechanic.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { BoardState } from '../shared/boardState.ts';
import { rectangularBoard } from '../shared/boardConfig.ts';

// 11x3 board (node = row*11+col). Two independent two-eyed groups, each
// exactly alive (liberties = its two eyes only, nothing else):
//
//   11111222222
//   10101221202
//   11111222222
//
// - Black (1) fills cols 0-4, eyes at (row1,col1) and (row1,col3).
// - White (2) is a wall (col5) fused to a frame (cols6-10), eyes at
//   (row1,col7) and (row1,col9). The wall has no liberties of its own (it's
//   flanked by black on one side and merges into the frame on the other), so
//   it doesn't change white's liberty count - it exists only so that white's
//   group is flush against black's group with no open buffer square that
//   would otherwise hand either side an extra, irrelevant liberty.
// - Both groups are flush against the board's edges on every other side, so
//   each group's only liberties are its own two eyes.
function twoEyeBoard() {
    const bc = rectangularBoard(11, 3);
    const board = new Array(bc.N).fill(0);
    const at = (r: number, c: number) => r * 11 + c;
    for (const c of [0, 1, 2, 3, 4]) board[at(0, c)] = 1;
    for (const c of [0, 2, 4])       board[at(1, c)] = 1;
    for (const c of [0, 1, 2, 3, 4]) board[at(2, c)] = 1;
    for (const c of [5, 6, 7, 8, 9, 10]) board[at(0, c)] = 2;
    for (const c of [5, 6, 8, 10])       board[at(1, c)] = 2;
    for (const c of [5, 6, 7, 8, 9, 10]) board[at(2, c)] = 2;
    return {
        bc, board,
        blackEye1: at(1, 1), blackEye2: at(1, 3),
        whiteEye1: at(1, 7), whiteEye2: at(1, 9),
    };
}

// "tl 1-10 2-01 1-10 2-01" + "sprot 01 00 10 11":
//   turn0 (black): protects white only - black can't plant in white's eye
//                  yet (unprotected), and nobody can capture white this turn.
//   turn1 (white): protects nobody.
//   turn2 (black): protects black only - black can plant a stone inside
//                  white's eye (protected self-atari, survives at 0 liberties).
//   turn3 (white): protects both - white can plant a stone inside black's eye.
function twoEyeTurnList() {
    return [
        { player: 1, stones: [1, 0], protected: [0, 1], friendly: [0, 0] },
        { player: 2, stones: [0, 1], protected: [0, 0], friendly: [0, 0] },
        { player: 1, stones: [1, 0], protected: [1, 0], friendly: [0, 0] },
        { player: 2, stones: [0, 1], protected: [1, 1], friendly: [0, 0] },
    ];
}

test('white kills a two-eyed black group by planting in both eyes, one protected and one a normal capture', () => {
    const { bc, board, blackEye1, blackEye2, whiteEye1 } = twoEyeBoard();
    const bs = new BoardState(2, 2, twoEyeTurnList(), [[null, null], [null, null]], [null, null], { 1: new Set([1]), 2: new Set([2]) }, false, 'area', [0, 0], 'situational', false, null, board, bc);

    assert.equal(bs.makeMove(null), true);          // turn0: black passes
    assert.equal(bs.makeMove(null), true);           // turn1: white passes
    assert.equal(bs.makeMove(whiteEye1), true);       // turn2: black plants in white's eye (protected)
    assert.equal(bs.board[whiteEye1], 1, 'black\'s stone survives at zero liberties, protected');
    assert.equal(bs.makeMove(blackEye1), true);       // turn3: white plants in black's eye (protected)
    assert.equal(bs.board[blackEye1], 2, 'white\'s stone survives at zero liberties, protected');
    assert.equal(bs.makeMove(null), true);            // turn0 again: black passes (white is protected, can't be touched)

    assert.equal(bs.board.filter(v => v === 1).length, 13, 'sanity: black group still fully on the board before the kill');
    assert.equal(bs.makeMove(blackEye2), true);       // turn1 again: white fills black's last liberty
    assert.equal(bs.board.filter(v => v === 1).length, 0, 'black\'s entire two-eyed group is captured');
});

test('black cannot finish killing the white group: its own planted stone gets swept by its own unprotected pass first', () => {
    const { bc, board, blackEye1, blackEye2, whiteEye1, whiteEye2 } = twoEyeBoard();
    const bs = new BoardState(2, 2, twoEyeTurnList(), [[null, null], [null, null]], [null, null], { 1: new Set([1]), 2: new Set([2]) }, false, 'area', [0, 0], 'situational', false, null, board, bc);

    bs.makeMove(null);          // turn0: black passes
    bs.makeMove(null);          // turn1: white passes
    bs.makeMove(whiteEye1);     // turn2: black plants in white's eye (protected)
    bs.makeMove(blackEye1);     // turn3: white plants in black's eye (protected)

    // turn0 again: black is not protected this turn (protected=[0,1] only
    // protects white/stone 2), so black's own leftover zero-liberty stone at
    // whiteEye1 - no longer protected - is swept by the pass itself
    // (earlySelfCapture applies unconditionally on a pass; see boardState.ts).
    bs.makeMove(null);
    assert.equal(bs.board[whiteEye1], 0, 'black\'s own planted stone is undone by its own pass, since it is unprotected this turn');

    bs.makeMove(blackEye2);     // turn1 again: white finishes killing black's group

    // turn2 again: black tries to finish white off, but whiteEye1 reverted to
    // empty above, so white's group still has both its eyes as liberties -
    // filling just whiteEye2 doesn't capture anything.
    const whiteCountBefore = bs.board.filter(v => v === 2).length;
    assert.deepEqual(bs.legalMovesData().captures[1][whiteEye2], new Set(), 'legal, but captures nothing');
    bs.makeMove(whiteEye2);
    assert.equal(bs.board.filter(v => v === 2).length, whiteCountBefore, 'white group survives entirely intact');
});

// 5x3 board (node = row*5+col), one two-eyed group (stone 3) flush against
// every board edge so its only liberties are its two eyes:
//
//   33333
//   3.3.3   (eyes at (row1,col1)=6 and (row1,col3)=8)
//   33333
//
// "tl 1-100 2-010 3-001" + "sprot 100 010 001": each stone is protected only on its
// own turn, and only its own turn - nobody's stone is ever protected on
// anyone else's turn. So no single player can kill this group alone (filling
// only one eye just leaves a dead stone sitting there, and a player's own
// turn only lets *their own* stone be planted) - but two different players,
// each contributing one protected stone into a different eye on their own
// turn, can kill it together in as few as two plies.
test('two different players can gang up to kill a third player\'s two-eyed group', () => {
    const bc = rectangularBoard(5, 3);
    const board = new Array(bc.N).fill(0);
    const at = (r: number, c: number) => r * 5 + c;
    for (const c of [0, 1, 2, 3, 4]) board[at(0, c)] = 3;
    for (const c of [0, 2, 4])       board[at(1, c)] = 3;
    for (const c of [0, 1, 2, 3, 4]) board[at(2, c)] = 3;
    const eye1 = at(1, 1), eye2 = at(1, 3);

    const turnList = [
        { player: 1, stones: [1, 0, 0], protected: [1, 0, 0], friendly: [0, 0, 0] },
        { player: 2, stones: [0, 1, 0], protected: [0, 1, 0], friendly: [0, 0, 0] },
        { player: 3, stones: [0, 0, 1], protected: [0, 0, 1], friendly: [0, 0, 0] },
    ];
    const bs = new BoardState(3, 3, turnList, [[null, null, null], [null, null, null], [null, null, null]], [null, null, null], { 1: new Set([1]), 2: new Set([2]), 3: new Set([3]) }, false, 'area', [0, 0, 0], 'situational', false, null, board, bc);

    assert.equal(bs.makeMove(eye1), true);   // turn0: player 1 plants in eye1 (protected self-atari)
    assert.equal(bs.board[eye1], 1, 'player 1\'s stone survives at zero liberties, protected');

    assert.equal(bs.board.filter(v => v === 3).length, 13, 'sanity: target group still fully on the board');
    assert.equal(bs.makeMove(eye2), true);   // turn1: player 2 plants in eye2 - stone 3's last liberty
    assert.equal(bs.board.filter(v => v === 3).length, 0, 'the entire two-eyed group is captured by the combined effort');
    assert.equal(bs.board[eye2], 2, 'player 2\'s winning stone remains');
});

// 5x5 board (node = row*5+col). White (2) holds the entire border ring; black
// (1) holds the middle 3x3 minus its very center (black's "eye"):
//
//   22222
//   21112
//   21.12
//   21112
//   22222
//
// White's ring only ever touches black or itself - never the empty center -
// so white already has ZERO liberties in this initial position (no protected
// turns needed; this is a direct consequence of earlyOppCapture/
// earlySelfLiberation, unrelated to the `protected` field).
test('black playing in its own eye is not suicide when it would capture an already-zero-liberty white ring first', () => {
    const bc = rectangularBoard(5, 5);
    const board = new Array(bc.N).fill(0);
    const at = (r: number, c: number) => r * 5 + c;
    for (let c = 0; c < 5; c++) { board[at(0, c)] = 2; board[at(4, c)] = 2; }
    for (let r = 1; r <= 3; r++) { board[at(r, 0)] = 2; board[at(r, 4)] = 2; }
    for (let r = 1; r <= 3; r++) for (let c = 1; c <= 3; c++) board[at(r, c)] = 1;
    const center = at(2, 2);
    board[center] = 0;

    const turnList = [
        { player: 1, stones: [1, 0], protected: [0, 0], friendly: [0, 0] },
        { player: 2, stones: [0, 1], protected: [0, 0], friendly: [0, 0] },
    ];
    // allowSuicide: false - if this move were actually treated as a suicide,
    // it would be rejected outright; it must be legal via the ordinary
    // "connects to a group with a real liberty" path instead.
    const bs = new BoardState(2, 2, turnList, [[null, null], [null, null]], [null, null], { 1: new Set([1]), 2: new Set([2]) }, false, 'area', [0, 0], 'situational', false, null, board, bc);

    assert.equal(bs.legalMovesData().passCapture.size, 16, 'sanity: the entire white ring is already at zero liberties');
    assert.notEqual(bs.legalMovesData().captures[1][center], null, 'legal - not suicide');

    assert.equal(bs.makeMove(center), true);
    assert.equal(bs.board.filter(v => v === 2).length, 0, 'white\'s whole ring is captured');
    assert.equal(bs.board.filter(v => v === 1).length, 9, 'black\'s group survives and grows by the new stone');
    assert.equal(bs.board[center], 1);
});
