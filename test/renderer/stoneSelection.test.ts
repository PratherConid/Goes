// Regression tests for the stone-selection popup (Renderer.selectingStone):
// when a turn offers multiple stones, _onBoardClick must decide whether to
// place, ask, or do nothing based on which stones are actually LEGAL at the
// clicked location - not merely offered that turn (see src/renderer.ts).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { setupDom, BOARD_PX } from './domSetup.ts';

setupDom();
const { Renderer } = await import('../../src/renderer.ts');
const { BoardState } = await import('../../shared/boardState.ts');
const { rectangularBoard } = await import('../../shared/boardConfig.ts');

// Fresh renderer mounted against a fresh jsdom document per call (see
// domSetup.ts's setupDom() comment on why this must happen every time).
function createRenderer(game: InstanceType<typeof BoardState>) {
    setupDom();
    const renderer = new Renderer(game);
    renderer.init();
    return renderer;
}

function clickCenter(mainSvg: SVGSVGElement) {
    mainSvg.dispatchEvent(new MouseEvent('click', { clientX: BOARD_PX / 2, clientY: BOARD_PX / 2, bubbles: true }));
}

test('two stones both legal at the clicked location: opens the popup with both', () => {
    const bc = rectangularBoard(3, 3);
    const game = new BoardState(2, 2, [
        { player: 1, stones: [1, 1], protected: [0, 0], friendly: [0, 0] },
        { player: 2, stones: [1, 1], protected: [0, 0], friendly: [0, 0] },
    ], [[null, null], [null, null]], [null, null], { 1: new Set([1]), 2: new Set([2]) }, false, 'area', [0, 0], 'situational', false, null, new Array(bc.N).fill(0), bc);
    const renderer: any = createRenderer(game);
    const mainSvg = document.getElementById('main-canvas') as unknown as SVGSVGElement;
    const plyNum = document.getElementById('ply-num') as HTMLSpanElement;

    clickCenter(mainSvg);

    assert.equal(renderer.selectingStone, true, 'should enter selection mode rather than place immediately');
    assert.equal(plyNum.textContent, '0/0', 'no move should have been made yet');
    const v = renderer._active.bs.getView();
    assert.equal(renderer._stonePopupCircles(v).length, 2, 'both legal stones should get a circle');
});

test('clicking a popup circle places a move with that specific stone', () => {
    const bc = rectangularBoard(3, 3);
    const game = new BoardState(2, 2, [
        { player: 1, stones: [1, 1], protected: [0, 0], friendly: [0, 0] },
        { player: 2, stones: [1, 1], protected: [0, 0], friendly: [0, 0] },
    ], [[null, null], [null, null]], [null, null], { 1: new Set([1]), 2: new Set([2]) }, false, 'area', [0, 0], 'situational', false, null, new Array(bc.N).fill(0), bc);
    const renderer: any = createRenderer(game);
    const mainSvg = document.getElementById('main-canvas') as unknown as SVGSVGElement;
    const plyNum = document.getElementById('ply-num') as HTMLSpanElement;

    clickCenter(mainSvg);
    const v = renderer._active.bs.getView();
    const circles = renderer._stonePopupCircles(v);
    assert.equal(circles[0].r, BOARD_PX / 24, 'popup circle radius is 1/24 of the board width');

    // Click the SECOND circle (stone 2) to prove the choice is actually threaded through.
    const target = circles[1];
    mainSvg.dispatchEvent(new MouseEvent('click', { clientX: target.x, clientY: target.y, bubbles: true }));

    assert.equal(renderer.selectingStone, false, 'selection mode should close after choosing');
    assert.equal(plyNum.textContent, '1/1', 'the move should now be committed');
    assert.equal(renderer._active.bs.moveInfos()[0].stone, target.stone, 'the placed stone should match the clicked circle');
});

test('clicking the board away from the popup circles cancels selection without committing a move', () => {
    const bc = rectangularBoard(3, 3);
    const game = new BoardState(2, 2, [
        { player: 1, stones: [1, 1], protected: [0, 0], friendly: [0, 0] },
        { player: 2, stones: [1, 1], protected: [0, 0], friendly: [0, 0] },
    ], [[null, null], [null, null]], [null, null], { 1: new Set([1]), 2: new Set([2]) }, false, 'area', [0, 0], 'situational', false, null, new Array(bc.N).fill(0), bc);
    const renderer: any = createRenderer(game);
    const mainSvg = document.getElementById('main-canvas') as unknown as SVGSVGElement;
    const plyNum = document.getElementById('ply-num') as HTMLSpanElement;
    const passBtn = document.getElementById('pass-btn') as HTMLButtonElement;

    clickCenter(mainSvg);
    assert.equal(renderer.selectingStone, true);
    assert.equal(passBtn.disabled, true, 'Pass is disabled while a stone-selection popup is open');

    // A click nowhere near any popup circle (top-left corner) cancels the selection.
    mainSvg.dispatchEvent(new MouseEvent('click', { clientX: 0, clientY: 0, bubbles: true }));
    assert.equal(renderer.selectingStone, false, 'a miss click should exit selection mode');
    assert.equal(renderer.pendingPos, null);
    assert.equal(plyNum.textContent, '0/0', 'cancelling must not have committed a move');

    // Board should be clickable again - clicking the same cell should re-open the popup there.
    clickCenter(mainSvg);
    assert.equal(renderer.selectingStone, true, 'board should be clickable again after cancel');
});

// 3x3 board, every cell is stone3 except center (empty). Stone3 is PROTECTED
// this turn, so it can never be captured - placing stone1 or stone2 at
// center is therefore a genuine suicide (no capture, no liberty gained).
function suicideTrapBoard(protectedStones: number[], offered1: number[], offered2: number[]) {
    const bc = rectangularBoard(3, 3);
    const board = new Array(bc.N).fill(3);
    board[4] = 0;
    const game = new BoardState(3, 2, [
        { player: 1, stones: offered1, protected: protectedStones, friendly: [0, 0, 0] },
        { player: 2, stones: offered2, protected: [0, 0, 0], friendly: [0, 0, 0] },
    ], [[null, null], [null, null], [null, null]], [null, null, null], { 1: new Set([1]), 2: new Set([2]), 3: new Set() }, false, 'area', [0, 0], 'situational', false, null, board, bc);
    return { game, center: 4 };
}

test('clicking a location illegal for every offered stone does nothing (no popup, no move)', () => {
    const { game, center } = suicideTrapBoard([0, 0, 1], [1, 1, 0], [0, 0, 1]);
    assert.equal(game.legalMovesData().legalsForLocation[center].size, 0,
        'sanity: suicide for both offered stones (stone3 is protected, so nothing gets captured)');

    const renderer: any = createRenderer(game);
    const mainSvg = document.getElementById('main-canvas') as unknown as SVGSVGElement;
    const plyNum = document.getElementById('ply-num') as HTMLSpanElement;

    clickCenter(mainSvg);
    assert.equal(renderer.selectingStone, false, 'should not open the popup for an all-illegal location');
    assert.equal(plyNum.textContent, '0/0', 'no move should have been made');
});

test('exactly one offered stone legal at the clicked location: auto-placed, no popup', () => {
    // Same trap as above, but stone1 is ALSO protected this turn - its own
    // suicide is then allowed (a protected placement always stays on the
    // board), making stone1 the sole legal stone while stone2 remains illegal.
    const { game, center } = suicideTrapBoard([1, 0, 1], [1, 1, 0], [0, 0, 1]);
    assert.deepEqual([...game.legalMovesData().legalsForLocation[center]], [1],
        'sanity: only stone1 is legal (protected, so its own suicide is allowed)');

    const renderer: any = createRenderer(game);
    const mainSvg = document.getElementById('main-canvas') as unknown as SVGSVGElement;
    const plyNum = document.getElementById('ply-num') as HTMLSpanElement;

    clickCenter(mainSvg);
    assert.equal(renderer.selectingStone, false, 'should not open the popup when only one stone is legal here');
    assert.equal(plyNum.textContent, '1/1', 'the single legal stone should be auto-placed');
    assert.equal(renderer._active.bs.moveInfos()[0].stone, 1);
});
