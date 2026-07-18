import { test } from 'node:test';
import assert from 'node:assert/strict';
import { setupDom, BOARD_PX } from './domSetup.ts';

// Renderer's module-scope `const conn = new ServerConnection()` needs
// WebSocket/location as globals at import time, so this must be dynamic and
// happen after the first setupDom() call.
setupDom();
const { Renderer } = await import('../../src/renderer.ts');
const { BoardState } = await import('../../shared/boardState.ts');
const { rectangularBoard } = await import('../../shared/boardConfig.ts');

// A fresh Renderer, mounted against a fresh jsdom document (setupDom() call)
// so this test's event listeners don't accumulate on nodes from a previous
// test - see domSetup.ts's setupDom() comment.
function createRenderer(forcedPassOnly = false) {
    setupDom();
    const bc = rectangularBoard(3, 3);
    const game = new BoardState(2, 2, [{ player: 1, stones: [1, 0], protected: [0, 0], friendly: [0, 0] }, { player: 2, stones: [0, 1], protected: [0, 0], friendly: [0, 0] }], [[null, null], [null, null]], [null, null], { 1: new Set([1]), 2: new Set([2]) }, forcedPassOnly, 'area', [0, 0], 'situational', false, null, new Array(bc.N).fill(0), bc);
    const renderer = new Renderer(game);
    renderer.init();
    return renderer;
}

function runCommand(text: string) {
    const cmdInput = document.getElementById('cmd-input') as HTMLInputElement;
    cmdInput.value = text;
    cmdInput.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
}

test('passBtn is enabled for a fresh local game with forcedPassOnly=false (regression: _isMyTurn bug fixed 2026-07-13)', () => {
    createRenderer(false);
    const passBtn = document.getElementById('pass-btn') as HTMLButtonElement;
    assert.equal(passBtn.disabled, false);
});

test('passBtn is disabled when forcedPassOnly=true and legal moves exist', () => {
    createRenderer(true);
    const passBtn = document.getElementById('pass-btn') as HTMLButtonElement;
    assert.equal(passBtn.disabled, true);
});

test('command input drives newCfg and the New Game panel via the real keydown listener', () => {
    createRenderer();
    // Navigate to the New Game node so newGameSetupHtml actually runs
    // (it's gated on currentSidePanel === SidePanelContent.NewGame) - ns/np
    // set newCfg, which that node (not Status) displays.
    (document.querySelector('#home-panel button[data-child="newGame"]') as HTMLButtonElement).click();

    runCommand('ns 3');
    runCommand('np 3');

    const newGameDetails = document.getElementById('new-game-setup-details') as HTMLDivElement;
    assert.match(newGameDetails.innerHTML, /Type of stones:<\/b> 3/);
    assert.match(newGameDetails.innerHTML, /Number of players:<\/b> 3/);
});

test('clicking the board places a stone at the clicked node', () => {
    createRenderer();
    const mainSvg = document.getElementById('main-canvas') as unknown as SVGSVGElement;
    const plyNum = document.getElementById('ply-num') as HTMLSpanElement;
    assert.equal(plyNum.textContent, '0/0');

    // rectangularBoard(3,3)'s center node sits exactly at board center - see
    // domSetup.ts's BOARD_PX comment.
    mainSvg.dispatchEvent(new MouseEvent('click', { clientX: BOARD_PX / 2, clientY: BOARD_PX / 2, bubbles: true }));

    assert.equal(plyNum.textContent, '1/1');
});
