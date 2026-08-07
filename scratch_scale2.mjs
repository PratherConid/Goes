import { setupDom } from './test/renderer/domSetup.ts';
import assert from 'node:assert/strict';

setupDom();
const { Renderer } = await import('./src/renderer.ts');
const { BoardState } = await import('./shared/boardState.ts');
const { cubeLatticeBoard } = await import('./shared/boardConfig.ts');

const bc = cubeLatticeBoard(3, 3, 3);
// Place a real stone at EVERY node - the worst case, since computeInitialScale must guarantee every
// possible stone position fits, not just the ones that happen to be occupied right now.
const board = new Array(bc.N).fill(1);
const game = new BoardState(
    2, 2,
    [
        { player: 1, stones: [1, 0], protected: [0, 0], friendly: [0, 0] },
        { player: 2, stones: [0, 1], protected: [0, 0], friendly: [0, 0] },
    ],
    [[null, null], [null, null]], [null, null], { 1: new Set([1]), 2: new Set([2]) }, false, 'area',
    [0, 0], 'situational', false, null, board, bc,
);
const r = new Renderer(game);
// Fix the test-harness mismatch: Renderer's constructor defaults _active.config to a 'rect' board
// (unrelated to the actual cubeLatticeBoard passed to BoardState), so computeStarPoints() would
// otherwise draw 5 star points at nonsensical positions for a totally different (9x9 rect) board's
// geometry - not a real production scenario, just this scratch script's own setup shortcut.
r._active.config.boardType = 'cublat';
r.init();

const svg = document.getElementById('main-canvas');
const size = Number(svg.getAttribute('width'));
console.log('canvas size:', size, 'circle count:', svg.querySelectorAll('circle').length);

let maxOverflow = 0;
for (const c of svg.querySelectorAll('circle')) {
    const cx = Number(c.getAttribute('cx')), cy = Number(c.getAttribute('cy')), rad = Number(c.getAttribute('r'));
    for (const edge of [cx - rad, size - (cx + rad), cy - rad, size - (cy + rad)]) {
        if (edge < 0) maxOverflow = Math.max(maxOverflow, -edge);
    }
}
console.log('max overflow past canvas edge (px):', maxOverflow);
assert.ok(maxOverflow < 1e-6, `stones should not overflow the canvas, got ${maxOverflow}px overflow`);

console.log('ALL OK');
