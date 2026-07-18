// Covers the online-game DOM paths that test/renderer/localGame.test.ts can't
// reach: a real ServerConnection driven by a scripted FakeWebSocket instead of
// a live server. Written as one sequential narrative (login, then game/start)
// sharing a single Renderer, rather than independent per-test Renderers -
// Renderer.init() registers handlers on the shared module-singleton `conn`
// with no way to unregister them, so constructing a second Renderer against
// the same `conn` within this file would leave both instances' handlers
// firing on every simulated event.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { setupDom, FakeWebSocket } from './domSetup.ts';

setupDom();
const { Renderer } = await import('../../src/renderer.ts');
const { BoardState } = await import('../../shared/boardState.ts');
const { rectangularBoard } = await import('../../shared/boardConfig.ts');

const bc = rectangularBoard(3, 3);
const initialGame = new BoardState(2, 2, [{ player: 1, stones: [1, 0], protected: [0, 0], friendly: [0, 0] }, { player: 2, stones: [0, 1], protected: [0, 0], friendly: [0, 0] }], [[null, null], [null, null]], [null, null], { 1: new Set([1]), 2: new Set([2]) }, false, 'area', [0, 0], 'situational', false, null, new Array(bc.N).fill(0), bc);
const renderer = new Renderer(initialGame);
renderer.init();

// The one FakeWebSocket constructed by renderer.ts's module-scope `const conn
// = new ServerConnection()`.
const ws = FakeWebSocket.last;
ws.simulateOpen();

function runCommand(text: string) {
    const cmdInput = document.getElementById('cmd-input') as HTMLInputElement;
    cmdInput.value = text;
    cmdInput.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
}
const tick = () => new Promise(r => setTimeout(r, 0));

test('LOGIN round trip updates the status panel\'s Your Name line', async () => {
    (document.querySelector('#home-panel button[data-child="status"]') as HTMLButtonElement).click();

    runCommand('login alice pw');
    const reqMsg = ws.lastSentRequest('LOGIN');
    assert.equal(reqMsg['name'], 'alice');

    ws.simulateMessage({ kind: 'res', reqId: reqMsg['reqId'], ok: true, data: { name: 'alice', finishedGames: [] } });
    await tick();

    const statusPanel = document.getElementById('status-panel') as HTMLDivElement;
    assert.match(statusPanel.innerHTML, /Your Name:<\/b> alice/);
});

test("game/start activates the online game and passBtn reflects _isMyTurn()'s 'client' branch", async () => {
    const config = {
        boardType: 'rect', boardArgs: [3, 3], numStones: 2, numPlayers: 2,
        turnList: [{ player: 1, stones: [1, 0], protected: [0, 0], friendly: [0, 0] }, { player: 2, stones: [0, 1], protected: [0, 0], friendly: [0, 0] }], stoneToPlayerMap: { 1: [1], 2: [2] },
        forcedPassOnly: false, scoreRule: 'area', allowSuicide: false,
        players: [
            { slot: 1, type: 'client', name: 'alice', emsim: 0, temp: 0 },
            { slot: 2, type: 'client', name: 'bob', emsim: 0, temp: 0 },
        ],
    };
    ws.simulateMessage({ kind: 'event', type: 'game/start', id: 'GAME1', config });
    await tick();

    const passBtn = document.getElementById('pass-btn') as HTMLButtonElement;
    // turnList stones=[1,2] -> slot 1 (alice, matches userName) moves first: her turn.
    assert.equal(passBtn.disabled, false);
});

test("it is not alice's turn once the opponent's slot is next", async () => {
    const config = {
        boardType: 'rect', boardArgs: [3, 3], numStones: 2, numPlayers: 2,
        turnList: [{ player: 1, stones: [1, 0], protected: [0, 0], friendly: [0, 0] }, { player: 2, stones: [0, 1], protected: [0, 0], friendly: [0, 0] }], stoneToPlayerMap: { 1: [1], 2: [2] },
        forcedPassOnly: false, scoreRule: 'area', allowSuicide: false,
        players: [
            { slot: 1, type: 'client', name: 'bob', emsim: 0, temp: 0 },
            { slot: 2, type: 'client', name: 'alice', emsim: 0, temp: 0 },
        ],
    };
    ws.simulateMessage({ kind: 'event', type: 'game/start', id: 'GAME2', config });
    await tick();

    const passBtn = document.getElementById('pass-btn') as HTMLButtonElement;
    // turnList stones=[1,2] -> slot 1 (bob) moves first: not alice's turn.
    assert.equal(passBtn.disabled, true);
});
