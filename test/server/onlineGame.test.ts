import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { startTestServer, startTestServerProcess, connect, waitForEvents, type TestServer, type TestClient } from './testServer.ts';

let server: TestServer;

before(async () => { server = await startTestServer(); });
after(async () => { await server.close(); });

// A 2-player, both-slots-claimed-later game where nobody ever has a legal
// PLACE move (1x1 board), so both players simply pass to a deterministic
// finish - avoids capture/liberty topology entirely, same approach used in
// test/boardState.finishedGame.test.ts. Player setup is no longer part of
// the config itself - see fixedRequest() below, sent as a separate
// onlinePlayerRequest field (see server/src/onlineGameManager.ts's
// OnlinePlayerRequest-based createGame()).
function passOnlyConfig() {
    return {
        boardType: 'rect', boardArgs: [1, 1], numStones: 2, numPlayers: 2,
        turnList: [{ player: 1, stones: [1, 0], protected: [0, 0], friendly: [0, 0] }, { player: 2, stones: [0, 1], protected: [0, 0], friendly: [0, 0] }], stoneToPlayerMap: { 1: [1], 2: [2] },
        forcedPassOnly: false, scoreRule: 'area', allowSuicide: false,
    };
}

// Builds a fixed-order OnlinePlayerRequest wire payload from [slot, {type, name}] entries.
function fixedRequest(entries: [number, { type: string; name: string }][]) {
    return {
        fixedOrder: entries.map(([slot, p]) => ({ slot, type: p.type, name: p.name, emsim: 0, temp: 0 })),
        randomOrder: [],
        fixed: true,
    };
}

async function registerAndLogin(name: string): Promise<TestClient> {
    const client = await connect(server.url);
    await client.req('REGISTER', { name, password: 'pw' });
    return client;
}

test('game/create + game/join broadcast byte-identical config to both observers (no personalized broadcasts)', async () => {
    const alice = await registerAndLogin('alice');
    const bob = await registerAndLogin('bob');

    const aliceStart = new Promise(resolve => alice.onEvent('game/start', resolve));
    const bobStart = new Promise(resolve => bob.onEvent('game/start', resolve));

    const { id, status } = await alice.req<{ id: string; status: string }>('game/create', {
        config: passOnlyConfig(), onlinePlayerRequest: fixedRequest([[1, { type: 'local', name: '' }]]),
    });
    assert.equal(status, 'waiting');

    // game/join's response data is just { position } - the resulting
    // 'waiting'/'playing' status is only observable via the broadcast below.
    const join = await bob.req<{ position: number }>('game/join', { id });
    assert.equal(join.position, 2);

    const [aliceMsg, bobMsg] = await Promise.all([aliceStart, bobStart]);
    assert.deepEqual(aliceMsg, bobMsg);

    await alice.close();
    await bob.close();
});

test('alternating passes reach a natural finish, both observers see both moves', async () => {
    const alice = await registerAndLogin('carol');
    const bob = await registerAndLogin('dave');

    const { id } = await alice.req<{ id: string }>('game/create', {
        config: passOnlyConfig(), onlinePlayerRequest: fixedRequest([[1, { type: 'local', name: '' }]]),
    });
    await bob.req('game/join', { id });

    // Each observer gets its own broadcast delivery (a separate socket), with
    // no ordering guarantee relative to either sender's own req() resolving -
    // wait for both clients to actually receive both events.
    const aliceMovesP = waitForEvents(alice, 'game/move', 2);
    const bobMovesP = waitForEvents(bob, 'game/move', 2);

    await alice.req('game/move', { id, moveIndex: null, clientIdx: 0 });
    await bob.req('game/move', { id, moveIndex: null, clientIdx: 1 });

    // both consecutive passes recorded -> game over (turnList.length = 2)
    const [aliceMoves, bobMoves] = await Promise.all([aliceMovesP, bobMovesP]);
    assert.deepEqual(aliceMoves, bobMoves);

    // Reconnect as alice and resync via game/subscribe - should see the finished state.
    await alice.close();
    const aliceAgain = await connect(server.url);
    await aliceAgain.req('LOGIN', { name: 'carol', password: 'pw' });
    const state = await aliceAgain.req<{ state: { status: string; moves: { pos: number | null; stone: number | null }[] } }>('game/subscribe', { id, position: 1 });
    assert.equal(state.state.status, 'finished');
    assert.deepEqual(state.state.moves, [{ pos: null, stone: null }, { pos: null, stone: null }]);

    await aliceAgain.close();
    await bob.close();
});

test('game/resign ends a 2-player game and broadcasts the resigned slot to both observers', async () => {
    const alice = await registerAndLogin('erin');
    const bob = await registerAndLogin('frank');

    // Both slots pre-claimed -> starts immediately ('playing').
    const { id, status } = await alice.req<{ id: string; status: string }>('game/create', {
        config: passOnlyConfig(),
        onlinePlayerRequest: fixedRequest([[1, { type: 'local', name: '' }], [2, { type: 'client', name: 'frank' }]]),
    });
    assert.equal(status, 'playing');
    await alice.req('game/subscribe', { id, position: 1 });
    await bob.req('game/subscribe', { id, position: 2 });

    const bobResign = new Promise(resolve => bob.onEvent('game/resign', resolve));
    await alice.req('game/resign', { id });
    const resignMsg: any = await bobResign;
    assert.deepEqual(resignMsg.slots, [1]);

    const state = await bob.req<{ state: { status: string } }>('game/subscribe', { id, position: 2 });
    assert.equal(state.state.status, 'finished');

    await alice.close();
    await bob.close();
});

test('a finished game survives a real server restart and shows up in a fresh LOGIN', async () => {
    const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'goes-test-restart-'));
    const first = await startTestServerProcess(dataDir);
    let stopped = false;
    try {
        const alice = await connect(first.url);
        await alice.req('REGISTER', { name: 'grace', password: 'pw' });
        const bob = await connect(first.url);
        await bob.req('REGISTER', { name: 'henry', password: 'pw' });

        const { id } = await alice.req<{ id: string }>('game/create', {
            config: passOnlyConfig(),
            onlinePlayerRequest: fixedRequest([[1, { type: 'local', name: '' }], [2, { type: 'client', name: 'henry' }]]),
        });
        await alice.req('game/move', { id, moveIndex: null, clientIdx: 0 });
        await bob.req('game/move', { id, moveIndex: null, clientIdx: 1 });

        await alice.close();
        await bob.close();
        await first.stop();
        stopped = true;

        const second = await startTestServerProcess(dataDir);
        try {
            const graceAgain = await connect(second.url);
            const login = await graceAgain.req<{ finishedGames: { id: string; finishedGame: unknown }[] }>(
                'LOGIN', { name: 'grace', password: 'pw' });
            assert.ok(login.finishedGames.some(g => g.id === id), 'finished game should survive the restart');
            await graceAgain.close();
        } finally {
            await second.stop();
        }
    } finally {
        if (!stopped) await first.stop();
    }
});

test('game/create rejects an invited username that does not exist', async () => {
    const alice = await registerAndLogin('ivan');
    await assert.rejects(
        alice.req('game/create', {
            config: passOnlyConfig(),
            onlinePlayerRequest: fixedRequest([[1, { type: 'local', name: '' }], [2, { type: 'pendingInvitedOnline', name: 'nobody' }]]),
        }),
        (e: any) => { assert.match(e.message, /does not exist/); assert.equal(e.statusCode, 400); return true; },
    );
    await alice.close();
});

test('game/create ignores a stale invite left in the inactive list (fixed vs random)', async () => {
    const alice = await registerAndLogin('nadia');
    // onlinePlayerRequest carries a 'pendingInvitedOnline' entry for a
    // nonexistent user in fixedOrder, but fixed:false means only randomOrder
    // is actually used - the leftover fixedOrder entry (e.g. from an earlier
    // attempt before switching modes) must not block this request.
    const { status } = await alice.req<{ status: string }>('game/create', {
        config: passOnlyConfig(),
        onlinePlayerRequest: {
            fixedOrder: [{ slot: 1, type: 'pendingInvitedOnline', name: 'nobody', emsim: 0, temp: 0 }],
            randomOrder: [{ type: 'local', name: '', emsim: 0, temp: 0 }, { type: 'local', name: '', emsim: 0, temp: 0 }],
            fixed: false,
        },
    });
    assert.equal(status, 'playing');
    await alice.close();
});

test('invite + accept starts the game and notifies both observers', async () => {
    const alice = await registerAndLogin('julia');
    const bob = await registerAndLogin('kevin');

    const bobInvite = new Promise<any>(resolve => bob.onEvent('game/invite', resolve));
    const { id, status } = await alice.req<{ id: string; status: string }>('game/create', {
        config: passOnlyConfig(),
        onlinePlayerRequest: fixedRequest([[1, { type: 'local', name: '' }], [2, { type: 'pendingInvitedOnline', name: 'kevin' }]]),
    });
    assert.equal(status, 'waiting');   // an unconfirmed invite never starts the game immediately

    const inviteMsg = await bobInvite;
    assert.equal(inviteMsg.id, id);
    assert.equal(inviteMsg.from, 'julia');

    const aliceStart = new Promise(resolve => alice.onEvent('game/start', resolve));
    const bobStart = new Promise(resolve => bob.onEvent('game/start', resolve));
    const respond = await bob.req<{ status: string }>('game/invite-respond', { id, accept: true });
    assert.equal(respond.status, 'playing');

    const [aliceMsg, bobMsg] = await Promise.all([aliceStart, bobStart]);
    assert.deepEqual(aliceMsg, bobMsg);

    await alice.close();
    await bob.close();
});

test('invite + refuse cancels the game and notifies everyone involved', async () => {
    const alice = await registerAndLogin('laura');
    const bob = await registerAndLogin('mike');

    const bobInvite = new Promise<any>(resolve => bob.onEvent('game/invite', resolve));
    const { id } = await alice.req<{ id: string }>('game/create', {
        config: passOnlyConfig(),
        onlinePlayerRequest: fixedRequest([[1, { type: 'local', name: '' }], [2, { type: 'pendingInvitedOnline', name: 'mike' }]]),
    });
    await bobInvite;

    const aliceFailed = new Promise<any>(resolve => alice.onEvent('game/invite-failed', resolve));
    const respond = await bob.req<{ status: string }>('game/invite-respond', { id, accept: false });
    assert.equal(respond.status, 'cancelled');

    const failedMsg = await aliceFailed;
    assert.equal(failedMsg.id, id);

    // The game is gone entirely - not just "still waiting" - so a later join attempt 404s.
    await assert.rejects(
        bob.req('game/join', { id }),
        (e: any) => { assert.equal(e.statusCode, 404); return true; },
    );

    await alice.close();
    await bob.close();
});
