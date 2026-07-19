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

// Regression tests for the offline-invitee rejection in wsServer.ts's
// game/create handler. These need a genuinely offline account - i.e. no live
// WebSocket connection anywhere in userToWs for that name - which the
// client-only protocol here can't prove deterministically: there's no
// LOGOUT acknowledgment, so a connection's own close() resolving only
// reflects what that client observed, never what the server has processed
// (its 'close' handler, which clears userToWs, runs on a separate event from
// a separate socket object). A real process restart sidesteps this rather
// than racing it: killing the server process destroys every live connection
// (and userToWs itself) structurally, so there is no ordering to prove -
// same startTestServerProcess() + dedicated temp dataDir pattern as the
// restart test above (an isolated dir, not any real/shared store).
test('game/create rejects invited usernames that are offline, proven via a real process restart', async () => {
    const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'goes-test-offline-invite-'));
    const first = await startTestServerProcess(dataDir);
    let stopped = false;
    try {
        for (const name of ['xavier', 'zoe', 'amir']) {
            const c = await connect(first.url);
            await c.req('REGISTER', { name, password: 'pw' });
            await c.close();
        }
        await first.stop();
        stopped = true;

        const second = await startTestServerProcess(dataDir);
        try {
            const alice = await connect(second.url);
            await alice.req('REGISTER', { name: 'walter', password: 'pw' });

            await assert.rejects(
                alice.req('game/create', {
                    config: passOnlyConfig(),
                    onlinePlayerRequest: fixedRequest([[1, { type: 'local', name: '' }], [2, { type: 'pendingInvitedOnline', name: 'xavier' }]]),
                }),
                (e: any) => {
                    assert.equal(e.message, 'Cannot create game. User xavier is offline.');
                    assert.equal(e.statusCode, 409);
                    return true;
                },
            );

            const threePlayerConfig = {
                ...passOnlyConfig(), numPlayers: 3, numStones: 3,
                turnList: [
                    { player: 1, stones: [1, 0, 0], protected: [0, 0, 0], friendly: [0, 0, 0] },
                    { player: 2, stones: [0, 1, 0], protected: [0, 0, 0], friendly: [0, 0, 0] },
                    { player: 3, stones: [0, 0, 1], protected: [0, 0, 0], friendly: [0, 0, 0] },
                ],
                stoneToPlayerMap: { 1: [1], 2: [2], 3: [3] },
            };
            await assert.rejects(
                alice.req('game/create', {
                    config: threePlayerConfig,
                    onlinePlayerRequest: fixedRequest([
                        [1, { type: 'local', name: '' }],
                        [2, { type: 'pendingInvitedOnline', name: 'zoe' }],
                        [3, { type: 'pendingInvitedOnline', name: 'amir' }],
                    ]),
                }),
                (e: any) => {
                    assert.match(e.message, /^Cannot create game\. Users .* are offline\.$/);
                    assert.match(e.message, /zoe/);
                    assert.match(e.message, /amir/);
                    assert.equal(e.statusCode, 409);
                    return true;
                },
            );

            // No game (and thus no invite) should have actually been created for
            // the rejected xavier invite above - confirm by logging back in as
            // xavier and checking no invite arrived.
            const xavierAgain = await connect(second.url);
            const xavierInvites: unknown[] = [];
            xavierAgain.onEvent('game/invite', m => xavierInvites.push(m));
            await xavierAgain.req('LOGIN', { name: 'xavier', password: 'pw' });
            await new Promise(r => setImmediate(r));
            assert.equal(xavierInvites.length, 0);

            await alice.close();
            await xavierAgain.close();
        } finally {
            await second.stop();
        }
    } finally {
        if (!stopped) await first.stop();
    }
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

test('a multi-invite decline waits for every invitee before cancelling, and a too-late accept is rejected with a specific message', async () => {
    const alice = await registerAndLogin('olga');
    const bob = await registerAndLogin('peter');
    const carol = await registerAndLogin('quinn');

    const bobInvite = new Promise<any>(resolve => bob.onEvent('game/invite', resolve));
    const carolInvite = new Promise<any>(resolve => carol.onEvent('game/invite', resolve));
    const { id } = await alice.req<{ id: string }>('game/create', {
        config: passOnlyConfig(),
        onlinePlayerRequest: fixedRequest([[1, { type: 'pendingInvitedOnline', name: 'peter' }], [2, { type: 'pendingInvitedOnline', name: 'quinn' }]]),
    });
    await Promise.all([bobInvite, carolInvite]);

    const aliceFailedEvents: any[] = [];
    alice.onEvent('game/invite-failed', m => aliceFailedEvents.push(m));

    const bobFailedEvents: any[] = [];
    bob.onEvent('game/invite-failed', m => bobFailedEvents.push(m));

    const bobRespond = await bob.req<{ status: string }>('game/invite-respond', { id, accept: false });
    assert.equal(bobRespond.status, 'cancelled');

    // Carol hasn't responded yet - the game must not be torn down/notified
    // to alice yet (only the first of two invitees has declined).
    await new Promise(r => setImmediate(r));
    assert.equal(aliceFailedEvents.length, 0);

    // Carol tries to accept, too late - the game was already refused by bob.
    // She never actually gets seated, and gets a specific message instead of
    // a raw 404 - this is also the response that finally tears the game down
    // and notifies alice.
    const aliceFailed = new Promise<any>(resolve => alice.onEvent('game/invite-failed', resolve));
    await assert.rejects(
        carol.req('game/invite-respond', { id, accept: true }),
        (e: any) => { assert.match(e.message, /already refused by another invited player/); assert.equal(e.statusCode, 409); return true; },
    );
    const failedMsg = await aliceFailed;
    assert.equal(failedMsg.id, id);
    // Bob already knows he declined (his own game/invite-respond call
    // already got {status:'cancelled'}) - he shouldn't get a redundant
    // game/invite-failed push. alice's push above arriving is proof the
    // same notify computation already ran, so if bob's were coming, it
    // would already be here too.
    assert.equal(bobFailedEvents.length, 0);

    // Fully torn down now - a further response 404s, matching the
    // single-invite case above.
    await assert.rejects(
        bob.req('game/invite-respond', { id, accept: true }),
        (e: any) => { assert.equal(e.statusCode, 404); return true; },
    );

    await alice.close();
    await bob.close();
    await carol.close();
});

test('inviting the same user into two slots resolves both from one response, with only one invite popup', async () => {
    const alice = await registerAndLogin('rachel');
    const bob = await registerAndLogin('sam');

    const bobInvites: any[] = [];
    const firstBobInvite = new Promise<any>(resolve =>
        bob.onEvent('game/invite', m => { bobInvites.push(m); if (bobInvites.length === 1) resolve(m); }));
    const { id, status } = await alice.req<{ id: string; status: string }>('game/create', {
        config: passOnlyConfig(),
        onlinePlayerRequest: fixedRequest([[1, { type: 'pendingInvitedOnline', name: 'sam' }], [2, { type: 'pendingInvitedOnline', name: 'sam' }]]),
    });
    assert.equal(status, 'waiting');
    await firstBobInvite;

    // Deduped by username - exactly one invite popup, not one per slot.
    await new Promise(r => setImmediate(r));
    assert.equal(bobInvites.length, 1);

    const aliceStart = new Promise(resolve => alice.onEvent('game/start', resolve));
    const respond = await bob.req<{ status: string }>('game/invite-respond', { id, accept: true });
    assert.equal(respond.status, 'playing');   // both slots resolved by this one response
    await aliceStart;

    await alice.close();
    await bob.close();
});
