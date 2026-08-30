import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import {
    startTestServer, startTestServerProcess, connect, waitForEvents, type TestServer, type TestClient,
} from './testServer.ts';
import { parseCleg } from '../../shared/cleg.ts';

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
        boardDescr: parseCleg('rectB(1, 1);'), numStones: 2, numPlayers: 2,
        turnList: [
            { player: 1, stones: [1, 0], protected: [0, 0], friendly: [0, 0] },
            { player: 2, stones: [0, 1], protected: [0, 0], friendly: [0, 0] },
        ],
        stoneToPlayerMap: { 1: [1], 2: [2] },
        forcedPassOnly: false, scoreRule: 'area', allowSuicide: false,
    };
}

// Unlike passOnlyConfig, real placements are legal (allowSuicide sidesteps liberty/capture
// bookkeeping entirely, so any empty cell is always a legal placement) - needed for the withdraw
// tests below, which need several distinct real moves (not just passes) to withdraw between.
function realTwoPlayerConfig() {
    return {
        boardDescr: parseCleg('rectB(3, 3);'), numStones: 2, numPlayers: 2,
        turnList: [
            { player: 1, stones: [1, 0], protected: [0, 0], friendly: [0, 0] },
            { player: 2, stones: [0, 1], protected: [0, 0], friendly: [0, 0] },
        ],
        stoneToPlayerMap: { 1: [1], 2: [2] },
        forcedPassOnly: false, scoreRule: 'area', allowSuicide: true,
    };
}

function realThreePlayerConfig() {
    return {
        boardDescr: parseCleg('rectB(5, 5);'), numStones: 3, numPlayers: 3,
        turnList: [
            { player: 1, stones: [1, 0, 0], protected: [0, 0, 0], friendly: [0, 0, 0] },
            { player: 2, stones: [0, 1, 0], protected: [0, 0, 0], friendly: [0, 0, 0] },
            { player: 3, stones: [0, 0, 1], protected: [0, 0, 0], friendly: [0, 0, 0] },
        ],
        stoneToPlayerMap: { 1: [1], 2: [2], 3: [3] },
        forcedPassOnly: false, scoreRule: 'area', allowSuicide: true,
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
    const state = await aliceAgain.req<
        { state: { status: string; moves: { pos: number | null; stone: number | null }[] } }
    >('game/subscribe', { id, position: 1 });
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

test('game/sendchat broadcasts to both observers, is seeded via game/subscribe, and rejects non-players', async () => {
    const alice = await registerAndLogin('tina');
    const bob = await registerAndLogin('ursula');

    const { id, status } = await alice.req<{ id: string; status: string }>('game/create', {
        config: passOnlyConfig(),
        onlinePlayerRequest: fixedRequest([[1, { type: 'local', name: '' }], [2, { type: 'client', name: 'ursula' }]]),
    });
    assert.equal(status, 'playing');
    await alice.req('game/subscribe', { id, position: 1 });
    await bob.req('game/subscribe', { id, position: 2 });

    const bobChat = new Promise(resolve => bob.onEvent('game/chatmessage', resolve));
    await alice.req('game/sendchat', { id, content: '  hello  ' });
    const chatMsg: any = await bobChat;
    assert.equal(chatMsg.player, 1);
    assert.equal(chatMsg.content, 'hello');   // trimmed server-side

    // A late subscriber (e.g. reconnect) sees the chat log via game/subscribe.
    const state = await bob.req<{ state: { chat: { player: number; content: string }[] } }>(
        'game/subscribe', { id, position: 2 },
    );
    assert.deepEqual(state.state.chat, [{ player: 1, time: chatMsg.time, content: 'hello' }]);

    // A connection that owns no slot in the game (pure non-player) is rejected.
    const carol = await registerAndLogin('victor');
    await assert.rejects(carol.req('game/sendchat', { id, content: 'nope' }));

    await alice.close();
    await bob.close();
    await carol.close();
});

test('game/sendchat is rejected once the game has finished', async () => {
    const alice = await registerAndLogin('wendy');
    const bob = await registerAndLogin('xavier');

    const { id, status } = await alice.req<{ id: string; status: string }>('game/create', {
        config: passOnlyConfig(),
        onlinePlayerRequest: fixedRequest([[1, { type: 'local', name: '' }], [2, { type: 'client', name: 'xavier' }]]),
    });
    assert.equal(status, 'playing');

    await alice.req('game/resign', { id });   // ends the game

    await assert.rejects(
        alice.req('game/sendchat', { id, content: 'gg' }),
        /Cannot send messages in a finished game/,
    );

    await alice.close();
    await bob.close();
});

test("a finished game's chat is included in a fresh LOGIN response", async () => {
    const alice = await registerAndLogin('yolanda');
    const bob = await registerAndLogin('zach');

    const { id, status } = await alice.req<{ id: string; status: string }>('game/create', {
        config: passOnlyConfig(),
        onlinePlayerRequest: fixedRequest([[1, { type: 'local', name: '' }], [2, { type: 'client', name: 'zach' }]]),
    });
    assert.equal(status, 'playing');

    await alice.req('game/sendchat', { id, content: 'well played' });
    await alice.req('game/resign', { id });   // ends the game

    await alice.close();
    const aliceAgain = await connect(server.url);
    const login = await aliceAgain.req<
        { finishedGames: { id: string; chat: { player: number; content: string }[] }[] }
    >('LOGIN', { name: 'yolanda', password: 'pw' });
    const record = login.finishedGames.find(g => g.id === id);
    assert.ok(record, 'finished game should be present in the LOGIN response');
    assert.deepEqual(record!.chat.map(c => c.content), ['well played']);

    await aliceAgain.close();
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
            onlinePlayerRequest: fixedRequest([
                [1, { type: 'local', name: '' }], [2, { type: 'pendingInvitedOnline', name: 'nobody' }],
            ]),
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
                    onlinePlayerRequest: fixedRequest([
                        [1, { type: 'local', name: '' }], [2, { type: 'pendingInvitedOnline', name: 'xavier' }],
                    ]),
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
        onlinePlayerRequest: fixedRequest([
            [1, { type: 'local', name: '' }], [2, { type: 'pendingInvitedOnline', name: 'kevin' }],
        ]),
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
        onlinePlayerRequest: fixedRequest([
            [1, { type: 'local', name: '' }], [2, { type: 'pendingInvitedOnline', name: 'mike' }],
        ]),
    });
    await bobInvite;

    const aliceFailed = new Promise<any>(resolve => alice.onEvent('game/invite-failed', resolve));
    const respond = await bob.req<{ status: string }>('game/invite-respond', { id, accept: false });
    assert.equal(respond.status, 'declined');

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

test(
    'a decline notifies every other invitee immediately, but the game stays open until everyone '
    + 'has responded, and a too-late accept is rejected with a specific message',
    async () => {
    const alice = await registerAndLogin('olga');
    const bob = await registerAndLogin('peter');
    const carol = await registerAndLogin('quinn');

    const bobInvite = new Promise<any>(resolve => bob.onEvent('game/invite', resolve));
    const carolInvite = new Promise<any>(resolve => carol.onEvent('game/invite', resolve));
    const { id } = await alice.req<{ id: string }>('game/create', {
        config: passOnlyConfig(),
        onlinePlayerRequest: fixedRequest([
            [1, { type: 'pendingInvitedOnline', name: 'peter' }], [2, { type: 'pendingInvitedOnline', name: 'quinn' }],
        ]),
    });
    await Promise.all([bobInvite, carolInvite]);

    const aliceFailed = new Promise<any>(resolve => alice.onEvent('game/invite-failed', resolve));
    const carolFailed = new Promise<any>(resolve => carol.onEvent('game/invite-failed', resolve));
    const bobFailedEvents: any[] = [];
    bob.onEvent('game/invite-failed', m => bobFailedEvents.push(m));

    const bobRespond = await bob.req<{ status: string }>('game/invite-respond', { id, accept: false });
    assert.equal(bobRespond.status, 'declined');

    // Both alice (the creator) and carol (the other, still-unresponded invitee) are notified
    // right away - the game is doomed the moment any required invitee refuses, regardless of who
    // else hasn't answered yet. Bob already knows he declined - no redundant push to him.
    const [aliceFailedMsg, carolFailedMsg] = await Promise.all([aliceFailed, carolFailed]);
    assert.equal(aliceFailedMsg.id, id);
    assert.equal(carolFailedMsg.id, id);
    assert.equal(bobFailedEvents.length, 0);

    // The pending game record itself isn't torn down yet - carol can still respond. Her accept is
    // still rejected (the game is doomed), but with a specific message, not a raw 404 - and it
    // doesn't trigger a second round of game/invite-failed pushes (already sent above).
    const secondAliceFailedEvents: any[] = [];
    alice.onEvent('game/invite-failed', m => secondAliceFailedEvents.push(m));
    await assert.rejects(
        carol.req('game/invite-respond', { id, accept: true }),
        (e: any) => {
            assert.match(e.message, /already refused by another invited player/);
            assert.equal(e.statusCode, 409);
            return true;
        },
    );
    await new Promise(r => setImmediate(r));
    assert.equal(secondAliceFailedEvents.length, 0);

    // Now that everyone (bob and carol) has responded, the game is fully torn down - a further
    // response 404s.
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
        onlinePlayerRequest: fixedRequest([
            [1, { type: 'pendingInvitedOnline', name: 'sam' }], [2, { type: 'pendingInvitedOnline', name: 'sam' }],
        ]),
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

test('a withdraw request applies once every remaining player agrees, broadcasting game/withdraw to everyone', async () => {
    const alice = await registerAndLogin('aaron');
    const bob = await registerAndLogin('bella');
    const carol = await registerAndLogin('cindy');

    const { id, status } = await alice.req<{ id: string; status: string }>('game/create', {
        config: realThreePlayerConfig(),
        onlinePlayerRequest: fixedRequest([
            [1, { type: 'local', name: '' }], [2, { type: 'client', name: 'bella' }], [3, { type: 'client', name: 'cindy' }],
        ]),
    });
    assert.equal(status, 'playing');
    await alice.req('game/subscribe', { id, position: 1 });
    await bob.req('game/subscribe', { id, position: 2 });
    await carol.req('game/subscribe', { id, position: 3 });

    await alice.req('game/move', { id, moveIndex: 0, clientIdx: 0 });
    await bob.req('game/move', { id, moveIndex: 2, clientIdx: 1 });
    await carol.req('game/move', { id, moveIndex: 4, clientIdx: 2 });
    await alice.req('game/move', { id, moveIndex: 10, clientIdx: 3 });
    // alice's last move is at ply index 3 - the Withdraw button auto-detects it, so this withdraws
    // just that one move (numWithdrawn: 1).

    const bobProposed = new Promise<any>(resolve => bob.onEvent('game/withdraw-proposed', resolve));
    const carolProposed = new Promise<any>(resolve => carol.onEvent('game/withdraw-proposed', resolve));
    const result = await alice.req<{ status: string; numWithdrawn: number }>('game/withdraw-request', { id });
    assert.equal(result.status, 'pending');
    assert.equal(result.numWithdrawn, 1);

    const [bobMsg, carolMsg] = await Promise.all([bobProposed, carolProposed]);
    assert.equal(bobMsg.from, 'aaron');
    assert.equal(bobMsg.numWithdrawn, 1);
    assert.equal(carolMsg.numWithdrawn, 1);

    const aliceWithdraw = new Promise<any>(resolve => alice.onEvent('game/withdraw', resolve));
    const carolWithdraw = new Promise<any>(resolve => carol.onEvent('game/withdraw', resolve));
    const bobRespond = await bob.req<{ status: string }>('game/withdraw-respond', { id, accept: true });
    assert.equal(bobRespond.status, 'waiting');   // carol hasn't responded yet
    const carolRespond = await carol.req<{ status: string }>('game/withdraw-respond', { id, accept: true });
    assert.equal(carolRespond.status, 'applied');   // carol was the last vote needed

    const [aliceMsg, carolWMsg] = await Promise.all([aliceWithdraw, carolWithdraw]);
    assert.equal(aliceMsg.toPly, 3);
    assert.equal(aliceMsg.numWithdrawn, 1);
    assert.deepEqual(aliceMsg, carolWMsg);

    const state = await alice.req<{ state: { moves: unknown[] } }>('game/subscribe', { id, position: 1 });
    assert.equal(state.state.moves.length, 3);

    await alice.close();
    await bob.close();
    await carol.close();
});

test('a decline notifies everyone else, leaves the game unmodified, and a too-late accept gets a specific error', async () => {
    const alice = await registerAndLogin('dexter');
    const bob = await registerAndLogin('elena');
    const carol = await registerAndLogin('felix');

    const { id, status } = await alice.req<{ id: string; status: string }>('game/create', {
        config: realThreePlayerConfig(),
        onlinePlayerRequest: fixedRequest([
            [1, { type: 'local', name: '' }], [2, { type: 'client', name: 'elena' }], [3, { type: 'client', name: 'felix' }],
        ]),
    });
    assert.equal(status, 'playing');
    await alice.req('game/subscribe', { id, position: 1 });
    await bob.req('game/subscribe', { id, position: 2 });
    await carol.req('game/subscribe', { id, position: 3 });

    await alice.req('game/move', { id, moveIndex: 0, clientIdx: 0 });
    await bob.req('game/move', { id, moveIndex: 2, clientIdx: 1 });
    await carol.req('game/move', { id, moveIndex: 4, clientIdx: 2 });
    await alice.req('game/move', { id, moveIndex: 10, clientIdx: 3 });
    await alice.req('game/withdraw-request', { id });

    const aliceFailed = new Promise<any>(resolve => alice.onEvent('game/withdraw-failed', resolve));
    const carolFailed = new Promise<any>(resolve => carol.onEvent('game/withdraw-failed', resolve));
    const bobRespond = await bob.req<{ status: string }>('game/withdraw-respond', { id, accept: false });
    assert.equal(bobRespond.status, 'declined');

    const [aliceMsg, carolMsg] = await Promise.all([aliceFailed, carolFailed]);
    assert.equal(aliceMsg.id, id);
    assert.equal(carolMsg.id, id);

    const state = await alice.req<{ state: { moves: unknown[] } }>('game/subscribe', { id, position: 1 });
    assert.equal(state.state.moves.length, 4, 'a declined withdrawal leaves the move list untouched');

    // The withdraw request record itself isn't torn down yet - carol can still respond, but her
    // accept is rejected (the request is doomed), with a specific message rather than a raw 404.
    await assert.rejects(
        carol.req('game/withdraw-respond', { id, accept: true }),
        (e: any) => { assert.match(e.message, /already declined/); assert.equal(e.statusCode, 409); return true; },
    );

    // Everyone (bob and carol) has now responded, so the request is fully torn down - a further
    // response 404s.
    await assert.rejects(
        carol.req('game/withdraw-respond', { id, accept: false }),
        (e: any) => { assert.equal(e.statusCode, 404); return true; },
    );

    await alice.close();
    await bob.close();
    await carol.close();
});

test('requesting withdrawal with no prior moves gets a specific error', async () => {
    const alice = await registerAndLogin('gabriel');
    const bob = await registerAndLogin('hannah');
    const { id, status } = await alice.req<{ id: string; status: string }>('game/create', {
        config: realTwoPlayerConfig(),
        onlinePlayerRequest: fixedRequest([[1, { type: 'local', name: '' }], [2, { type: 'client', name: 'hannah' }]]),
    });
    assert.equal(status, 'playing');

    await assert.rejects(
        alice.req('game/withdraw-request', { id }),
        (e: any) => {
            assert.equal(e.message, 'Cannot withdraw your move when you have not made any moves');
            assert.equal(e.statusCode, 409);
            return true;
        },
    );

    await alice.close();
    await bob.close();
});

test('a second withdraw request while one is already pending is rejected', async () => {
    const alice = await registerAndLogin('isabel');
    const bob = await registerAndLogin('jasper');
    const { id, status } = await alice.req<{ id: string; status: string }>('game/create', {
        config: realTwoPlayerConfig(),
        onlinePlayerRequest: fixedRequest([[1, { type: 'local', name: '' }], [2, { type: 'client', name: 'jasper' }]]),
    });
    assert.equal(status, 'playing');
    await alice.req('game/move', { id, moveIndex: 0, clientIdx: 0 });

    const first = await alice.req<{ status: string }>('game/withdraw-request', { id });
    assert.equal(first.status, 'pending');

    await assert.rejects(
        alice.req('game/withdraw-request', { id }),
        (e: any) => {
            assert.equal(e.message, 'Cannot start withdrawal request: another withdrawal request in progress');
            assert.equal(e.statusCode, 409);
            return true;
        },
    );

    await alice.close();
    await bob.close();
});

test('game/move and game/resign are rejected while a withdraw vote is pending, and the lock lifts once resolved', async () => {
    const alice = await registerAndLogin('kara');
    const bob = await registerAndLogin('liam');
    const { id, status } = await alice.req<{ id: string; status: string }>('game/create', {
        config: realTwoPlayerConfig(),
        onlinePlayerRequest: fixedRequest([[1, { type: 'local', name: '' }], [2, { type: 'client', name: 'liam' }]]),
    });
    assert.equal(status, 'playing');
    await alice.req('game/move', { id, moveIndex: 0, clientIdx: 0 });

    const req = await alice.req<{ status: string }>('game/withdraw-request', { id });
    assert.equal(req.status, 'pending');

    const lockedError = (e: any) => {
        assert.match(e.message, /withdrawal request is in progress/); assert.equal(e.statusCode, 409); return true;
    };
    await assert.rejects(bob.req('game/move', { id, moveIndex: 1, clientIdx: 1 }), lockedError);
    await assert.rejects(bob.req('game/resign', { id }), lockedError);

    // bob accepts - the withdrawal applies (restoring the pre-alice-move state, so it's still
    // alice's turn) and the lock lifts, so normal play resumes.
    const applied = await bob.req<{ status: string }>('game/withdraw-respond', { id, accept: true });
    assert.equal(applied.status, 'applied');
    await alice.req('game/move', { id, moveIndex: 1, clientIdx: 0 });

    await alice.close();
    await bob.close();
});
