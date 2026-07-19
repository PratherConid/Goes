import { WebSocketServer, WebSocket } from 'ws';
import type { Server } from 'node:http';
import { OnlineGameManager } from './onlineGameManager.js';
import { GameConfig, OnlinePlayerRequest } from '@shared/types.js';
import { engineManager, aiMove } from './engineManager.js';
import { loadUserStore, registerUser, verifyLogin, userExists } from './userStore.js';
import { loadGameRecordStore } from './gameRecordStore.js';

// ── Wire protocol ──────────────────────────────────────────────────────────────
//
// Client → server:  { kind:'req', reqId, type, ...payload }
// Server → client:  { kind:'res', reqId, ok:true,  data }
//                   { kind:'res', reqId, ok:false, error, statusCode }
//                   { kind:'event', type:'game/pending-games', id, config } (push, personalized)
//                   { kind:'event', type:'game/start',  id, config }        (push, personalized)
//                   { kind:'event', type:'game/move',   id, moveIndex, stone } (push)
//                   { kind:'event', type:'game/resign', id, slots }          (push)
//                   { kind:'event', type:'game/invite', id, from }           (push, personalized - to one invited user)
//                   { kind:'event', type:'game/invite-failed', id, message } (push, personalized - to everyone involved)
//                   { kind:'event', type:'game/engine-error', id, message }  (push)
//
// While waiting, game/pending-games is broadcast after every join so clients see
// who has joined. After a game starts only the minimal change is forwarded.
// Reconnects use game/subscribe, which returns full OnlineStateResponse for catchup.

interface ReqMessage {
    kind: 'req';
    reqId: number;
    type: string;
    [k: string]: unknown;
}

// Result of handling a request: list of (data + optional broadcast) pairs, plus
// an optional game ID to trigger server-engine advancement after the ack.
interface BroadcastMsg {
    id: string; type: string;
    payload: object;   // same to every observer - usernames are the single point of
                        // reference now, so the client figures out which slot (if any)
                        // is its own by comparing PlayerInfo.name to its own userName.
}
// Personalized events sent directly to specific usernames (not a game's
// observer set - see broadcastEvent) - e.g. an invite going to someone who
// isn't a participant yet. handleRequest itself never calls send()
// directly (the sole pre-existing exception is FLOGIN's auth/kicked, a
// narrow single-recipient case tightly coupled to also closing that
// connection); every case instead returns data that the one dispatch site
// below actually sends, after the request's own ack - `pushes` extends that
// same convention to arbitrary multi-recipient notifications.
interface PersonalizedPush { to: string; type: string; payload: object; }
interface Handled { results: { data: unknown; broadcast?: BroadcastMsg }[]; engineGame?: string; pushes?: PersonalizedPush[]; }

function send(ws: WebSocket, obj: unknown) {
    if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(obj));
}

// This is the one place server/src turns `dataDir` into live state and wires
// everything up - userStore/gameRecordStore/OnlineGameManager are all
// stateless/argument-driven modules with no initialization of their own.
export function attachWebSocket(server: Server, dataDir: string): void {
    const userStoreState  = loadUserStore(dataDir);
    const gameRecordState = loadGameRecordStore(dataDir);
    const onlineGameManager = new OnlineGameManager(gameRecordState);

    // Per-connection numeric IDs used to key local-game engine processes.
    let _wsCounter = 0;
    const _wsIds = new WeakMap<WebSocket, number>();

    // The name each connection is currently logged in as, set by REGISTER/LOGIN/FLOGIN.
    // A WeakMap needs no explicit cleanup on disconnect - the entry is GC'd along
    // with the WebSocket object once the connection closes.
    const wsToUser = new WeakMap<WebSocket, string>();

    // The live connection currently logged in as each username - the one place the
    // server still deals in raw WebSocket objects, used only to resolve who to
    // actually .send() to. Must be a plain Map (not WeakMap): queried by string key.
    const userToWs = new Map<string, WebSocket>();

    // Records a successful REGISTER/LOGIN/FLOGIN for `ws`. If this connection was
    // previously logged in as a different name, releases that old mapping first
    // (guarded so it only clears an entry that still points at this connection).
    function setLogin(ws: WebSocket, name: string): void {
        const prev = wsToUser.get(ws);
        if (prev && prev !== name && userToWs.get(prev) === ws) userToWs.delete(prev);
        wsToUser.set(ws, name);
        userToWs.set(name, ws);
    }

    // The positions this connection owns in game `id`, or throw 403 if none.
    // Requires the connection to be logged in (a username is the only identity
    // getPositions understands now).
    function requirePositions(id: string, ws: WebSocket): number[] {
        const userName = wsToUser.get(ws);
        if (!userName) throw Object.assign(new Error('Not logged in'), { statusCode: 401 });
        const positions = onlineGameManager.getPositions(id, userName);
        if (!positions.length)
            throw Object.assign(new Error('You are not a player in this game'), { statusCode: 403 });
        return positions;
    }

    // Push a lightweight event to every joined connection for game `id`. Usernames
    // with no currently-live connection (per userToWs) are silently skipped.
    function broadcastEvent(id: string, type: string, payload: object) {
        const msg = JSON.stringify({ kind: 'event', type, id, ...payload });
        for (const name of onlineGameManager.getObservers(id)) {
            const ws = userToWs.get(name);
            if (ws && ws.readyState === ws.OPEN) ws.send(msg);
        }
    }

    // Build a game/pending-games or game/start broadcast carrying the game's config as-is.
    function buildBroadcast(id: string, type: string): BroadcastMsg {
        return { id, type, payload: { config: onlineGameManager.getConfig(id) } };
    }

    // Advances serverEngine turns until a human slot or game over.
    // Releases the engine process once the game is over.
    async function advanceServerEngine(id: string): Promise<void> {
        const slot = onlineGameManager.getEngineSlot(id);
        if (slot === null) {
            if (onlineGameManager.isGameOver(id)) engineManager.release(id);
            return;
        }
        let url: string;
        try {
            url = await engineManager.getOrCreate(id);
        } catch (e: any) {
            console.error('[engine] failed to start process for game', id, e);
            broadcastEvent(id, 'game/engine-error', { message: e?.message ?? 'Engine failed to start' });
            return;
        }
        const params = onlineGameManager.getEngineRequestParams(id, slot);
        if (!params) return;
        try {
            const result = await aiMove(url, params) as { move: number | null; stone: number | null; session_id?: string };
            onlineGameManager.applyEngineMove(id, slot, result.move, result.stone, result.session_id);
            broadcastEvent(id, 'game/move', { moveIndex: result.move, stone: result.stone });
            await advanceServerEngine(id);
        } catch (e: any) {
            console.error('[engine] server engine error for game', id, e);
            broadcastEvent(id, 'game/engine-error', { message: e?.message ?? 'Engine move failed' });
        }
    }

    // JSON-safe {id, finishedGame} list for a successful REGISTER/LOGIN/FLOGIN response,
    // so the client can populate its own finishedGames without having watched them live.
    function buildFinishedGamesPayload(name: string): { id: string; finishedGame: unknown }[] {
        return onlineGameManager.getFinishedGamesFor(name)
            .map(({ id, finishedGame }) => ({ id, finishedGame: finishedGame.toJSON() }));
    }

    // Dispatch one request; returns the ack data (+ optional broadcast), or throws { statusCode }.
    async function handleRequest(ws: WebSocket, msg: ReqMessage): Promise<Handled> {
        switch (msg.type) {
            case 'ai/move': {
                const body  = msg['body'] as Record<string, unknown>;
                const gameId = body['game_id'] as string | undefined;
                if (!gameId)
                    throw Object.assign(new Error('game_id is required for ai/move'), { statusCode: 400 });
                const wsId = _wsIds.get(ws)!;
                const url  = await engineManager.getOrCreate(`local:${wsId}:${gameId}`);
                const { game_id: _gid, ...engineBody } = body;
                return { results: [{ data: await aiMove(url, engineBody) }] };
            }
            case 'ai/health':
                return { results: [{ data: { status: engineManager.ready ? 'ok' : 'unavailable' } }] };
            case 'REGISTER': {
                const name = msg['name'] as string, password = msg['password'] as string;
                if (!name || !password)
                    throw Object.assign(new Error('name and password are required'), { statusCode: 400 });
                const result = await registerUser(userStoreState, name, password);
                if (!result.ok) throw Object.assign(new Error(result.error), { statusCode: 409 });
                setLogin(ws, name);  // auto-login after successful registration
                return { results: [{ data: { name, finishedGames: buildFinishedGamesPayload(name) } }] };
            }
            case 'LOGIN': {
                const name = msg['name'] as string, password = msg['password'] as string;
                if (!name || !password)
                    throw Object.assign(new Error('name and password are required'), { statusCode: 400 });
                if (!await verifyLogin(userStoreState, name, password))
                    throw Object.assign(new Error('Invalid username or password'), { statusCode: 401 });
                if (userToWs.has(name) && userToWs.get(name) !== ws)
                    throw Object.assign(new Error('Already logged in elsewhere - use flogin to take over'), { statusCode: 409 });
                setLogin(ws, name);
                return { results: [{ data: { name, finishedGames: buildFinishedGamesPayload(name) } }] };
            }
            case 'FLOGIN': {
                const name = msg['name'] as string, password = msg['password'] as string;
                if (!name || !password)
                    throw Object.assign(new Error('name and password are required'), { statusCode: 400 });
                if (!await verifyLogin(userStoreState, name, password))
                    throw Object.assign(new Error('Invalid username or password'), { statusCode: 401 });
                const existing = userToWs.get(name);
                if (existing && existing !== ws) {
                    send(existing, { kind: 'event', type: 'auth/kicked', name });
                    existing.close();
                }
                setLogin(ws, name);
                return { results: [{ data: { name, finishedGames: buildFinishedGamesPayload(name) } }] };
            }
            case 'game/create': {
                const userName = wsToUser.get(ws);
                if (!userName) throw Object.assign(new Error('Not logged in'), { statusCode: 401 });
                const config = GameConfig.fromJSON(msg['config'] as any);
                const request = OnlinePlayerRequest.fromJSON(msg['onlinePlayerRequest'] as any);
                // The server, not the client, decides the name for slots this connection
                // controls - overwrite rather than trust whatever the client's JSON contained.
                for (const pi of request.fixedOrder.values())  if (pi.type === 'local') pi.name = userName;
                for (const pi of request.randomOrder)          if (pi.type === 'local') pi.name = userName;
                // Refuse the whole request (no game created at all) if any invited
                // username isn't a real account, or isn't currently online - checked
                // before createGame() runs, since an offline invitee would otherwise
                // never learn about the invite (the game/invite push below is only
                // delivered to a live connection, with no persistence/catch-up for a
                // missed one). Only the list request.fixed actually selects is checked
                // - createGame() itself ignores the other one entirely (see its own
                // fixed-branch), so a stale invite left over in the inactive list (e.g.
                // from before switching fixed/random modes) must not block a request
                // that no longer uses it.
                const activeEntries = request.fixed ? [...request.fixedOrder.values()] : request.randomOrder;
                const offlineInvited = new Set<string>();
                for (const pi of activeEntries) {
                    if (pi.type !== 'pendingInvitedOnline') continue;
                    if (!userExists(userStoreState, pi.name))
                        throw Object.assign(new Error(`Invited user "${pi.name}" does not exist`), { statusCode: 400 });
                    if (!userToWs.has(pi.name)) offlineInvited.add(pi.name);
                }
                if (offlineInvited.size > 0) {
                    const names = [...offlineInvited];
                    const label = names.length === 1
                        ? `User ${names[0]} is offline.`
                        : `Users ${names.join(', ')} are offline.`;
                    throw Object.assign(new Error(`Cannot create game. ${label}`), { statusCode: 409 });
                }
                const result = onlineGameManager.createGame(config, request);
                // Slot ownership already follows from pi.name (set above); just mark the
                // creator as observing the game for broadcast purposes.
                onlineGameManager.addObserver(result.id, userName);
                // Personally notify each invited user (if currently connected) - they
                // aren't observers yet (see OnlineGameManager.respondToInvite's doc
                // comment), so the regular observer-set broadcast below won't reach
                // them. Deduped by username - a user invited into multiple slots
                // (respondToInvite() resolves all of them at once) should still only
                // get a single invite popup.
                const pushes = [...new Set(
                    [...onlineGameManager.getConfig(result.id).players.values()]
                        .filter(pi => pi.type === 'pendingInvitedOnline')
                        .map(pi => pi.name)
                )].map(name => ({ to: name, type: 'game/invite', payload: { id: result.id, from: userName } }));
                return {
                    results: [{ data: { id: result.id, status: result.status }, broadcast: buildBroadcast(result.id, result.status === 'playing' ? 'game/start' : 'game/pending-games') }],
                    engineGame: result.status === 'playing' ? result.id : undefined,
                    pushes,
                };
            }
            case 'game/invite-respond': {
                const userName = wsToUser.get(ws);
                if (!userName) throw Object.assign(new Error('Not logged in'), { statusCode: 401 });
                const id = msg['id'] as string;
                const accept = msg['accept'] as boolean;
                const result = onlineGameManager.respondToInvite(id, userName, accept);
                if (result.status === 'cancelled') {
                    return {
                        results: [{ data: { status: result.status } }],
                        pushes: result.notify.map(name => ({ to: name, type: 'game/invite-failed', payload: { id, message: `Creation of invited online game ${id} failed due to user refusal` } })),
                    };
                }
                return {
                    results: [{ data: { status: result.status }, broadcast: buildBroadcast(id, result.status === 'playing' ? 'game/start' : 'game/pending-games') }],
                    engineGame: result.status === 'playing' ? id : undefined,
                };
            }
            case 'game/join': {
                const userName = wsToUser.get(ws);
                if (!userName) throw Object.assign(new Error('Not logged in'), { statusCode: 401 });
                const id = msg['id'] as string;
                const result = onlineGameManager.joinGame(id, userName);
                onlineGameManager.addObserver(id, userName);
                return {
                    results: [{ data: { position: result.position }, broadcast: buildBroadcast(id, result.status === 'playing' ? 'game/start' : 'game/pending-games') }],
                    engineGame: result.status === 'playing' ? id : undefined,
                };
            }
            case 'game/move': {
                const id = msg['id'] as string;
                const positions = requirePositions(id, ws);
                const moveIndex = (msg['moveIndex'] as number | null) ?? null;
                const stone = (msg['stone'] as number | null) ?? null;
                onlineGameManager.applyMove(id, positions, moveIndex, stone, msg['clientIdx'] as number);
                return { results: [{ data: { ok: true }, broadcast: { id, type: 'game/move', payload: { moveIndex, stone } } }], engineGame: id };
            }
            case 'game/resign': {
                const id = msg['id'] as string;
                const positions = requirePositions(id, ws);
                const slot = onlineGameManager.resign(id, positions);
                return { results: [{ data: { ok: true }, broadcast: { id, type: 'game/resign', payload: { slots: [slot] } } }], engineGame: id };
            }
            case 'game/subscribe': {
                // Re-bind this connection after a reconnect; reply with full state + personalised config for catchup.
                const userName = wsToUser.get(ws);
                if (!userName) throw Object.assign(new Error('Not logged in'), { statusCode: 401 });
                const id = msg['id'] as string;
                const position = msg['position'] as number;
                if (!onlineGameManager.acceptJoin(id, userName, position))
                    throw Object.assign(new Error('Not your slot'), { statusCode: 403 });
                return { results: [{ data: { state: onlineGameManager.getState(id), config: onlineGameManager.getConfig(id) } }] };
            }
            default:
                throw Object.assign(new Error(`Unknown request type: ${msg.type}`), { statusCode: 400 });
        }
    }

    const wss = new WebSocketServer({ server, path: '/ws' });

    wss.on('connection', (ws: WebSocket) => {
        const wsId = ++_wsCounter;
        _wsIds.set(ws, wsId);

        ws.on('message', (raw) => {
            let msg: ReqMessage;
            try {
                msg = JSON.parse(raw.toString());
            } catch {
                return;  // ignore malformed frames
            }
            if (msg?.kind !== 'req' || typeof msg.reqId !== 'number') return;

            handleRequest(ws, msg)
                .then(({ results, engineGame, pushes }) => {
                    send(ws, { kind: 'res', reqId: msg.reqId, ok: true, data: results[0].data });
                    for (const { broadcast } of results) {
                        if (broadcast) broadcastEvent(broadcast.id, broadcast.type, broadcast.payload);
                    }
                    for (const { to, type, payload } of pushes ?? []) {
                        const targetWs = userToWs.get(to);
                        if (targetWs) send(targetWs, { kind: 'event', type, ...payload });
                    }
                    if (engineGame) void advanceServerEngine(engineGame);
                })
                .catch((e: any) => {
                    // A thrown error can still carry `pushes` (see e.g.
                    // OnlineGameManager.respondToInvite()'s already-refused-game
                    // branch) - forward them exactly like the success path above,
                    // before sending the error response to the original requester.
                    for (const { to, type, payload } of e?.pushes ?? []) {
                        const targetWs = userToWs.get(to);
                        if (targetWs) send(targetWs, { kind: 'event', type, ...payload });
                    }
                    send(ws, {
                        kind: 'res', reqId: msg.reqId, ok: false,
                        error: e?.message ?? 'Internal error',
                        statusCode: e?.statusCode ?? 500,
                    });
                });
        });

        ws.on('close', () => {
            // Only clear the userToWs entry if it still points at this connection - a
            // stale close event from a connection that was just force-logged-out by
            // FLOGIN must not clobber the new connection's mapping.
            const name = wsToUser.get(ws);
            if (name && userToWs.get(name) === ws) userToWs.delete(name);
            engineManager.releasePrefix(`local:${wsId}:`);
        });
    });

    console.log('[ws] WebSocket server attached at /ws');
}
