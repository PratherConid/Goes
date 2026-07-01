import { WebSocketServer, WebSocket } from 'ws';
import type { Server } from 'node:http';
import { onlineGameManager } from './onlineGameManager.js';
import { GameConfig } from '@shared/types.js';
import { aiMove, aiHealth } from './engineProxy.js';

// ── Wire protocol ──────────────────────────────────────────────────────────────
//
// Client → server:  { kind:'req', reqId, type, ...payload }
// Server → client:  { kind:'res', reqId, ok:true,  data }
//                   { kind:'res', reqId, ok:false, error, statusCode }
//                   { kind:'event', type:'game/pending-games', id, config } (push, personalized)
//                   { kind:'event', type:'game/start',  id, config }        (push, personalized)
//                   { kind:'event', type:'game/move',   id, moveIndex }      (push)
//                   { kind:'event', type:'game/resign', id, slots }          (push)
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

// The positions this connection owns in game `id`, or throw 403 if none.
function requirePositions(id: string, ws: WebSocket): number[] {
    const positions = onlineGameManager.getPositions(id, ws);
    if (!positions.length)
        throw Object.assign(new Error('You are not a player in this game'), { statusCode: 403 });
    return positions;
}

// Push a lightweight event to every joined connection for game `id`.
function broadcastEvent(id: string, type: string, payload: object) {
    const sockets = onlineGameManager.getSockets(id) as WebSocket[];
    const msg = JSON.stringify({ kind: 'event', type, id, ...payload });
    for (const ws of sockets) if (ws.readyState === ws.OPEN) ws.send(msg);
}

function send(ws: WebSocket, obj: unknown) {
    if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(obj));
}

// Result of handling a request: list of (data + optional broadcast) pairs, plus
// an optional game ID to trigger server-engine advancement after the ack.
interface BroadcastMsg {
    id: string; type: string;
    payload?: object;                                   // uniform: same to all sockets
    perSocket?: { ws: WebSocket; payload: object }[];   // personalized: one entry per socket
}
interface Handled { results: { data: unknown; broadcast?: BroadcastMsg }[]; engineGame?: string; }

// Build a personalised game/pending-games or game/start broadcast for all observers.
function buildBroadcast(id: string, type: string): BroadcastMsg {
    const sockets = onlineGameManager.getSockets(id) as WebSocket[];
    return {
        id, type,
        perSocket: sockets.map(rcv => ({
            ws: rcv,
            payload: { config: onlineGameManager.getPersonalizedConfig(id, rcv) },
        })),
    };
}

// Advances serverEngine turns until a human slot or game over.
async function advanceServerEngine(id: string): Promise<void> {
    const slot = onlineGameManager.getEngineSlot(id);
    if (slot === null) return;
    const params = onlineGameManager.getEngineRequestParams(id, slot);
    if (!params) return;
    try {
        const result = await aiMove(params) as { move: number | null; session_id?: string };
        onlineGameManager.applyEngineMove(id, slot, result.move, result.session_id);
        broadcastEvent(id, 'game/move', { moveIndex: result.move });
        await advanceServerEngine(id);
    } catch (e) {
        console.error('[engine] server engine error for game', id, e);
    }
}

// Dispatch one request; returns the ack data (+ optional broadcast), or throws { statusCode }.
async function handleRequest(ws: WebSocket, msg: ReqMessage): Promise<Handled> {
    switch (msg.type) {
        case 'ai/move':
            return { results: [{ data: await aiMove(msg['body']) }] };
        case 'ai/health':
            return { results: [{ data: await aiHealth() }] };
        case 'game/create': {
            const config = GameConfig.fromJSON(msg['config'] as any);
            const result = onlineGameManager.createGame(config);
            // Bind creator's ws to all pre-assigned client slots (getPositions can't be used here
            // because pi.socket is null until acceptJoin is called).
            for (const [slot, pi] of onlineGameManager.getConfig(result.id).players)
                if (pi.type === 'client') onlineGameManager.acceptJoin(result.id, ws, slot);
            onlineGameManager.addObserver(result.id, ws);
            return {
                results: [{ data: { id: result.id, status: result.status }, broadcast: buildBroadcast(result.id, result.status === 'playing' ? 'game/start' : 'game/pending-games') }],
                engineGame: result.status === 'playing' ? result.id : undefined,
            };
        }
        case 'game/join': {
            const id = msg['id'] as string;
            const playerName = (msg['playerName'] as string) ?? 'Anonymous';
            const result = onlineGameManager.joinGame(id, playerName);
            onlineGameManager.acceptJoin(id, ws, result.position);
            return {
                results: [{ data: { position: result.position }, broadcast: buildBroadcast(id, result.status === 'playing' ? 'game/start' : 'game/pending-games') }],
                engineGame: result.status === 'playing' ? id : undefined,
            };
        }
        case 'game/move': {
            const id = msg['id'] as string;
            const positions = requirePositions(id, ws);
            const moveIndex = (msg['moveIndex'] as number | null) ?? null;
            onlineGameManager.applyMove(id, positions, moveIndex, msg['clientIdx'] as number);
            return { results: [{ data: { ok: true }, broadcast: { id, type: 'game/move', payload: { moveIndex } } }], engineGame: id };
        }
        case 'game/resign': {
            const id = msg['id'] as string;
            const positions = requirePositions(id, ws);
            const slot = onlineGameManager.resign(id, positions);
            return { results: [{ data: { ok: true }, broadcast: { id, type: 'game/resign', payload: { slots: [slot] } } }], engineGame: id };
        }
        case 'game/subscribe': {
            // Re-bind this connection after a reconnect; reply with full state + personalised config for catchup.
            const id = msg['id'] as string;
            const position = msg['position'] as number;
            onlineGameManager.acceptJoin(id, ws, position);
            return { results: [{ data: { state: onlineGameManager.getState(id), config: onlineGameManager.getPersonalizedConfig(id, ws) } }] };
        }
        default:
            throw Object.assign(new Error(`Unknown request type: ${msg.type}`), { statusCode: 400 });
    }
}

export function attachWebSocket(server: Server) {
    const wss = new WebSocketServer({ server, path: '/ws' });

    wss.on('connection', (ws: WebSocket) => {
        ws.on('message', (raw) => {
            let msg: ReqMessage;
            try {
                msg = JSON.parse(raw.toString());
            } catch {
                return;  // ignore malformed frames
            }
            if (msg?.kind !== 'req' || typeof msg.reqId !== 'number') return;

            handleRequest(ws, msg)
                .then(({ results, engineGame }) => {
                    send(ws, { kind: 'res', reqId: msg.reqId, ok: true, data: results[0].data });
                    for (const { broadcast } of results) {
                        if (broadcast) {
                            const { type, id } = broadcast;
                            if (broadcast.perSocket)
                                for (const { ws: rcv, payload } of broadcast.perSocket)
                                    send(rcv, { kind: 'event', type, id, ...payload });
                            else broadcastEvent(id, type, broadcast.payload!);
                        }
                    }
                    if (engineGame) void advanceServerEngine(engineGame);
                })
                .catch((e: any) => send(ws, {
                    kind: 'res', reqId: msg.reqId, ok: false,
                    error: e?.message ?? 'Internal error',
                    statusCode: e?.statusCode ?? 500,
                }));
        });

        ws.on('close', () => onlineGameManager.removeConnection(ws));
    });

    console.log('[ws] WebSocket server attached at /ws');
}
