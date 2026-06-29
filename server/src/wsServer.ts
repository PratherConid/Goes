import { WebSocketServer, WebSocket } from 'ws';
import type { Server } from 'node:http';
import { onlineGameManager } from './onlineGameManager.js';
import type { GameConfig } from '@shared/types.js';
import { aiMove, aiHealth } from './engineProxy.js';

// ── Wire protocol ──────────────────────────────────────────────────────────────
//
// Client → server:  { kind:'req', reqId, type, ...payload }
// Server → client:  { kind:'res', reqId, ok:true,  data }
//                   { kind:'res', reqId, ok:false, error, statusCode }
//                   { kind:'event', type:'game/pending-players', id, players } (push, personalized)
//                   { kind:'event', type:'game/start',  id, config, players } (push, personalized)
//                   { kind:'event', type:'game/move',   id, moveIndex }        (push)
//                   { kind:'event', type:'game/resign', id, slots }            (push)
//
// While waiting, game/pending-players is broadcast after every join so clients see
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

// Result of handling a request: the data to ack, and an optional broadcast to fire
// after the ack so the requester's ack arrives first (avoids a client-side race).
interface BroadcastMsg {
    id: string; type: string;
    payload?: object;                                   // uniform: same to all sockets
    perSocket?: { ws: WebSocket; payload: object }[];   // personalized: one entry per socket
}
interface Handled { data: unknown; broadcast?: BroadcastMsg; }

// Build the personalized players payload for one socket in an active game (game/start).
function buildStartPayload(id: string, ws: WebSocket): object {
    const mySlots = new Set(onlineGameManager.getPositions(id, ws));
    const players = (onlineGameManager.getState(id).players as ({ slot: number; name: string } | null)[])
        .filter((p): p is { slot: number; name: string } => p !== null)
        .map(p => ({ ...p, type: mySlots.has(p.slot) ? 'local' : 'server' }));
    return { config: onlineGameManager.getConfig(id), players };
}

// Build the personalized players payload for one socket in a pending game.
function buildPendingPlayersPayload(id: string, ws: WebSocket): object {
    const players = onlineGameManager.getPendingPlayers(id) ?? new Map();
    const mySlots = new Set(onlineGameManager.getPositions(id, ws));
    return {
        players: [...players.entries()].map(([slot, pi]) => ({
            slot, name: pi.name,
            type: mySlots.has(slot) ? 'local' : 'server',
        })),
    };
}

// Dispatch one request; returns the ack data (+ optional broadcast), or throws { statusCode }.
async function handleRequest(ws: WebSocket, msg: ReqMessage): Promise<Handled> {
    switch (msg.type) {
        case 'ai/move':
            return { data: await aiMove(msg['body']) };
        case 'ai/health':
            return { data: await aiHealth() };
        case 'game/create': {
            const config = msg['config'] as GameConfig;
            const playerName = (msg['playerName'] as string) ?? 'Anonymous';
            const result = onlineGameManager.createGame(config, playerName);
            onlineGameManager.acceptJoin(result.id, ws, result.position);
            const sockets = onlineGameManager.getSockets(result.id) as WebSocket[];
            return {
                data: result,
                broadcast: { id: result.id, type: 'game/pending-players',
                    perSocket: sockets.map(rcv => ({ ws: rcv, payload: buildPendingPlayersPayload(result.id, rcv) })) },
            };
        }
        case 'game/join': {
            const id = msg['id'] as string;
            const playerName = (msg['playerName'] as string) ?? 'Anonymous';
            const result = onlineGameManager.joinGame(id, playerName);
            onlineGameManager.acceptJoin(id, ws, result.position);
            const sockets = onlineGameManager.getSockets(id) as WebSocket[];
            const broadcast = result.status === 'playing'
                ? { id, type: 'game/start',
                    perSocket: sockets.map(rcv => ({ ws: rcv, payload: buildStartPayload(id, rcv) })) }
                : { id, type: 'game/pending-players',
                    perSocket: sockets.map(rcv => ({ ws: rcv, payload: buildPendingPlayersPayload(id, rcv) })) };
            return { data: { position: result.position }, broadcast };
        }
        case 'game/move': {
            const id = msg['id'] as string;
            const positions = requirePositions(id, ws);
            const moveIndex = (msg['moveIndex'] as number | null) ?? null;
            onlineGameManager.applyMove(id, positions, moveIndex, msg['clientIdx'] as number);
            return { data: { ok: true }, broadcast: { id, type: 'game/move', payload: { moveIndex } } };
        }
        case 'game/resign': {
            const id = msg['id'] as string;
            const positions = requirePositions(id, ws);
            onlineGameManager.resign(id, positions);
            return { data: { ok: true }, broadcast: { id, type: 'game/resign', payload: { slots: positions } } };
        }
        case 'game/subscribe': {
            // Re-bind this connection after a reconnect; reply with full state for catchup sync.
            const id = msg['id'] as string;
            const position = msg['position'] as number;
            onlineGameManager.acceptJoin(id, ws, position);
            return { data: { state: onlineGameManager.getState(id), config: onlineGameManager.getConfig(id) } };
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
                .then(({ data, broadcast }) => {
                    send(ws, { kind: 'res', reqId: msg.reqId, ok: true, data });
                    if (broadcast) {
                        const { type, id } = broadcast;
                        if (broadcast.perSocket)
                            for (const { ws: rcv, payload } of broadcast.perSocket)
                                send(rcv, { kind: 'event', type, id, ...payload });
                        else broadcastEvent(id, type, broadcast.payload!);
                    }
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
