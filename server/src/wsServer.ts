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
//                   { kind:'event', type:'game/start',  id, config, players }  (push)
//                   { kind:'event', type:'game/move',   id, moveIndex }         (push)
//                   { kind:'event', type:'game/resign', id, slots }             (push)
//
// After a game starts the server forwards only the minimal change (move index or
// resigned slots). Clients maintain their own BoardState incrementally. Reconnects
// use game/subscribe, which returns the full OnlineStateResponse for a catchup sync.

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
interface BroadcastMsg { id: string; type: string; payload: object; }
interface Handled { data: unknown; broadcast?: BroadcastMsg; }

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
            return { data: result };   // ack only; game is 'waiting', nobody is notified yet
        }
        case 'game/join': {
            const id = msg['id'] as string;
            const playerName = (msg['playerName'] as string) ?? 'Anonymous';
            const result = onlineGameManager.joinGame(id, playerName);
            onlineGameManager.acceptJoin(id, ws, result.position);
            const broadcast = result.status === 'playing' ? {
                id, type: 'game/start',
                payload: { config: onlineGameManager.getConfig(id), players: onlineGameManager.getState(id).players },
            } : undefined;
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
                    if (broadcast) broadcastEvent(broadcast.id, broadcast.type, broadcast.payload);
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
