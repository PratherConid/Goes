import { WebSocketServer, WebSocket } from 'ws';
import type { Server } from 'node:http';
import { onlineGameManager } from './onlineGameManager.js';
import type { OnlineGameConfig } from './onlineGameManager.js';
import { aiMove, aiHealth } from './engineProxy.js';

// ── Wire protocol ──────────────────────────────────────────────────────────────
//
// Client → server:  { kind:'req', reqId, type, ...payload }
// Server → client:  { kind:'res', reqId, ok:true,  data }
//                   { kind:'res', reqId, ok:false, error, statusCode }
//                   { kind:'event', type:'game/state', id, state }   (push, no reqId)

interface ReqMessage {
    kind: 'req';
    reqId: number;
    type: string;
    [k: string]: unknown;
}

// Each game's subscribed connections, mapped to the player position (join index)
// that connection owns. Identity is the connection itself: moves/resigns are
// attributed to the position bound here, never to a client-supplied value.
const subscribers = new Map<string, Map<WebSocket, number>>();

function subscribe(id: string, ws: WebSocket, position: number) {
    let map = subscribers.get(id);
    if (!map) { map = new Map(); subscribers.set(id, map); }
    map.set(ws, position);
}

// The player position this connection owns in game `id`, or throw 403 if none.
function requirePosition(id: string, ws: WebSocket): number {
    const position = subscribers.get(id)?.get(ws);
    if (position === undefined)
        throw Object.assign(new Error('You are not a player in this game'), { statusCode: 403 });
    return position;
}

// Push the current state of game `id` to every subscribed connection.
function broadcastState(id: string) {
    const map = subscribers.get(id);
    if (!map || map.size === 0) return;
    let state: unknown;
    try { state = onlineGameManager.getState(id); } catch { return; }
    const msg = JSON.stringify({ kind: 'event', type: 'game/state', id, state });
    for (const ws of map.keys()) if (ws.readyState === ws.OPEN) ws.send(msg);
}

function send(ws: WebSocket, obj: unknown) {
    if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(obj));
}

// Dispatch one request; returns the data to send back, or throws { statusCode }.
async function handleRequest(ws: WebSocket, msg: ReqMessage): Promise<unknown> {
    switch (msg.type) {
        case 'ai/move':
            return aiMove(msg['body']);
        case 'ai/health':
            return aiHealth();
        case 'game/create': {
            const config = msg['config'] as OnlineGameConfig;
            const playerName = (msg['playerName'] as string) ?? 'Anonymous';
            const result = onlineGameManager.createGame(config, playerName);
            subscribe(result.id, ws, result.position);
            return result;
        }
        case 'game/join': {
            const id = msg['id'] as string;
            const playerName = (msg['playerName'] as string) ?? 'Anonymous';
            const result = onlineGameManager.joinGame(id, playerName);
            subscribe(id, ws, result.position);
            broadcastState(id);   // notify the host (and others) that someone joined / game started
            // Include the current state so the joiner applies it directly: the broadcast
            // above races ahead of this response and is dropped client-side (onlineGameId
            // isn't set yet), so we must not rely on it for the joiner's initial state.
            return { ...result, state: onlineGameManager.getState(id) };
        }
        case 'game/move': {
            const id = msg['id'] as string;
            const position = requirePosition(id, ws);   // who you are = your connection
            const result = onlineGameManager.applyMove(
                id, position,
                (msg['moveIndex'] as number | null) ?? null,
                msg['clientIdx'] as number,
            );
            broadcastState(id);
            return result;
        }
        case 'game/resign': {
            const id = msg['id'] as string;
            const position = requirePosition(id, ws);
            const result = onlineGameManager.resign(id, position);
            broadcastState(id);
            return result;
        }
        case 'game/subscribe': {
            // Re-bind this connection to its player position after a reconnect,
            // then reply with the current state.
            const id = msg['id'] as string;
            subscribe(id, ws, msg['position'] as number);
            return onlineGameManager.getState(id);
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
                .then(data => send(ws, { kind: 'res', reqId: msg.reqId, ok: true, data }))
                .catch((e: any) => send(ws, {
                    kind: 'res', reqId: msg.reqId, ok: false,
                    error: e?.message ?? 'Internal error',
                    statusCode: e?.statusCode ?? 500,
                }));
        });

        ws.on('close', () => {
            for (const map of subscribers.values()) map.delete(ws);
        });
    });

    console.log('[ws] WebSocket server attached at /ws');
}
