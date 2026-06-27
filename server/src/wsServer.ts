import { WebSocketServer, WebSocket } from 'ws';
import type { Server } from 'node:http';
import { onlineGameManager } from './onlineGameManager.js';
import type { OnlineGameConfig } from '@shared/types.js';
import { aiMove, aiHealth } from './engineProxy.js';

// ── Wire protocol ──────────────────────────────────────────────────────────────
//
// Client → server:  { kind:'req', reqId, type, ...payload }
// Server → client:  { kind:'res', reqId, ok:true,  data }
//                   { kind:'res', reqId, ok:false, error, statusCode }
//                   { kind:'event', type:'game/state', id, state, config }  (push, no reqId)
//
// Create/join only ack (id/position); a connection's renderer state changes only
// when it receives a game/state event — which the server broadcasts when the game
// starts and after every move/resign. The event carries the config so each client
// can build its board the first time it sees a started game.

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

// Current state + static config for game `id` (the payload of a game/state event).
function stateWithConfig(id: string) {
    return { state: onlineGameManager.getState(id), config: onlineGameManager.getConfig(id) };
}

// Push the current state of game `id` to every subscribed connection.
function broadcastState(id: string) {
    const map = subscribers.get(id);
    if (!map || map.size === 0) return;
    let payload: { state: unknown; config: unknown };
    try { payload = stateWithConfig(id); } catch { return; }
    const msg = JSON.stringify({ kind: 'event', type: 'game/state', id, ...payload });
    for (const ws of map.keys()) if (ws.readyState === ws.OPEN) ws.send(msg);
}

function send(ws: WebSocket, obj: unknown) {
    if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(obj));
}

// Result of handling a request: the data to ack, and optionally a game id to
// broadcast *after* the ack is sent (so the requester's ack arrives first and its
// onlineGameId is set before the game/state event — avoids a client-side race).
interface Handled { data: unknown; broadcastId?: string; }

// Dispatch one request; returns the ack data (+ optional broadcast), or throws { statusCode }.
async function handleRequest(ws: WebSocket, msg: ReqMessage): Promise<Handled> {
    switch (msg.type) {
        case 'ai/move':
            return { data: await aiMove(msg['body']) };
        case 'ai/health':
            return { data: await aiHealth() };
        case 'game/create': {
            const config = msg['config'] as OnlineGameConfig;
            const playerName = (msg['playerName'] as string) ?? 'Anonymous';
            const result = onlineGameManager.createGame(config, playerName);
            subscribe(result.id, ws, result.position);
            return { data: result };   // ack only; game is 'waiting', nobody is notified yet
        }
        case 'game/join': {
            const id = msg['id'] as string;
            const playerName = (msg['playerName'] as string) ?? 'Anonymous';
            if (subscribers.get(id)?.has(ws))
                throw Object.assign(new Error('Connection already joined this game'), { statusCode: 409 });
            const result = onlineGameManager.joinGame(id, playerName);
            subscribe(id, ws, result.position);
            // Ack only. Broadcast lazily: just the join that starts the game (status
            // becomes 'playing') notifies everyone, *after* this ack so the joiner's
            // onlineGameId is set when the game/state event arrives.
            return { data: { position: result.position },
                     broadcastId: result.status === 'playing' ? id : undefined };
        }
        case 'game/move': {
            const id = msg['id'] as string;
            const position = requirePosition(id, ws);   // who you are = your connection
            onlineGameManager.applyMove(
                id, position,
                (msg['moveIndex'] as number | null) ?? null,
                msg['clientIdx'] as number,
            );
            return { data: { ok: true }, broadcastId: id };   // state arrives via the broadcast
        }
        case 'game/resign': {
            const id = msg['id'] as string;
            const position = requirePosition(id, ws);
            onlineGameManager.resign(id, position);
            return { data: { ok: true }, broadcastId: id };
        }
        case 'game/subscribe': {
            // Re-bind this connection to its player position after a reconnect, and
            // reply with the current state + config so it can resync.
            const id = msg['id'] as string;
            subscribe(id, ws, msg['position'] as number);
            return { data: stateWithConfig(id) };
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
                .then(({ data, broadcastId }) => {
                    send(ws, { kind: 'res', reqId: msg.reqId, ok: true, data });
                    if (broadcastId) broadcastState(broadcastId);   // after the ack
                })
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
