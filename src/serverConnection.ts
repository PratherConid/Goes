// Persistent WebSocket connection to the main server.
//
// Replaces all HTTP fetches to the main server with request/response messages
// correlated by reqId, plus server-pushed events (e.g. online-game state).
//
// Wire protocol (see server/src/wsServer.ts):
//   C→S  { kind:'req', reqId, type, ...payload }
//   S→C  { kind:'res', reqId, ok, data | (error, statusCode) }
//        { kind:'event', type, ...fields }

export interface RequestError extends Error {
    statusCode?: number;
}

interface Pending {
    resolve: (data: unknown) => void;
    reject: (err: RequestError) => void;
}

// Handle for an in-flight request, allowing it to be cancelled (its eventual
// response is dropped rather than resolving/rejecting the promise).
export interface RequestHandle<T = unknown> {
    promise: Promise<T>;
    cancel: () => void;
}

type EventHandler = (msg: any) => void;

export class ServerConnection {
    private ws: WebSocket | null = null;
    private nextReqId = 1;
    private pending = new Map<number, Pending>();
    private sendQueue: string[] = [];
    private eventHandlers = new Map<string, EventHandler[]>();
    private reconnectDelay = 500;            // ms, grows on repeated failures
    private readonly url: string;

    constructor(path = '/ws') {
        const proto = location.protocol === 'https:' ? 'wss' : 'ws';
        this.url = `${proto}://${location.host}${path}`;
        this._connect();
    }

    // Register a handler for a pushed event type (e.g. 'game/state').
    onEvent(type: string, handler: EventHandler): void {
        const list = this.eventHandlers.get(type) ?? [];
        list.push(handler);
        this.eventHandlers.set(type, list);
    }

    // Send a request and resolve with the response data (rejects on ok:false).
    request<T = unknown>(type: string, payload: Record<string, unknown> = {}): RequestHandle<T> {
        const reqId = this.nextReqId++;
        let settled = false;
        const promise = new Promise<T>((resolve, reject) => {
            this.pending.set(reqId, {
                resolve: d => { settled = true; resolve(d as T); },
                reject:  e => { settled = true; reject(e); },
            });
        });
        this._send({ kind: 'req', reqId, type, ...payload });
        const cancel = () => {
            // Drop the pending entry so the eventual response is ignored.
            if (!settled) this.pending.delete(reqId);
        };
        return { promise, cancel };
    }

    private _connect(): void {
        const ws = new WebSocket(this.url);
        this.ws = ws;
        ws.onopen = () => {
            this.reconnectDelay = 500;
            const queued = this.sendQueue;
            this.sendQueue = [];
            for (const m of queued) ws.send(m);
            this._emit('open', {});
        };
        ws.onmessage = (ev) => this._onMessage(ev.data);
        ws.onclose = () => {
            this.ws = null;
            // Fail all in-flight requests so callers don't hang.
            for (const [, p] of this.pending) p.reject(Object.assign(new Error('Connection closed'), { statusCode: 0 }));
            this.pending.clear();
            this._emit('close', {});
            setTimeout(() => this._connect(), this.reconnectDelay);
            this.reconnectDelay = Math.min(this.reconnectDelay * 2, 10000);
        };
        ws.onerror = () => { /* close handler drives reconnect */ };
    }

    private _onMessage(raw: string): void {
        let msg: any;
        try { msg = JSON.parse(raw); } catch { return; }
        if (msg.kind === 'res') {
            const p = this.pending.get(msg.reqId);
            if (!p) return;                       // cancelled or unknown
            this.pending.delete(msg.reqId);
            if (msg.ok) p.resolve(msg.data);
            else p.reject(Object.assign(new Error(msg.error ?? 'Request failed'), { statusCode: msg.statusCode }));
        } else if (msg.kind === 'event') {
            this._emit(msg.type, msg);
        }
    }

    private _emit(type: string, msg: any): void {
        for (const h of this.eventHandlers.get(type) ?? []) h(msg);
    }

    private _send(obj: unknown): void {
        const str = JSON.stringify(obj);
        if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(str);
        else this.sendQueue.push(str);            // flushed on (re)connect
    }
}
