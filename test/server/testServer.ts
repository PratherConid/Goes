// Shared test infra for spinning up real server instances (not a test file itself).
//
// Two flavors:
//   startTestServer()         - fast, in-process (dynamic import). server/src has no
//                                module-level singletons - attachWebSocket(server, dataDir)
//                                loads fresh user/game-record/online-game state from
//                                dataDir on every call - so this is safe to call more
//                                than once per process (e.g. against the same dataDir,
//                                to simulate a restart) without state leaking between calls.
//   startTestServerProcess()  - spawns a genuine child process, for tests that want to
//                                exercise the real CLI entrypoint (argv parsing) or a
//                                true OS-level process restart rather than an in-process
//                                re-import.
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { spawn } from 'node:child_process';
import type { AddressInfo } from 'node:net';
import { WebSocket } from 'ws';

const __dirname   = path.dirname(fileURLToPath(import.meta.url));
const repoRoot     = path.resolve(__dirname, '../..');
const serverEntry  = path.join(repoRoot, 'server', 'src', 'index.ts');

export interface TestServer {
    url: string;
    dataDir: string;
    close(): Promise<void>;
}

// Starts a real server in-process on an ephemeral port, pointed at a fresh temp
// data dir (or `dataDir`, to reuse one - e.g. after a startTestServerProcess()
// restart in the same test).
export async function startTestServer(dataDir?: string): Promise<TestServer> {
    const dir = dataDir ?? fs.mkdtempSync(path.join(os.tmpdir(), 'goes-test-'));
    const mod = await import(pathToFileURL(serverEntry).href) as {
        startServer(port: number, dataDir: string, autoStart: boolean): Promise<import('node:http').Server>;
    };
    const server = await mod.startServer(0, dir, true);
    const address = server.address() as AddressInfo;
    return {
        url: `ws://localhost:${address.port}/ws`,
        dataDir: dir,
        // closeAllConnections() first: http.Server.close() alone waits for every
        // open socket (including upgraded WS connections) to end on its own, so a
        // test that failed before closing its client would otherwise hang teardown
        // for the whole file.
        close: () => new Promise<void>(resolve => { server.closeAllConnections(); server.close(() => resolve()); }),
    };
}

// Spawns server/src/index.ts as a real child process (via tsx, resolved from
// server/node_modules) pointed at `dataDir`, waits for it to report the
// assigned ephemeral port, and returns a handle to stop it. Use this (not
// startTestServer) when a test needs to prove data survived an actual restart.
export function startTestServerProcess(dataDir: string): Promise<{ url: string; stop(): Promise<void> }> {
    return new Promise((resolve, reject) => {
        const child = spawn(process.execPath, ['--import', 'tsx', serverEntry, '0', dataDir, 'true'], {
            cwd: path.join(repoRoot, 'server'),
            env: process.env,
            stdio: ['ignore', 'pipe', 'pipe'],
        });
        let buf = '';
        const onData = (chunk: Buffer) => {
            buf += chunk.toString();
            const m = buf.match(/Server listening on port (\d+)/);
            if (!m) return;
            child.stdout!.off('data', onData);
            resolve({
                url: `ws://localhost:${m[1]}/ws`,
                stop: () => new Promise<void>(res => { child.once('exit', () => res()); child.kill(); }),
            });
        };
        child.stdout!.on('data', onData);
        child.stderr!.on('data', d => process.stderr.write(d));
        child.on('error', reject);
        child.on('exit', code => {
            if (code !== 0 && code !== null) reject(new Error(`server process exited early with code ${code}`));
        });
    });
}

export interface TestClient {
    req<T = unknown>(type: string, payload?: Record<string, unknown>): Promise<T>;
    onEvent(type: string, cb: (msg: any) => void): void;
    // Resolves once the server has actually processed the close (its 'close'
    // handler has run and cleared userToWs) - awaiting this before a next
    // LOGIN as the same user avoids a race against "already logged in
    // elsewhere", since close() itself only starts the close handshake.
    close(): Promise<void>;
}

// Resolves once `client` has received `n` events of `type`. A broadcast to
// two different connections is two independent socket deliveries with no
// ordering guarantee relative to either connection's own in-flight request -
// awaiting the *sender's* req() resolving is not enough to know a broadcast
// has reached some *other* connection yet; wait on this instead.
export function waitForEvents(client: TestClient, type: string, n: number): Promise<any[]> {
    return new Promise(resolve => {
        const collected: any[] = [];
        client.onEvent(type, m => { collected.push(m); if (collected.length === n) resolve(collected); });
    });
}

// Opens a real WS connection and returns a small client mirroring
// src/serverConnection.ts's request/response + event protocol handling
// (reqId correlation, ok/error dispatch), simplified for test use - no
// reconnect or send-queueing, since tests connect after the server is up.
export function connect(url: string): Promise<TestClient> {
    return new Promise((resolve, reject) => {
        const ws = new WebSocket(url);
        let nextReqId = 1;
        const pending = new Map<number, { resolve: (d: unknown) => void; reject: (e: Error) => void }>();
        const eventHandlers = new Map<string, ((msg: any) => void)[]>();

        ws.on('message', (raw: Buffer) => {
            const msg = JSON.parse(raw.toString());
            if (msg.kind === 'res') {
                const p = pending.get(msg.reqId);
                if (!p) return;
                pending.delete(msg.reqId);
                if (msg.ok) p.resolve(msg.data);
                else p.reject(Object.assign(new Error(msg.error ?? 'Request failed'), { statusCode: msg.statusCode }));
            } else if (msg.kind === 'event') {
                for (const cb of eventHandlers.get(msg.type) ?? []) cb(msg);
            }
        });
        ws.on('error', reject);
        ws.once('open', () => resolve({
            req: (type, payload = {}) => new Promise((res, rej) => {
                const reqId = nextReqId++;
                pending.set(reqId, { resolve: res as (d: unknown) => void, reject: rej });
                ws.send(JSON.stringify({ kind: 'req', reqId, type, ...payload }));
            }),
            onEvent: (type, cb) => {
                const list = eventHandlers.get(type) ?? [];
                list.push(cb);
                eventHandlers.set(type, list);
            },
            close: () => new Promise<void>(res => { ws.once('close', () => res()); ws.close(); }),
        }));
    });
}
