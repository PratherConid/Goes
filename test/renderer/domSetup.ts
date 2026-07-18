// Shared test infra for mounting src/renderer.ts's Renderer class in a jsdom
// environment (not a test file itself).
//
// src/renderer.ts constructs a live `ServerConnection` (which opens a
// `WebSocket`) at MODULE IMPORT TIME, so setupDom() must run - and the
// resulting globals must already be in place - before any test file does
// `await import('../../src/renderer.ts')`. A static top-level import would
// run too early (hoisted before setupDom()'s body), so that import must be
// dynamic.
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { JSDOM } from 'jsdom';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const indexHtmlPath = path.join(__dirname, '..', '..', 'index.html');
const gamePresetsDir = path.join(__dirname, '..', '..', 'public', 'game_presets');

// src/renderer.ts's _loadPresets() calls fetch('/game_presets/<name>.json') at
// startup. There's no real HTTP server in this environment, and Node's global
// fetch rejects relative URLs outright, so route just that one path to the
// real files on disk instead - lets tests exercise real preset data rather
// than the fetch always failing (caught, but noisy - see _loadPresets()'s
// per-preset try/catch). Anything else rejects, mirroring a real fetch
// against an unhandled route.
function installFetchMock(): void {
    const mockFetch = (async (input: unknown) => {
        const url = typeof input === 'string' ? input : String(input);
        const match = url.match(/^\/game_presets\/([\w.-]+\.json)$/);
        if (!match) throw new TypeError(`domSetup's fetch mock has no route for: ${url}`);
        const body = await fs.promises.readFile(path.join(gamePresetsDir, match[1]), 'utf8');
        return { ok: true, status: 200, json: async () => JSON.parse(body) };
    }) as typeof fetch;
    Object.defineProperty(globalThis, 'fetch', { value: mockFetch, configurable: true, writable: true });
}

// Fixed size the clientWidth/clientHeight/getBoundingClientRect stub below
// reports for every element, since jsdom has no real layout engine. Tests
// that click the board can use this to compute where a given board node
// lands on screen (see boardLayout()'s formula in src/renderer.ts, which
// this satisfies for an odd x odd rectangular board: the center node sits
// exactly at (BOARD_PX/2, BOARD_PX/2)).
export const BOARD_PX = 600;

// Minimal fake WebSocket installed as globalThis.WebSocket. Implements only
// what src/serverConnection.ts actually uses. Tests drive a connection's
// lifecycle explicitly (readyState + on*/onmessage callbacks) rather than
// this class simulating a real handshake - there is no real server here.
export class FakeWebSocket {
    static readonly CONNECTING = 0;
    static readonly OPEN = 1;
    static readonly CLOSING = 2;
    static readonly CLOSED = 3;
    static instances: FakeWebSocket[] = [];
    static get last(): FakeWebSocket {
        const ws = FakeWebSocket.instances[FakeWebSocket.instances.length - 1];
        if (!ws) throw new Error('No FakeWebSocket has been constructed yet');
        return ws;
    }

    readyState: number = FakeWebSocket.CONNECTING;
    sent: string[] = [];
    onopen: (() => void) | null = null;
    onmessage: ((ev: { data: string }) => void) | null = null;
    onclose: (() => void) | null = null;
    onerror: (() => void) | null = null;

    constructor(public url: string) {
        FakeWebSocket.instances.push(this);
    }

    send(data: string): void { this.sent.push(data); }
    close(): void { this.readyState = FakeWebSocket.CLOSED; this.onclose?.(); }

    // Test helper: move to OPEN and fire the onopen handler, mirroring what a
    // real WebSocket does once its handshake completes.
    simulateOpen(): void {
        this.readyState = FakeWebSocket.OPEN;
        this.onopen?.();
    }

    // Test helper: deliver a server->client frame.
    simulateMessage(obj: unknown): void {
        this.onmessage?.({ data: JSON.stringify(obj) });
    }

    // Test helper: pop+parse the most recently sent request frame (as sent by
    // ServerConnection.request()), asserting its `type` matches if given.
    lastSentRequest(expectedType?: string): { reqId: number; type: string; [k: string]: unknown } {
        const raw = this.sent[this.sent.length - 1];
        if (!raw) throw new Error('No frame has been sent on this FakeWebSocket');
        const msg = JSON.parse(raw);
        if (expectedType && msg.type !== expectedType)
            throw new Error(`Expected last sent request type '${expectedType}', got '${msg.type}'`);
        return msg;
    }
}

// Builds a FRESH jsdom document and reassigns the DOM globals to it every
// call - safe (and expected) to call once per test, not just once per file.
// A Renderer's constructor binds directly to whatever `document` is current
// at construction time (document.getElementById(...) into private fields),
// with no teardown/removeEventListener anywhere; reusing one shared document
// across tests would accumulate every previous test's Renderer's event
// listeners on the same nodes. A fresh document per test sidesteps that
// entirely - old listeners stay bound to now-orphaned nodes nothing dispatches
// to anymore. (The WebSocket class reassignment below is idempotent - always
// the same FakeWebSocket class - so repeating it is harmless.)
export function setupDom(): void {
    const html = fs.readFileSync(indexHtmlPath, 'utf8');
    const dom = new JSDOM(html, { url: 'http://localhost/' });
    const w = dom.window as unknown as typeof globalThis;

    // Node defines its own built-in `navigator` as a getter-only global
    // property, so a plain Object.assign() throws on it - defineProperty
    // overrides it instead.
    for (const [key, value] of Object.entries({
        window: w,
        document: dom.window.document,
        location: dom.window.location,
        navigator: dom.window.navigator,
        HTMLElement: dom.window.HTMLElement,
        HTMLButtonElement: dom.window.HTMLButtonElement,
        Event: dom.window.Event,
        MouseEvent: dom.window.MouseEvent,
        KeyboardEvent: dom.window.KeyboardEvent,
    })) {
        Object.defineProperty(globalThis, key, { value, configurable: true, writable: true });
    }

    // jsdom doesn't implement these.
    globalThis.requestAnimationFrame = ((cb: FrameRequestCallback) => setTimeout(() => cb(Date.now()), 0)) as any;
    globalThis.cancelAnimationFrame = ((id: number) => clearTimeout(id)) as any;

    // jsdom has no real layout engine: clientWidth/clientHeight are always 0
    // and getComputedStyle can't resolve the page's viewport-relative calc()
    // padding, which would make _renderMainBoard's sizing math NaN. Pin both
    // to fixed, predictable values so board-layout math (used by tests that
    // click the board to place a stone) is computable. Installed on
    // Element.prototype (not HTMLElement.prototype) since the main board and
    // history-panel thumbnails are SVGElements, which extend Element directly,
    // not HTMLElement - HTMLElement-only stubs would silently not apply to them.
    Object.defineProperty(dom.window.Element.prototype, 'clientWidth', { configurable: true, get: () => BOARD_PX });
    Object.defineProperty(dom.window.Element.prototype, 'clientHeight', { configurable: true, get: () => BOARD_PX });
    (dom.window.Element.prototype as any).getBoundingClientRect = () =>
        ({ x: 0, y: 0, left: 0, top: 0, right: BOARD_PX, bottom: BOARD_PX, width: BOARD_PX, height: BOARD_PX, toJSON() {} });
    globalThis.getComputedStyle = (() => ({ paddingLeft: '0px', paddingRight: '0px', paddingTop: '0px', paddingBottom: '0px' })) as any;

    (globalThis as any).WebSocket = FakeWebSocket;
    installFetchMock();
}
