// Thin HTTP proxy to the C++ inference engine (goes_server).
// The client↔main-server hop is WebSocket, but the main-server↔engine hop stays HTTP.

const AI_PORT = process.env.AI_PORT ?? '8765';
const AI_URL  = `http://localhost:${AI_PORT}`;

// Forward a move request to the engine. Throws { statusCode } on failure so the
// WS layer can turn it into an error response.
export async function aiMove(body: unknown): Promise<unknown> {
    let resp: Response;
    try {
        resp = await fetch(`${AI_URL}/move`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
    } catch {
        throw Object.assign(new Error('AI engine unavailable'), { statusCode: 503 });
    }
    if (!resp.ok) throw Object.assign(new Error('AI engine error'), { statusCode: resp.status });
    return resp.json();
}

// Engine health check. Returns { status: 'unavailable' } instead of throwing when
// the engine is unreachable, so the client can show the engine state without error.
export async function aiHealth(): Promise<unknown> {
    try {
        const resp = await fetch(`${AI_URL}/health`);
        return await resp.json();
    } catch {
        return { status: 'unavailable' };
    }
}
