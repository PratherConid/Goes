// Manages one goes_server process per game (online or local).
// Online games are keyed by game ID; local games by "local:<wsId>:<gameId>".

import { spawn } from 'node:child_process';
import type { ChildProcess } from 'node:child_process';

interface ManagedEngine {
    proc: ChildProcess;
    url: string;
}

class EngineManager {
    private engines = new Map<string, ManagedEngine>();
    private _aiExe  = '';
    private _ckptDir = '';
    private _nextPort = 8765;

    // Call once from index.ts after the binary path is resolved.
    init(exePath: string, ckptDir: string): void {
        this._aiExe   = exePath;
        this._ckptDir = ckptDir;
    }

    get ready(): boolean { return this._aiExe !== ''; }

    // Return the URL for the engine process assigned to `key`, spawning one if needed.
    async getOrCreate(key: string): Promise<string> {
        const existing = this.engines.get(key);
        if (existing) return existing.url;
        if (!this._aiExe)
            throw Object.assign(new Error('Engine binary not found'), { statusCode: 503 });

        const port = this._nextPort++;
        const url  = `http://localhost:${port}`;
        const proc = spawn(
            this._aiExe,
            ['--checkpoint-dir', this._ckptDir, '--port', String(port)],
            { stdio: 'inherit' },
        );
        proc.on('error', err => console.error(`[engine:${key}] error:`, err.message));
        proc.on('exit', code => {
            console.log(`[engine:${key}] exited with code ${code}`);
            this.engines.delete(key);
        });

        // Poll /health until the process is ready (up to 5 s).
        for (let i = 0; i < 50; i++) {
            await new Promise(r => setTimeout(r, 100));
            try {
                const resp = await fetch(`${url}/health`);
                if (resp.ok) { this.engines.set(key, { proc, url }); return url; }
            } catch {}
        }
        proc.kill();
        throw Object.assign(
            new Error(`Engine process for "${key}" failed to start`),
            { statusCode: 503 },
        );
    }

    release(key: string): void {
        const e = this.engines.get(key);
        if (!e) return;
        e.proc.kill();
        this.engines.delete(key);
    }

    // Kill all engines whose key starts with `prefix` (used on ws disconnect).
    releasePrefix(prefix: string): void {
        for (const key of [...this.engines.keys()])
            if (key.startsWith(prefix)) this.release(key);
    }

    releaseAll(): void {
        for (const key of [...this.engines.keys()]) this.release(key);
    }
}

export const engineManager = new EngineManager();

// Forward a move request to the engine at `url`.
export async function aiMove(url: string, body: unknown): Promise<unknown> {
    let resp: Response;
    try {
        resp = await fetch(`${url}/move`, {
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
