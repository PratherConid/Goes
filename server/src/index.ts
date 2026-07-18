import express from 'express';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { existsSync } from 'node:fs';
import { attachWebSocket } from './wsServer.js';
import { engineManager } from './engineManager.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Builds the Express+WebSocket app against `dataDir` (attachWebSocket is the
// one place that turns dataDir into live user/game-record/online-game state)
// and, iff `autoStart`, starts listening on `port` and wires up
// graceful-shutdown signal handlers. When `autoStart` is false, the returned
// http.Server is fully built but not yet listening - the caller decides
// if/when to server.listen() themselves (e.g. a test choosing its own port).
export async function startServer(port: number, dataDir: string, autoStart: boolean): Promise<http.Server> {
    // ── AI engine initialisation ──────────────────────────────────────────
    // No shared engine process is pre-spawned. Instead, EngineManager spawns
    // one goes_server process per game on demand. We only resolve the binary
    // path here.
    const projectRoot = path.resolve(__dirname, '../..');
    const isWin       = process.platform === 'win32';
    const exeName     = isWin ? 'goes_server.exe' : 'goes_server';
    const candidates  = [
        path.join(projectRoot, 'ai', 'build', 'Release', exeName),
        path.join(projectRoot, 'ai', 'build', exeName),
    ];
    const aiExe   = candidates.find(p => existsSync(p));
    const ckptDir = path.join(projectRoot, 'ai', 'checkpoints');

    if (aiExe) {
        engineManager.init(aiExe, ckptDir);
        console.log(`[ai] Engine binary found: ${aiExe}`);
    } else {
        console.warn('[ai] goes_server binary not found — AI features unavailable');
    }

    // ── Express app + WebSocket ─────────────────────────────────────────
    const app = express();
    app.use(express.static(path.resolve(__dirname, '../../dist')));
    const server = http.createServer(app);
    attachWebSocket(server, dataDir);

    if (!autoStart) return server;

    process.on('exit',    () => engineManager.releaseAll());
    process.on('SIGINT',  () => { engineManager.releaseAll(); process.exit(); });
    process.on('SIGTERM', () => { engineManager.releaseAll(); process.exit(); });

    return new Promise(resolve => server.listen(port, () => resolve(server)));
}

// Only parse argv and run when this file is executed directly
// (`tsx src/index.ts <port> <dataDir> <autoStart>`), not when a test imports
// this module purely to call startServer() itself.
if (process.argv[1] === fileURLToPath(import.meta.url)) {
    const [portArg, dataDirArg, autoStartArg] = process.argv.slice(2);
    if (portArg === undefined || dataDirArg === undefined || autoStartArg === undefined) {
        console.error('Usage: index.ts <port> <dataDir> <autoStart: true|false>');
        process.exit(1);
    }
    const port = Number(portArg);
    if (!Number.isInteger(port) || port < 0) {
        console.error(`Invalid port: ${portArg}`);
        process.exit(1);
    }
    const autoStart = autoStartArg === 'true';
    startServer(port, dataDirArg, autoStart).then(server => {
        if (!autoStart) { console.log('Server built (autoStart=false), not listening'); return; }
        const address = server.address();
        const actualPort = typeof address === 'object' && address ? address.port : port;
        console.log(`Server listening on port ${actualPort}`);
    });
}
