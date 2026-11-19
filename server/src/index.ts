import express from 'express';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { existsSync, readFileSync } from 'node:fs';
import { attachWebSocket } from './wsServer.js';
import { engineManager } from './engineManager.js';
import { loadFont, BUNDLED_FONT_NAMES } from '@shared/glyphRaster.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Loads every bundled font (public/fonts/<name>.ttf) and registers it with shared/glyphRaster.ts's
// own loadFont() - the server evaluates cleg programs itself, so strB(...) has to work here just as
// much as it does in the renderer.
function loadBundledFonts(publicDir: string): void {
    for (const name of BUNDLED_FONT_NAMES) {
        const buf = readFileSync(path.join(publicDir, 'fonts', `${name}.ttf`));
        // Buffer.buffer is the underlying ArrayBuffer, which (for a small file) may be a slice of a
        // larger pooled allocation shared with unrelated data - byteOffset/byteLength narrow it back
        // down to exactly this file's own bytes.
        loadFont(name, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer);
    }
}

// Builds the Express+WebSocket app against `dataDir` and, iff `autoStart`, starts listening on
// `port` and wires up graceful-shutdown signal handlers. When `autoStart` is false, the returned
// http.Server is fully built but not yet listening - the caller decides if/when to server.listen()
// themselves, e.g. to choose its own port. `checkpointDir` is the directory the AI engine searches
// for checkpoints (one subdirectory per training run, as goes_train writes them).
export async function startServer(port: number, dataDir: string, autoStart: boolean,
                                  checkpointDir: string): Promise<http.Server> {
    // ── AI engine initialisation ──────────────────────────────────────────
    // Only the binary path is resolved here; EngineManager does the spawning.
    const projectRoot = path.resolve(__dirname, '../..');
    loadBundledFonts(path.join(projectRoot, 'public'));

    const isWin       = process.platform === 'win32';
    const exeName     = isWin ? 'goes_server.exe' : 'goes_server';
    const candidates  = [
        path.join(projectRoot, 'ai', 'build', 'Release', exeName),
        path.join(projectRoot, 'ai', 'build', exeName),
    ];
    const aiExe   = candidates.find(p => existsSync(p));

    if (aiExe) {
        engineManager.init(aiExe, checkpointDir);
        console.log(`[ai] Engine binary found: ${aiExe}`);
        console.log(`[ai] Checkpoint directory: ${checkpointDir}`);
        if (!existsSync(checkpointDir)) console.warn(`[ai] Checkpoint directory does not exist: ${checkpointDir}`);
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
// (`tsx src/index.ts <port> <dataDir> <autoStart> <checkpointDir>`), not when it's imported for
// startServer(). A relative checkpointDir resolves against the current working directory, like dataDir.
if (process.argv[1] === fileURLToPath(import.meta.url)) {
    const [portArg, dataDirArg, autoStartArg, checkpointDirArg] = process.argv.slice(2);
    if (portArg === undefined || dataDirArg === undefined || autoStartArg === undefined || checkpointDirArg === undefined) {
        console.error('Usage: index.ts <port> <dataDir> <autoStart: true|false> <checkpointDir>');
        process.exit(1);
    }
    const port = Number(portArg);
    if (!Number.isInteger(port) || port < 0) {
        console.error(`Invalid port: ${portArg}`);
        process.exit(1);
    }
    const autoStart = autoStartArg === 'true';
    startServer(port, dataDirArg, autoStart, path.resolve(checkpointDirArg)).then(server => {
        if (!autoStart) { console.log('Server built (autoStart=false), not listening'); return; }
        const address = server.address();
        const actualPort = typeof address === 'object' && address ? address.port : port;
        console.log(`Server listening on port ${actualPort}`);
    });
}
