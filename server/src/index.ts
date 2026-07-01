import express from 'express';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { existsSync } from 'node:fs';
import { attachWebSocket } from './wsServer.js';
import { engineManager } from './engineManager.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = process.env.PORT ?? 3000;

// ── AI engine initialisation ──────────────────────────────────────────────────
// No shared engine process is pre-spawned. Instead, EngineManager spawns one
// goes_server process per game on demand. We only resolve the binary path here.

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

process.on('exit',    () => engineManager.releaseAll());
process.on('SIGINT',  () => { engineManager.releaseAll(); process.exit(); });
process.on('SIGTERM', () => { engineManager.releaseAll(); process.exit(); });

// ── Express app + WebSocket ────────────────────────────────────────────────────

app.use(express.static(path.resolve(__dirname, '../../dist')));

const server = http.createServer(app);
attachWebSocket(server);

server.listen(PORT, () => console.log(`Server listening on port ${PORT}`));
