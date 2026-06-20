import express from 'express';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import gameRouter from './routes/game.js';
import aiRouter from './routes/ai.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = process.env.PORT ?? 3000;

// ── AI engine startup ─────────────────────────────────────────────────────────

const projectRoot  = path.resolve(__dirname, '../..');
const isWin        = process.platform === 'win32';
const exeName      = isWin ? 'goes_server.exe' : 'goes_server';
// cmake puts the binary in Release/ on Windows (MSVC) and directly in build/ on Linux
const candidates   = [
    path.join(projectRoot, 'ai', 'build', 'Release', exeName),
    path.join(projectRoot, 'ai', 'build', exeName),
];
const aiExe        = candidates.find(p => existsSync(p)) ?? candidates[0];
const ckptDir      = path.join(projectRoot, 'ai', 'checkpoints');

const aiProc = (() => {
    try {
        const proc = spawn(aiExe, ['--checkpoint-dir', ckptDir], { stdio: 'inherit' });
        proc.on('error', err => console.error('[ai] Failed to start goes_server:', err.message));
        proc.on('exit',  code => console.log(`[ai] goes_server exited with code ${code}`));
        console.log(`[ai] Spawned goes_server (pid ${proc.pid}): ${aiExe}`);
        return proc;
    } catch (err) {
        console.error('[ai] Could not spawn goes_server:', err);
        return null;
    }
})();

// Clean up the AI process on server exit
process.on('exit',    () => aiProc?.kill());
process.on('SIGINT',  () => { aiProc?.kill(); process.exit(); });
process.on('SIGTERM', () => { aiProc?.kill(); process.exit(); });

// ── Express app ───────────────────────────────────────────────────────────────

app.use(express.json());
app.use('/api/game', gameRouter);
app.use('/api/ai',   aiRouter);
app.use(express.static(path.resolve(__dirname, '../../dist')));

app.listen(PORT, () => console.log(`Server listening on port ${PORT}`));
