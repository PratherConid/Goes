import { Router } from 'express';
import { onlineGameManager } from '../onlineGameManager.js';
import type { OnlineGameConfig } from '../onlineGameManager.js';

const router = Router();

router.post('/', (req, res) => {
    try {
        const { playerName, ...config } = req.body as OnlineGameConfig & { playerName: string };
        res.json(onlineGameManager.createGame(config, playerName ?? 'Anonymous'));
    } catch (e: any) {
        res.status(e.statusCode ?? 500).json({ error: e.message });
    }
});

router.post('/:id/join', (req, res) => {
    try {
        const { playerName } = req.body as { playerName: string };
        res.json(onlineGameManager.joinGame(req.params['id']!, playerName ?? 'Anonymous'));
    } catch (e: any) {
        res.status(e.statusCode ?? 500).json({ error: e.message });
    }
});

router.get('/:id/state', (req, res) => {
    try {
        res.json(onlineGameManager.getState(req.params['id']!));
    } catch (e: any) {
        res.status(e.statusCode ?? 500).json({ error: e.message });
    }
});

router.post('/:id/move', (req, res) => {
    try {
        const { position, moveIndex, clientIdx } = req.body as {
            position: number;
            moveIndex: number | null;
            clientIdx: number;
        };
        res.json(onlineGameManager.applyMove(req.params['id']!, position, moveIndex ?? null, clientIdx));
    } catch (e: any) {
        res.status(e.statusCode ?? 500).json({ error: e.message });
    }
});

export default router;
