import { Router } from 'express';

const AI_PORT = process.env.AI_PORT ?? '8765';
const AI_URL  = `http://localhost:${AI_PORT}`;

const router = Router();

router.post('/move', async (req, res) => {
    try {
        const resp = await fetch(`${AI_URL}/move`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(req.body),
        });
        if (!resp.ok) { res.status(resp.status).json({ error: 'AI engine error' }); return; }
        res.json(await resp.json());
    } catch {
        res.status(503).json({ error: 'AI engine unavailable' });
    }
});

router.get('/health', async (_req, res) => {
    try {
        const resp = await fetch(`${AI_URL}/health`);
        res.json(await resp.json());
    } catch {
        res.status(503).json({ status: 'unavailable' });
    }
});

export default router;
