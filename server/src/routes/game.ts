import { Router } from 'express';

const router = Router();

router.post('/', (_req, res) => { res.sendStatus(501); });
router.get('/:id', (_req, res) => { res.sendStatus(501); });
router.post('/:id/move', (_req, res) => { res.sendStatus(501); });

export default router;
