import { BoardState } from '@shared/boardState.js';
import type { BoardConfig } from '@shared/boardConfig.js';

export class GameManager {
    private games = new Map<string, BoardState>();

    createGame(id: string, numPlayers: number, turnStoneList: number[], forcedPassOnly: boolean, board: number[], config: BoardConfig): BoardState {
        throw new Error('Not implemented');
    }

    getGame(id: string): BoardState | undefined {
        return this.games.get(id);
    }
}

export const gameManager = new GameManager();
