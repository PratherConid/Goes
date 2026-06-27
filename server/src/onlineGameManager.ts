import { BoardState } from '@shared/boardState.js';
import { PrescribedBoardMap, PrescribedBoardFns, PrescribedBoard } from '@shared/boardConfig.js';
import type { BoardConfig } from '@shared/boardConfig.js';
import type { OnlineGameConfig, OnlineStateResponse } from '@shared/types.js';

interface OnlineGame {
    id: string;
    config: OnlineGameConfig;
    pendingNames: string[];                      // in join order, before slot assignment
    players: { name: string; slot: number }[];   // populated when game starts
    moves: (number | null)[];
    boardState: BoardState | null;               // null until game starts
    bc: BoardConfig;
    status: 'waiting' | 'playing' | 'finished';
    // Resignations are tracked on boardState.resignedPlayers (a resigned player auto-passes
    // and is excluded from scoring); the game also ends if ≤1 player remains.
}

const boardTypeToFn = new Map<string, (...args: number[]) => BoardConfig>();
for (const key of Object.keys(PrescribedBoardMap)) {
    const numKey = Number(key) as PrescribedBoard;
    const [, typeStr] = PrescribedBoardMap[numKey];
    boardTypeToFn.set(typeStr, PrescribedBoardFns[numKey]);
}

function makeId(len: number): string {
    const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
    let id = '';
    for (let i = 0; i < len; i++) id += chars[Math.floor(Math.random() * chars.length)];
    return id;
}

class OnlineGameManager {
    // In-memory store mapping game ID to game state
    // All games are lost on server restart; no persistence.
    private games = new Map<string, OnlineGame>();

    createGame(config: OnlineGameConfig, playerName: string): { id: string; position: number } {
        const fn = boardTypeToFn.get(config.boardType);
        if (!fn) throw Object.assign(new Error(`Unknown board type: ${config.boardType}`), { statusCode: 400 });
        let id: string;
        do { id = makeId(6); } while (this.games.has(id));
        this.games.set(id, {
            id, config,
            pendingNames: [playerName],
            players: [],
            moves: [],
            boardState: null,
            bc: fn(...config.boardArgs),
            status: 'waiting',
        });
        return { id, position: 0 };
    }

    joinGame(id: string, playerName: string): { position: number; config: OnlineGameConfig; status: OnlineGame['status'] } {
        const game = this.games.get(id);
        if (!game) throw Object.assign(new Error('Game not found'), { statusCode: 404 });
        if (game.status !== 'waiting') throw Object.assign(new Error('Game already started'), { statusCode: 409 });
        if (game.pendingNames.length >= game.config.numPlayers)
            throw Object.assign(new Error('Game is full'), { statusCode: 409 });
        const position = game.pendingNames.length;
        game.pendingNames.push(playerName);
        if (game.pendingNames.length === game.config.numPlayers) this._startGame(game);
        return { position, config: game.config, status: game.status };
    }

    private _startGame(game: OnlineGame) {
        // Randomly assign player slots (Fisher-Yates shuffle of [1..numPlayers])
        const slots = Array.from({ length: game.config.numPlayers }, (_, i) => i + 1);
        for (let i = slots.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [slots[i], slots[j]] = [slots[j], slots[i]];
        }
        game.players = game.pendingNames.map((name, i) => ({ name, slot: slots[i] }));
        game.boardState = new BoardState(
            game.config.numStones, game.config.numPlayers,
            game.config.turnStoneList, game.config.stoneToPlayerMap,
            game.config.forcedPassOnly, new Array(game.bc.N).fill(0), game.bc,
        );
        game.status = 'playing';
    }

    getConfig(id: string): OnlineGameConfig {
        const game = this.games.get(id);
        if (!game) throw Object.assign(new Error('Game not found'), { statusCode: 404 });
        return game.config;
    }

    getState(id: string): OnlineStateResponse {
        const game = this.games.get(id);
        if (!game) throw Object.assign(new Error('Game not found'), { statusCode: 404 });
        let currentStone: number | null = null;
        let winners: number[] = [];
        let resignedPlayers: number[] = [];
        if (game.boardState) {
            const v = game.boardState.getView();
            winners = v.winners;
            resignedPlayers = v.resignedPlayers;
            currentStone = game.status === 'playing' ? v.nextPlayer : null;
        }
        return {
            status: game.status,
            numPlayersRequired: game.config.numPlayers,
            numJoined: game.pendingNames.length,
            players: game.pendingNames.map((_, i) => game.players[i] ?? null),
            moves: game.moves,
            currentStone,
            winners,
            resignedPlayers,
        };
    }

    applyMove(id: string, position: number, moveIndex: number | null, clientIdx: number): OnlineStateResponse {
        const game = this.games.get(id);
        if (!game) throw Object.assign(new Error('Game not found'), { statusCode: 404 });
        if (game.status !== 'playing') throw Object.assign(new Error('Game is not in progress'), { statusCode: 409 });
        if (game.moves.length !== clientIdx) throw Object.assign(new Error('Move index mismatch'), { statusCode: 409 });
        const v = game.boardState!.getView();
        if (game.config.stoneToPlayerMap[v.nextPlayer] !== game.players[position]?.slot)
            throw Object.assign(new Error('Not your turn'), { statusCode: 403 });
        if (!game.boardState!.makeMove(moveIndex)) throw Object.assign(new Error('Illegal move'), { statusCode: 400 });
        game.moves.push(moveIndex);
        this._advancePastResigned(game);   // auto-pass for any resigned player now on turn
        return this.getState(id);
    }

    resign(id: string, position: number): OnlineStateResponse {
        const game = this.games.get(id);
        if (!game) throw Object.assign(new Error('Game not found'), { statusCode: 404 });
        if (game.status !== 'playing') throw Object.assign(new Error('Game is not in progress'), { statusCode: 409 });
        const player = game.players[position];
        if (!player) throw Object.assign(new Error('Invalid position'), { statusCode: 400 });
        const bs = game.boardState!;
        bs.resign(player.slot);   // mark resigned, exclude from scoring, refresh winners
        // End immediately if at most one player is left; otherwise auto-pass any resigned
        // player now on turn (which may itself end the game by a final pass).
        if (game.config.numPlayers - bs.resignedPlayers.length <= 1) game.status = 'finished';
        else this._advancePastResigned(game);
        return this.getState(id);
    }

    // Auto-pass on behalf of any resigned player whose turn it is, until a non-resigned
    // player is to move or the game ends. Marks the game finished when it ends.
    private _advancePastResigned(game: OnlineGame) {
        const bs = game.boardState!;
        while (!bs.getView().gameOver) {
            const player = game.config.stoneToPlayerMap[bs.nextPlayer];
            if (player === undefined || !bs.resignedPlayers.includes(player)) break;
            if (!bs.makeMove(null)) break;   // resigned pass always succeeds; guard anyway
            game.moves.push(null);
        }
        if (bs.getView().gameOver) game.status = 'finished';
    }
}

export const onlineGameManager = new OnlineGameManager();
