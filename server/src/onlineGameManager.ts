import { BoardState } from '@shared/boardState.js';
import { PrescribedBoardMap, PrescribedBoardFns, PrescribedBoard } from '@shared/boardConfig.js';
import type { BoardConfig } from '@shared/boardConfig.js';
import type { OnlineGameConfig, OnlineStateResponse } from '@shared/types.js';

interface ServerPendingGame {
    id: string;
    config: OnlineGameConfig;
    pendingNames: string[];                      // in join order, before slot assignment
    bc: BoardConfig;
}

interface OnlineGame {
    id: string;
    config: OnlineGameConfig;
    players: { name: string; slot: number }[];
    boardState: BoardState;                      // always present (game has started)
    // Not purely derivable from boardState: when resignation leaves ≤1 player the game ends
    // immediately without a game-over move being recorded in boardState.
    status: 'playing' | 'finished';
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
    // In-memory store. All games are lost on server restart; no persistence.
    private pendingGames = new Map<string, ServerPendingGame>();
    private activeGames  = new Map<string, OnlineGame>();

    createGame(config: OnlineGameConfig, playerName: string): { id: string; position: number } {
        const fn = boardTypeToFn.get(config.boardType);
        if (!fn) throw Object.assign(new Error(`Unknown board type: ${config.boardType}`), { statusCode: 400 });
        let id: string;
        do { id = makeId(12); } while (this.pendingGames.has(id) || this.activeGames.has(id));
        this.pendingGames.set(id, {
            id, config,
            pendingNames: [playerName],
            bc: fn(...config.boardArgs),
        });
        return { id, position: 0 };
    }

    joinGame(id: string, playerName: string): { position: number; config: OnlineGameConfig; status: 'waiting' | 'playing' } {
        const pending = this.pendingGames.get(id);
        if (!pending) {
            if (this.activeGames.has(id)) throw Object.assign(new Error('Game already started'), { statusCode: 409 });
            throw Object.assign(new Error('Game not found'), { statusCode: 404 });
        }
        if (pending.pendingNames.length >= pending.config.numPlayers)
            throw Object.assign(new Error('Game is full'), { statusCode: 409 });
        const position = pending.pendingNames.length;
        pending.pendingNames.push(playerName);
        if (pending.pendingNames.length === pending.config.numPlayers) {
            this._startGame(pending);
            return { position, config: pending.config, status: 'playing' };
        }
        return { position, config: pending.config, status: 'waiting' };
    }

    private _startGame(pending: ServerPendingGame) {
        // Randomly assign player slots (Fisher-Yates shuffle of [1..numPlayers])
        const slots = Array.from({ length: pending.config.numPlayers }, (_, i) => i + 1);
        for (let i = slots.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [slots[i], slots[j]] = [slots[j], slots[i]];
        }
        const players = pending.pendingNames.map((name, i) => ({ name, slot: slots[i] }));
        const boardState = new BoardState(
            pending.config.numStones, pending.config.numPlayers,
            pending.config.turnStoneList, pending.config.stoneToPlayerMap,
            pending.config.forcedPassOnly, new Array(pending.bc.N).fill(0), pending.bc,
        );
        this.pendingGames.delete(pending.id);
        this.activeGames.set(pending.id, { id: pending.id, config: pending.config, players, boardState, status: 'playing' });
    }

    getConfig(id: string): OnlineGameConfig {
        const game = this.pendingGames.get(id) ?? this.activeGames.get(id);
        if (!game) throw Object.assign(new Error('Game not found'), { statusCode: 404 });
        return game.config;
    }

    getState(id: string): OnlineStateResponse {
        const pending = this.pendingGames.get(id);
        if (pending) {
            return {
                status: 'waiting',
                numPlayersRequired: pending.config.numPlayers,
                numJoined: pending.pendingNames.length,
                players: pending.pendingNames.map(() => null),
                moves: [],
                currentStone: null,
                winners: [],
                resignedPlayers: [],
            };
        }
        const game = this.activeGames.get(id);
        if (!game) throw Object.assign(new Error('Game not found'), { statusCode: 404 });
        const v = game.boardState.getView();
        return {
            status: game.status,
            numPlayersRequired: game.config.numPlayers,
            numJoined: game.players.length,
            players: game.players,
            moves: game.boardState.lastMoves.map(m => m.pos),
            currentStone: game.status === 'playing' ? v.nextPlayer : null,
            winners: v.winners,
            resignedPlayers: v.resignedPlayers,
        };
    }

    applyMove(id: string, position: number, moveIndex: number | null, clientIdx: number): OnlineStateResponse {
        const game = this.activeGames.get(id);
        if (!game) throw Object.assign(new Error('Game not found'), { statusCode: 404 });
        if (game.status !== 'playing') throw Object.assign(new Error('Game is not in progress'), { statusCode: 409 });
        if (game.boardState.lastMoves.length !== clientIdx) throw Object.assign(new Error('Move index mismatch'), { statusCode: 409 });
        const v = game.boardState.getView();
        if (game.config.stoneToPlayerMap[v.nextPlayer] !== game.players[position]?.slot)
            throw Object.assign(new Error('Not your turn'), { statusCode: 403 });
        if (!game.boardState.makeMove(moveIndex)) throw Object.assign(new Error('Illegal move'), { statusCode: 400 });
        game.boardState.advanceResigned();
        if (game.boardState.getView().gameOver) game.status = 'finished';
        return this.getState(id);
    }

    resign(id: string, position: number): OnlineStateResponse {
        const game = this.activeGames.get(id);
        if (!game) throw Object.assign(new Error('Game not found'), { statusCode: 404 });
        if (game.status !== 'playing') throw Object.assign(new Error('Game is not in progress'), { statusCode: 409 });
        const player = game.players[position];
        if (!player) throw Object.assign(new Error('Invalid position'), { statusCode: 400 });
        const bs = game.boardState;
        bs.resign(player.slot);   // mark resigned, exclude from scoring, refresh winners
        // End immediately if at most one player is left; otherwise auto-pass any resigned
        // player now on turn (which may itself end the game by a final pass).
        if (game.config.numPlayers - bs.resignedPlayers.length <= 1) game.status = 'finished';
        else {
            bs.advanceResigned();
            if (bs.getView().gameOver) game.status = 'finished';
        }
        return this.getState(id);
    }
}

export const onlineGameManager = new OnlineGameManager();
