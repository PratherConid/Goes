import { BoardState } from '@shared/boardState.js';
import { PrescribedBoardMap, PrescribedBoardFns, PrescribedBoard } from '@shared/boardConfig.js';
import type { BoardConfig } from '@shared/boardConfig.js';
import { PlayerInfo, makeId } from '@shared/types.js';
import type { GameConfig, OnlineStateResponse } from '@shared/types.js';

interface ServerPendingGame {
    id: string;
    config: GameConfig;
    pendingNames: string[];                      // in join order, before slot assignment
    bc: BoardConfig;
    joinedPlayers: Map<object, number[]>;        // ws → positions; tracks connections before game starts
}

interface OnlineGame {
    id: string;
    config: GameConfig;
    players: Map<number, PlayerInfo>;            // key = slot; insertion order = join order
    boardState: BoardState;                      // always present (game has started)
}

const boardTypeToFn = new Map<string, (...args: number[]) => BoardConfig>();
for (const key of Object.keys(PrescribedBoardMap)) {
    const numKey = Number(key) as PrescribedBoard;
    const [, typeStr] = PrescribedBoardMap[numKey];
    boardTypeToFn.set(typeStr, PrescribedBoardFns[numKey]);
}


class OnlineGameManager {
    // In-memory store. All games are lost on server restart; no persistence.
    private pendingGames = new Map<string, ServerPendingGame>();
    private activeGames  = new Map<string, OnlineGame>();

    createGame(config: GameConfig, playerName: string): { id: string; position: number } {
        const fn = boardTypeToFn.get(config.boardType);
        if (!fn) throw Object.assign(new Error(`Unknown board type: ${config.boardType}`), { statusCode: 400 });
        let id: string;
        do { id = makeId(12); } while (this.pendingGames.has(id) || this.activeGames.has(id));
        this.pendingGames.set(id, {
            id, config,
            pendingNames: [playerName],
            bc: fn(...config.boardArgs),
            joinedPlayers: new Map(),
        });
        return { id, position: 0 };
    }

    joinGame(id: string, playerName: string): { position: number; config: GameConfig; status: 'waiting' | 'playing' } {
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
        const players = new Map<number, PlayerInfo>();
        for (let i = 0; i < pending.pendingNames.length; i++)
            players.set(slots[i], new PlayerInfo('client', pending.pendingNames[i]));
        const boardState = new BoardState(
            pending.config.numStones, pending.config.numPlayers,
            pending.config.turnStoneList, pending.config.stoneToPlayerMap,
            pending.config.forcedPassOnly, new Array(pending.bc.N).fill(0), pending.bc,
        );
        this.pendingGames.delete(pending.id);
        const game: OnlineGame = { id: pending.id, config: pending.config, players, boardState };
        this.activeGames.set(pending.id, game);
        // Transfer socket connections from pending joinedPlayers into active PlayerInfo
        const playerKeys = [...game.players.keys()];
        for (const [ws, positions] of pending.joinedPlayers)
            for (const pos of positions) {
                const slot = playerKeys[pos];
                const pi = slot !== undefined ? game.players.get(slot) : undefined;
                if (pi) pi.socket = ws as WebSocket | null;
            }
    }

    // Record that a connection (identified by ws) owns a position in a game.
    // For pending games: stored in joinedPlayers. For active games: stored in PlayerInfo.socket.
    // Also used for reconnects: re-binds a connection to its position after disconnect.
    acceptJoin(id: string, ws: unknown, position: number): void {
        const pending = this.pendingGames.get(id);
        if (pending) {
            const existing = pending.joinedPlayers.get(ws as object);
            if (existing) existing.push(position);
            else pending.joinedPlayers.set(ws as object, [position]);
            return;
        }
        const game = this.activeGames.get(id);
        if (!game) return;
        const slot = [...game.players.keys()][position];
        const pi = slot !== undefined ? game.players.get(slot) : undefined;
        if (pi) pi.socket = ws as WebSocket | null;
    }

    // Returns the positions (join indices) owned by ws in game id, or [] if none.
    getPositions(id: string, ws: unknown): number[] {
        const pending = this.pendingGames.get(id);
        if (pending) return pending.joinedPlayers.get(ws as object) ?? [];
        const game = this.activeGames.get(id);
        if (!game) return [];
        const positions: number[] = [];
        let i = 0;
        for (const pi of game.players.values()) {
            if (pi.socket === (ws as WebSocket | null)) positions.push(i);
            i++;
        }
        return positions;
    }

    // Returns all WebSocket connections currently joined to a game (for broadcasting).
    getSockets(id: string): unknown[] {
        const pending = this.pendingGames.get(id);
        if (pending) return [...pending.joinedPlayers.keys()];
        const game = this.activeGames.get(id);
        if (!game) return [];
        return [...game.players.values()].map(pi => pi.socket).filter(s => s !== null);
    }

    // Removes a closed connection from all games it was part of.
    removeConnection(ws: unknown): void {
        for (const pending of this.pendingGames.values())
            pending.joinedPlayers.delete(ws as object);
        for (const game of this.activeGames.values())
            for (const pi of game.players.values())
                if (pi.socket === (ws as WebSocket | null)) pi.socket = null;
    }

    getConfig(id: string): GameConfig {
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
            status: v.gameOver ? 'finished' : 'playing',
            numPlayersRequired: game.config.numPlayers,
            numJoined: game.players.size,
            players: [...game.players.entries()].map(([slot, pi]) => ({ name: pi.name, slot })),
            moves: game.boardState.lastMoves.map(m => m.pos),
            currentStone: v.gameOver ? null : v.nextPlayer,
            winners: v.winners,
            resignedPlayers: v.resignedPlayers,
        };
    }

    applyMove(id: string, positions: number[], moveIndex: number | null, clientIdx: number): OnlineStateResponse {
        const game = this.activeGames.get(id);
        if (!game) throw Object.assign(new Error('Game not found'), { statusCode: 404 });
        if (game.boardState.gameOver()) throw Object.assign(new Error('Game is not in progress'), { statusCode: 409 });
        if (game.boardState.lastMoves.length !== clientIdx) throw Object.assign(new Error('Move index mismatch'), { statusCode: 409 });
        const v = game.boardState.getView();
        const playerKeys = [...game.players.keys()];
        const currentSlot = game.config.stoneToPlayerMap[v.nextPlayer];
        if (!positions.some(pos => playerKeys[pos] === currentSlot))
            throw Object.assign(new Error('Not your turn'), { statusCode: 403 });
        if (!game.boardState.makeMove(moveIndex)) throw Object.assign(new Error('Illegal move'), { statusCode: 400 });
        game.boardState.advanceResigned();
        return this.getState(id);
    }

    resign(id: string, positions: number[]): OnlineStateResponse {
        const game = this.activeGames.get(id);
        if (!game) throw Object.assign(new Error('Game not found'), { statusCode: 404 });
        if (game.boardState.gameOver()) throw Object.assign(new Error('Game is not in progress'), { statusCode: 409 });
        const playerKeys = [...game.players.keys()];
        const bs = game.boardState;
        for (const position of positions) {
            const slot = playerKeys[position];
            if (slot === undefined) throw Object.assign(new Error('Invalid position'), { statusCode: 400 });
            bs.resign(slot);   // mark resigned, exclude from scoring, refresh winners
        }
        bs.advanceResigned();
        return this.getState(id);
    }
}

export const onlineGameManager = new OnlineGameManager();
