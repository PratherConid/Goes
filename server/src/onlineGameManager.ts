import { BoardState } from '@shared/boardState.js';
import { PrescribedBoardMap, PrescribedBoardFns, PrescribedBoard } from '@shared/boardConfig.js';
import type { BoardConfig } from '@shared/boardConfig.js';
import { PlayerInfo, makeId } from '@shared/types.js';
import type { GameConfig, OnlineStateResponse } from '@shared/types.js';

interface ServerPendingGame {
    id: string;
    config: GameConfig;
    players: Map<number, PlayerInfo>;            // key = slot (randomly assigned on join)
    pendingSlots: number[];                      // unassigned slots, pre-shuffled
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
        // Pre-shuffle all slots; assign first to creator, remainder go to pendingSlots
        const slots = Array.from({ length: config.numPlayers }, (_, i) => i + 1);
        for (let i = slots.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [slots[i], slots[j]] = [slots[j], slots[i]];
        }
        const creatorSlot = slots[0];
        this.pendingGames.set(id, {
            id, config,
            players: new Map([[creatorSlot, new PlayerInfo('client', playerName)]]),
            pendingSlots: slots.slice(1),
        });
        return { id, position: creatorSlot };
    }

    joinGame(id: string, playerName: string): { position: number; config: GameConfig; status: 'waiting' | 'playing' } {
        const pending = this.pendingGames.get(id);
        if (!pending) {
            if (this.activeGames.has(id)) throw Object.assign(new Error('Game already started'), { statusCode: 409 });
            throw Object.assign(new Error('Game not found'), { statusCode: 404 });
        }
        if (pending.pendingSlots.length === 0)
            throw Object.assign(new Error('Game is full'), { statusCode: 409 });
        const slot = pending.pendingSlots.shift()!;
        pending.players.set(slot, new PlayerInfo('client', playerName));
        if (pending.pendingSlots.length === 0) {
            this._startGame(pending);
            return { position: slot, config: pending.config, status: 'playing' };
        }
        return { position: slot, config: pending.config, status: 'waiting' };
    }

    private _startGame(pending: ServerPendingGame) {
        const bc = boardTypeToFn.get(pending.config.boardType)!(...pending.config.boardArgs);
        const boardState = new BoardState(
            pending.config.numStones, pending.config.numPlayers,
            pending.config.turnStoneList, pending.config.stoneToPlayerMap,
            pending.config.forcedPassOnly, new Array(bc.N).fill(0), bc,
        );
        this.pendingGames.delete(pending.id);
        // Slots were already randomly assigned at join time; move players map directly.
        this.activeGames.set(pending.id, { id: pending.id, config: pending.config, players: pending.players, boardState });
    }

    // Record that a connection (identified by ws) owns a slot in a game.
    // Also used for reconnects: re-binds a connection to its slot after disconnect.
    acceptJoin(id: string, ws: unknown, slot: number): void {
        const game = this.pendingGames.get(id) ?? this.activeGames.get(id);
        if (!game) return;
        const pi = game.players.get(slot);
        if (pi) pi.socket = ws as WebSocket | null;
    }

    // Returns the slots owned by ws in game id, or [] if none.
    getPositions(id: string, ws: unknown): number[] {
        const game = this.pendingGames.get(id) ?? this.activeGames.get(id);
        if (!game) return [];
        return [...game.players.entries()]
            .filter(([, pi]) => pi.socket === (ws as WebSocket | null))
            .map(([slot]) => slot);
    }

    // Returns all WebSocket connections currently joined to a game (for broadcasting).
    getSockets(id: string): unknown[] {
        const game = this.pendingGames.get(id) ?? this.activeGames.get(id);
        if (!game) return [];
        return [...game.players.values()].map(pi => pi.socket).filter(s => s !== null);
    }

    // Removes a closed connection from all games it was part of.
    removeConnection(ws: unknown): void {
        for (const g of [...this.pendingGames.values(), ...this.activeGames.values()])
            for (const pi of g.players.values())
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
                players: new Array(pending.players.size).fill(null),
                moves: [],
                winners: [],
                resignedPlayers: [],
            };
        }
        const game = this.activeGames.get(id);
        if (!game) throw Object.assign(new Error('Game not found'), { statusCode: 404 });
        const v = game.boardState.getView();
        return {
            status: v.gameOver ? 'finished' : 'playing',
            players: [...game.players.entries()].map(([slot, pi]) => ({ name: pi.name, slot })),
            moves: game.boardState.lastMoves.map(m => m.pos),
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
        const currentSlot = game.config.stoneToPlayerMap[v.nextPlayer];
        if (!positions.includes(currentSlot))
            throw Object.assign(new Error('Not your turn'), { statusCode: 403 });
        if (!game.boardState.makeMove(moveIndex)) throw Object.assign(new Error('Illegal move'), { statusCode: 400 });
        game.boardState.advanceResigned();
        return this.getState(id);
    }

    resign(id: string, positions: number[]): OnlineStateResponse {
        const game = this.activeGames.get(id);
        if (!game) throw Object.assign(new Error('Game not found'), { statusCode: 404 });
        if (game.boardState.gameOver()) throw Object.assign(new Error('Game is not in progress'), { statusCode: 409 });
        const bs = game.boardState;
        for (const slot of positions) {
            if (!game.players.has(slot)) throw Object.assign(new Error('Invalid position'), { statusCode: 400 });
            bs.resign(slot);
        }
        bs.advanceResigned();
        return this.getState(id);
    }
}

export const onlineGameManager = new OnlineGameManager();
