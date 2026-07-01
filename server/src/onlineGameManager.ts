import { BoardState } from '@shared/boardState.js';
import { PrescribedBoardMap, PrescribedBoardFns, PrescribedBoard } from '@shared/boardConfig.js';
import type { BoardConfig } from '@shared/boardConfig.js';
import { PlayerInfo, GameConfig, makeId } from '@shared/types.js';
import type { OnlineStateResponse, PendingGame } from '@shared/types.js';

// Server-side pending game: extends PendingGame with a set of all connected
// websockets (creator + joiners). Used by getSockets for broadcasting.
interface ServerPendingGame extends PendingGame {
    observers: Set<unknown>;
}

interface OnlineGame {
    id: string;
    config: GameConfig;
    boardState: BoardState;
    engineSessions: Map<number, string>;   // slot → AI session ID for serverEngine slots
    observers: Set<unknown>;               // all connected websockets; used for broadcasting
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

    createGame(config: GameConfig): { id: string; status: 'waiting' | 'playing' } {
        const fn = boardTypeToFn.get(config.boardType);
        if (!fn) throw Object.assign(new Error(`Unknown board type: ${config.boardType}`), { statusCode: 400 });
        let id: string;
        do { id = makeId(12); } while (this.pendingGames.has(id) || this.activeGames.has(id));

        // Validate and normalise incoming player setup: 'local' → 'client'.
        const serverConfig = config.copy();
        for (const [slot, pi] of serverConfig.players) {
            if (slot < 1 || slot > config.numPlayers)
                throw Object.assign(new Error(`Invalid slot ${slot} for ${config.numPlayers}-player game`), { statusCode: 400 });
            if (pi.type === 'local')
                serverConfig.players.set(slot, new PlayerInfo('client', pi.name));
        }

        if (this._pendingSlots(serverConfig).length === 0) {
            // All slots pre-assigned — start immediately.
            const pending: ServerPendingGame = { id, config: serverConfig, observers: new Set() };
            this._startGame(pending);
            return { id, status: 'playing' };
        }
        this.pendingGames.set(id, { id, config: serverConfig, observers: new Set() });
        return { id, status: 'waiting' };
    }

    joinGame(id: string, playerName: string): { position: number; status: 'waiting' | 'playing' } {
        const pending = this.pendingGames.get(id);
        if (!pending) {
            if (this.activeGames.has(id)) throw Object.assign(new Error('Game already started'), { statusCode: 409 });
            throw Object.assign(new Error('Game not found'), { statusCode: 404 });
        }
        const slots = this._pendingSlots(pending.config);
        if (slots.length === 0)
            throw Object.assign(new Error('Game is full'), { statusCode: 409 });
        const slot = slots[0];
        pending.config.players.set(slot, new PlayerInfo('client', playerName));
        if (slots.length === 1) {
            this._startGame(pending);
            return { position: slot, status: 'playing' };
        }
        return { position: slot, status: 'waiting' };
    }

    private _pendingSlots(config: GameConfig): number[] {
        return Array.from({ length: config.numPlayers }, (_, i) => i + 1)
            .filter(s => !config.players.has(s));
    }

    private _startGame(pending: ServerPendingGame) {
        const bc = boardTypeToFn.get(pending.config.boardType)!(...pending.config.boardArgs);
        const boardState = new BoardState(
            pending.config.numStones, pending.config.numPlayers,
            pending.config.turnStoneList, pending.config.stoneToPlayerMap,
            pending.config.forcedPassOnly, new Array(bc.N).fill(0), bc,
        );
        this.pendingGames.delete(pending.id);
        this.activeGames.set(pending.id, {
            id: pending.id, config: pending.config,
            boardState, engineSessions: new Map(), observers: pending.observers,
        });
    }

    // Record that a connection (identified by ws) owns a slot in a game.
    // Also used for reconnects: re-binds a connection to its slot after disconnect.
    acceptJoin(id: string, ws: unknown, slot: number): void {
        const game = this.pendingGames.get(id) ?? this.activeGames.get(id);
        if (!game) return;
        const pi = game.config.players.get(slot);
        if (pi) pi.socket = ws as WebSocket | null;
        game.observers.add(ws);
    }

    addObserver(id: string, ws: unknown): void {
        const game = this.pendingGames.get(id) ?? this.activeGames.get(id);
        game?.observers.add(ws);
    }

    // Returns the slots owned by ws in game id, or [] if none.
    getPositions(id: string, ws: unknown): number[] {
        const game = this.pendingGames.get(id) ?? this.activeGames.get(id);
        if (!game) return [];
        return [...game.config.players.entries()]
            .filter(([, pi]) => pi.socket === (ws as WebSocket | null))
            .map(([slot]) => slot);
    }

    // Returns all WebSocket connections for a game (players + observers), deduplicated.
    getSockets(id: string): unknown[] {
        const game = this.pendingGames.get(id) ?? this.activeGames.get(id);
        if (!game) return [];
        return [...game.observers];
    }

    // Removes a closed connection from all games it was part of.
    removeConnection(ws: unknown): void {
        for (const g of [...this.pendingGames.values(), ...this.activeGames.values()]) {
            for (const pi of g.config.players.values())
                if (pi.socket === (ws as WebSocket | null)) pi.socket = null;
            g.observers.delete(ws);
        }
    }

    getConfig(id: string): GameConfig {
        const game = this.pendingGames.get(id) ?? this.activeGames.get(id);
        if (!game) throw Object.assign(new Error('Game not found'), { statusCode: 404 });
        return game.config;
    }

    // Returns a copy of the game config with player types personalised for `ws`:
    // owned slots → 'local', serverEngine slots → 'serverEngine', others → 'server'.
    getPersonalizedConfig(id: string, ws: unknown): GameConfig {
        const game = this.pendingGames.get(id) ?? this.activeGames.get(id);
        if (!game) throw Object.assign(new Error('Game not found'), { statusCode: 404 });
        const mySlots = new Set(this.getPositions(id, ws));
        const config = game.config.copy();
        for (const [slot, pi] of config.players) {
            if (mySlots.has(slot))
                config.players.set(slot, new PlayerInfo('local', pi.name, null, pi.emsim, pi.temp));
            else if (pi.type !== 'serverEngine')
                config.players.set(slot, new PlayerInfo('server', pi.name));
        }
        return config;
    }

    getState(id: string): OnlineStateResponse {
        const pending = this.pendingGames.get(id);
        if (pending) {
            return { status: 'waiting', moves: [], winners: [], resignedPlayers: [] };
        }
        const game = this.activeGames.get(id);
        if (!game) throw Object.assign(new Error('Game not found'), { statusCode: 404 });
        const v = game.boardState.getView();
        return {
            status: v.gameOver ? 'finished' : 'playing',
            moves: game.boardState.lastMoves.map(m => m.pos),
            winners: v.winners,
            resignedPlayers: v.resignedPlayers,
        };
    }

    isGameOver(id: string): boolean {
        return this.activeGames.get(id)?.boardState.gameOver() ?? false;
    }

    applyMove(id: string, positions: number[], moveIndex: number | null, clientIdx: number): void {
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
    }

    // Returns the slot that should move next if it is a serverEngine slot; null otherwise.
    getEngineSlot(id: string): number | null {
        const game = this.activeGames.get(id);
        if (!game || game.boardState.gameOver()) return null;
        const v = game.boardState.getView();
        const slot = game.config.stoneToPlayerMap[v.nextPlayer];
        const pi = game.config.players.get(slot);
        return (pi?.type === 'serverEngine') ? slot : null;
    }

    // Returns the body needed to call aiMove for a serverEngine slot.
    getEngineRequestParams(id: string, slot: number): {
        config: GameConfig; board: number[]; moves: (number | null)[];
        session_id: string | null; num_simulations: number; temperature: number;
    } | null {
        const game = this.activeGames.get(id);
        if (!game) return null;
        const v = game.boardState.getView();
        const pi = game.config.players.get(slot)!;
        return {
            config: game.config,
            board: v.history[v.plyCount].board,
            moves: game.boardState.lastMoves.map(m => m.pos),
            session_id: game.engineSessions.get(slot) ?? null,
            num_simulations: pi.emsim || 0,
            temperature: pi.temp || 0,
        };
    }

    // Applies a move from the server-side engine (bypasses player-auth check).
    applyEngineMove(id: string, slot: number, moveIndex: number | null, sessionId?: string): void {
        const game = this.activeGames.get(id);
        if (!game) return;
        if (sessionId) game.engineSessions.set(slot, sessionId);
        if (!game.boardState.makeMove(moveIndex)) throw new Error(`Engine returned illegal move ${moveIndex} for slot ${slot}`);
        game.boardState.advanceResigned();
    }

    // Resigns the next slot among `positions` in the turn order (skipping already-resigned slots).
    // Returns the slot that was resigned.
    resign(id: string, positions: number[]): number {
        const game = this.activeGames.get(id);
        if (!game) throw Object.assign(new Error('Game not found'), { statusCode: 404 });
        if (game.boardState.gameOver()) throw Object.assign(new Error('Game is not in progress'), { statusCode: 409 });
        const { turnStoneList, stoneToPlayerMap } = game.config;
        const posSet = new Set(positions);
        const resignedSet = new Set(game.boardState.resignedPlayers);
        const startIdx = turnStoneList.indexOf(game.boardState.nextPlayer);
        let slot: number | null = null;
        for (let i = 0; i < turnStoneList.length; i++) {
            const candidate = stoneToPlayerMap[turnStoneList[(startIdx + i) % turnStoneList.length]];
            if (posSet.has(candidate) && !resignedSet.has(candidate)) { slot = candidate; break; }
        }
        if (slot === null) throw Object.assign(new Error('No resignable slot'), { statusCode: 409 });
        game.boardState.resign(slot);
        game.boardState.advanceResigned();
        return slot;
    }
}

export const onlineGameManager = new OnlineGameManager();
