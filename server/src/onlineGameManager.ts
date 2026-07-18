import { BoardState } from '@shared/boardState.js';
import { PrescribedBoardMap, PrescribedBoardFns, PrescribedBoard } from '@shared/boardConfig.js';
import type { BoardConfig } from '@shared/boardConfig.js';
import { PlayerInfo, GameConfig, FinishedGame, OnlinePlayerRequest, makeId } from '@shared/types.js';
import type { OnlineStateResponse, PendingGame, ReplayMove } from '@shared/types.js';
import { recordFinishedGame, getFinishedGames } from './gameRecordStore.js';
import type { GameRecordStoreState } from './gameRecordStore.js';

// Server-side pending game: extends PendingGame with a set of all connected
// usernames (creator + joiners). Used by getObservers for broadcasting.
interface ServerPendingGame extends PendingGame {
    observers: Set<string>;
    // Whether this game's initial player batch was assigned via fixedOrder
    // (true) or randomOrder (false) - see OnlinePlayerRequest. Determines how
    // joinGame picks a slot for a later joiner: lowest-empty (fixed) or
    // random-empty (not fixed), keeping the whole random-mode experience
    // consistent rather than only randomizing the initial batch.
    fixed: boolean;
}

export interface OnlineGame {
    id: string;
    config: GameConfig;
    boardState: BoardState;
    engineSessions: Map<number, string>;   // slot → AI session ID for serverEngine slots
    observers: Set<string>;                // all connected usernames; used for broadcasting
}

const boardTypeToFn = new Map<string, (...args: number[]) => BoardConfig>();
for (const key of Object.keys(PrescribedBoardMap)) {
    const numKey = Number(key) as PrescribedBoard;
    const [, typeStr] = PrescribedBoardMap[numKey];
    boardTypeToFn.set(typeStr, PrescribedBoardFns[numKey]);
}


export class OnlineGameManager {
    // pendingGames/activeGames are lost on server restart; no persistence.
    // finishedGames is persisted via gameRecordState (see _maybeFinish) and
    // reconstructed by replay at startup, in the constructor below.
    private pendingGames  = new Map<string, ServerPendingGame>();
    private activeGames   = new Map<string, OnlineGame>();
    private finishedGames = new Map<string, OnlineGame>();
    private gameRecordState: GameRecordStoreState;

    constructor(gameRecordState: GameRecordStoreState) {
        this.gameRecordState = gameRecordState;
        for (const { id, finishedGame, observers } of gameRecordState.loadedRecords) {
            try {
                const fn = boardTypeToFn.get(finishedGame.config.boardType);
                if (!fn) throw new Error(`Unknown board type: ${finishedGame.config.boardType}`);
                const bc = fn(...finishedGame.config.boardArgs);
                const boardState = BoardState.fromFinishedGame(finishedGame, bc);
                this.finishedGames.set(id, {
                    id, config: finishedGame.config, boardState, engineSessions: new Map(), observers,
                });
            } catch (e) {
                console.warn('[onlineGameManager] failed to reconstruct finished game', id, e);
            }
        }
    }

    // Finds a game (any lifecycle stage) by id.
    private _findGame(id: string): ServerPendingGame | OnlineGame | undefined {
        return this.pendingGames.get(id) ?? this.activeGames.get(id) ?? this.finishedGames.get(id);
    }

    // Moves `game` from activeGames to finishedGames and persists it (fire-and-forget)
    // the first time it's observed as finished.
    private _maybeFinish(game: OnlineGame): void {
        if (!game.boardState.gameOver() || this.finishedGames.has(game.id)) return;
        this.activeGames.delete(game.id);
        game.engineSessions.clear();   // ephemeral AI session IDs have no value once the game is over
        this.finishedGames.set(game.id, game);
        const finishedGame = new FinishedGame(
            game.config, game.boardState.moveInfos().map(m => ({ pos: m.pos, stone: m.stone })), new Map(game.boardState.resigns),
        );
        void recordFinishedGame(this.gameRecordState, game.id, finishedGame, game.observers).catch(e =>
            console.error('[onlineGameManager] failed to record finished game', game.id, e));
    }

    // Resolves `request` (fixedOrder copied as-is, or randomOrder assigned to
    // randomly chosen slots) into config.players - the server, not the
    // client, is the sole authority for this; an incoming config's own
    // `players` map (if any) is ignored entirely.
    createGame(config: GameConfig, request: OnlinePlayerRequest): { id: string; status: 'waiting' | 'playing' } {
        const fn = boardTypeToFn.get(config.boardType);
        if (!fn) throw Object.assign(new Error(`Unknown board type: ${config.boardType}`), { statusCode: 400 });
        let id: string;
        do { id = makeId(12); } while (this.pendingGames.has(id) || this.activeGames.has(id));

        const serverConfig = config.copy();
        let resolved: Map<number, PlayerInfo>;
        try {
            resolved = request.resolve(config.numPlayers);
        } catch (e: any) {
            throw Object.assign(new Error(e.message), { statusCode: 400 });
        }
        const normalize = (pi: PlayerInfo) => pi.type === 'local' ? new PlayerInfo('client', pi.name) : pi;
        serverConfig.players = new Map([...resolved].map(([slot, pi]) => [slot, normalize(pi)]));

        if (this._readyToStart(serverConfig)) {
            // All slots pre-assigned and confirmed — start immediately.
            const pending: ServerPendingGame = { id, config: serverConfig, observers: new Set(), fixed: request.fixed };
            this._startGame(pending);
            return { id, status: 'playing' };
        }
        this.pendingGames.set(id, { id, config: serverConfig, observers: new Set(), fixed: request.fixed });
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
        // Fixed games fill the lowest-numbered empty slot (deterministic);
        // random-order games keep picking randomly for later joiners too, so
        // the whole random-mode experience stays consistent (see
        // ServerPendingGame.fixed's doc comment).
        const slot = pending.fixed ? slots[0] : slots[Math.floor(Math.random() * slots.length)];
        pending.config.players.set(slot, new PlayerInfo('client', playerName));
        if (this._readyToStart(pending.config)) {
            this._startGame(pending);
            return { position: slot, status: 'playing' };
        }
        return { position: slot, status: 'waiting' };
    }

    // The invite-side counterpart to joinGame() - joinGame fills any open
    // slot for any caller; this instead resolves the ONE slot specifically
    // reserved for userName (a 'pendingInvitedOnline' PlayerInfo), mirroring
    // acceptJoin()'s ownership check below. A refusal cancels the whole
    // pending game - there is no per-slot failure, per the feature's spec -
    // and `notify` covers everyone who needs to know: current observers
    // (creator + anyone already accepted/joined) plus every username still
    // referenced in config.players (covers other invitees who haven't
    // responded yet, and thus aren't observers - invite recipients only
    // become observers once they accept, same "commit on join" timing as a
    // regular game/join).
    respondToInvite(id: string, userName: string, accept: boolean):
        { status: 'waiting' | 'playing' } | { status: 'cancelled'; notify: string[] } {
        const pending = this.pendingGames.get(id);
        if (!pending) throw Object.assign(new Error('Game not found'), { statusCode: 404 });
        const slot = [...pending.config.players.entries()]
            .find(([, pi]) => pi.type === 'pendingInvitedOnline' && pi.name === userName)?.[0];
        if (slot === undefined) throw Object.assign(new Error('No pending invite for you in this game'), { statusCode: 403 });

        if (!accept) {
            const notify = new Set([...pending.observers, ...[...pending.config.players.values()].map(pi => pi.name)]);
            this.pendingGames.delete(id);
            return { status: 'cancelled', notify: [...notify] };
        }

        pending.config.players.set(slot, new PlayerInfo('client', userName));
        pending.observers.add(userName);
        if (this._readyToStart(pending.config)) { this._startGame(pending); return { status: 'playing' }; }
        return { status: 'waiting' };
    }

    private _pendingSlots(config: GameConfig): number[] {
        return Array.from({ length: config.numPlayers }, (_, i) => i + 1)
            .filter(s => !config.players.has(s));
    }

    // A 'pendingInvitedOnline' slot must NOT count as "filled" for starting
    // purposes even though it's present in config.players (it deliberately
    // still counts as filled for _pendingSlots()'s own purpose - open slots
    // - since it must not be claimable via game/join either).
    private _hasUnconfirmedInvites(config: GameConfig): boolean {
        return [...config.players.values()].some(pi => pi.type === 'pendingInvitedOnline');
    }

    private _readyToStart(config: GameConfig): boolean {
        return this._pendingSlots(config).length === 0 && !this._hasUnconfirmedInvites(config);
    }

    private _startGame(pending: ServerPendingGame) {
        const bc = boardTypeToFn.get(pending.config.boardType)!(...pending.config.boardArgs);
        const boardState = new BoardState(
            pending.config.numStones, pending.config.numPlayers,
            pending.config.turnList, pending.config.playerStonePlaceLimit, pending.config.globalStonePlaceLimit,
            pending.config.stoneToPlayerMap,
            pending.config.forcedPassOnly, pending.config.scoreRule, pending.config.komi, pending.config.koRule, pending.config.allowSuicide,
            pending.config.maxPlies, new Array(bc.N).fill(0), bc,
        );
        this.pendingGames.delete(pending.id);
        this.activeGames.set(pending.id, {
            id: pending.id, config: pending.config,
            boardState, engineSessions: new Map(), observers: pending.observers,
        });
    }

    // Reconnect path: verifies `userName` actually owns `slot` in game `id` before
    // re-binding (the client's claimed slot is otherwise unverified). Returns false
    // (no mutation) on mismatch so the caller can reject with 403.
    acceptJoin(id: string, userName: string, slot: number): boolean {
        const game = this._findGame(id);
        if (!game) return false;
        const pi = game.config.players.get(slot);
        if (!pi || pi.name !== userName) return false;
        game.observers.add(userName);
        return true;
    }

    addObserver(id: string, userName: string): void {
        this._findGame(id)?.observers.add(userName);
    }

    // Returns the slots owned by userName in game id, or [] if none.
    getPositions(id: string, userName: string): number[] {
        const game = this._findGame(id);
        if (!game) return [];
        return [...game.config.players.entries()]
            .filter(([, pi]) => pi.name === userName)
            .map(([slot]) => slot);
    }

    // Returns all usernames observing a game (players + spectators), deduplicated.
    getObservers(id: string): string[] {
        const game = this._findGame(id);
        if (!game) return [];
        return [...game.observers];
    }

    getConfig(id: string): GameConfig {
        const game = this._findGame(id);
        if (!game) throw Object.assign(new Error('Game not found'), { statusCode: 404 });
        return game.config;
    }

    // Returns {id, finishedGame} for every finished game `userName` observed - sent
    // to the client at login so it can populate its own finishedGames without
    // having watched those games live.
    getFinishedGamesFor(userName: string): { id: string; finishedGame: FinishedGame }[] {
        const result: { id: string; finishedGame: FinishedGame }[] = [];
        for (const id of getFinishedGames(this.gameRecordState, userName)) {
            const game = this.finishedGames.get(id);
            if (!game) continue;   // shouldn't happen, but don't crash on a bookkeeping mismatch
            result.push({ id, finishedGame: new FinishedGame(
                game.config, game.boardState.moveInfos().map(m => ({ pos: m.pos, stone: m.stone })), new Map(game.boardState.resigns),
            ) });
        }
        return result;
    }


    getState(id: string): OnlineStateResponse {
        const pending = this.pendingGames.get(id);
        if (pending) {
            return { status: 'waiting', moves: [], winners: [], resignedPlayers: [] };
        }
        const game = this.activeGames.get(id) ?? this.finishedGames.get(id);
        if (!game) throw Object.assign(new Error('Game not found'), { statusCode: 404 });
        const v = game.boardState.getView();
        return {
            status: v.gameOver ? 'finished' : 'playing',
            moves: game.boardState.moveInfos().map(m => ({ pos: m.pos, stone: m.stone })),
            winners: v.winners,
            resignedPlayers: v.resignedPlayers,
        };
    }

    isGameOver(id: string): boolean {
        return this.finishedGames.has(id) || (this.activeGames.get(id)?.boardState.gameOver() ?? false);
    }

    applyMove(id: string, positions: number[], moveIndex: number | null, stone: number | null, clientIdx: number): void {
        const game = this.activeGames.get(id);
        if (!game) throw Object.assign(new Error('Game not found'), { statusCode: 404 });
        if (game.boardState.gameOver()) throw Object.assign(new Error('Game is not in progress'), { statusCode: 409 });
        if (game.boardState.getView().plyCount !== clientIdx) throw Object.assign(new Error('Move index mismatch'), { statusCode: 409 });
        if (!positions.includes(game.boardState.nextTurn.player))
            throw Object.assign(new Error('Not your turn'), { statusCode: 403 });
        if (!game.boardState.makeMove(moveIndex, stone ?? undefined)) throw Object.assign(new Error('Illegal move'), { statusCode: 400 });
        game.boardState.advanceResigned();
        this._maybeFinish(game);
    }

    // Returns the slot that should move next if it is a serverEngine slot; null otherwise.
    getEngineSlot(id: string): number | null {
        const game = this.activeGames.get(id);
        if (!game || game.boardState.gameOver()) return null;
        const slot = game.boardState.nextTurn.player;
        const pi = game.config.players.get(slot);
        return (pi?.type === 'serverEngine') ? slot : null;
    }

    // Returns the body needed to call aiMove for a serverEngine slot.
    getEngineRequestParams(id: string, slot: number): {
        config: GameConfig; board: number[]; moves: ReplayMove[]; resigns: [number, number[]][];
        session_id: string | null; num_simulations: number; temperature: number;
    } | null {
        const game = this.activeGames.get(id);
        if (!game) return null;
        const v = game.boardState.getView();
        const pi = game.config.players.get(slot)!;
        return {
            config: game.config,
            board: v.situations[v.plyCount].board,
            moves: game.boardState.moveInfos().map(m => ({ pos: m.pos, stone: m.stone })),
            // Same [ply, players[]][] wire shape as FinishedGame.toJSON()'s resigns field - see
            // ai/src/server.cpp's replay_tail(), which interleaves these with move replay the
            // same way BoardState.fromFinishedGame() does (a resigned player's own earlier real
            // placements must still replay; only their post-resignation forced-passes need the
            // resignation to already be known).
            resigns: [...game.boardState.resigns.entries()],
            session_id: game.engineSessions.get(slot) ?? null,
            num_simulations: pi.emsim || 0,
            temperature: pi.temp || 0,
        };
    }

    // Applies a move from the server-side engine (bypasses player-auth check).
    applyEngineMove(id: string, slot: number, moveIndex: number | null, stone: number | null, sessionId?: string): void {
        const game = this.activeGames.get(id);
        if (!game) return;
        if (sessionId) game.engineSessions.set(slot, sessionId);
        if (!game.boardState.makeMove(moveIndex, stone ?? undefined))
            throw new Error(`Engine returned illegal move ${moveIndex} (stone ${stone}) for slot ${slot}`);
        game.boardState.advanceResigned();
        this._maybeFinish(game);
    }

    // Resigns the next slot among `positions` in the turn order (skipping already-resigned slots).
    // Returns the slot that was resigned.
    resign(id: string, positions: number[]): number {
        const game = this.activeGames.get(id);
        if (!game) throw Object.assign(new Error('Game not found'), { statusCode: 404 });
        if (game.boardState.gameOver()) throw Object.assign(new Error('Game is not in progress'), { statusCode: 409 });
        const { turnList } = game.config;
        const posSet = new Set(positions);
        const resignedSet = new Set(game.boardState.resignedPlayers);
        const startIdx = (game.boardState.situations.length - 1) % turnList.length;
        let slot: number | null = null;
        for (let i = 0; i < turnList.length; i++) {
            const candidate = turnList[(startIdx + i) % turnList.length].player;
            if (posSet.has(candidate) && !resignedSet.has(candidate)) { slot = candidate; break; }
        }
        if (slot === null) throw Object.assign(new Error('No resignable slot'), { statusCode: 409 });
        game.boardState.resign(slot);
        game.boardState.advanceResigned();
        this._maybeFinish(game);
        return slot;
    }
}
