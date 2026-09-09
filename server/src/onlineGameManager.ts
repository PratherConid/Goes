import { BoardState } from '@shared/boardState.js';
import { buildBoardFromCleg, typecheckClegAsBoard } from '@shared/clegEval.js';
import { PlayerInfo, OnlinePlayerRequest, makeId } from '@shared/types.js';
import type { OnlineStateResponse, PendingGame, ReplayMove, ChatMessage } from '@shared/types.js';
import { GameConfig, FinishedGame } from '@shared/gameConfig.js';
import { httpError } from './httpError.js';
import { recordFinishedGame, getFinishedGames } from './gameRecordStore.js';
import type { GameRecordStoreState } from './gameRecordStore.js';

// Server-side pending game: extends PendingGame with a set of all connected
// usernames (creator + joiners), used for broadcasting.
interface ServerPendingGame extends PendingGame {
    observers: Set<string>;
    // Whether this game's initial player batch was assigned via fixedOrder
    // (true) or randomOrder (false) - see OnlinePlayerRequest. Determines how
    // joinGame picks a slot for a later joiner: lowest-empty (fixed) or
    // random-empty (not fixed), keeping the whole random-mode experience
    // consistent rather than only randomizing the initial batch.
    fixed: boolean;
    // The set of invited usernames who have declined so far - see
    // respondToInvite()'s doc comment for how this interacts with
    // unrespondedInvited. A non-empty set means the game is doomed; empty-vs-
    // non-empty (not membership) is what respondToInvite() checks to decide
    // whether a given decline is the first one (the one that triggers
    // notify).
    refused: Set<string>;
    // username -> every slot that user was invited to and hasn't yet
    // responded to (as a whole - see respondToInvite()). Keying by username
    // rather than by individual slot lets one response resolve every slot
    // that user holds in this game at once, and lets a genuinely-redundant
    // second response 403 for free (the username's entry is gone after the
    // first call).
    unrespondedInvited: Map<string, number[]>;
}

// A pending withdrawal vote on an active game - see requestWithdraw()/respondToWithdraw(). Mirrors
// ServerPendingGame's refused/unrespondedInvited shape.
interface WithdrawRequest {
    moveIndex: number;              // ply to withdraw to (situations index right before this move)
    numWithdrawn: number;           // moveInfos().length - moveIndex at request time, for the popup text
    unresponded: Map<string, number[]>;   // username -> owned slots, mirrors unrespondedInvited
    refused: Set<string>;
}

export interface OnlineGame {
    id: string;
    config: GameConfig;
    boardState: BoardState;
    engineSessions: Map<number, string>;   // slot → AI session ID for serverEngine slots
    observers: Set<string>;                // all connected usernames; used for broadcasting
    chat: ChatMessage[];
    withdrawRequest: WithdrawRequest | null;
}

const MAX_CHAT_LENGTH = 2000;


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
        for (const { id, finishedGame, observers, chat } of gameRecordState.loadedRecords) {
            try {
                const bc = buildBoardFromCleg(finishedGame.config.boardDescr);
                const boardState = BoardState.fromFinishedGame(finishedGame, bc);
                this.finishedGames.set(id, {
                    id, config: finishedGame.config, boardState, engineSessions: new Map(), observers, chat,
                    withdrawRequest: null,
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
        void recordFinishedGame(
            this.gameRecordState, game.id, this._snapshotFinished(game), game.observers, game.chat,
        ).catch(e => console.error('[onlineGameManager] failed to record finished game', game.id, e));
    }

    // The replay-only projection of a game (config + moves + resignations) that both persistence
    // and the login-time finished-game payload hand out.
    private _snapshotFinished(game: OnlineGame): FinishedGame {
        return new FinishedGame(
            game.config, game.boardState.moveInfos().map(m => ({ pos: m.pos, stone: m.stone })),
            new Map(game.boardState.resigns),
        );
    }

    // Resolves `request` (fixedOrder copied as-is, or randomOrder assigned to
    // randomly chosen slots) into config.players - the server, not the
    // client, is the sole authority for this; an incoming config's own
    // `players` map (if any) is ignored entirely.
    createGame(config: GameConfig, request: OnlinePlayerRequest): { id: string; status: 'waiting' | 'playing' } {
        try {
            typecheckClegAsBoard(config.boardDescr);
        } catch (e) {
            throw httpError(400, e instanceof Error ? e.message : String(e));
        }
        let id: string;
        do { id = makeId(12); } while (this.pendingGames.has(id) || this.activeGames.has(id));

        const serverConfig = config.copy();
        let resolved: Map<number, PlayerInfo>;
        try {
            resolved = request.resolve(config.numPlayers);
        } catch (e: any) {
            throw httpError(400, e.message);
        }
        const normalize = (pi: PlayerInfo) => pi.type === 'local' ? new PlayerInfo('client', pi.name) : pi;
        serverConfig.players = new Map([...resolved].map(([slot, pi]) => [slot, normalize(pi)]));

        const unrespondedInvited = new Map<string, number[]>();
        for (const [slot, pi] of serverConfig.players)
            if (pi.type === 'pendingInvitedOnline')
                unrespondedInvited.set(pi.name, [...(unrespondedInvited.get(pi.name) ?? []), slot]);

        const pending: ServerPendingGame = {
            id, config: serverConfig, observers: new Set(), fixed: request.fixed,
            refused: new Set(), unrespondedInvited,
        };
        if (this._readyToStart(serverConfig)) {
            // All slots pre-assigned and confirmed — start immediately.
            this._startGame(pending);
            return { id, status: 'playing' };
        }
        this.pendingGames.set(id, pending);
        return { id, status: 'waiting' };
    }

    joinGame(id: string, playerName: string): { position: number; status: 'waiting' | 'playing' } {
        const pending = this.pendingGames.get(id);
        if (!pending) {
            if (this.activeGames.has(id)) throw httpError(409, 'Game already started');
            throw httpError(404, 'Game not found');
        }
        const slots = this._pendingSlots(pending.config);
        if (slots.length === 0) throw httpError(409, 'Game is full');
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
    // slot for any caller; this instead resolves EVERY slot specifically
    // reserved for userName (unrespondedInvited.get(userName) - a user can
    // hold more than one invited slot in the same game), mirroring
    // acceptJoin()'s ownership check below.
    //
    // The FIRST decline for a game notifies everyone else immediately (the game is doomed the
    // moment any required invitee refuses, regardless of who else hasn't answered yet) - `notify`
    // covers current observers (creator + anyone already accepted/joined) plus every username
    // still referenced in config.players (declined/never-seated invitees are deliberately left in
    // place rather than removed - see below - so they're included too), minus the decliner
    // themselves (they already know). A later decline or too-late accept doesn't repeat this -
    // `notify` is only returned on the first one (isFirstDecline).
    //
    // The pending game record itself is NOT torn down on that first decline - it stays around so
    // stragglers who haven't responded yet still get a normal accept/decline experience (a
    // specific "already refused" rejection for accept, see below) rather than a raw 404. It's only
    // actually deleted once unrespondedInvited is empty (everyone has responded, one way or
    // another), which may happen on this same call (single-invitee games, or the last straggler's
    // own response) or later.
    //
    // Once `refused` is non-empty, a further accept can no longer actually seat anyone (the game
    // is doomed regardless) - it throws instead, so the caller gets a specific message rather than
    // silently joining a dead game. No `notify` is attached to that throw - notification already
    // happened on the first decline, so a late accept never needs to trigger it again.
    //
    // config.players is intentionally never mutated for a decline/too-late accept - the slot
    // simply stays 'pendingInvitedOnline' forever, which is harmless since the pending game itself
    // is torn down once every invite is accounted for anyway.
    respondToInvite(id: string, userName: string, accept: boolean):
        { status: 'waiting' | 'playing' } | { status: 'declined'; notify?: string[] } {
        const pending = this.pendingGames.get(id);
        if (!pending) throw httpError(404, 'Game not found');
        const slots = pending.unrespondedInvited.get(userName);
        if (!slots) throw httpError(403, 'No pending invite for you in this game');
        pending.unrespondedInvited.delete(userName);

        if (accept && pending.refused.size === 0) {
            for (const slot of slots) pending.config.players.set(slot, new PlayerInfo('client', userName));
            pending.observers.add(userName);
            if (this._readyToStart(pending.config)) { this._startGame(pending); return { status: 'playing' }; }
            return { status: 'waiting' };
        }

        // Decline, or an accept arriving after some OTHER invitee already
        // declined - either way userName's slot(s) are never seated.
        const isFirstDecline = !accept && pending.refused.size === 0;
        if (!accept) pending.refused.add(userName);

        if (pending.unrespondedInvited.size === 0) this.pendingGames.delete(id);

        if (accept)  // only reachable here when the game was already refused
            throw httpError(409, `Game ${id} already refused by another invited player`);

        if (!isFirstDecline) return { status: 'declined' };
        const notify = [...new Set([...pending.observers, ...[...pending.config.players.values()].map(pi => pi.name)])]
            .filter(name => name !== userName);
        return { status: 'declined', notify };
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
        const bc = buildBoardFromCleg(pending.config.boardDescr);
        const boardState = new BoardState(
            pending.config.numStones, pending.config.numPlayers,
            pending.config.turnList, pending.config.playerStonePlaceLimit, pending.config.globalStonePlaceLimit,
            pending.config.stoneToPlayerMap,
            pending.config.forcedPassOnly, pending.config.scoreRule, pending.config.komi, pending.config.koRule,
            pending.config.allowSuicide, pending.config.maxPlies, new Array(bc.N).fill(0), bc,
        );
        this.pendingGames.delete(pending.id);
        this.activeGames.set(pending.id, {
            id: pending.id, config: pending.config,
            boardState, engineSessions: new Map(), observers: pending.observers,
            chat: [], withdrawRequest: null,
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
        if (!game) throw httpError(404, 'Game not found');
        return game.config;
    }

    // Returns {id, finishedGame, chat} for every finished game `userName` observed - sent to the
    // client at login so it can populate its own finishedGames without having watched those games
    // live.
    getFinishedGamesFor(userName: string): { id: string; finishedGame: FinishedGame; chat: ChatMessage[] }[] {
        const result: { id: string; finishedGame: FinishedGame; chat: ChatMessage[] }[] = [];
        for (const id of getFinishedGames(this.gameRecordState, userName)) {
            const game = this.finishedGames.get(id);
            if (!game) continue;   // shouldn't happen, but don't crash on a bookkeeping mismatch
            result.push({ id, chat: game.chat, finishedGame: this._snapshotFinished(game) });
        }
        return result;
    }


    getState(id: string): OnlineStateResponse {
        if (this.pendingGames.has(id)) {
            return { status: 'waiting', moves: [], winners: [], resignedPlayers: [], chat: [] };
        }
        const game = this.activeGames.get(id) ?? this.finishedGames.get(id);
        if (!game) throw httpError(404, 'Game not found');
        const v = game.boardState.getView();
        return {
            status: v.gameOver ? 'finished' : 'playing',
            moves: game.boardState.moveInfos().map(m => ({ pos: m.pos, stone: m.stone })),
            winners: v.winners,
            resignedPlayers: v.resignedPlayers,
            chat: game.chat,
        };
    }

    isGameOver(id: string): boolean {
        return this.finishedGames.has(id) || (this.activeGames.get(id)?.boardState.gameOver() ?? false);
    }

    // The active game `id`, or a 404/409 explaining why it can't be acted on right now. A pending
    // withdrawal vote locks the game against moves/resignations/further withdraw requests, each of
    // which wants its own message for that case; chatting stays allowed and doesn't come through here.
    private _requirePlayable(id: string, withdrawLockMessage: string): OnlineGame {
        const game = this.activeGames.get(id);
        if (!game) throw httpError(404, 'Game not found');
        if (game.boardState.gameOver()) throw httpError(409, 'Game is not in progress');
        if (game.withdrawRequest) throw httpError(409, withdrawLockMessage);
        return game;
    }

    applyMove(id: string, positions: number[], moveIndex: number | null, stone: number | null, clientIdx: number): void {
        const game = this._requirePlayable(id, 'A withdrawal request is in progress');
        if (game.boardState.getView().plyCount !== clientIdx) throw httpError(409, 'Move index mismatch');
        if (!positions.includes(game.boardState.nextTurn.player)) throw httpError(403, 'Not your turn');
        if (!game.boardState.makeMove(moveIndex, stone ?? undefined)) throw httpError(400, 'Illegal move');
        game.boardState.advanceResigned();
        this._maybeFinish(game);
    }

    // Returns the slot that should move next if it is a serverEngine slot; null otherwise.
    getEngineSlot(id: string): number | null {
        const game = this.activeGames.get(id);
        if (!game || game.boardState.gameOver() || game.withdrawRequest) return null;
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
        const game = this._requirePlayable(id, 'A withdrawal request is in progress');
        const { turnList } = game.config;
        const posSet = new Set(positions);
        const resignedSet = new Set(game.boardState.resignedPlayers);
        const startIdx = (game.boardState.situations.length - 1) % turnList.length;
        let slot: number | null = null;
        for (let i = 0; i < turnList.length; i++) {
            const candidate = turnList[(startIdx + i) % turnList.length].player;
            if (posSet.has(candidate) && !resignedSet.has(candidate)) { slot = candidate; break; }
        }
        if (slot === null) throw httpError(409, 'No resignable slot');
        game.boardState.resign(slot);
        game.boardState.advanceResigned();
        this._maybeFinish(game);
        return slot;
    }

    // Finds userName's last move (scanning moveInfos() backwards for a ply whose mover is one of
    // userName's slots) when toPly is omitted; otherwise validates the explicit toPly, which is
    // not required to be a move userName made. Builds `unresponded` from every 'client'-type slot
    // except userName and any slot(s) already fully resigned - mirrors createGame()'s
    // unrespondedInvited construction. Applies immediately (no voting needed) if that leaves
    // nobody to ask.
    requestWithdraw(id: string, userName: string, toPly?: number):
        | { status: 'applied'; toPly: number; numWithdrawn: number }
        | { status: 'pending'; numWithdrawn: number; notify: string[] } {
        const game = this._requirePlayable(
            id, 'Cannot start withdrawal request: another withdrawal request in progress',
        );

        const moves = game.boardState.moveInfos();
        let moveIndex: number;
        if (toPly !== undefined) {
            if (!Number.isInteger(toPly) || toPly < 0 || toPly >= moves.length)
                throw httpError(400, 'Invalid withdraw target');
            moveIndex = toPly;
        } else {
            const { turnList } = game.config;
            const positions = this.getPositions(id, userName);
            let found: number | null = null;
            for (let i = moves.length - 1; i >= 0; i--) {
                if (positions.includes(turnList[i % turnList.length].player)) { found = i; break; }
            }
            if (found === null)
                throw httpError(409, 'Cannot withdraw your move when you have not made any moves');
            moveIndex = found;
        }

        const numWithdrawn = moves.length - moveIndex;
        const resignedSet = new Set(game.boardState.resignedPlayers);
        const unresponded = new Map<string, number[]>();
        for (const [slot, pi] of game.config.players) {
            if (pi.type !== 'client' || pi.name === userName || resignedSet.has(slot)) continue;
            unresponded.set(pi.name, [...(unresponded.get(pi.name) ?? []), slot]);
        }

        if (unresponded.size === 0) {
            this._applyWithdraw(game, moveIndex);
            return { status: 'applied', toPly: moveIndex, numWithdrawn };
        }
        game.withdrawRequest = { moveIndex, numWithdrawn, unresponded, refused: new Set() };
        return { status: 'pending', numWithdrawn, notify: [...unresponded.keys()] };
    }

    private _applyWithdraw(game: OnlineGame, moveIndex: number): void {
        game.boardState.withdrawTo(moveIndex);
        game.boardState.advanceResigned();
        this._maybeFinish(game);
    }

    // The withdraw-vote counterpart to respondToInvite() - same first-decline-notifies-everyone /
    // torn-down-once-everyone-has-responded shape (see that method's own comment for the reasoning).
    respondToWithdraw(id: string, userName: string, accept: boolean):
        | { status: 'waiting' }
        | { status: 'applied'; toPly: number; numWithdrawn: number }
        | { status: 'declined'; notify?: string[] } {
        const game = this.activeGames.get(id);
        if (!game) throw httpError(404, 'Game not found');
        const wr = game.withdrawRequest;
        if (!wr) throw httpError(404, 'No withdrawal request in progress');
        const slots = wr.unresponded.get(userName);
        if (!slots) throw httpError(403, 'No pending withdrawal request for you to respond to');
        wr.unresponded.delete(userName);

        if (accept && wr.refused.size === 0) {
            if (wr.unresponded.size === 0) {
                const { moveIndex, numWithdrawn } = wr;
                game.withdrawRequest = null;
                this._applyWithdraw(game, moveIndex);
                return { status: 'applied', toPly: moveIndex, numWithdrawn };
            }
            return { status: 'waiting' };
        }

        const isFirstDecline = !accept && wr.refused.size === 0;
        if (!accept) wr.refused.add(userName);
        if (wr.unresponded.size === 0) game.withdrawRequest = null;

        if (accept)  // only reachable when some other voter already declined
            throw httpError(409, `Withdrawal request for game ${id} already declined`);

        if (!isFirstDecline) return { status: 'declined' };
        const notify = [...game.observers].filter(name => name !== userName);
        return { status: 'declined', notify };
    }

    // In-progress-only, like applyMove/resign, except that a pending withdrawal vote doesn't block
    // chatting. `player` is assumed to be authorized by the caller.
    sendChat(id: string, player: number, content: string): ChatMessage {
        const game = this.activeGames.get(id);
        if (!game) {
            // _maybeFinish() moves a game out of activeGames the instant it's over, so
            // activeGames.get(id) alone can't tell "finished" apart from "never existed".
            if (this.finishedGames.has(id))
                throw httpError(409, 'Cannot send messages in a finished game');
            throw httpError(404, 'Game not found');
        }
        const trimmed = content.trim().slice(0, MAX_CHAT_LENGTH);
        if (!trimmed) throw httpError(400, 'Chat message cannot be empty');
        const msg: ChatMessage = { player, time: Date.now(), content: trimmed };
        game.chat.push(msg);
        return msg;
    }
}
