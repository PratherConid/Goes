export const enum MoveType {
    NOMOVE = 0,
    PLACE  = 1,
    PASS   = 2,
}

export interface StoneInfo { name: string; color: string; }

export const STONE_MAP: Record<number, StoneInfo> = {
    1: { name: 'black',   color: '#1a1a1a' },
    2: { name: 'white',   color: '#ffffff' },
    3: { name: 'red',     color: '#b91e1e' },
    4: { name: 'green',   color: '#1eb91e' },
    5: { name: 'blue',    color: '#1e1eb9' },
    6: { name: 'cyan',    color: '#1eb9b9' },
    7: { name: 'magenta', color: '#b91eb9' },
    8: { name: 'yellow',  color: '#b9b91e' },
};

export interface MoveInfo {
    moveType: MoveType;
    pos: number | null;
    stone: number | null;   // the stone color actually placed (null for PASS)
    captures: number[];
    // Number of consecutive pass moves ending with (and including, if this move
    // itself is a pass) this move. Resets to 0 on a PLACE move. The game ends
    // once this reaches turnList.length, since a stone appearing multiple
    // times in turnList (one player controlling several stones) must pass
    // on each of its turns, not just once, before the round can be considered over.
    consecutivePasses: number;
    // True iff this move was the pass that completed a full round of
    // consecutive passes (consecutivePasses reached turnList.length) - a
    // per-ply-intrinsic fact, set once when the move is created and never
    // retroactively mutated afterward. This is only ONE of
    // BoardState.gameOver()'s three conditions (see there, which also checks
    // maxPlies and resignedPlayers live) - e.g. a maxPlies-triggered PLACE
    // move has allPassed=false even though it ends the game.
    allPassed: boolean;
}

// Genesis value of HistoryEntry.moveInfo (history[0], before any real move has been made) -
// also what BoardState.lastMove() naturally returns at that point. Shared/immutable - never mutate.
export const NO_MOVE: MoveInfo = { moveType: MoveType.NOMOVE, pos: null, stone: null, captures: [], consecutivePasses: 0, allPassed: false };

// One slot in the turn order: `player` is who plays this turn - the sole
// source of truth for turn ownership. `stones` has length numStones (0 = not
// offered, 1 = offered): the set of stone colors the player may choose from
// this turn (at least one must be offered) - see BoardState.makeMove.
// stoneToPlayerMap is separate and used only for scoring (see BoardState).
// `protected` has length numStones (0 = normal, 1 = protected): stones of a
// protected color can never be removed from the board on this turn, even at
// zero liberties (see calculateLegalMoves).
// `friendly` has length numStones (0 = normal, 1 = friendly): stones of a
// friendly color don't count as blocking anyone else's liberties this turn -
// a group adjacent to one treats that cell as a liberty, not an occupied
// neighbor (see groupLiberty).
export interface TurnInfo { player: number; stones: number[]; protected: number[]; friendly: number[]; }

export type ScoreRule = 'stone' | 'territoryonly' | 'area' | 'territory';

// 'situational': a repeated board position is only illegal when it's also the
//                same player's turn to move as the earlier occurrence.
// 'positional':  any repeated board position is illegal, regardless of whose
//                turn is next.
export type KoRule = 'positional' | 'situational';

export interface ScoreData {
    stoneCount: Record<number, number>;   // stones on the board, per stone type (1..numStones)
    territory: Record<number, number>;    // territory points, per stone type (1..numStones)
    territoryOwner: number[];             // length N; stone type whose territory this node belongs
                                           // to, or 0 if occupied or neutral (dame) - same 0-sentinel
                                           // convention as `board` (0 = empty/none)
    // Cumulative stones captured so far, indexed [player-1] - unlike
    // stoneCount/territory (stone-indexed, board-derived every ply), this is
    // player-indexed and a running total across the whole game (captured
    // stones are simply gone, so it can't be recomputed from the current
    // board) - see BoardState.makeMove/captureCount(). Used for the
    // 'territory' ScoreRule (real-world Japanese-style scoring: territory +
    // prisoners); folded in at the player-aggregation layer (_computeWinners),
    // the same way komi is, rather than inside computePoints (which stays
    // stone-indexed).
    captureCount: number[];
}

// The board-only, ko-tree-searchable part of a ply - what BoardState.sortedSituations (an
// AVLTree) is ordered by (see compareState()), and all it ever reads.
export interface Situation {
    board: number[];
    plyCount: number;
    lenTurnList: number;
}

// The rest of a ply's per-ply record - not needed by the ko-rule AVL tree, so kept separate
// from Situation (see BoardState.history/situations).
export interface HistoryEntry {
    // The move that produced this ply, or NO_MOVE for history[0] (the genesis entry, before any
    // real move) - see BoardState.lastMove()/moveInfos.
    moveInfo: MoveInfo;
    legalMoves: LegalMovesData;
    score: ScoreData;
    // Cumulative count of stones placed through this ply, indexed
    // [stone-1][player-1] (same indexing as GameConfig.playerStonePlaceLimit,
    // for direct cell-by-cell comparison) - see BoardState.makeMove.
    playerStonePlaceCnt: number[][];
}

// 'local': a slot in a local (non-online) game - never sent to the server.
// 'client': a human participant's slot in an online game, identified by PlayerInfo.name
//           matching that connection's own username (see e.g. Renderer._isMyTurn()).
// 'serverEngine': an AI-controlled slot in an online game, driven by the server itself.
// 'pendingInvitedOnline': an online slot reserved for a specific invited username
//           (PlayerInfo.name) who hasn't yet accepted - not claimable via game/join,
//           and a game holding one never auto-starts, until it's either converted to
//           'client' (accepted) or the whole game is cancelled (refused). See
//           OnlineGameManager.respondToInvite() (server/src/onlineGameManager.ts).
// 'localEngine': a slot in a *local* game driven by the client's own AI engine calls
//           (Renderer._fireEngineMove()) - like 'serverEngine' but client- rather than
//           server-driven, and auto-advanced (Renderer._render()) rather than requiring
//           the 'em' command each time.
export type PlayerType = 'local' | 'serverEngine' | 'client' | 'pendingInvitedOnline' | 'localEngine';

export class PlayerInfo {
    type: PlayerType;
    name: string;
    emsim: number;  // AI simulations per move (0 = server default); for serverEngine slots
    temp: number;   // AI temperature (0 = server default); for serverEngine slots

    constructor(type: PlayerType, name: string, emsim = 0, temp = 0) {
        this.type   = type;
        this.name   = name;
        this.emsim  = emsim;
        this.temp   = temp;
    }
}

// A not-yet-resolved online-game player setup, sent to the server in
// game/create instead of a pre-populated GameConfig.players map - the server
// is the sole authority for turning this into actual slot assignments (see
// OnlineGameManager.createGame): fixedOrder is copied directly (slot-by-slot,
// as specified), while randomOrder is assigned to randomly chosen slots.
// `fixed` selects which of the two lists is actually used; the other is
// simply ignored (not required to be empty).
export class OnlinePlayerRequest {
    fixedOrder: Map<number, PlayerInfo>;
    randomOrder: PlayerInfo[];
    fixed: boolean;

    constructor(fixedOrder: Map<number, PlayerInfo> = new Map(), randomOrder: PlayerInfo[] = [], fixed = true) {
        this.fixedOrder  = fixedOrder;
        this.randomOrder = randomOrder;
        this.fixed       = fixed;
    }

    copy(): OnlinePlayerRequest {
        return new OnlinePlayerRequest(
            new Map([...this.fixedOrder.entries()].map(([s, pi]) => [s, new PlayerInfo(pi.type, pi.name, pi.emsim, pi.temp)])),
            this.randomOrder.map(pi => new PlayerInfo(pi.type, pi.name, pi.emsim, pi.temp)),
            this.fixed,
        );
    }

    toJSON() {
        return {
            fixedOrder: [...this.fixedOrder.entries()].map(([slot, pi]) => ({ slot, type: pi.type, name: pi.name, emsim: pi.emsim, temp: pi.temp })),
            randomOrder: this.randomOrder.map(pi => ({ type: pi.type, name: pi.name, emsim: pi.emsim, temp: pi.temp })),
            fixed: this.fixed,
        };
    }

    static fromJSON(raw: any): OnlinePlayerRequest {
        return new OnlinePlayerRequest(
            new Map((raw.fixedOrder ?? []).map((p: any) => [p.slot, new PlayerInfo(p.type, p.name, p.emsim ?? 0, p.temp ?? 0)])),
            (raw.randomOrder ?? []).map((p: any) => new PlayerInfo(p.type, p.name, p.emsim ?? 0, p.temp ?? 0)),
            raw.fixed ?? true,
        );
    }

    // Resolves fixedOrder (copied as-is) or randomOrder (assigned to randomly
    // chosen slots) into a slot map - POTENTIALLY INCOMPLETE: a slot nothing
    // was specified for is simply absent, not filled with any default.
    // Callers (OnlineGameManager.createGame() for online games,
    // Renderer._createLocalGame() for local ones) each do their own
    // type-specific normalization afterward (e.g. 'local' -> 'client' online,
    // 'serverEngine' -> 'localEngine' local, filling empty slots) - this only
    // does the structural fixedOrder/randomOrder -> slots part, identical
    // either way.
    resolve(numPlayers: number): Map<number, PlayerInfo> {
        const result = new Map<number, PlayerInfo>();
        if (this.fixed) {
            for (const [slot, pi] of this.fixedOrder) {
                if (slot < 1 || slot > numPlayers)
                    throw new Error(`Invalid slot ${slot} for ${numPlayers}-player game`);
                result.set(slot, pi);
            }
        } else {
            if (this.randomOrder.length > numPlayers)
                throw new Error(`Too many players (${this.randomOrder.length}) for a ${numPlayers}-player game`);
            const slots = Array.from({ length: numPlayers }, (_, i) => i + 1);
            for (let i = slots.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [slots[i], slots[j]] = [slots[j], slots[i]];
            }
            this.randomOrder.forEach((pi, i) => result.set(slots[i], pi));
        }
        return result;
    }
}

export class GameConfig {
    boardType: string;
    boardArgs: number[];
    numStones: number;
    numPlayers: number;
    turnList: TurnInfo[];
    // How many times each player may place each stone color over the whole
    // game, indexed [stone-1][player-1]; null = unlimited. Enforced in
    // calculateLegalMoves against BoardState's live playerStonePlaceCnt (see
    // HistoryEntry.playerStonePlaceCnt) - once a player hits their limit for a
    // color, that color is simply no longer offered to them for the rest of
    // the game, same as if it were never in TurnInfo.stones.
    playerStonePlaceLimit: (number | null)[][];
    // How many times each stone color may ever be placed in TOTAL, summed
    // across every player (unlike playerStonePlaceLimit, which is per-player);
    // length numStones, indexed [stone-1]; null = unlimited. There's no
    // separate running-count field for this - BoardState derives it on the fly
    // by summing playerStonePlaceCnt[stone-1] across all players.
    globalStonePlaceLimit: (number | null)[];
    stoneToPlayerMap: Record<number, Set<number>>;
    forcedPassOnly: boolean;
    scoreRule: ScoreRule;
    komi: number[];
    koRule: KoRule;
    allowSuicide: boolean;
    // Maximum number of plies before the game is automatically ended (see
    // BoardState.makeMove); null = unlimited.
    maxPlies: number | null;
    players: Map<number, PlayerInfo>;  // slot → player; empty slots are pending/unassigned

    constructor(
        boardType: string,
        boardArgs: number[],
        numStones: number,
        numPlayers: number,
        turnList: TurnInfo[],
        playerStonePlaceLimit: (number | null)[][],
        globalStonePlaceLimit: (number | null)[],
        stoneToPlayerMap: Record<number, Set<number>>,
        forcedPassOnly: boolean,
        scoreRule: ScoreRule,
        komi: number[],
        koRule: KoRule,
        allowSuicide: boolean,
        maxPlies: number | null,
        players: Map<number, PlayerInfo> = new Map(),
    ) {
        if (komi.some(k => k < 0)) throw new Error(`komi values must be >= 0, got [${komi.join(', ')}]`);
        this.boardType        = boardType;
        this.boardArgs        = boardArgs;
        this.numStones        = numStones;
        this.numPlayers       = numPlayers;
        this.turnList         = turnList;
        this.playerStonePlaceLimit = playerStonePlaceLimit;
        this.globalStonePlaceLimit = globalStonePlaceLimit;
        this.stoneToPlayerMap = stoneToPlayerMap;
        this.forcedPassOnly   = forcedPassOnly;
        this.scoreRule        = scoreRule;
        this.komi             = komi;
        this.koRule           = koRule;
        this.allowSuicide     = allowSuicide;
        this.maxPlies         = maxPlies;
        this.players          = players;
    }

    copy(): GameConfig {
        return new GameConfig(
            this.boardType,
            [...this.boardArgs],
            this.numStones,
            this.numPlayers,
            this.turnList.map(t => ({ ...t, stones: [...t.stones], protected: [...t.protected], friendly: [...t.friendly] })),
            this.playerStonePlaceLimit.map(row => [...row]),
            [...this.globalStonePlaceLimit],
            Object.fromEntries(Object.entries(this.stoneToPlayerMap).map(([k, v]) => [k, new Set(v)])),
            this.forcedPassOnly,
            this.scoreRule,
            this.komi,
            this.koRule,
            this.allowSuicide,
            this.maxPlies,
            new Map([...this.players.entries()].map(
                ([s, pi]) => [s, new PlayerInfo(pi.type, pi.name, pi.emsim, pi.temp)]
            )),
        );
    }

    // Serialise players Map as an array, and each stoneToPlayerMap Set as a plain
    // array, so JSON.stringify works (Set serializes to "{}" otherwise).
    toJSON() {
        return {
            boardType: this.boardType, boardArgs: this.boardArgs,
            numStones: this.numStones, numPlayers: this.numPlayers,
            turnList: this.turnList,
            playerStonePlaceLimit: this.playerStonePlaceLimit,
            globalStonePlaceLimit: this.globalStonePlaceLimit,
            stoneToPlayerMap: Object.fromEntries(Object.entries(this.stoneToPlayerMap).map(([k, v]) => [k, [...v]])),
            forcedPassOnly: this.forcedPassOnly, scoreRule: this.scoreRule, komi: this.komi, koRule: this.koRule,
            allowSuicide: this.allowSuicide,
            maxPlies: this.maxPlies,
            players: [...this.players.entries()].map(([slot, pi]) =>
                ({ slot, type: pi.type, name: pi.name, emsim: pi.emsim, temp: pi.temp })),
        };
    }

    static fromJSON(raw: any): GameConfig {
        const players = new Map<number, PlayerInfo>(
            ((raw.players ?? []) as { slot: number; type: PlayerType; name: string; emsim: number; temp: number }[])
                .map(p => [p.slot, new PlayerInfo(p.type, p.name, p.emsim ?? 0, p.temp ?? 0)])
        );
        const stoneToPlayerMap: Record<number, Set<number>> = Object.fromEntries(
            Object.entries(raw.stoneToPlayerMap).map(([k, v]) => [k, new Set(v as number[])])
        );
        const playerStonePlaceLimit = (raw.playerStonePlaceLimit
            ?? Array.from({ length: raw.numStones }, () => new Array(raw.numPlayers).fill(null))) as (number | null)[][];
        const globalStonePlaceLimit = (raw.globalStonePlaceLimit
            ?? new Array(raw.numStones).fill(null)) as (number | null)[];
        return new GameConfig(
            raw.boardType, raw.boardArgs, raw.numStones, raw.numPlayers,
            raw.turnList, playerStonePlaceLimit, globalStonePlaceLimit, stoneToPlayerMap, raw.forcedPassOnly,
            (raw.scoreRule ?? 'area') as ScoreRule,
            (raw.komi ?? new Array(raw.numPlayers).fill(0)) as number[],
            (raw.koRule ?? 'situational') as KoRule,
            (raw.allowSuicide ?? false) as boolean,
            (raw.maxPlies ?? null) as number | null, players,
        );
    }
}

// Position + chosen stone for one replayed ply (see BoardState.fromFinishedGame(),
// OnlineStateResponse). A pass has both fields null.
export interface ReplayMove { pos: number | null; stone: number | null; }

// Minimal, principled record of a finished game: just enough to deterministically
// reconstruct the final BoardState by replay (see BoardState.fromFinishedGame()),
// rather than persisting a full point-in-time snapshot of derived state.
export class FinishedGame {
    config: GameConfig;
    moves: ReplayMove[];
    resigns: Map<number, number[]>;   // ply -> players (1-indexed) who resigned at that ply

    constructor(config: GameConfig, moves: ReplayMove[], resigns: Map<number, number[]>) {
        this.config  = config;
        this.moves   = moves;
        this.resigns = resigns;
    }

    toJSON() {
        return { config: this.config.toJSON(), moves: this.moves, resigns: [...this.resigns.entries()] };
    }

    static fromJSON(raw: any): FinishedGame {
        return new FinishedGame(GameConfig.fromJSON(raw.config), raw.moves, new Map(raw.resigns));
    }
}

export interface PendingGame {
    id: string;
    config: GameConfig;   // always set; config.players tracks assigned slots
}

// Result of calculateLegalMoves() (see boardState.ts). A turn may offer more
// than one stone color (TurnInfo.stones); legality/captures are precomputed
// for every offered stone, since which color is placed can change what's
// legal and what gets captured (protected/friendly are per-stone-color).
//   passCapture       - nodes captured by a pass. Color-independent: it's
//                        simply every non-protected zero-liberty group on the
//                        board, regardless of which stone the mover could
//                        have played - see calculateLegalMoves.
//   captures          - captures[stone][loc]: per-(stone, board location)
//                        legal PLACE move data. `stone` is the literal stone
//                        color (1-indexed; index 0 unused). null = illegal for
//                        that (stone, loc) pair; Set<number> = legal, and is
//                        the FULL final set of nodes that specific placement
//                        captures (already includes that stone's own
//                        early-opponent-capture cleanup - callers never need
//                        to union anything else in).
//   legalsForStone    - legalsForStone[stone] = set of legal locations for
//                        that stone color (1-indexed; index 0 unused).
//   legalsForLocation - legalsForLocation[loc] = set of stone colors legal at
//                        that location.
//   placeLegals       - total count of legal (stone, loc) pairs; see
//                        BoardState.noTradLegal().
export class LegalMovesData {
    passCapture: Set<number>;
    captures: (Set<number> | null)[][];
    legalsForStone: Set<number>[];
    legalsForLocation: Set<number>[];
    placeLegals: number;

    constructor(
        passCapture: Set<number>, captures: (Set<number> | null)[][],
        legalsForStone: Set<number>[], legalsForLocation: Set<number>[], placeLegals: number,
    ) {
        this.passCapture       = passCapture;
        this.captures           = captures;
        this.legalsForStone     = legalsForStone;
        this.legalsForLocation  = legalsForLocation;
        this.placeLegals        = placeLegals;
    }
}

export interface OnlineStateResponse {
    status: 'waiting' | 'playing' | 'finished';
    moves: ReplayMove[];
    winners: number[] | null;
    resignedPlayers: number[];
}

export interface BoardView {
    N: number;
    pos: number[][];           // N×2
    boardDimension: [[number, number], [number, number]];
    numStones: number;
    numPlayers: number;
    turnList: TurnInfo[];
    playerStonePlaceLimit: (number | null)[][];
    globalStonePlaceLimit: (number | null)[];
    stoneToPlayerMap: Record<number, Set<number>>;
    forcedPassOnly: boolean;
    scoreRule: ScoreRule;
    komi: number[];
    koRule: KoRule;
    allowSuicide: boolean;
    maxPlies: number | null;
    nextTurn: TurnInfo;       // the turnList entry for the upcoming ply (see BoardState.nextTurn)
    lastMove: MoveInfo;
    moveInfos: MoveInfo[];    // the full per-ply move list (see BoardState.moveInfos)
    score: ScoreData;
    winners: number[] | null;
    resignedPlayers: number[];
    plyCount: number;
    situations: Situation[];
    history: HistoryEntry[];
    gameOver: boolean;
    passEnabled: boolean;
}

const _ID_CHARS = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
export function makeId(len: number): string {
    let id = '';
    for (let i = 0; i < len; i++) id += _ID_CHARS[Math.floor(Math.random() * _ID_CHARS.length)];
    return id;
}
