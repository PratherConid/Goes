// GameConfig/FinishedGame live in their own module, separate from shared/types.ts, specifically to
// avoid a circular-import hazard: both classes need real (not just type) values from BOTH
// shared/types.ts (PlayerInfo) AND shared/cleg.ts (parseCleg/unparseCleg, for boardDescr's own
// text<->AST conversion - see that field's own doc comment below). shared/cleg.ts itself imports
// real values from shared/boardConfig.ts, which imports real values (e.g. the BoardArgType enum)
// from shared/types.ts, eagerly at its own module top level - so if shared/types.ts imported
// cleg.ts back (which it would need to, for GameConfig's own toJSON()/fromJSON()), a native ESM
// loader (Node's test runner, a dev server) can end up evaluating shared/boardConfig.ts's top-level
// code BEFORE shared/types.ts has defined BoardArgType yet, crashing with "cannot read properties
// of undefined". Keeping GameConfig/FinishedGame here instead breaks that cycle entirely:
// shared/types.ts no longer needs to import anything real from here (PendingGame's own `config:
// GameConfig` field below is a type-only reference), so there's no edge back into this module's own
// dependency chain.
import { PlayerInfo, type TurnInfo, type ScoreRule, type KoRule, type PlayerType, type ReplayMove } from './types.js';
import { parseCleg, unparseCleg, type ClegProgram } from './cleg.js';

export class GameConfig {
    // The board's own construction program - a cleg (shared/cleg.ts) AST, run via
    // buildBoardFromCleg (shared/cleg.ts) to get the actual BoardConfig when this GameConfig
    // becomes a real game. In memory this is always the parsed AST, not source text (see
    // ClegProgram's own doc comment) - the Configure Board popup (src/renderer.ts) round-trips it
    // through parseCleg/unparseCleg directly for editing. On the wire/on disk, though, toJSON()/
    // fromJSON() below store it as cleg source text (via unparseCleg/parseCleg) rather than the raw
    // AST, so a serialized GameConfig (public/game_presets/*.json, a persisted finished game, a
    // game/create payload) is human-readable/editable as plain cleg, not a JSON AST dump.
    boardDescr: ClegProgram;
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
        boardDescr: ClegProgram,
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
        this.boardDescr       = boardDescr;
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
            // No deep clone needed - unlike the old boardArgs/boardModifiers (mutated in place by
            // dimension-editing UI/the 'mod' command), boardDescr is always replaced wholesale (see
            // adoptBoardDescr below), never mutated node-by-node, so aliasing the same ClegProgram
            // is safe.
            this.boardDescr,
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
    // array, so JSON.stringify works (Set serializes to "{}" otherwise). boardDescr goes out as
    // cleg source text (unparseCleg), not the raw AST - see this field's own doc comment.
    toJSON() {
        return {
            boardDescr: unparseCleg(this.boardDescr),
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

    // Applies a board-only preset (a boardDescr, e.g. from public/board_presets/) to this
    // GameConfig in place, leaving every other field (turnList, players, scoring rules, etc.)
    // untouched - unlike fromJSON(), which builds a whole new GameConfig from a full preset. No
    // clone needed - see copy()'s own doc comment on why aliasing a ClegProgram is safe.
    adoptBoardDescr(descr: ClegProgram): void {
        this.boardDescr = descr;
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
            parseCleg(raw.boardDescr as string), raw.numStones, raw.numPlayers,
            raw.turnList, playerStonePlaceLimit, globalStonePlaceLimit, stoneToPlayerMap, raw.forcedPassOnly,
            (raw.scoreRule ?? 'area') as ScoreRule,
            (raw.komi ?? new Array(raw.numPlayers).fill(0)) as number[],
            (raw.koRule ?? 'situational') as KoRule,
            (raw.allowSuicide ?? false) as boolean,
            (raw.maxPlies ?? null) as number | null, players,
        );
    }
}

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
