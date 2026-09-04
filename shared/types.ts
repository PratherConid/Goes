// GameConfig/FinishedGame live in shared/gameConfig.ts, not here - see that file's own top comment
// for why (a real circular-import hazard through shared/clegEval.ts and shared/boardConfig.ts). The
// only reference to GameConfig left in this file (PendingGame's own `config: GameConfig` field
// below) is type-only, so this file has no real runtime dependency on gameConfig.ts at all.
import type { GameConfig } from './gameConfig.js';

/** General-purpose runtime assertion, used throughout shared/ and beyond - unified here rather than
 * duplicated per-file (see git history). */
export function assert(cond: boolean, msg: string): asserts cond {
    if (!cond) throw new Error(`Assertion failed: ${msg}`);
}

/** One edge {n1, n2} of a board's adjacency graph, always normalized so n1 <= n2 - see makeBoardEdge
 * - so the same edge has a unique representation regardless of which endpoint was found first. */
export interface BoardEdge {
    n1: number;
    n2: number;
}

/** Builds a BoardEdge from two node indices in either order, normalizing n1 <= n2. */
export function makeBoardEdge(a: number, b: number): BoardEdge {
    return a <= b ? { n1: a, n2: b } : { n1: b, n2: a };
}

/** One (n+1)-node simplex (n+1 pairwise-adjacent vertices - a clique) of a board's adjacency
 * graph, `nodes` always sorted ascending - see makeBoardSimplex - so the same simplex has a
 * unique representation regardless of which vertex was found first (matches
 * shared/topology.ts's findSimplices() own increasing-order convention). A simplex's full
 * symmetry group (every permutation of its n+1 members is some relabeling of it) means a plain
 * ascending sort loses no information - unlike BoardQuad below, whose cycle structure a sort
 * would destroy. */
export interface BoardSimplex {
    nodes: number[];
}

/** Builds a BoardSimplex from n+1 node indices in any order, normalizing to ascending order. */
export function makeBoardSimplex(nodes: number[]): BoardSimplex {
    return { nodes: [...nodes].sort((a, b) => a - b) };
}

/** One quad (induced 4-cycle - 4 distinct vertices forming a cycle with no diagonal edges) {n1, n2,
 * n3, n4} of a board's adjacency graph, n1-n2-n3-n4-n1 always a genuine cycle (n1-n2, n2-n3, n3-n4,
 * n4-n1 are the 4 real graph edges; n1-n3/n2-n4 are the 2 absent diagonals) - see makeBoardQuad.
 * Unlike BoardSimplex above (whose members can always be sorted ascending with no loss, since a
 * simplex's own full symmetry group means every permutation of its members is some relabeling of
 * it), a quad's cycle structure is real information a plain ascending sort would destroy (turning
 * a diagonal into an apparent edge). So a BoardQuad is instead normalized to whichever of its own
 * cycle's 8 rotation/reflection-equivalent relabelings (see makeBoardQuad) is lexicographically
 * smallest - still giving every quad a unique representation regardless of which vertex/direction
 * it was found from (the same guarantee BoardSimplex/BoardEdge have), while
 * keeping n1-n2-n3-n4-n1 a genuine cycle (unlike shared/topology.ts's findQuads(), whose own
 * [a, b, c, d] is *a* valid cycle order, but not necessarily this canonical one). */
export interface BoardQuad {
    n1: number;
    n2: number;
    n3: number;
    n4: number;
}

/** Builds a BoardQuad from four node indices already in cycle order (a-b-c-d-a, as
 * shared/topology.ts's findQuads() reports them) - normalizes to the lexicographically smallest of
 * the cycle's own 8 rotation/reflection-equivalent relabelings (see BoardQuad's own doc comment),
 * NOT a plain sort of all 4 values. Since every rotation starts with a different one of the 4
 * (distinct) node indices, the smallest-starting rotation is unique - so only the 2 candidates
 * starting at the minimum (one per direction) ever need comparing, and since all 4 inputs are
 * distinct, those two candidates' second element can never tie either. */
export function makeBoardQuad(a: number, b: number, c: number, d: number): BoardQuad {
    const seq = [a, b, c, d];
    const i = seq.indexOf(Math.min(...seq));
    const fwd = [0, 1, 2, 3].map(k => seq[(i + k) % 4]);
    const bwd = [0, 1, 2, 3].map(k => seq[(i - k + 4) % 4]);
    const [n1, n2, n3, n4] = fwd[1] < bwd[1] ? fwd : bwd;
    return { n1, n2, n3, n4 };
}

/**
 * A board's node positions in their natural embedding dimension (embDim - 2 for most boards, 3 for
 * shared/boardConfig.ts's cubeLatticeBoard/tetrahedronBoard/dodecahedronBoard/icosahedronBoard,
 * arbitrary for hypercuboidBoard/sierpinskiSimplex/orthoplexBoard). The linear map that projects
 * these down to a 3D render position (x, y, z) is no longer part of this class - the client builds
 * that (a Viewport's own `projMat`, always `defaultProjMat(embDim)` - see src/camera.ts) once
 * per active game, not board construction time, since projection is purely a rendering concern
 * shared/boardConfig.ts's own board builders (and the server, and the C++ AI engine) never need.
 * Kept as its own class (rather than a bare number[][]) so geometric operations that care about real
 * dimensionality (e.g. shared/boardConfig.ts's convex-hull-based rectify()) still have `embDim`
 * alongside `pos` without recomputing it from `pos[0].length` every time.
 */
export class Embedding {
    embDim: number;
    pos: number[][];       // N x embDim

    constructor(embDim: number, pos: number[][]) {
        assert(pos.every(p => p.length === embDim), 'Embedding: pos row length must equal embDim');
        this.embDim = embDim;
        this.pos = pos;
    }
}

export interface BoardConfig {
    emb: Embedding;    // natural-dimension node positions
    adj: number[][];  // N×N symmetric adjacency matrix, entries 0/1
    N: number;
}

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
export const NO_MOVE: MoveInfo = {
    moveType: MoveType.NOMOVE, pos: null, stone: null, captures: [], consecutivePasses: 0, allPassed: false,
};

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
            fixedOrder: [...this.fixedOrder.entries()].map(
                ([slot, pi]) => ({ slot, type: pi.type, name: pi.name, emsim: pi.emsim, temp: pi.temp }),
            ),
            randomOrder: this.randomOrder.map(pi => ({ type: pi.type, name: pi.name, emsim: pi.emsim, temp: pi.temp })),
            fixed: this.fixed,
        };
    }

    static fromJSON(raw: any): OnlinePlayerRequest {
        return new OnlinePlayerRequest(
            new Map((raw.fixedOrder ?? []).map(
                (p: any) => [p.slot, new PlayerInfo(p.type, p.name, p.emsim ?? 0, p.temp ?? 0)],
            )),
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

/**
 * The kind of value a single positional command-line token for a prescribed board type parses
 * into: `Number` is a plain integer; `CommaSeparatedNumbers` is a comma-joined list of integers
 * packed into one token - needed for a variable-arity board type (currently only
 * `hypercuboidBoard`, whose dimension count isn't fixed); `ZeroOneList` is a plain string of `0`/`1`
 * characters with NO separator (e.g. `"0011"`), packing a 0/1 indicator list into one token -
 * needed for `mengerSpongeFlake`'s own `indicator` argument (see its own doc comment). A plain
 * string enum (rather than the numeric-backed convention `PrescribedBoard` uses, shared/
 * boardConfig.ts) so that it reads the same way as `BoardModifier`'s own string-literal `kind`
 * tags wherever a `BoardArgEntry` (below) ends up serialized to JSON (e.g.
 * `public/board_presets/*.json`) - see `parseBoardArgToken`.
 */
export enum BoardArgType { Number = 'Number', CommaSeparatedNumbers = 'CommaSeparatedNumbers', ZeroOneList = 'ZeroOneList' }

/**
 * One parsed board-arg token, tagged with its own `BoardArgType` so callers never have to
 * separately track "which type produced this value" alongside a flattened, anonymous number -
 * this is the type every "list of board args" in this codebase is a list OF (`BoardModifier`'s own
 * `Prod.boardArgs`, shared/boardConfig.ts's `PrescribedBoardFns`' own rest
 * args, ...): exactly one `BoardArgEntry` per POSITIONAL arg (never flattened), so `entries[i]`
 * always corresponds to `PrescribedBoardMap[pb][0][i]` (that positional arg's own declared
 * `BoardArgType`) for any board type `pb`. Earlier revisions of this design flattened every entry
 * into one anonymous `number[]` instead (a `CommaSeparatedNumbers`/`ZeroOneList` token contributing
 * several numbers at once, with no record of which - see git history) - callers then had to
 * re-derive the grouping via positional-index conventions (e.g. "the tail after index 1 is the
 * list"), which broke outright for `mengerSpongeFlake`'s own `(order, dim, indicator)` shape, where
 * the list's own length depends on ANOTHER arg's value (`dim`), making it fundamentally impossible
 * to unambiguously re-derive from a flat array alone.
 */
export type BoardArgEntry =
    | { kind: BoardArgType.Number; value: number }
    | { kind: BoardArgType.CommaSeparatedNumbers; values: number[] }
    | { kind: BoardArgType.ZeroOneList; values: number[] };

/** Constructs a `BoardArgEntry` of each kind - shorthand shared by every caller that builds one by hand (as opposed to parsing one via `parseBoardArgToken`), e.g. default-arg tables. */
export const numArg = (value: number): BoardArgEntry => ({ kind: BoardArgType.Number, value });
export const csvArg = (values: number[]): BoardArgEntry => ({ kind: BoardArgType.CommaSeparatedNumbers, values });
export const zolArg = (values: number[]): BoardArgEntry => ({ kind: BoardArgType.ZeroOneList, values });

/**
 * Reads a `BoardArgEntry` back to the single number it must hold, throwing if it's actually a list
 * entry - shorthand for the (very common) case of a `PrescribedBoardFns` entry whose underlying
 * function takes a plain `number` for this positional arg.
 */
export function boardArgNumber(e: BoardArgEntry): number {
    assert(e.kind === BoardArgType.Number, `expected a Number board arg, got ${e.kind}`);
    return e.value;
}
/**
 * Reads a `BoardArgEntry` back to the number list it must hold, throwing if it's actually a
 * `Number` entry - shorthand for a `PrescribedBoardFns` entry whose underlying function takes a
 * `number[]` for this positional arg (`CommaSeparatedNumbers` or `ZeroOneList`, whichever - a
 * function taking a plain number list generally doesn't care which token syntax produced it).
 */
export function boardArgList(e: BoardArgEntry): number[] {
    assert(e.kind !== BoardArgType.Number, `expected a list board arg, got ${e.kind}`);
    return e.values;
}

/**
 * Parses a single command-line token into the `BoardArgEntry` it represents, per `type` - see
 * `BoardArgType`'s own doc comment. Shared by shared/boardConfig.ts's `parseBoardTypeArgs`
 * (the `prod`/`beginprod` modifier syntax) and `src/renderer.ts`'s `'bd'` command, so there is
 * exactly one place that knows how to interpret a board-arg token - callers never sniff the token's
 * own shape (e.g. "does it contain a comma") themselves. `ZeroOneList` is the one case that
 * validates its own token eagerly (throwing on any non-`0`/`1` character) rather than leaving that
 * to the caller, since - unlike `Number`/`CommaSeparatedNumbers`, where "is this a valid integer" is
 * exactly what `Number.isInteger` already checks downstream - a malformed `ZeroOneList` token (e.g.
 * containing a stray letter or comma) would otherwise silently produce `NaN` entries with no clear
 * origin.
 */
export function parseBoardArgToken(type: BoardArgType, token: string): BoardArgEntry {
    switch (type) {
        case BoardArgType.Number: return numArg(Number(token));
        case BoardArgType.CommaSeparatedNumbers: return csvArg(token.split(',').map(Number));
        case BoardArgType.ZeroOneList:
            if (!/^[01]+$/.test(token))
                throw new Error(`expected a string of 0/1 characters, got "${token}"`);
            return zolArg(token.split('').map(Number));
    }
}

/**
 * The exact inverse of `parseBoardArgToken` (modulo `CommaSeparatedNumbers`/`ZeroOneList`
 * distinguishing themselves only via separator, `,` vs none, on the way back out too) - reconstructs
 * the command-line token `e` was parsed from. Used wherever a `BoardArgEntry` needs to be rendered
 * back to displayable/re-parseable text (e.g. `src/sidePanel.ts`'s `fmtModifiers()`, for a `Prod`
 * modifier's own `boardArgs`).
 */
export function formatBoardArgEntry(e: BoardArgEntry): string {
    switch (e.kind) {
        case BoardArgType.Number: return String(e.value);
        case BoardArgType.CommaSeparatedNumbers: return e.values.join(',');
        case BoardArgType.ZeroOneList: return e.values.join('');
    }
}

/** Deep-clones a single `BoardArgEntry` - a plain `{ ...e }` alone would still share a
 * `CommaSeparatedNumbers`/`ZeroOneList` entry's own `values` array between the original and the
 * clone (same reasoning as `cloneBoardModifier` below, which clones a `BoardArgEntry[]` this way
 * wherever a `BoardModifier` carries one). */
export function cloneBoardArgEntry(e: BoardArgEntry): BoardArgEntry {
    return e.kind === BoardArgType.Number ? { ...e } : { ...e, values: [...e.values] };
}

/** The kinds of object a Selector can denote: `node`, `edge`, `quad` (a fixed arity each), and
 * `` `simp${n}` `` for any integer n >= 2 (a complete (n+1)-node subgraph - `simp2` is what used
 * to be called `tri`, still accepted as sugar in selector/type-annotation text - see
 * shared/selector.ts's own top comment for the full grammar this drives). Use simpType(n)/simpN(t)
 * below to build/read the simp case rather than string-templating '`simp${n}`' by hand. */
export type SelectorType = 'node' | 'edge' | 'quad' | `simp${number}`;

/** Builds the SelectorType for a simp selector of the given arity (n+1 nodes). */
export function simpType(n: number): SelectorType {
    return `simp${n}` as SelectorType;
}

/** Extracts n from a simp SelectorType (simpType's own inverse), or null for node/edge/quad. */
export function simpN(t: SelectorType): number | null {
    return t.startsWith('simp') ? Number(t.slice(4)) : null;
}

/**
 * The literal, already-materialized payload of a `raw` Selector (see below) - one branch per
 * SelectorType, holding exactly what that kind's own shared/selector.ts evaluator (selectNode/
 * selectEdge/selectSimp/selectQuad) itself returns: a real `Set<number>` for nodes (numbers
 * have genuine equality, so a JS Set works as an actual set), plain arrays for edge/simp/quad
 * (which don't - see ClegValue's own 'set' variant in shared/clegBase.ts for why every other edge/
 * simp/quad collection in this codebase is a plain array, never a JS Set, deduplicated by a real key
 * function rather than reference equality).
 */
export type SelectedVals =
    | { kind: 'node'; value: Set<number> }
    | { kind: 'edge'; value: BoardEdge[] }
    | { kind: 'simp'; n: number; value: BoardSimplex[] }
    | { kind: 'quad'; value: BoardQuad[] };

/**
 * A tiny S-expression language for selecting a subset of a board's nodes, edges, simplices, or
 * quads (a "simplex"/"quad" here is exactly what shared/topology.ts's findSimplices()/
 * findQuads() finds - see BoardSimplex/BoardQuad above) - see shared/selector.ts for the full
 * grammar (`(union SEL...)` / `(inter SEL...)` / `(diff SEL SEL)` / `(compl SEL)` /
 * `(more [<num>] SEL)` /
 * `(all <node|edge|simp N|quad>)` / `(none <node|edge|simp N|quad>)` / `(deg <eq|gt|lt> <num>)` /
 * `(conva <node|edge|simp N|quad> SEL)` / `(conve <node|edge|simp N|quad> SEL)` /
 * `(rrmn <num> SEL)` / `(rrmp <num> SEL)` / `(rpkn <num> SEL)` / `(rpkp <num> SEL)`) and its own
 * parsing (parseNodeSelector/
 * parseEdgeSelector/parseTriangleSelector/parseQuadSelector) and evaluation (selectNode/
 * selectEdge/selectSimp/selectQuad) functions. Every Selector carries its own `type` (which kind
 * it denotes) - inferred bottom-up by the one context-free parser (see shared/selector.ts's own
 * top comment for why), then checked by whichever entry point was actually called.
 */
export type Selector =
    // `union`/`inter` take a variadic list of operands - one or more, all the same kind (their own
    // `type`); a zero-operand `(union)`/`(inter)` isn't parseable (nothing to infer `type` from - see
    // shared/selector.ts's own top comment), though this field-level type doesn't itself forbid a
    // hand-built Selector with an empty `items`.
    | { op: 'union' | 'inter'; type: SelectorType; items: Selector[] }
    | { op: 'diff'; type: SelectorType; a: Selector; b: Selector }
    | { op: 'compl'; type: SelectorType; a: Selector }
    // `steps` is the optional repeat count written after `more` in the grammar (`(more [<num>]
    // SEL)`) - undefined means it was omitted (defaults to 1 at evaluation, see selectNode/
    // selectEdge), kept undefined here (rather than eagerly filled in to 1) so formatSelector can
    // round-trip the exact text a Selector was parsed from.
    | { op: 'more'; type: 'node' | 'edge'; steps?: number; a: Selector }
    | { op: 'all' | 'none'; type: SelectorType }
    | { op: 'deg'; type: 'node'; cmp: 'eq' | 'gt' | 'lt'; n: number }
    | { op: 'conva' | 'conve'; type: SelectorType; from: SelectorType; a: Selector }
    | { op: 'rrmn'; type: SelectorType; count: number; a: Selector }
    | { op: 'rrmp'; type: SelectorType; frac: number; a: Selector }
    // The pick-instead-of-remove counterparts of rrmn/rrmp just above - same shape, but keeps
    // exactly `count`/(portion of the size) elements instead of dropping them.
    | { op: 'rpkn'; type: SelectorType; count: number; a: Selector }
    | { op: 'rpkp'; type: SelectorType; frac: number; a: Selector }
    // Wraps an already-materialized SelectedVals directly, rather than computing a selection from
    // scratch - `type` must agree with `items.kind` (both name the same SelectorType; kept as two
    // fields rather than one, like every other variant here, so generic code can still read `.type`
    // without switching on `.op` first). Lets a selection built some other way (e.g. evaluated
    // in cleg, or combined via ordinary set operations) be reused wherever a Selector is expected.
    | { op: 'raw'; type: SelectorType; items: SelectedVals };

/**
 * One selected face plus which kind genericForm should build a lattice for - see
 * shared/boardConfig.ts's genericForm() for the actual construction. A `tri`/`quad` selection is
 * already unambiguous on its own for genericForm's own purposes (unlike LocalReplaceSelector's own
 * situation just below, where a bare `quad` selection can't tell QuadCentralize/QuadCentering/
 * QuadOctarize apart), but FormSelector still tags the kind explicitly, for the same API shape
 * LocalReplaceSelector below already has. `sel`, on either branch, defaults the same way
 * triangleForm/quadForm's own `sel?: Selector` parameter already documents (omitted = every object
 * of the matching kind).
 */
export type FormSelector =
    | { kind: 'TriForm'; sel?: Selector }
    | { kind: 'QuadForm'; sel?: Selector }
    // QuadDiagForm: like QuadForm, but genericForm builds a diagonally-oriented square lattice for
    // each selected quad instead of a plain w-by-w grid - see genericForm's own doc comment
    // (shared/boardConfig.ts) for the actual node/edge construction.
    | { kind: 'QuadDiagForm'; sel?: Selector }
    // QuadKnightForm/QuadBishopForm: same w-by-w node grid as QuadForm, but genericForm connects
    // only nodes a knight's move apart (QuadKnightForm) or diagonally adjacent (QuadBishopForm,
    // same directions QuadDiagForm's own primary-to-center edges run in, but here directly between
    // grid nodes - no extra center nodes) instead of QuadForm's own axis-aligned grid edges - see
    // genericForm's own doc comment (shared/boardConfig.ts) for the actual construction.
    | { kind: 'QuadKnightForm'; sel?: Selector }
    | { kind: 'QuadBishopForm'; sel?: Selector };

/**
 * One selected face plus which LOCAL shape to replace it with - see shared/boardConfig.ts's
 * genericLocalReplace() for the actual construction. This exists (rather than a bare Selector, which
 * suffices for genericForm's own `sels`) because a `quad` selection alone no longer determines a
 * unique replacement: QuadCentralize's own single-hub "pyramid" and QuadOctarize's own two-apex
 * octahedron both consume a quad selector, so the branch itself has to say which one applies. `sel`,
 * on every branch, defaults the same way each single-kind thin wrapper below already documents its
 * own `sel?` parameter as defaulting (omitted = every object of the matching kind). No separate
 * TriCentralize branch - it's SimpCentralize's own n=2 case, same as shared/boardConfig.ts's/
 * shared/clegEval.ts's own triCentralize thin wrappers over simpCentralize(bc, 2, ...).
 */
export type LocalReplaceSelector =
    | { kind: 'QuadCentralize'; sel?: Selector }
    // n: the simplex arity centralized (n+1 corners per hub) - sel, if given, must itself already
    // be a simp `n` selector (checked at runtime by shared/boardConfig.ts's own genericLocalReplace).
    | { kind: 'SimpCentralize'; n: number; sel?: Selector }
    | { kind: 'QuadOctarize'; sel?: Selector }
    // QuadCentering/SimpCentering: same hub-and-spoke construction as QuadCentralize/SimpCentralize
    // (one new hub node, connected to all of that face's own corners), but the face's own original
    // edges are NOT added back - only the new hub-to-corner edges survive, so the selected face's own
    // corners end up connected only through the hub, not to each other directly.
    | { kind: 'QuadCentering'; sel?: Selector }
    | { kind: 'SimpCentering'; n: number; sel?: Selector };

/**
 * A BoardConfig-transforming operation - see shared/boardConfig.ts's applyModifier()/
 * applyModifiers() for how these are built and applied. A `mod`-typed cleg value (shared/clegBase.ts)
 * always wraps one of these directly, built by whichever of cleg's own rectify()/edgeSplit()/.../
 * nis()/eis() builtins matches - cleg's own `prod(a, b)` combines two already-built boards directly
 * (no BoardModifier of its own involved), and has no `Repeat` equivalent at all (a cleg program just
 * writes out a repeated call, or a real `for` loop, instead) - so unlike every other variant here,
 * nothing constructs a `Prod`- or `Repeat`-kind BoardModifier value anymore.
 */
export type BoardModifier =
    | { kind: 'Rectify' }
    | { kind: 'Truncate' }
    | { kind: 'EdgeSplit'; splitN: number }
    | { kind: 'MergeClose'; dist: number }
    | { kind: 'TriangleForm'; w: number; sel?: Selector }
    | { kind: 'QuadForm'; w: number; sel?: Selector }
    | { kind: 'QuadDiagForm'; w: number; sel?: Selector }
    | { kind: 'QuadKnightForm'; w: number; sel?: Selector }
    | { kind: 'QuadBishopForm'; w: number; sel?: Selector }
    // sels: one FormSelector per face-and-kind to look for - see genericForm's own doc comment
    // (shared/boardConfig.ts), which this wraps.
    | { kind: 'Form'; w: number; sels: FormSelector[] }
    // selectors: one LocalReplaceSelector per face-and-shape to replace - see genericLocalReplace's
    // own doc comment (shared/boardConfig.ts). Folds what used to be five separate kinds here
    // (SimpCentralize/TriCentralize/QuadCentralize/Centralize/QuadOctarize) into one.
    | { kind: 'LocalReplace'; selectors: LocalReplaceSelector[] }
    | { kind: 'GlobalCentralize' }
    | { kind: 'Scale'; factor: number }
    | { kind: 'NodeInducedSubgraph'; sel: Selector }
    | { kind: 'EdgeInducedSubgraph'; sel: Selector };

// Position + chosen stone for one replayed ply (see BoardState.fromFinishedGame(),
// OnlineStateResponse). A pass has both fields null.
export interface ReplayMove { pos: number | null; stone: number | null; }

// One chat entry (see ActiveGame.chat/OnlineGame.chat) - time is epoch ms (Date.now()), stored
// but never displayed (see Renderer._refreshChatLog()).
export interface ChatMessage { player: number; time: number; content: string; }

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
    chat: ChatMessage[];
}

export interface BoardView {
    N: number;
    emb: Embedding;             // natural-dim node positions
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
