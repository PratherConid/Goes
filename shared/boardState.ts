import type { BoardConfig } from './boardConfig.js';
import { MoveType, STONE_MAP } from './types.js';
import { NO_MOVE } from './types.js';
import type { MoveInfo, Situation, HistoryEntry, BoardView, ScoreData, ScoreRule, KoRule, TurnInfo, FinishedGame, ReplayMove } from './types.js';
import { LegalMovesData } from './types.js';
import { AVLTree } from './avlTree.js';

function assert(cond: boolean, msg: string): asserts cond {
    if (!cond) throw new Error(`Assertion failed: ${msg}`);
}

// ── comparison / search ──────────────────────────────────────────────────────

// Returns: s1 > s2 → 1, s1 == s2 → 0, s1 < s2 → -1.
// koRule 'situational': two states are only "the same" (for the leading,
// non-strict comparison) when the board AND the player to move next match.
// koRule 'positional': only the board matters - the situational leading
// comparison is skipped entirely.
// With strict = true, only returns 0 if plyCount is also equal.
function compareState(s1: Situation, s2: Situation, koRule: KoRule, strict = false): number {
    const { board: b1, plyCount: t1, lenTurnList: ltl } = s1;
    const { board: b2, plyCount: t2 } = s2;
    if (koRule === 'situational') {
        if (t1 % ltl < t2 % ltl) return -1;
        if (t1 % ltl > t2 % ltl) return  1;
    }
    for (let i = 0; i < b1.length; i++) {
        if (b1[i] < b2[i]) return -1;
        if (b1[i] > b2[i]) return  1;
    }
    if (!strict) return 0;
    if (t1 < t2) return -1;
    if (t1 > t2) return  1;
    return 0;
}

// ── group / liberty ──────────────────────────────────────────────────────────

interface GroupEntry { group: number[]; liberties: Set<number>; nonLiberties: Set<number>; }

// Returns a map from stone color to list of (group, liberties, nonLiberties) triples:
//   group        - list of node indices belonging to the group
//   liberties    - set of empty node indices adjacent to the group, plus any
//                  occupied neighbor whose color is friendly this turn (see
//                  friendlyStones - such a neighbor doesn't block a liberty)
//   nonLiberties - set of occupied, non-friendly node indices (belonging to another
//                  group) adjacent to the group - used to detect when capturing a
//                  neighboring group frees a liberty for this one (see calculateLegalMoves).
function groupLiberty(board: number[], adj: number[][], N: number, friendlyStones: number[])
    : Map<number, GroupEntry[]>
{
    const isFriendly = (stone: number) => friendlyStones[stone - 1] === 1;
    // affColor[i]: stone color of node i (0 = unvisited/empty)
    // affGid[i]:   group ID of node i (0-indexed, unique within each color)
    const affColor = new Int32Array(N);
    const affGid   = new Int32Array(N);
    const groups   = new Map<number, number[][]>();
    const colorGid = new Map<number, number>();

    for (let i = 0; i < N; i++) {
        if (board[i] === 0 || affColor[i] !== 0) continue;
        const color = board[i];
        if (!groups.has(color)) { groups.set(color, []); colorGid.set(color, 0); }
        const gid  = colorGid.get(color)!;
        const allel: number[] = [i];
        affColor[i] = color; affGid[i] = gid;
        let st: number[] = [i];
        while (st.length > 0) {
            const nxt: number[] = [];
            for (const node of st) {
                const row = adj[node];
                for (let j = 0; j < N; j++) {
                    if (row[j] && affColor[j] === 0 && board[j] === color) {
                        affColor[j] = color; affGid[j] = gid;
                        allel.push(j); nxt.push(j);
                    }
                }
            }
            st = nxt;
        }
        groups.get(color)!.push(allel);
        colorGid.set(color, gid + 1);
    }

    // Compute liberties/nonLiberties: for each group, the sets of adjacent
    // empty nodes and adjacent occupied nodes (belonging to another group).
    const liberties    = new Map<number, Set<number>[]>();
    const nonLiberties = new Map<number, Set<number>[]>();
    for (const [c, grps] of groups) {
        liberties.set(c, grps.map(() => new Set<number>()));
        nonLiberties.set(c, grps.map(() => new Set<number>()));
    }

    for (let i = 0; i < N; i++) {
        if (affColor[i] === 0) continue;
        for (let j = 0; j < N; j++) {
            if (!adj[i][j]) continue;
            if (affColor[j] === 0) {
                liberties.get(affColor[i])![affGid[i]].add(j);
            } else if (affColor[j] !== affColor[i]) {
                if (isFriendly(affColor[j])) liberties.get(affColor[i])![affGid[i]].add(j);
                else nonLiberties.get(affColor[i])![affGid[i]].add(j);
            }
        }
    }

    // Note: a group may legitimately have zero liberties here - a protected
    // color (see calculateLegalMoves) can be left on the board at zero
    // liberties instead of being captured.
    const result = new Map<number, GroupEntry[]>();
    for (const [c, grps] of groups)
        result.set(c, grps.map((g, idx) => ({
            group: g, liberties: liberties.get(c)![idx], nonLiberties: nonLiberties.get(c)![idx],
        })));
    return result;
}

// ── legal move calculation ───────────────────────────────────────────────────

// Calculate legal PLACE moves given the board, adjacency matrix, turn list, and sorted history.
// A turn may offer more than one stone color (TurnInfo.stones); legality and
// captures are computed separately for every offered color, since which color
// is placed can change both (protected/friendly are per-stone-color). See
// LegalMovesData in types.ts for the returned shape.
function calculateLegalMoves(
    board: number[], adj: number[][], turnList: TurnInfo[],
    sortedSituations: AVLTree<Situation>, allowSuicide: boolean, koRule: KoRule,
    playerStonePlaceLimit: (number | null)[][], playerStonePlaceCnt: number[][],
    globalStonePlaceLimit: (number | null)[],
): LegalMovesData {
    // Private copy: this function mutates `board` in place below (early-capture
    // cleanup) and must never touch the caller's live array (e.g. BoardState.board)
    // before a move is actually committed via makeMove().
    board = board.slice();
    const cmpNonStrict = (a: Situation, b: Situation) => compareState(a, b, koRule, false);
    const N = board.length;
    const turnIdx = (sortedSituations.size - 1) % turnList.length;
    const turnInfo = turnList[turnIdx];
    const numStones = turnInfo.stones.length;
    let offeredStones: number[] = [];
    for (let s = 1; s <= numStones; s++) if (turnInfo.stones[s - 1] === 1) offeredStones.push(s);
    // A stone the mover has already placed as many times as their per-player
    // limit allows, or that's already been placed (by anyone) as many times as
    // its global limit allows, is treated exactly as if it were never offered
    // this turn - see GameConfig.playerStonePlaceLimit/globalStonePlaceLimit's
    // doc comments in types.ts. The global count isn't separately tracked - it's
    // derived by summing playerStonePlaceCnt[stone-1] across every player.
    offeredStones = offeredStones.filter(s => {
        const playerLimit = playerStonePlaceLimit[s - 1][turnInfo.player - 1];
        if (playerLimit !== null && playerStonePlaceCnt[s - 1][turnInfo.player - 1] >= playerLimit) return false;
        const globalLimit = globalStonePlaceLimit[s - 1];
        if (globalLimit !== null) {
            const globalCnt = playerStonePlaceCnt[s - 1].reduce((a, b) => a + b, 0);
            if (globalCnt >= globalLimit) return false;
        }
        return true;
    });
    const protectedStones = turnInfo.protected;
    const isProtected = (stone: number) => protectedStones[stone - 1] === 1;
    const friendlyStones = turnInfo.friendly;
    const isFriendly = (stone: number) => friendlyStones[stone - 1] === 1;
    const groupDict = groupLiberty(board, adj, N, friendlyStones);

    // Nodes captured by a pass: every non-protected zero-liberty group on the
    // board, regardless of color - this doesn't depend on which stone the
    // mover could have played (see LegalMovesData doc comment in types.ts for
    // why the opponent/self split below doesn't change this union).
    const passCapture = new Set<number>();
    for (const [color, groups] of groupDict) {
        if (isProtected(color)) continue;
        for (const { group, liberties } of groups) {
            if (liberties.size === 0) for (const node of group) passCapture.add(node);
        }
    }

    const captures: (Set<number> | null)[][] = Array.from({ length: numStones + 1 }, () => new Array(N).fill(null));
    const legalsForStone: Set<number>[] = Array.from({ length: numStones + 1 }, () => new Set<number>());
    const legalsForLocation: Set<number>[] = Array.from({ length: N }, () => new Set<number>());
    let placeLegals = 0;

    for (const nextPlayer of offeredStones) {
        // A group may be sitting at zero liberties because it was protected on
        // some earlier turn and was never actually removed. If it's no longer
        // protected and isn't this candidate's own color, it's captured by
        // whichever move actually gets played, regardless of position -
        // equivalent to (and cheaply derived from) passCapture, minus this
        // candidate's own color's contribution.
        const earlyOppCapture = new Set([...passCapture].filter(n => board[n] !== nextPlayer));

        const me    = groupDict.get(nextPlayer) ?? [];
        const other = [...groupDict.entries()]
            .filter(([c]) => c !== nextPlayer)
            .flatMap(([c, es]) => es.map(e => ({ color: c, ...e })));

        // `me` groups whose only remaining liberty (if any) is illusory: one of
        // their occupied neighbors is in earlyOppCapture, which will actually be
        // vacated once applied (see makeMove) - so such a group isn't really down
        // to just its current raw liberty count, and connecting to it is safe.
        const earlySelfLiberation = new Set<GroupEntry>();
        for (const g of me) {
            if ([...g.nonLiberties].some(n => earlyOppCapture.has(n))) earlySelfLiberation.add(g);
        }

        for (let i = 0; i < N; i++) {
            if (board[i] !== 0) continue;

            // whether the newly placed stone is adjacent to an empty node
            let iLib = false;
            for (let j = 0; j < N; j++) {
                if (adj[i][j] && board[j] === 0) { iLib = true; break; }
            }

            const nb = board.slice();
            nb[i] = nextPlayer;

            // Theorem: if any opponent group is captured, the move must be legal.
            // Proof: every remaining opponent group still has a liberty because either
            // it was not adjacent to i (liberty unchanged) or it was adjacent and now
            // has at least one liberty freed by the captured stones. Qed.
            // (Protected-colored groups are never captured: even when their last
            // liberty would be filled, they simply stay on the board at zero
            // liberties, so they never contribute to `capture`. Likewise, if the
            // mover's own color is friendly this turn, the new stone doesn't take
            // away anyone's liberty - it can't fill an opponent group's last
            // liberty, so no capture happens here at all.)
            let capture = false;
            const posCaptures = new Set<number>();
            if (!isFriendly(nextPlayer)) {
                for (const { color, group, liberties } of other) {
                    if (isProtected(color)) continue;
                    if (liberties.has(i) && liberties.size === 1) {
                        capture = true;
                        for (const node of group) { nb[node] = 0; posCaptures.add(node); }
                    }
                }
            }

            if (!iLib && !capture) {
                // the new stone may connect previous friendly groups;
                // if any such group has >1 liberty, the move is still legal
                // - or exactly 1 (about to be filled by this placement), if that
                // group is in earlySelfLiberation (it'll gain a real liberty back
                // once earlyOppCapture is applied, so it's not really down to
                // zero liberties even though this move fills its only raw one).
                let ok = false;
                for (const g of me) {
                    const { liberties } = g;
                    if (!liberties.has(i)) continue;
                    if (liberties.size > 1) { ok = true; break; }
                    if (liberties.size === 1 && earlySelfLiberation.has(g)) { ok = true; break; }
                }
                if (!ok) {
                    if (isProtected(nextPlayer)) {
                        // The mover's own color can't be removed either: the placed
                        // stone (and any connected group) simply stays on the board
                        // at zero liberties, and the move is legal regardless of
                        // allowSuicide - nothing to capture.
                    } else if (!allowSuicide) {
                        continue;
                    } else {
                        // Suicide: the new stone and any friendly group(s) it connects to (per
                        // the check above, none have another liberty) form one zero-liberty
                        // group - captured immediately, mirroring the opponent-capture case above.
                        nb[i] = 0;
                        posCaptures.add(i);
                        for (const { group, liberties } of me) {
                            if (!liberties.has(i)) continue;
                            for (const node of group) { nb[node] = 0; posCaptures.add(node); }
                        }
                    }
                }
            }

            // check whether the resulting board would repeat a previous state
            const ns: Situation = {
                board: nb,
                plyCount: sortedSituations.size, lenTurnList: turnList.length,
            };
            if (sortedSituations.has(ns, cmpNonStrict)) continue;

            // A leftover zero-liberty `me` group might additionally be freed by
            // THIS candidate's own captures - e.g. one of its occupied neighbors
            // is an opponent group this specific placement captures. If none of
            // its neighbors were freed (by this move's own captures or by
            // earlyOppCapture), it's still dead and is captured too.
            const earlySelfCaptures = new Set<number>();
            if (!isProtected(nextPlayer)) {
                for (const { group, liberties, nonLiberties } of me) {
                    if (liberties.size !== 0) continue;
                    const freed = [...nonLiberties].some(n => posCaptures.has(n) || earlyOppCapture.has(n));
                    if (!freed) for (const node of group) earlySelfCaptures.add(node);
                }
            }

            captures[nextPlayer][i] = new Set([...posCaptures, ...earlySelfCaptures, ...earlyOppCapture]);
            legalsForStone[nextPlayer].add(i);
            legalsForLocation[i].add(nextPlayer);
            placeLegals++;
        }
    }

    return new LegalMovesData(passCapture, captures, legalsForStone, legalsForLocation, placeLegals);
}

function cloneLegalMovesData(lmd: LegalMovesData): LegalMovesData {
    return new LegalMovesData(
        new Set(lmd.passCapture),
        lmd.captures.map(row => row.map(s => s ? new Set(s) : null)),
        lmd.legalsForStone.map(s => new Set(s)),
        lmd.legalsForLocation.map(s => new Set(s)),
        lmd.placeLegals,
    );
}

// ── BoardState ───────────────────────────────────────────────────────────────

export class BoardState {
    numStones:    number;
    numPlayers:   number;
    turnList:     TurnInfo[];
    playerStonePlaceLimit: (number | null)[][]; // [stone-1][player-1]; null = unlimited - see calculateLegalMoves
    // Total placements of each stone color allowed across ALL players combined
    // (unlike playerStonePlaceLimit); length numStones, indexed [stone-1]; null
    // = unlimited. No separate count field - derived in calculateLegalMoves by
    // summing playerStonePlaceCnt[stone-1] across every player.
    globalStonePlaceLimit: (number | null)[];
    stoneToPlayerMap: Record<number, Set<number>>;
    forcedPassOnly: boolean;
    scoreRule: ScoreRule;
    komi:         number[];
    koRule: KoRule;
    allowSuicide: boolean;
    maxPlies: number | null; // max plies before the game auto-ends (see makeMove); null = unlimited
    nextTurn:     TurnInfo; // the turnList entry for the upcoming ply - nextTurn.stones lists the offered stone colors (see STONE_MAP), nextTurn.player is the player whose turn is next
    board:        number[];
    pos:          number[][];
    adj:          number[][];
    N:            number;
    boardDimension: [[number, number], [number, number]];

    situations:       Situation[];
    sortedSituations: AVLTree<Situation>;

    // Only non-null once the game is over (see _refreshWinners).
    winners: number[] | null = null;
    // Map from ply index to the players (1-indexed) that resigned at that ply, in
    // resignation order. A resigned player may only pass and is excluded from scoring.
    resigns:              Map<number, number[]>  = new Map();
    // invariant: history.length === situations.length
    history: HistoryEntry[] = [];

    // turnList:
    //   the turn order: turnList[i] says which player plays on turn i (mod
    //   turnList.length) - the sole source of truth for turn ownership - see
    //   nextTurn/advanceResigned/makeMove. e.g.
    //   [{player:1,stones:[1,0]},{player:2,stones:[0,1]}] for standard two-player.
    //   Each entry's `stones` offers a set of stone colors the player may choose
    //   from that turn - see BoardState.makeMove.
    //   Each entry's `protected` (length numStones) marks stone colors that can
    //   never be removed from the board on that turn, and `friendly` (length
    //   numStones) marks stone colors that don't block anyone else's liberties
    //   that turn - see calculateLegalMoves and groupLiberty.
    // stoneToPlayerMap:
    //   map from stone type to the set of players it scores for, used only for
    //   scoring (_computeWinners) - independent of turnList, so a stone's scoring
    //   owner(s) need not be the same as whoever's turn places it. A stone maps to
    //   multiple players when they should each get its full point value (not
    //   split); a stone absent from the map, or mapped to an empty set, scores
    //   for no one.
    //   players are 1-indexed
    // forcedPassOnly: only allow pass when there are no legal PLACE moves
    // scoreRule:      how computePoints() converts ScoreData into per-stone points
    //                 ('stone' | 'territoryonly' | 'area')
    // koRule:         superko variant enforced by calculateLegalMoves()'s repeat-position
    //                 check ('positional' | 'situational')
    // allowSuicide:   whether a move that leaves the mover's own group with zero
    //                 liberties is legal (captures that own group immediately, rather
    //                 than being rejected)
    // board:          initial board state - array of length N; 0 = empty, positive = stone color
    constructor(
        numStones: number, numPlayers: number, turnList: TurnInfo[], playerStonePlaceLimit: (number | null)[][],
        globalStonePlaceLimit: (number | null)[],
        stoneToPlayerMap: Record<number, Set<number>>,
        forcedPassOnly: boolean, scoreRule: ScoreRule, komi: number[], koRule: KoRule,
        allowSuicide: boolean, maxPlies: number | null, board: number[], bc: BoardConfig,
    ) {
        assert(numStones > 0, `numStones must be > 0, got ${numStones}`);
        assert(turnList.length > 0, 'turnList must be non-empty');
        for (const t of turnList) {
            assert(t.player >= 1 && t.player <= numPlayers, `turnList player ${t.player} out of range [1, ${numPlayers}]`);
            assert(t.stones.length === numStones, `turnList stones length ${t.stones.length} must equal numStones ${numStones}`);
            assert(t.stones.every(p => p === 0 || p === 1), 'turnList stones values must be 0 or 1');
            assert(t.stones.some(p => p === 1), 'turnList stones must have at least one available stone');
            assert(t.protected.length === numStones, `turnList protected length ${t.protected.length} must equal numStones ${numStones}`);
            assert(t.protected.every(p => p === 0 || p === 1), 'turnList protected values must be 0 or 1');
            assert(t.friendly.length === numStones, `turnList friendly length ${t.friendly.length} must equal numStones ${numStones}`);
            assert(t.friendly.every(p => p === 0 || p === 1), 'turnList friendly values must be 0 or 1');
        }
        assert(playerStonePlaceLimit.length === numStones,
            `playerStonePlaceLimit length ${playerStonePlaceLimit.length} must equal numStones ${numStones}`);
        for (const row of playerStonePlaceLimit) {
            assert(row.length === numPlayers, `playerStonePlaceLimit sublist length ${row.length} must equal numPlayers ${numPlayers}`);
            assert(row.every(v => v === null || (Number.isInteger(v) && v >= 0)),
                'playerStonePlaceLimit values must be null or a non-negative integer');
        }
        assert(globalStonePlaceLimit.length === numStones,
            `globalStonePlaceLimit length ${globalStonePlaceLimit.length} must equal numStones ${numStones}`);
        assert(globalStonePlaceLimit.every(v => v === null || (Number.isInteger(v) && v >= 0)),
            'globalStonePlaceLimit values must be null or a non-negative integer');
        assert(maxPlies === null || (Number.isInteger(maxPlies) && maxPlies >= 1),
            'maxPlies must be null or a positive integer');
        assert(komi.every(k => k >= 0), 'komi values must be >= 0');
        this.numStones     = numStones;
        this.numPlayers    = numPlayers;
        this.turnList      = turnList;
        this.playerStonePlaceLimit = playerStonePlaceLimit;
        this.globalStonePlaceLimit = globalStonePlaceLimit;
        this.stoneToPlayerMap = stoneToPlayerMap;
        this.forcedPassOnly = forcedPassOnly;
        this.scoreRule     = scoreRule;
        this.komi          = komi;
        this.koRule        = koRule;
        this.allowSuicide  = allowSuicide;
        this.maxPlies      = maxPlies;
        this.nextTurn       = turnList[0];
        this.board         = board;
        this.pos           = bc.pos;
        this.adj           = bc.adj;
        this.N             = bc.N;
        this.boardDimension = bc.boardDimension;

        this.situations       = [];
        // sortedSituations is queried with two different comparators, both
        // koRule-dependent (see compareState) and so built fresh here rather
        // than shared module constants:
        //
        // 1. Ko-rule enforcement (has): two positions are "the same" when the
        //    board (and, under 'situational', the player-to-move) match,
        //    regardless of which ply they occurred on. This needs a
        //    non-strict comparator (compareState(..., koRule, false)),
        //    built on demand from koRule where it's needed - see
        //    calculateLegalMoves.
        //
        // 2. Withdrawal (remove): each entry must be identified exactly so that
        //    withdrawing a later ply never accidentally removes an earlier ply that
        //    happens to share the same board and player-to-move. PASS moves can
        //    produce such duplicates, because unlike PLACE moves, they are not
        //    constrained by the superko rule. If one board state occurs twice in the history,
        //    a non-strict comparator would treat these as equal, so insertNode
        //    silently ignores the second one, and removeNode then removes the
        //    first (wrong) entry. The strict comparator below breaks the tie
        //    with the actual plyCount, ensuring every entry is unique and
        //    remove always hits the right node.
        //
        // Resolution: build the tree with the strict comparator (correct
        // insert/remove); calculateLegalMoves separately builds a non-strict
        // comparator from the same koRule to pass explicitly to has() for
        // ko-rule queries. Searching with a coarser comparator in a tree
        // ordered by a finer one is valid because the non-strict comparator is
        // a prefix of the strict one (same leading key, for a given koRule):
        // nodes that are non-strictly equal are contiguous in the tree, so the
        // descent still reaches one of them in O(log n).
        this.sortedSituations = new AVLTree((a, b) => compareState(a, b, koRule, true));
        this._afterMove(
            NO_MOVE,
            Array.from({ length: numStones }, () => new Array(numPlayers).fill(0)),
            new Array(numPlayers).fill(0));
    }

    // Running count of stones placed so far, indexed [stone-1][player-1] (same
    // as playerStonePlaceLimit) - just the last history entry's snapshot, so
    // there's no separate live field to keep in sync: makeMove() computes each
    // new entry's count directly, and withdrawMove() "rewinds" it for free
    // simply by popping history. Compared against playerStonePlaceLimit in
    // calculateLegalMoves.
    playerStonePlaceCnt(): number[][] {
        return this.history[this.history.length - 1].playerStonePlaceCnt;
    }

    // Cumulative stones captured so far, indexed [player-1] - same "just the
    // last history entry's snapshot" pattern as playerStonePlaceCnt() above:
    // makeMove() computes each new entry's count directly, and withdrawMove()
    // rewinds it for free simply by popping history.
    captureCount(): number[] {
        return this.history[this.history.length - 1].score.captureCount;
    }

    // Reconstructs the final BoardState of a finished game by replaying its moves
    // from an empty board. No advanceResigned() calls during replay - resignation
    // auto-passes are already explicit entries in fg.moves (each auto-pass in the
    // original game called makeMove(null), captured in moveInfos), so re-deriving
    // them here would double-apply. Resignations themselves aren't moves, so they're
    // the one piece of side-channel information FinishedGame carries alongside moves.
    static fromFinishedGame(fg: FinishedGame, bc: BoardConfig): BoardState {
        const state = new BoardState(
            fg.config.numStones, fg.config.numPlayers, fg.config.turnList, fg.config.playerStonePlaceLimit,
            fg.config.globalStonePlaceLimit,
            fg.config.stoneToPlayerMap, fg.config.forcedPassOnly, fg.config.scoreRule, fg.config.komi, fg.config.koRule,
            fg.config.allowSuicide, fg.config.maxPlies, new Array(bc.N).fill(0), bc,
        );
        for (let ply = 0; ply < fg.moves.length; ply++) {
            for (const player of fg.resigns.get(ply) ?? []) state.resign(player);
            state.makeMove(fg.moves[ply].pos, fg.moves[ply].stone ?? undefined);
        }
        for (const player of fg.resigns.get(fg.moves.length) ?? []) state.resign(player);
        return state;
    }

    // Appends a new Situation (and, in lockstep, a new HistoryEntry - see legalMovesData()) for
    // the ply that was just played. moveInfo is the move that produced this ply (NO_MOVE for the
    // constructor's genesis call).
    private _afterMove(moveInfo: MoveInfo, playerStonePlaceCnt: number[][], captureCount: number[]) {
        const sit: Situation = {
            board: this.board.slice(),
            plyCount: this.situations.length,
            lenTurnList: this.turnList.length,
        };
        this.situations.push(sit);
        this.sortedSituations.insert(sit);
        // Takes playerStonePlaceCnt/moveInfo.allPassed as parameters (rather than reading them
        // back via this.playerStonePlaceCnt()/this.lastMove()) since this.history doesn't have
        // this ply's entry yet - it's only pushed once, below, alongside score/legalMoves.
        const legalMoves = this._computeLegalMoves(moveInfo.allPassed, playerStonePlaceCnt);
        // _countScore() is pure/board-only (like TS's territory flood-fill),
        // so captureCount - a running total, not derivable from the board -
        // is merged in from the caller, the same way playerStonePlaceCnt is.
        const score = { ...this._countScore(), captureCount };
        this.history.push({ moveInfo, legalMoves, score, playerStonePlaceCnt });
        this._refreshWinners();
    }

    // Pure: computes the legal-move table for the current position (this.board/sortedSituations/etc.).
    // Gated on the mover's allPassed flag rather than gameOver(): allPassed only reflects the last
    // move's own intrinsic, never-retroactively-mutated flag (see MoveInfo.allPassed), so this
    // method's result for a given ply is fixed forever from the moment it's computed - no caller
    // ever needs to recompute a cached history entry, not even the last one. Trade-off: a
    // game that ends via resign()/maxPlies (gameOver() true, allPassed false, since neither
    // retroactively touches history) keeps reporting the position's actual board-legal moves
    // here rather than an empty/terminal table - harmless, since makeMove() independently blocks
    // further moves via its own gameOver() check regardless of what this returns.
    private _computeLegalMoves(allPassed: boolean, playerStonePlaceCnt: number[][]): LegalMovesData {
        return allPassed
            ? new LegalMovesData(
                new Set(),
                Array.from({ length: this.numStones + 1 }, () => new Array(this.board.length).fill(null)),
                Array.from({ length: this.numStones + 1 }, () => new Set<number>()),
                Array.from({ length: this.board.length }, () => new Set<number>()),
                0,
              )
            : calculateLegalMoves(
                this.board, this.adj, this.turnList, this.sortedSituations, this.allowSuicide, this.koRule,
                this.playerStonePlaceLimit, playerStonePlaceCnt, this.globalStonePlaceLimit);
    }

    // Pure: computes the score (stone count + territory, both per stone type)
    // for the current board. Territory is found by flood-filling each maximal
    // connected region of empty nodes (same multi-frontier BFS pattern as
    // groupLiberty): a region belongs to a stone type only if every node
    // bordering it is that same type; regions bordering zero or several
    // distinct types are neutral (dame) and score nobody. Deliberately
    // doesn't produce captureCount - that's a running total across the whole
    // game, not derivable from the current board - so callers merge it in
    // separately (see _afterMove).
    private _countScore(): Omit<ScoreData, 'captureCount'> {
        const stoneCount: Record<number, number> = {};
        const territory: Record<number, number> = {};
        for (let p = 1; p <= this.numStones; p++) {
            stoneCount[p] = this.board.filter(v => v === p).length;
            territory[p] = 0;
        }

        const territoryOwner = new Array<number>(this.N).fill(0);
        const visited = new Array<boolean>(this.N).fill(false);
        for (let i = 0; i < this.N; i++) {
            if (this.board[i] !== 0 || visited[i]) continue;
            const region: number[] = [i];
            visited[i] = true;
            const borderStones = new Set<number>();
            let st: number[] = [i];
            while (st.length > 0) {
                const nxt: number[] = [];
                for (const node of st) {
                    for (let j = 0; j < this.N; j++) {
                        if (!this.adj[node][j]) continue;
                        if (this.board[j] !== 0) { borderStones.add(this.board[j]); continue; }
                        if (!visited[j]) { visited[j] = true; region.push(j); nxt.push(j); }
                    }
                }
                st = nxt;
            }
            if (borderStones.size === 1) {
                const stone = [...borderStones][0];
                territory[stone] += region.length;
                for (const node of region) territoryOwner[node] = stone;
            }
        }

        return { stoneCount, territory, territoryOwner };
    }

    // Pure: converts ScoreData into a per-stone-type point map under the given
    // scoring rule - 'stone' counts stones on the board only, 'territoryonly'
    // counts territory only, 'area' counts stones + territory (Chinese-style;
    // today's default), 'territory' counts territory only here too (real-world
    // Japanese-style scoring is territory + prisoners, but captures are
    // player-indexed, not stone-indexed like this map - see
    // ScoreData.captureCount and _computeWinners, which folds captures in
    // separately at the player-aggregation layer, the same way komi is).
    static computePoints(rule: ScoreRule, score: ScoreData): Record<number, number> {
        const points: Record<number, number> = {};
        for (const key of Object.keys(score.stoneCount)) {
            const s = Number(key);
            const stoneCount = score.stoneCount[s]!;
            const territory  = score.territory[s] ?? 0;
            points[s] = rule === 'stone'                                     ? stoneCount
                      : rule === 'territoryonly' || rule === 'territory'     ? territory
                      :                                                        stoneCount + territory;
        }
        return points;
    }

    // Pure: computes winners from the points (under this.scoreRule) of the last
    // entry in `history`, plus each player's komi. Only meaningful once the
    // game is over (see _refreshWinners).
    private _computeWinners(history: HistoryEntry[]): number[] {
        const points = BoardState.computePoints(this.scoreRule, history[history.length - 1].score);
        // Resigned players are excluded from scoring and cannot be winners.
        const players = Array.from({ length: this.numPlayers }, (_, i) => i + 1)
            .filter(p => !this.resignedPlayers.includes(p));
        const playerCount: Record<number, number> = {};
        for (const p of players) playerCount[p] = this.komi[p - 1] ?? 0;
        if (this.scoreRule === 'territory') {
            const captureCount = history[history.length - 1].score.captureCount;
            for (const p of players) playerCount[p] += captureCount[p - 1] ?? 0;
        }
        for (const [stone, count] of Object.entries(points)) {
            for (const player of this.stoneToPlayerMap[Number(stone)] ?? []) {
                if (playerCount[player] === undefined) continue;   // resigned player's points don't score
                playerCount[player] += count;
            }
        }
        const max = players.length > 0 ? Math.max(...players.map(p => playerCount[p]!)) : 0;
        return players.filter(p => playerCount[p] === max);
    }

    // Refreshes `winners`: only non-null once the game is over, null otherwise
    // (e.g. after withdrawMove() un-ends a finished game).
    private _refreshWinners() {
        this.winners = this.gameOver() ? this._computeWinners(this.history) : null;
    }

    // history is never empty (the constructor seeds a genesis entry with moveInfo = NO_MOVE),
    // so this always resolves without a fallback.
    lastMove(): MoveInfo {
        return this.history[this.history.length - 1].moveInfo;
    }

    // Current score (stone count + territory, both per stone type), i.e. the
    // last history entry's.
    score(): ScoreData {
        return this.history[this.history.length - 1].score;
    }

    // Real per-ply moves only (unlike history/situations, which also include the genesis
    // ply-0 entry before any move) - derived from history, skipping history[0]'s NO_MOVE
    // sentinel. No separate live field to keep in sync, same pattern as score()/etc. above.
    moveInfos(): MoveInfo[] {
        return this.history.slice(1).map(e => e.moveInfo);
    }

    // True iff the game has ended, via any of three independent conditions:
    // too few non-resigned players remain, maxPlies has been reached, or the
    // last move completed a full round of consecutive passes. The first two
    // are checked live against always-current state (resignedPlayers/
    // situations.length) rather than stamped onto a move, so - unlike the old
    // design - nothing ever needs retroactive fixing up after resign() or
    // withdrawMove() (see _computeLegalMoves()).
    gameOver(): boolean {
        if (this.numPlayers - this.resignedPlayers.length <= 1) return true;
        if (this.maxPlies !== null && this.situations.length - 1 >= this.maxPlies) return true;
        return this.lastMove().allPassed;
    }

    // Legal-move table for the position just reached, i.e. history's last entry's legalMoves -
    // just the last history entry's snapshot, same "no separate live field to keep in sync"
    // pattern as score()/playerStonePlaceCnt()/captureCount() above.
    legalMovesData(): LegalMovesData {
        return this.history[this.history.length - 1].legalMoves;
    }

    // Returns true iff there are no legal PLACE moves for the current player.
    noTradLegal(): boolean { return this.legalMovesData().placeLegals === 0; }

    // The list of legal (stone, position) PLACE move pairs.
    legalPlaceList(): { pos: number; stone: number }[] {
        const result: { pos: number; stone: number }[] = [];
        this.legalMovesData().legalsForLocation.forEach((stones, pos) => {
            for (const stone of stones) result.push({ pos, stone });
        });
        return result;
    }

    get resignedPlayers(): number[] { return [...this.resigns.values()].flat(); }

    // Mark a player (1-indexed) as resigned: thereafter they may only pass and are
    // excluded from scoring. Recomputes winners immediately.
    resign(player: number) {
        if (this.resignedPlayers.includes(player)) return;
        const ply = this.situations.length - 1;
        const list = this.resigns.get(ply);
        if (list) list.push(player);
        else this.resigns.set(ply, [player]);
        // Resigning may end the game (too few non-resigned players remain) or
        // change who counts as a winner if it's already over - gameOver()
        // checks resignedPlayers live, so nothing needs stamping onto a move.
        this._refreshWinners();
    }

    // Auto-pass on behalf of any resigned player whose turn it is, until a non-resigned
    // player is to move or the game ends.
    advanceResigned(): void {
        while (!this.gameOver()) {
            if (!this.resignedPlayers.includes(this.nextTurn.player)) break;
            if (!this.makeMove(null)) break;
        }
    }

    // Sets nextTurn to the turnList entry for the upcoming ply.
    private _advanceTurn(): void {
        this.nextTurn = this.turnList[this.situations.length % this.turnList.length];
    }

    // Make a move. Pass null for a pass move (no stone needed). For a PLACE
    // move, `stone` selects which offered color to play; it may be omitted
    // only when the current turn offers exactly one stone (the unambiguous
    // case) - otherwise the caller must choose. Returns true if the move was legal.
    // Fields are updated immediately after each move.
    makeMove(k: number | null, stone?: number): boolean {
        // No computation needed here: history's last entry is never
        // stale (see _computeLegalMoves()), so there's nothing to refresh.
        if (this.gameOver()) return false;
        // A resigned player may only pass, and always may (ignoring forcedPassOnly).
        const resigned = this.resignedPlayers.includes(this.nextTurn.player);
        if (resigned && k !== null) return false;
        let consecutivePasses = 0;
        if (k === null) {
            if (!resigned && this.forcedPassOnly && !this.noTradLegal()) return false;
            consecutivePasses = this.lastMove().consecutivePasses + 1;
            if (consecutivePasses >= this.turnList.length) {
                this._advanceTurn();
                const moveInfo: MoveInfo = { moveType: MoveType.PASS, pos: null, stone: null, captures: [], consecutivePasses, allPassed: true };
                this._afterMove(moveInfo, this.playerStonePlaceCnt().map(row => row.slice()), this.captureCount().slice()); return true;
            }
        } else {
            if (this.noTradLegal()) return false;
            if (stone === undefined) {
                const offeredIdx = this.nextTurn.stones.reduce<number[]>((acc, v, i) => (v === 1 ? [...acc, i] : acc), []);
                if (offeredIdx.length !== 1) return false;
                stone = offeredIdx[0] + 1;
            }
            if (this.nextTurn.stones[stone - 1] !== 1) return false;
            if (this.legalMovesData().captures[stone][k] === null) return false;
        }

        // A PLACE move's captures are fully precomputed (see calculateLegalMoves);
        // a pass's captures are simply passCapture.
        const captures = k === null ? this.legalMovesData().passCapture : this.legalMovesData().captures[stone!][k]!;
        const nb = this.board.slice();
        if (k !== null) nb[k] = stone!;
        for (const c of captures) nb[c] = 0;
        this.board = nb;
        // Must read this.nextTurn.player (the mover) before _advanceTurn() below
        // changes it to the upcoming ply's player.
        const newPlayerStonePlaceCnt = this.playerStonePlaceCnt().map(row => row.slice());
        if (k !== null) newPlayerStonePlaceCnt[stone! - 1][this.nextTurn.player - 1]++;
        const newCaptureCount = this.captureCount().slice();
        newCaptureCount[this.nextTurn.player - 1] += captures.size;
        this._advanceTurn();
        const moveType = k === null ? MoveType.PASS : MoveType.PLACE;
        // allPassed is always false here: this branch is only reached for a PLACE move or a
        // non-round-completing PASS (the round-completing case returns early above) - maxPlies
        // ending the game is handled entirely by gameOver()'s own live check, no stamping needed.
        const moveInfo: MoveInfo = { moveType, pos: k, stone: k === null ? null : stone!, captures: [...captures], consecutivePasses: k === null ? consecutivePasses : 0, allPassed: false };
        this._afterMove(moveInfo, newPlayerStonePlaceCnt, newCaptureCount);
        return true;
    }

    // Withdraw one move. Fields are updated immediately.
    withdrawMove() {
        if (this.situations.length <= 1) return;
        const sit = this.situations[this.situations.length - 1];
        this.situations.pop();
        this.board = this.situations[this.situations.length - 1].board.slice();
        // playerStonePlaceCnt() reads straight from the (now-popped-back-to)
        // last history entry, so it's already rewound - nothing more to do here.
        this.nextTurn = this.turnList[(this.situations.length - 1) % this.turnList.length];
        this.sortedSituations.remove(sit);
        // No recompute needed: history's entries never go stale (see
        // _computeLegalMoves()) - gameOver()'s resignedPlayers/maxPlies checks are
        // live/derived rather than stamped onto MoveInfo, and allPassed is a
        // per-ply-intrinsic fact that's never retroactively mutated, so the entry
        // this pop uncovers is already exactly correct. A resignation is permanent
        // (never un-resigned by withdrawMove) and gameOver() reflects that live too.
        this.history.pop();
        this._refreshWinners();
    }

    // Make a uniformly random legal move (or pass if no PLACE moves exist).
    randomMove() {
        if (this.gameOver()) return;
        const resigned = this.resignedPlayers.includes(this.nextTurn.player);
        const legals = this.legalPlaceList();
        const choice = (!resigned && legals.length > 0)
            ? legals[Math.floor(Math.random() * legals.length)] : null;
        const success = choice === null ? this.makeMove(null) : this.makeMove(choice.pos, choice.stone);
        assert(success, 'random move failed');
    }

    // Monte-Carlo evaluation of the current position over `n` random playouts.
    // Returns a map from player index to number of wins; a tie among k players counts as 1/k each.
    randomEvaluate(n: number): Record<number, number> {
        const players = Array.from({ length: this.numPlayers }, (_, i) => i + 1);
        const wins: Record<number, number> = {};
        for (const p of players) wins[p] = 0;
        for (let i = 0; i < n; i++) {
            const copy = this._copy();
            while (!copy.gameOver()) copy.randomMove();
            // gameOver() just became true, so winners is guaranteed defined.
            const val = 1 / copy.winners!.length;
            for (const w of copy.winners!) wins[w] += val;
        }
        return wins;
    }

    private _copy(): BoardState {
        const c = new BoardState(
            this.numStones, this.numPlayers, this.turnList.map(t => ({...t, stones: [...t.stones], protected: [...t.protected], friendly: [...t.friendly]})),
            this.playerStonePlaceLimit.map(row => [...row]),
            [...this.globalStonePlaceLimit],
            Object.fromEntries(Object.entries(this.stoneToPlayerMap).map(([k, v]) => [k, new Set(v)])),
            this.forcedPassOnly, this.scoreRule, this.komi, this.koRule, this.allowSuicide, this.maxPlies, this.board.slice(),
            { pos: this.pos, adj: this.adj, N: this.N, boardDimension: this.boardDimension },
        );
        // replace history/situations with deep copies
        c.situations       = this.situations.map(e => ({ ...e, board: e.board.slice() }));
        c.sortedSituations = this.sortedSituations.clone(e => ({ ...e, board: e.board.slice() }));
        // c.playerStonePlaceCnt() reads from c.history, already deep-cloned below.
        // Point into c's own (already deep-cloned) turnList, not this.nextTurn -
        // otherwise the copy would alias the original's turnList entry.
        c.nextTurn = c.turnList[(this.situations.length - 1) % this.turnList.length];
        c.winners       = this.winners ? [...this.winners] : null;
        c.resigns = new Map([...this.resigns.entries()].map(([k, v]) => [k, [...v]]));
        c.history = this.history.map(e => ({
            moveInfo: { ...e.moveInfo },
            legalMoves: cloneLegalMovesData(e.legalMoves),
            score: e.score,
            playerStonePlaceCnt: e.playerStonePlaceCnt.map(row => row.slice()),
        }));
        return c;
    }

    getView(): BoardView {
        const lm = this.lastMove();
        return {
            N: this.N,
            pos: this.pos,
            boardDimension: this.boardDimension,
            numStones: this.numStones,
            numPlayers: this.numPlayers,
            turnList: this.turnList,
            playerStonePlaceLimit: this.playerStonePlaceLimit,
            globalStonePlaceLimit: this.globalStonePlaceLimit,
            stoneToPlayerMap: this.stoneToPlayerMap,
            forcedPassOnly: this.forcedPassOnly,
            scoreRule: this.scoreRule,
            komi: this.komi,
            koRule: this.koRule,
            allowSuicide: this.allowSuicide,
            maxPlies: this.maxPlies,
            nextTurn: this.nextTurn,
            lastMove: lm,
            moveInfos: this.moveInfos(),
            score: this.score(),
            winners: this.winners ? [...this.winners] : null,
            resignedPlayers: [...this.resignedPlayers],
            plyCount: this.situations.length - 1,
            situations: this.situations,
            history: this.history,
            gameOver: this.gameOver(),
            // A resigned player may always pass; otherwise the forced-pass-only rule applies.
            passEnabled: this.resignedPlayers.includes(this.nextTurn.player)
                || !this.forcedPassOnly || this.noTradLegal(),
        };
    }
}

export { MoveType, STONE_MAP };
