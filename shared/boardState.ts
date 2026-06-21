import type { BoardConfig } from './boardConfig.js';
import { MoveType, STONE_MAP } from './types.js';
import type { MoveInfo, HistoryEntry, BoardView } from './types.js';
import { AVLTree } from './avlTree.js';

function assert(cond: boolean, msg: string): asserts cond {
    if (!cond) throw new Error(`Assertion failed: ${msg}`);
}

// ── comparison / search ──────────────────────────────────────────────────────

// Returns: s1 > s2 → 1, s1 == s2 → 0, s1 < s2 → -1.
// With strict = true, only returns 0 if plyCount is also equal.
function compareState(s1: HistoryEntry, s2: HistoryEntry, strict = false): number {
    const { board: b1, plyCount: t1, lenTurnList: ltl } = s1;
    const { board: b2, plyCount: t2 } = s2;
    if (t1 % ltl < t2 % ltl) return -1;
    if (t1 % ltl > t2 % ltl) return  1;
    for (let i = 0; i < b1.length; i++) {
        if (b1[i] < b2[i]) return -1;
        if (b1[i] > b2[i]) return  1;
    }
    if (!strict) return 0;
    if (t1 < t2) return -1;
    if (t1 > t2) return  1;
    return 0;
}

const cmpNonStrict = (a: HistoryEntry, b: HistoryEntry) => compareState(a, b, false);
const cmpStrict    = (a: HistoryEntry, b: HistoryEntry) => compareState(a, b, true);

// ── group / liberty ──────────────────────────────────────────────────────────

interface GroupEntry { group: number[]; liberties: Set<number>; }

// Returns a map from stone color to list of (group, liberties) pairs:
//   group     - list of node indices belonging to the group
//   liberties - set of empty node indices adjacent to the group
function groupLiberty(board: number[], adj: number[][], N: number)
    : Map<number, GroupEntry[]>
{
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

    // Compute liberties: liberties[color][gid] = set of adjacent empty nodes
    const liberties = new Map<number, Set<number>[]>();
    for (const [c, grps] of groups)
        liberties.set(c, grps.map(() => new Set<number>()));

    for (let i = 0; i < N; i++) {
        if (affColor[i] !== 0) continue;
        for (let j = 0; j < N; j++) {
            if (!adj[i][j] || affColor[j] === 0) continue;
            liberties.get(affColor[j])![affGid[j]].add(i);
        }
    }

    for (const [c, grps] of groups)
        for (let idx = 0; idx < grps.length; idx++)
            assert(liberties.get(c)![idx].size > 0, `color ${c} group ${idx} has no liberties`);

    const result = new Map<number, GroupEntry[]>();
    for (const [c, grps] of groups)
        result.set(c, grps.map((g, idx) => ({ group: g, liberties: liberties.get(c)![idx] })));
    return result;
}

// ── legal move calculation ───────────────────────────────────────────────────

// Calculate legal PLACE moves given the board, adjacency matrix, turn list, and sorted history.
// Returns an array of length N:
//   null      - position occupied, or playing there is illegal (suicide without capture,
//               or would repeat a previous board state)
//   Set<number> - playing here is legal; the set contains the captured node indices
//                 (empty set means no captures)
function calculateLegalMoves(
    board: number[], adj: number[][], turnStoneList: number[],
    sortedHistory: AVLTree<HistoryEntry>,
): (Set<number> | null)[] {
    const N = board.length;
    const legal: (Set<number> | null)[] = new Array(N).fill(null);
    const groupDict = groupLiberty(board, adj, N);
    const nextPlayer     = turnStoneList[(sortedHistory.size - 1) % turnStoneList.length];
    const nextNextPlayer = turnStoneList[ sortedHistory.size      % turnStoneList.length];
    const me    = groupDict.get(nextPlayer) ?? [];
    const other = [...groupDict.entries()]
        .filter(([c]) => c !== nextPlayer)
        .flatMap(([, es]) => es);

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
        let capture = false;
        const captures = new Set<number>();
        for (const { group, liberties } of other) {
            if (liberties.has(i) && liberties.size === 1) {
                capture = true;
                for (const node of group) { nb[node] = 0; captures.add(node); }
            }
        }

        if (!iLib && !capture) {
            // the new stone may connect previous friendly groups;
            // if any such group has >1 liberty, the move is still legal
            let ok = false;
            for (const { liberties } of me) {
                if (liberties.has(i) && liberties.size > 1) { ok = true; break; }
            }
            if (!ok) continue;
        }

        // check whether the resulting board would repeat a previous state
        const ns: HistoryEntry = {
            board: nb, nextPlayer: nextNextPlayer,
            plyCount: sortedHistory.size, lenTurnList: turnStoneList.length,
        };
        if (sortedHistory.has(ns, cmpNonStrict)) continue;

        legal[i] = captures;
    }
    return legal;
}

// ── BoardState ───────────────────────────────────────────────────────────────

export class BoardState {
    numStones:    number;
    numPlayers:   number;
    turnStoneList: number[];
    stoneToPlayerMap: Record<number, number>;
    forcedPassOnly: boolean;
    nextPlayer:   number;
    board:        number[];
    pos:          number[][];
    adj:          number[][];
    N:            number;
    boardDimension: [[number, number], [number, number]];

    history:       HistoryEntry[];
    sortedHistory: AVLTree<HistoryEntry>;
    lastMoves:     MoveInfo[];

    stoneCount:           Record<number, number> = {};
    winners:              number[]               = [];
    legalMovesWithTake:   (Set<number> | null)[] = [];
    // invariant: legalMoveHistory.length === history.length
    legalMoveHistory: (Set<number> | null)[][] = [];

    // turnStoneList:
    //   the sequence of stone colors placed each turn, e.g. [1, 2] for standard two-player
    //   stone types are 1-indexed
    // stoneToPlayerMap:
    //   map from stone type to player
    //   players are 1-indexed
    // forcedPassOnly: only allow pass when there are no legal PLACE moves
    // board:          initial board state - array of length N; 0 = empty, positive = stone color
    constructor(
        numStones: number, numPlayers: number, turnStoneList: number[], stoneToPlayerMap: Record<number, number>,
        forcedPassOnly: boolean, board: number[], bc: BoardConfig,
    ) {
        assert(numStones > 0, `numStones must be > 0, got ${numStones}`);
        assert(turnStoneList.length > 0, 'turnStoneList must be non-empty');
        for (const p of turnStoneList)
            assert(p >= 1 && p <= numStones, `turnStoneList entry ${p} out of range [1, ${numStones}]`);
        this.numStones     = numStones;
        this.numPlayers    = numPlayers;
        this.turnStoneList = turnStoneList;
        this.stoneToPlayerMap = stoneToPlayerMap;
        this.forcedPassOnly = forcedPassOnly;
        this.nextPlayer    = turnStoneList[0];
        this.board         = board;
        this.pos           = bc.pos;
        this.adj           = bc.adj;
        this.N             = bc.N;
        this.boardDimension = bc.boardDimension;

        this.history       = [];
        // sortedHistory serves two roles that require different comparators:
        //
        // 1. Ko-rule enforcement (has): two positions are "the same" when the board
        //    and the player-to-move match, regardless of which ply they occurred on.
        //    This needs cmpNonStrict, which compares only (plyCount%turnListLen, board).
        //
        // 2. Retraction (remove): each entry must be identified exactly so that
        //    retracting a later ply never accidentally removes an earlier ply that
        //    happens to share the same board and player-to-move. PASS moves can
        //    produce such duplicates, because unlike PLACE moves, they are not
        //    constrained by the superko rule. If one board state occurs twice in the history,
        //    cmpNonStrict would treat these as equal, so insertNode silently
        //    ignores the second one, and removeNode then removes the first
        //    (wrong) entry. cmpStrict breaks the tie with the actual plyCount,
        //    ensuring every entry is unique and remove always hits the right node.
        //
        // Resolution: build the tree with cmpStrict (correct insert/remove), and
        // pass cmpNonStrict explicitly to has() for ko-rule queries (see
        // calculateLegalMoves). Searching with a coarser comparator in a tree
        // ordered by a finer one is valid because cmpNonStrict is a prefix of
        // cmpStrict: nodes that are non-strictly equal are contiguous in the tree,
        // so the descent still reaches one of them in O(log n).
        this.sortedHistory = new AVLTree(cmpStrict);
        this.lastMoves     = [];
        this._addToHistoryAndAfterMove();
    }

    private _addToHistoryAndAfterMove() {
        const s: HistoryEntry = {
            board: this.board.slice(),
            nextPlayer: this.nextPlayer,
            plyCount: this.history.length,
            lenTurnList: this.turnStoneList.length,
        };
        this.history.push(s);
        this.sortedHistory.insert(s);
        this._afterMove();
        this.legalMoveHistory.push(this.legalMovesWithTake);
    }

    private _afterMove() {
        if (this.lastMove().moveType === MoveType.GAMEOVER)
            this.legalMovesWithTake = new Array(this.board.length).fill(null);
        else
            this.legalMovesWithTake = calculateLegalMoves(
                this.board, this.adj, this.turnStoneList, this.sortedHistory);
        this._countStones();
    }

    private _countStones() {
        this.stoneCount = {};
        for (let p = 1; p <= this.numStones; p++)
            this.stoneCount[p] = this.board.filter(v => v === p).length;
        const players = Array.from({ length: this.numPlayers }, (_, i) => i + 1);
        const playerCount: Record<number, number> = {};
        for (const p of players) playerCount[p] = 0;
        for (const [stone, count] of Object.entries(this.stoneCount))
            playerCount[this.stoneToPlayerMap[Number(stone)]!] += count;
        const max = Math.max(...Object.values(playerCount));
        this.winners = players.filter(p => playerCount[p] === max);
    }

    lastMove(): MoveInfo {
        return this.lastMoves.length > 0
            ? this.lastMoves[this.lastMoves.length - 1]
            : { moveType: MoveType.NOMOVE, pos: null, captures: [], passedPlayers: new Set() };
    }

    // Returns true iff there are no legal PLACE moves for the current player.
    noTradLegal(): boolean { return this.legalMovesWithTake.every(m => m === null); }

    // The list of legal PLACE move positions (node indices).
    legalMoveList(): number[] {
        return this.legalMovesWithTake
            .map((m, i) => m !== null ? i : -1)
            .filter(i => i >= 0);
    }

    // Make a move. Pass null for a pass move. Returns true if the move was legal.
    // Fields are updated immediately after each move.
    makeMove(k: number | null): boolean {
        if (this.lastMove().moveType === MoveType.GAMEOVER) {
            this._afterMove(); return false;
        }
        if (k === null) {
            if (this.forcedPassOnly && !this.noTradLegal()) return false;
            const passed = new Set(this.lastMove().passedPlayers);
            passed.add(this.nextPlayer);
            if (passed.size >= new Set(this.turnStoneList).size) {
                this.nextPlayer = this.turnStoneList[this.history.length % this.turnStoneList.length];
                this.lastMoves.push({ moveType: MoveType.GAMEOVER, pos: null, captures: [], passedPlayers: passed });
                this._addToHistoryAndAfterMove(); return true;
            }
            this.nextPlayer = this.turnStoneList[this.history.length % this.turnStoneList.length];
            this.lastMoves.push({ moveType: MoveType.PASS, pos: null, captures: [], passedPlayers: passed });
            this._addToHistoryAndAfterMove();
            return true;
        }
        if (this.noTradLegal()) return false;
        const captures = this.legalMovesWithTake[k];
        if (captures === null) return false;
        const nb = this.board.slice();
        nb[k] = this.nextPlayer;
        for (const c of captures) nb[c] = 0;
        this.board = nb;
        this.nextPlayer = this.turnStoneList[this.history.length % this.turnStoneList.length];
        this.lastMoves.push({ moveType: MoveType.PLACE, pos: k, captures: [...captures], passedPlayers: new Set() });
        this._addToHistoryAndAfterMove();
        return true;
    }

    // Retract one move. Fields are updated immediately.
    retractMove() {
        if (this.history.length <= 1) return;
        const state = this.history[this.history.length - 1];
        this.history.pop();
        this.lastMoves.pop();
        this.board      = this.history[this.history.length - 1].board.slice();
        this.nextPlayer = this.history[this.history.length - 1].nextPlayer;
        this.sortedHistory.remove(state);
        this.legalMoveHistory.pop();
        this._afterMove();
    }

    // Make a uniformly random legal move (or pass if no PLACE moves exist).
    randomMove() {
        if (this.lastMove().moveType === MoveType.GAMEOVER) return;
        const legals = this.legalMoveList();
        const k = legals.length > 0 ? legals[Math.floor(Math.random() * legals.length)] : null;
        const success = this.makeMove(k);
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
            while (copy.lastMove().moveType !== MoveType.GAMEOVER) copy.randomMove();
            const val = 1 / copy.winners.length;
            for (const w of copy.winners) wins[w] += val;
        }
        return wins;
    }

    private _copy(): BoardState {
        const c = new BoardState(
            this.numStones, this.numPlayers, [...this.turnStoneList], {...this.stoneToPlayerMap},
            this.forcedPassOnly, this.board.slice(),
            { pos: this.pos, adj: this.adj, N: this.N, boardDimension: this.boardDimension },
        );
        // replace history with deep copies
        c.history       = this.history.map(e => ({ ...e, board: e.board.slice() }));
        c.sortedHistory = this.sortedHistory.clone(e => ({ ...e, board: e.board.slice() }));
        c.lastMoves     = this.lastMoves.map(m => ({ ...m, passedPlayers: new Set(m.passedPlayers) }));
        c.nextPlayer    = this.nextPlayer;
        c.stoneCount    = { ...this.stoneCount };
        c.winners       = [...this.winners];
        c.legalMovesWithTake = this.legalMovesWithTake.map(s => s ? new Set(s) : null);
        c.legalMoveHistory   = this.legalMoveHistory.map(row => row.map(s => s ? new Set(s) : null));
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
            turnStoneList: this.turnStoneList,
            stoneToPlayerMap: this.stoneToPlayerMap,
            forcedPassOnly: this.forcedPassOnly,
            nextPlayer: this.nextPlayer,
            lastMove: lm,
            stoneCount: { ...this.stoneCount },
            winners: [...this.winners],
            plyCount: this.history.length - 1,
            history: this.history,
            legalMoves: this.legalMovesWithTake,
            legalMoveHistory: this.legalMoveHistory,
            gameOver: lm.moveType === MoveType.GAMEOVER,
            passEnabled: !this.forcedPassOnly || this.noTradLegal(),
        };
    }
}

export { MoveType, STONE_MAP };
