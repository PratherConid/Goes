import { BoardState, MoveType, STONE_MAP } from '@shared/boardState.js';
import type { BoardView } from '@shared/types.js';
import type { BoardConfig } from '@shared/boardConfig.js';
import {
    PrescribedBoard, PrescribedBoardMap, PrescribedBoardFns,
} from '@shared/boardConfig.js';
import { ServerConnection, type RequestHandle } from './serverConnection.js';

// Single persistent WebSocket connection to the main server, shared by the
// EngineManager (AI proxy) and the online-game commands.
const conn = new ServerConnection();

interface OnlineGameConfig {
    boardType: string;
    boardArgs: number[];
    numStones: number;
    numPlayers: number;
    turnStoneList: number[];
    stoneToPlayerMap: Record<number, number>;
    forcedPassOnly: boolean;
}

interface OnlineStateResponse {
    status: 'waiting' | 'playing' | 'finished';
    numPlayersRequired: number;
    numJoined: number;
    players: ({ name: string; slot: number } | null)[];
    moves: (number | null)[];
    currentStone: number | null;
    winners: number[];
}

enum GameMode { local = 'local', online = 'online' }


const _cmdToBoard = new Map(
    (Object.entries(PrescribedBoardMap) as [string, [number, string, string, string]][])
        .map(([k, [numArgs, cmd, argStr, desc]]) =>
            [cmd, { boardType: Number(k), numArgs, fn: PrescribedBoardFns[Number(k) as PrescribedBoard], argStr, desc }])
);


function _defaultPlayerName(): string {
    const existing = localStorage.getItem('playerName');
    if (existing) return existing;
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let name = '';
    for (let i = 0; i < 16; i++) name += chars[Math.floor(Math.random() * chars.length)];
    localStorage.setItem('playerName', name);
    return name;
}

const COLOR_GRID    = '#4c4c4c';
const COLOR_ILLEGAL = '#ca9d44';
const COLOR_BOARD   = '#e5b24c';

// ── layout helper ────────────────────────────────────────────────────────────

// Given a canvas size w×h, compute how to map board coordinates to screen pixels so that
// the board is centred and as large as possible without exceeding 150 px per cell.
// +1.26 adds 1.5 stone diameter of margin so the outermost stones are not flush against the edges.
//
// Returns:
//   originX, originY - screen pixel for board coordinate (0, 0):
//                      sx = originX + bx * cell
//                      sy = originY - by * cell  (board y-up → canvas y-down)
//   cell             - pixels per board-coordinate unit
//   stone_r          - stone radius in pixels (= 0.42 * cell)
function boardLayout(view: BoardView, w: number, h: number) {
    const [[xMin, yMin], [xMax, yMax]] = view.boardDimension;
    const spanX = xMax - xMin || 1, spanY = yMax - yMin || 1;
    const cell = Math.min(w / (spanX + 1.26), h / (spanY + 1.26), 150);
    const stone_r = cell * 0.42;
    const originX = w / 2 - (xMin + xMax) / 2 * cell;
    const originY = h / 2 + (yMin + yMax) / 2 * cell;
    return { originX, originY, cell, stone_r };
}

// ── board canvas drawing ─────────────────────────────────────────────────────

// Render a board state onto `ctx`.
// legalMoves: if non-null, empty nodes where legalMoves[i] is null are marked with COLOR_ILLEGAL.
function drawBoardFull(
    ctx: CanvasRenderingContext2D,
    view: BoardView,
    adj: number[][],
    board: number[],
    canvasW: number, canvasH: number,
    legalMoves: (Set<number> | null)[] | null,
) {
    const { originX, originY, cell, stone_r } = boardLayout(view, canvasW, canvasH);
    const N = view.N;

    // grid lines
    ctx.strokeStyle = COLOR_GRID;
    ctx.lineWidth   = 1;
    for (let i = 0; i < N; i++) {
        for (let j = i + 1; j < N; j++) {
            if (!adj[i][j]) continue;
            const [x1, y1] = view.pos[i], [x2, y2] = view.pos[j];
            ctx.beginPath();
            ctx.moveTo(originX + x1 * cell, originY - y1 * cell);
            ctx.lineTo(originX + x2 * cell, originY - y2 * cell);
            ctx.stroke();
        }
    }

    // stones / illegal markers
    for (let i = 0; i < N; i++) {
        const [x, y] = view.pos[i];
        const sx = originX + x * cell, sy = originY - y * cell;
        const stone = board[i];
        if (stone > 0) {
            ctx.beginPath();
            ctx.arc(sx, sy, stone_r, 0, 2 * Math.PI);
            ctx.fillStyle = STONE_MAP[stone].color;
            ctx.fill();
            ctx.strokeStyle = '#333';
            ctx.lineWidth   = 1;
            ctx.stroke();
        } else if (legalMoves !== null && legalMoves[i] === null) {
            ctx.beginPath();
            ctx.arc(sx, sy, stone_r, 0, 2 * Math.PI);
            ctx.fillStyle = COLOR_ILLEGAL;
            ctx.fill();
        }
    }
}

// ── EngineManager ────────────────────────────────────────────────────────────
//
// Manages a sequence of engine move requests without blocking the main thread.
// The caller drives the state machine by alternating poll() and submit() calls
// from the render loop (_checkAsync), so no async/await appears in Renderer.
//
// State machine:
//
//   idle ──register()──► needsRequest ──submit()──► waiting
//    ▲                        ▲                        │
//    │                        │           WS response completes / error
//    │                        │                        │
//    │              remainingMoves > 0            hasResult
//    │                        │                        │
//    └── remainingMoves == 0 ─┴──────poll()────────────┘
//
//   cancel() transitions any state back to idle; if waiting, it cancels the
//   in-flight WS request so its eventual response is dropped (no-op).
//   A request error also transitions waiting → idle directly.

class EngineManager {
    private _state: 'idle' | 'needsRequest' | 'waiting' | 'hasResult' = 'idle';
    private _pendingMove: number | null = null;
    private _pendingHandle: RequestHandle | null = null;
    private _onResult: () => void;
    remainingMoves = 0;
    sessionId: string | null = null;

    get running() { return this._state !== 'idle'; }

    constructor(onResult: () => void) {
        this._onResult = onResult;
    }

    // Begin a new sequence of numMoves engine moves. Transitions idle → needsRequest
    // and triggers a render so _checkAsync sees 'needsRequest' and calls submit().
    // Returns false (no-op) if already running.
    register(numMoves: number): boolean {
        if (this._state !== 'idle') return false;
        this.remainingMoves = numMoves;
        this._state = 'needsRequest';
        this._onResult();
        return true;
    }

    // Fire one engine request over the WS. Called by _checkAsync only when poll()
    // returned 'needsRequest'.
    // Transitions needsRequest → waiting; on completion → hasResult (or idle on error).
    submit(body: Record<string, unknown>): void {
        this._state = 'waiting';
        const handle = conn.request<{ move: number | null; session_id?: string }>('ai/move', { body });
        this._pendingHandle = handle;
        handle.promise
            .then(data => { this._pendingHandle = null; this._pendingMove = data.move; if (data.session_id) this.sessionId = data.session_id; this._state = 'hasResult'; this._onResult(); })
            .catch(e => {
                this._pendingHandle = null;
                console.error('em:', e); this._state = 'idle'; this._onResult();
            });
    }

    // Called each render loop iteration by _checkAsync:
    //   null           - nothing to do (idle or request in-flight)
    //   'needsRequest' - build a request body and call submit()
    //   { move }       - apply this move; state already advanced:
    //                    remainingMoves decremented, and if > 0 transitioned to
    //                    needsRequest so the next poll() will prompt another submit()
    poll(): null | 'needsRequest' | { move: number | null } {
        if (this._state === 'needsRequest') return 'needsRequest';
        if (this._state === 'hasResult') {
            const move = this._pendingMove;
            this._pendingMove = null;
            this.remainingMoves--;
            this._state = this.remainingMoves > 0 ? 'needsRequest' : 'idle';
            return { move };
        }
        return null;
    }

    // Abort the current sequence and return to idle.
    // If a request is in-flight (waiting), it is cancelled; its eventual response
    // is dropped (the promise never settles, so its handlers never run).
    cancel(): void {
        if (this._pendingHandle) { this._pendingHandle.cancel(); this._pendingHandle = null; }
        this._state = 'idle';
        this._pendingMove = null;
        this.remainingMoves = 0;
    }
}

// ── Renderer class ───────────────────────────────────────────────────────────

export class Renderer {
    game: BoardState;
    // invariant: displayPlyNum < game.history.length
    displayPlyNum = 0;
    aiEngineReady = false;
    selfPlay   = false;
    autoForced = false;
    numStonesForNew  = 2;
    numPlayersForNew = 2;
    boardTypeForNew = PrescribedBoard.rectangularBoard;
    boardDimensionForNew : Record<PrescribedBoard, number[]> = {
        [PrescribedBoard.rectangularBoard]:         [9, 9],
        [PrescribedBoard.rectangularDiagonalBoard]: [9, 9],
        [PrescribedBoard.cubicalBoard]:             [5, 5, 2],
        [PrescribedBoard.hypercubeBoard]:           [5, 5, 2, 2],
        [PrescribedBoard.triangularBoard]:          [13],
        [PrescribedBoard.twistedSquareBoard]:       [4, 4, 3],
        [PrescribedBoard.glueTwistedSquareBoard]:   [4, 4, 3],
    };
    // Board type/dims of the live game (used to build /move requests to the AI engine)
    boardTypeForCurrent  = PrescribedBoard.rectangularBoard;
    boardDimsForCurrent: number[] = [9, 9];
    turnStoneListForNew: number[] = [1, 2];
    stoneToPlayerMap: Record<number, number> = {1: 1, 2: 2};
    forcedPassOnlyForNew = true;
    nShowHistory = 10;
    idxShowHistory = 0;
    activeTab: 'history' | 'status' | 'commands' = 'history';
    randomEvaled: Record<number, number> | null = null;
    emNumSims: number = 200;
    emTemperature: number = 0;

    // Online multiplayer state
    gameMode: GameMode = GameMode.local;
    createdOnlineGames = new Map<string, OnlineGameConfig>();
    playerName: string = _defaultPlayerName();
    onlineGameId: string | null = null;
    onlinePosition: number | null = null;
    onlinePlayerSlot: number | null = null;
    onlineMovesSeen = 0;
    private onlineGameFinished = false;

    private mainCanvas:   HTMLCanvasElement;
    private histBoards:   HTMLDivElement;
    private passBtn:      HTMLButtonElement;
    private resignBtn:    HTMLButtonElement;
    private bwEndBtn:     HTMLButtonElement;
    private bw10Btn:      HTMLButtonElement;
    private bwBtn:        HTMLButtonElement;
    private fwBtn:        HTMLButtonElement;
    private fw10Btn:      HTMLButtonElement;
    private fwEndBtn:     HTMLButtonElement;
    private turnStone:    HTMLDivElement;
    private plyNum:       HTMLSpanElement;
    private cmdInput:     HTMLInputElement;
    private cmdOutput:    HTMLDivElement;
    private statusPanel:   HTMLDivElement;
    private commandsPanel: HTMLDivElement;
    private historyPanel:  HTMLDivElement;
    private selfPlayTimer: number | null = null;
    private engineManager = new EngineManager(() => this._render());

    constructor(game: BoardState) {
        this.game = game;
        this.mainCanvas   = document.getElementById('main-canvas')    as HTMLCanvasElement;
        this.histBoards   = document.getElementById('history-boards') as HTMLDivElement;
        this.passBtn      = document.getElementById('pass-btn')        as HTMLButtonElement;
        this.resignBtn    = document.getElementById('resign-btn')      as HTMLButtonElement;
        this.bwEndBtn     = document.getElementById('bwend-btn')      as HTMLButtonElement;
        this.bw10Btn      = document.getElementById('bw10-btn')       as HTMLButtonElement;
        this.bwBtn        = document.getElementById('bw-btn')         as HTMLButtonElement;
        this.fwBtn        = document.getElementById('fw-btn')         as HTMLButtonElement;
        this.fw10Btn      = document.getElementById('fw10-btn')       as HTMLButtonElement;
        this.fwEndBtn     = document.getElementById('fwend-btn')      as HTMLButtonElement;
        this.turnStone    = document.getElementById('turn-stone')     as HTMLDivElement;
        this.plyNum       = document.getElementById('ply-num')        as HTMLSpanElement;
        this.cmdInput     = document.getElementById('cmd-input')      as HTMLInputElement;
        this.cmdOutput    = document.getElementById('cmd-output')     as HTMLDivElement;
        this.statusPanel   = document.getElementById('status-panel')    as HTMLDivElement;
        this.commandsPanel = document.getElementById('commands-panel')  as HTMLDivElement;
        this.historyPanel  = document.getElementById('history-panel')   as HTMLDivElement;
    }

    init() {
        this._initCommandsPanel();
        this.mainCanvas.addEventListener('click', e => this._onBoardClick(e));
        this.bwEndBtn.addEventListener('click', () => {
            this.displayPlyNum = 0;
            this._render();
        });
        this.bw10Btn.addEventListener('click', () => {
            this.displayPlyNum = Math.max(this.displayPlyNum - 10, 0);
            this._render();
        });
        this.bwBtn.addEventListener('click', () => {
            this.displayPlyNum = Math.max(this.displayPlyNum - 1, 0);
            this._render();
        });
        this.fwBtn.addEventListener('click', () => {
            this.displayPlyNum = Math.min(this.displayPlyNum + 1, this.game.history.length - 1);
            this._render();
        });
        this.fw10Btn.addEventListener('click', () => {
            this.displayPlyNum = Math.min(this.displayPlyNum + 10, this.game.history.length - 1);
            this._render();
        });
        this.fwEndBtn.addEventListener('click', () => {
            this.displayPlyNum = this.game.history.length - 1;
            this._render();
        });
        this.passBtn.addEventListener('click', () => {
            const v = this.game.getView();
            if (v.passEnabled && !v.gameOver) {
                if (this.gameMode === GameMode.online) {
                    if (this._isMyOnlineTurn()) void this._submitOnlineMove(null);
                } else {
                    this.engineManager.cancel();
                    this.game.makeMove(null);
                    this.displayPlyNum = this.game.getView().plyCount;
                    this.randomEvaled = null;
                    this._render();
                }
            }
        });
        this.resignBtn.addEventListener('click', () => { void this._resignOnline(); });
        this.cmdInput.addEventListener('keydown', e => {
            if (e.key === 'Enter') { this._parseCommand(this.cmdInput.value.trim()); this.cmdInput.value = ''; this._render(); }
        });
        document.querySelectorAll<HTMLButtonElement>('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const tab = btn.dataset['tab'] as 'history' | 'status' | 'commands';
                this.activeTab = tab;
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.historyPanel.style.display  = tab === 'history'  ? 'flex'   : 'none';
                this.statusPanel.style.display   = tab === 'status'   ? 'block'  : 'none';
                this.commandsPanel.style.display = tab === 'commands' ? 'block'  : 'none';
                this._render();
            });
        });
        document.querySelectorAll<HTMLButtonElement>('.nav-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const action = btn.dataset['action']!;
                const step   = parseInt(btn.dataset['step'] ?? '0');
                const v      = this.game.getView();
                const n      = v.history.length;
                if      (action === 'prev')  this.idxShowHistory = Math.max(0, this.idxShowHistory - step);
                else if (action === 'next')  this.idxShowHistory = Math.min(this.idxShowHistory + step, n - 1);
                else if (action === 'start') this.idxShowHistory = 0;
                else if (action === 'end')   this.idxShowHistory = n - 1;
                this._render();
            });
        });
        window.addEventListener('resize', () => this._render());
        this._render();
        conn.request<{ status?: string }>('ai/health').promise.then(data => {
            this.aiEngineReady = data?.status === 'ok';
            this._render();
        }).catch(() => { this.aiEngineReady = false; });

        // Online-game state arrives via server push (replaces polling). Apply only
        // pushes for the game we're currently in.
        conn.onEvent('game/state', (msg: { id: string; state: OnlineStateResponse }) => {
            if (msg.id === this.onlineGameId) this._applyOnlineState(msg.state);
        });
        // Re-subscribe to the active game after a (re)connect so pushes resume and
        // the server re-binds this connection to our player position.
        conn.onEvent('open', () => {
            if (this.onlineGameId !== null && this.onlinePosition !== null)
                conn.request('game/subscribe', { id: this.onlineGameId, position: this.onlinePosition }).promise.catch(() => {});
        });
    }

    private _checkAsync() {
        while (true) {
            const outcome = this.engineManager.poll();
            if (outcome === null) break;
            if (outcome === 'needsRequest') {
                const v = this.game.getView();
                if (v.gameOver) { console.warn('em: game is already over'); this.engineManager.cancel(); break; }
                if (this.displayPlyNum !== v.plyCount) { console.warn('em: not at live position (navigate to end first)'); this.engineManager.cancel(); break; }
                const moves = this.game.lastMoves.map(m => m.pos);
                this.engineManager.submit({
                    board_type:          PrescribedBoardMap[this.boardTypeForCurrent][1],
                    board_args:          this.boardDimsForCurrent,
                    board:               v.history[v.plyCount].board,
                    num_stones:          v.numStones,
                    num_players:         v.numPlayers,
                    turn_stone_list:     v.turnStoneList,
                    stone_to_player_map: v.stoneToPlayerMap,
                    forced_pass_only:    v.forcedPassOnly,
                    moves,
                    session_id:          this.engineManager.sessionId,
                    num_simulations:     this.emNumSims,
                    temperature:         this.emTemperature,
                });
                break;
            }
            if (!this.game.makeMove(outcome.move)) {
                console.error('em: engine returned an illegal move', outcome.move);
                this.engineManager.cancel();
                break;
            }
            this.displayPlyNum = this.game.getView().plyCount;
            this.randomEvaled = null;
            // loop: poll() has already advanced state to 'needsRequest' or 'idle'
        }
    }

    private _render() {
        this._checkAsync();
        const v = this.game.getView();
        this._renderMainBoard(v);
        this._renderControlBar(v);
        this._renderHistoryPanel(v);
        if (this.activeTab === 'status') this._renderStatus(v);
        if (this.autoForced && !this.selfPlay && !v.gameOver && this.gameMode === GameMode.local) {
            const legals = this.game.legalMoveList();
            if (legals.length === 0 || legals.length === 1) {
                if (this.engineManager.running) return;
                this.game.makeMove(legals.length === 0 ? null : legals[0]);
                this.displayPlyNum = this.game.getView().plyCount;
                requestAnimationFrame(() => this._render());
            }
        }
    }

    private _renderMainBoard(v: BoardView) {
        const wrap  = this.mainCanvas.parentElement!;
        const style = getComputedStyle(wrap);
        const w = wrap.clientWidth  - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
        const h = wrap.clientHeight - parseFloat(style.paddingTop)  - parseFloat(style.paddingBottom);
        const size = Math.max(Math.min(w, h), 1);
        this.mainCanvas.width  = size;
        this.mainCanvas.height = size;
        const ctx = this.mainCanvas.getContext('2d')!;
        ctx.fillStyle = COLOR_BOARD;
        ctx.fillRect(0, 0, size, size);
        drawBoardFull(ctx, v, this.game.adj, v.history[this.displayPlyNum].board,
                      size, size, v.legalMoveHistory[this.displayPlyNum]);
    }

    private _renderControlBar(v: BoardView) {
        this.turnStone.style.background = STONE_MAP[v.nextPlayer]?.color ?? '#888';
        this.plyNum.textContent = `${this.displayPlyNum}/${v.plyCount}`;
        this.bwEndBtn.disabled = this.displayPlyNum === 0;
        this.bw10Btn.disabled  = this.displayPlyNum === 0;
        this.bwBtn.disabled    = this.displayPlyNum === 0;
        this.fwBtn.disabled    = this.displayPlyNum === v.plyCount;
        this.fw10Btn.disabled  = this.displayPlyNum === v.plyCount;
        this.fwEndBtn.disabled = this.displayPlyNum === v.plyCount;
        this.passBtn.disabled = this.displayPlyNum !== v.plyCount || !v.passEnabled || v.gameOver
            || (this.gameMode === GameMode.online && !this._isMyOnlineTurn());
        this.resignBtn.hidden = this.gameMode !== GameMode.online;
        this.resignBtn.disabled = this.gameMode !== GameMode.online || this.onlineGameFinished;
    }

    private _renderHistoryPanel(v: BoardView) {
        const n = v.history.length;
        this.idxShowHistory = Math.max(0, Math.min(this.idxShowHistory, n - 1));
        const nAvail = n - this.idxShowHistory;
        const nShow  = Math.min(nAvail, this.nShowHistory);

        // rebuild entry DOM (simple approach: always rebuild)
        this.histBoards.innerHTML = '';
        for (let idx = 0; idx < this.nShowHistory; idx++) {
            const entry = document.createElement('div');
            entry.className = 'history-entry';
            this.histBoards.appendChild(entry);

            if (idx >= nShow) continue;  // empty slot - show background box only

            const left = document.createElement('div');
            left.className = 'history-entry-left';
            const circle = document.createElement('div');
            circle.className = 'hist-stone';
            const plyLabel = document.createElement('div');
            plyLabel.className = 'hist-ply';
            const canvas = document.createElement('canvas');
            canvas.className = 'hist-canvas';
            left.append(circle, plyLabel);
            entry.append(left, canvas);

            const t = n - 1 - this.idxShowHistory - idx;
            const he = v.history[t];
            circle.style.background = STONE_MAP[he.nextPlayer]?.color ?? '#888';
            plyLabel.textContent = String(he.plyCount);

            // size canvas after layout
            requestAnimationFrame(() => {
                const rect = canvas.parentElement!.getBoundingClientRect();
                const cw = Math.max(1, Math.floor(rect.width  - left.offsetWidth - 8));
                const ch = Math.max(1, Math.floor(rect.height - 8));
                canvas.width  = cw;
                canvas.height = ch;
                const ctx2 = canvas.getContext('2d')!;
                ctx2.fillStyle = COLOR_BOARD;
                ctx2.fillRect(0, 0, cw, ch);
                drawBoardFull(ctx2, v, this.game.adj, he.board, cw, ch, null);
            });
        }
    }

    private _initCommandsPanel() {
        const row = (cmd: string, desc: string) =>
            `<tr><td>${cmd}</td><td>${desc}</td></tr>`;
        const head = (label: string) =>
            `<tr><th colspan="2">${label}</th></tr>`;
        this.commandsPanel.innerHTML = `<table>
            <colgroup><col style="width:40%"><col style="width:60%"></colgroup>
            ${head('Game')}
            ${row('new',              'Start new local game')}
            ${row('em [&lt;n&gt;]',  'Engine move (optional n consecutive moves)')}
            ${row('cem',             'Cancel current engine move')}
            ${row('temp &lt;f&gt;',  'Set engine temperature (0 = argmax visits)')}
            ${row('s',           'Toggle self-play (random moves)')}
            ${row('af',          'Toggle auto-forced: auto-execute forced moves')}
            ${row('w &lt;n&gt;', 'Withdraw n moves')}
            ${row('wcd',         'Withdraw moves until ply equals display position')}
            ${row('re &lt;n&gt;','Random evaluation over n playouts')}
            ${head('Display')}
            ${row('fw &lt;n&gt;','Step display forward n plies')}
            ${row('bw &lt;n&gt;','Step display backward n plies')}
            ${row('h &lt;n&gt;', 'Show n entries in the history panel')}
            ${head('New Game Setup')}
            ${row('fpo',                      'Toggle forced-pass-only for new games')}
            ${row('bt &lt;name&gt',           'Set board type for new game')}
            ${row('bd &lt;num&gt; &lt;num&gt; …', 'Set board dimension for new game')}
            ${row('ns &lt;n&gt;',             'Set number of stone types for new games')}
            ${row('np &lt;n&gt;',             'Set number of players for new games')}
            ${row('tsl &lt;p1&gt; &lt;p2&gt; …','Set turn stone list for new games. Stone types are 1-indexed')}
            ${row('spm &lt;p1&gt; &lt;p2&gt; …','Set stone to player map. Players are 1-indexed')}
            ${head('Online Multiplayer')}
            ${row('setname &lt;name&gt;', 'Set your display name for online games')}
            ${row('newo',                 'Create online game with current config; prints game ID')}
            ${row('joino &lt;ID&gt;',     'Join an existing online game by ID')}
            ${head('Board Types')}
            ${[..._cmdToBoard.entries()].map(([cmd, { argStr, desc }]) => row(`${cmd} ${argStr}`, desc)).join('\n            ')}
        </table>`;
    }

    private _renderStatus(v: BoardView) {
        const lm  = v.lastMove;
        const lastMover = v.turnStoneList[(v.plyCount - 1 + v.turnStoneList.length) % v.turnStoneList.length];
        const sideName  = (p: number) => STONE_MAP[p]?.name ?? `P${p}`;

        let lastMoveStr = '';
        if      (lm.moveType === MoveType.NOMOVE)   lastMoveStr = 'None';
        else if (lm.moveType === MoveType.ILLEGAL)  lastMoveStr = `Illegal@${lm.pos}`;
        else if (lm.moveType === MoveType.PLACE)    lastMoveStr = `${sideName(lastMover)}@${lm.pos}, captures ${lm.captures.length}`;
        else if (lm.moveType === MoveType.PASS)     lastMoveStr = `${sideName(lastMover)}@Pass`;
        else if (lm.moveType === MoveType.GAMEOVER) {
            const winnerNames = v.winners.map(w => `P${w}`);
            lastMoveStr = v.winners.length === 1
                ? `Game over, ${winnerNames[0]} wins`
                : `Game over, tied: ${winnerNames.join(', ')}`;
        }

        const stoneLine = Object.entries(v.stoneCount)
            .map(([p, c]) => `${sideName(Number(p))}: ${c}`)
            .join('  ');

        const evalStr = this.randomEvaled
            ? Object.entries(this.randomEvaled).map(([p, w]) => `P${p} ${w.toFixed(1)}`).join(' | ')
            : 'None';

        const fmtMap = (map: Record<number, number>) =>
            Object.entries(map).map(([s, p]) => `${sideName(Number(s))}→P${p}`).join(', ');
        const createdGamesSection = this.createdOnlineGames.size > 0 ? `
            <div><b>Created online games:</b> ${[...this.createdOnlineGames.keys()].join(', ')}</div>
            <hr style="margin:6px 0">` : '';
        const onlineSection = this.gameMode === GameMode.online ? `
            <div><b>Online game:</b> ${this.onlineGameId}</div>
            <div><b>Your player slot:</b> ${this.onlinePlayerSlot ?? '(waiting for start)'}</div>
            <div><b>Your name:</b> ${this.playerName || '(not set)'}</div>
            <hr style="margin:6px 0">` : '';
        this.statusPanel.innerHTML = `${createdGamesSection}${onlineSection}
            <div><b>To move:</b> ${sideName(v.nextPlayer)}</div>
            <div><b>Last move:</b> ${lastMoveStr}</div>
            <div><b>Stones:</b> ${stoneLine}</div>
            <div><b>Ply:</b> ${v.plyCount}</div>
            <div><b>AI engine:</b> ${this.aiEngineReady ? 'ready' : 'unavailable'}</div>
            <div><b>Engine sims per move:</b> ${this.emNumSims ?? 'default'}</div>
            <div><b>Engine temperature:</b> ${this.emTemperature}</div>
            <div><b>Self play:</b> ${this.selfPlay}</div>
            <div><b>Auto forced:</b> ${this.autoForced}</div>
            <div><b>Show history:</b> ${this.nShowHistory}</div>
            <div><b>Evaluation:</b> ${evalStr}</div>
            <hr style="margin:6px 0">
            <div><b>Current Game Setup</div>
            <div><b>&emsp;Type of stones (current):</b> ${v.numStones}</div>
            <div><b>&emsp;Number of players (current):</b> ${v.numPlayers}</div>
            <div><b>&emsp;Turn stone list (current):</b> [${v.turnStoneList.join(', ')}]</div>
            <div><b>&emsp;Stone to player map (current):</b> ${fmtMap(v.stoneToPlayerMap)}</div>
            <div><b>&emsp;Forced pass only (current):</b> ${v.forcedPassOnly}</div>
            <hr style="margin:6px 0">
            <div><b>New Game Setup</div>
            <div><b>&emsp;Type of stones:</b> ${this.numStonesForNew}</div>
            <div><b>&emsp;Number of players:</b> ${this.numPlayersForNew}</div>
            <div><b>&emsp;Turn stone list:</b> [${this.turnStoneListForNew.join(', ')}]</div>
            <div><b>&emsp;Stone to player map:</b> ${fmtMap(this.stoneToPlayerMap)}</div>
            <div><b>&emsp;Forced pass only:</b> ${this.forcedPassOnlyForNew}</div>
            <div><b>&emsp;Board type:</b> ${PrescribedBoardMap[this.boardTypeForNew][1]}</div>
            <div><b>&emsp;Board dimension:</b> ${this.boardDimensionForNew[this.boardTypeForNew]}</div>
        `;
    }

    private _onBoardClick(e: MouseEvent) {
        const v    = this.game.getView();
        const rect = this.mainCanvas.getBoundingClientRect();
        const mx   = e.clientX - rect.left;
        const my   = e.clientY - rect.top;
        const { originX, originY, cell, stone_r } =
            boardLayout(v, this.mainCanvas.width, this.mainCanvas.height);

        let bestDist = Infinity, bestId = -1;
        for (let i = 0; i < v.N; i++) {
            const [bx, by] = v.pos[i];
            const sx = originX + bx * cell, sy = originY - by * cell;
            const dist = Math.hypot(mx - sx, my - sy);
            if (dist < bestDist) { bestDist = dist; bestId = i; }
        }
        if (bestId >= 0 && bestDist < stone_r * 1.3) {
            if (this.displayPlyNum !== v.plyCount) return;
            if (this.gameMode === GameMode.online) {
                if (this._isMyOnlineTurn()) void this._submitOnlineMove(bestId);
            } else {
                this.engineManager.cancel();
                this.game.makeMove(bestId);
                this.displayPlyNum = this.game.getView().plyCount;
                this.randomEvaled = null;
                this._render();
            }
        }
    }

    private _newGame(bc: BoardConfig) {
        this.engineManager.cancel();
        this.engineManager.sessionId = null;
        this.game = new BoardState(
            this.numStonesForNew, this.numPlayersForNew, this.turnStoneListForNew, this.stoneToPlayerMap,
            this.forcedPassOnlyForNew,
            new Array(bc.N).fill(0), bc,
        );
        this.boardTypeForCurrent = this.boardTypeForNew;
        this.boardDimsForCurrent = [...this.boardDimensionForNew[this.boardTypeForNew]];
        this.idxShowHistory = 0;
        this.displayPlyNum  = 0;
        this.randomEvaled   = null;
    }

    private _parseCommand(raw: string) {
        const parts = raw.trim().split(/\s+/);
        this.cmdOutput.textContent = '';
        if (!parts[0]) return;
        const cmd = parts[0];
        const posInt = (s: string | undefined) => { const n = Number(s); return Number.isInteger(n) && n > 0 ? n : null; };

        if (cmd === 'setname') {
            if (parts[1]) {
                this.playerName = parts.slice(1).join(' ');
                localStorage.setItem('playerName', this.playerName);
                this._setCmdOutput(`Name set to: ${this.playerName}`);
            } else {
                this._setCmdOutput(`Current name: ${this.playerName || '(not set)'}`);
            }
        }
        else if (cmd === 'newo') {
            void this._createOnlineGame();
        }
        else if (cmd === 'joino') {
            if (this.gameMode === GameMode.online && !this.onlineGameFinished)
                { this._setCmdOutput('Cannot join a game while an online game is ongoing'); return; }
            if (!parts[1]) { this._setCmdOutput('Usage: joino <ID>'); return; }
            void this._joinOnlineGame(parts[1].toUpperCase());
        }
        else if (cmd === 'em') {
            if (this.gameMode === GameMode.online) { this._setCmdOutput('Engine moves are disabled in online mode'); return; }
            const n = posInt(parts[1] ?? '1');
            if (n === null) { this._setCmdOutput('em: n must be a positive integer'); return; }
            if (!this.engineManager.register(n)) console.warn('em: engine move already in progress');
            return;  // EngineManager triggers _render() when done
        }
        else if (cmd === 'cem') {
            this.engineManager.cancel();
        }
        else if (cmd === 'emsim') {
            const n = posInt(parts[1]);
            if (n === null) { this._setCmdOutput('Usage: emsim <n>  (positive integer)'); return; }
            this.emNumSims = n;
        }
        else if (cmd === 'temp') {
            const t = parseFloat(parts[1]);
            if (!Number.isFinite(t) || t < 0) { this._setCmdOutput('Usage: temp <float>  (non-negative; 0 = argmax)'); return; }
            this.emTemperature = t;
        }
        else if (cmd === 's') {
            this.selfPlay = !this.selfPlay;
            if (this.selfPlay) this._startSelfPlay();
            else this._stopSelfPlay();
        }
        else if (cmd === 'fpo')  this.forcedPassOnlyForNew = !this.forcedPassOnlyForNew;
        else if (cmd === 'af')   this.autoForced = !this.autoForced;
        else if (cmd === 'bt') {
            if (!parts[1]) { this._setCmdOutput('Usage: bt <board-type>'); return; }
            if (!_cmdToBoard.has(parts[1])) { this._setCmdOutput(`Unknown board type: ${parts[1]}`); return; }
            this.boardTypeForNew = _cmdToBoard.get(parts[1])!.boardType;
        }
        else if (cmd === 'bd') {
            if (!parts[1]) { this._setCmdOutput('Usage: bd <num> <num> …'); return; }
            const nums = parts.slice(1).map(Number);
            if (nums.some(n => !Number.isInteger(n) || n <= 0)) { this._setCmdOutput('bd: all arguments must be positive integers'); return; }
            for (const [idx, val] of nums.entries())
                if (idx < (PrescribedBoardMap[this.boardTypeForNew][0]))
                    this.boardDimensionForNew[this.boardTypeForNew][idx] = val;
        }
        else if (cmd === 'ns') {
            const n = Number(parts[1]);
            if (!parts[1] || !Number.isInteger(n) || n < 1 || n > 8) { this._setCmdOutput('Usage: ns <n>  (1–8)'); return; }
            this.numStonesForNew = n;
            this.turnStoneListForNew = Array.from({ length: n }, (_, i) => i + 1);
            this.stoneToPlayerMap = Object.fromEntries(Array.from({ length: n }, (_, i) => [i + 1, i % this.numPlayersForNew + 1]));
        }
        else if (cmd === 'np') {
            const n = Number(parts[1]);
            if (!parts[1] || !Number.isInteger(n) || n < 1 || n > 8) { this._setCmdOutput('Usage: np <n>  (1–8)'); return; }
            this.numPlayersForNew = n;
            this.stoneToPlayerMap = Object.fromEntries(Array.from({ length: this.numStonesForNew }, (_, i) => [i + 1, i % n + 1]));
        }
        else if (cmd === 'tsl') {
            if (parts.length < 2) { this._setCmdOutput('Usage: tsl <p1> <p2> …'); return; }
            const stones = parts.slice(1).map(Number);
            if (!stones.every(p => Number.isInteger(p) && p >= 1 && p <= this.numStonesForNew))
                { this._setCmdOutput(`tsl: each value must be an integer between 1 and ${this.numStonesForNew}`); return; }
            this.turnStoneListForNew = stones;
        }
        else if (cmd === 'spm') {
            if (parts.length < 2) { this._setCmdOutput('Usage: spm <p1> <p2> …'); return; }
            const players = parts.slice(1).map(Number);
            if (!players.every(p => Number.isInteger(p) && p >= 1 && p <= this.numPlayersForNew))
                { this._setCmdOutput(`spm: each value must be an integer between 1 and ${this.numPlayersForNew}`); return; }
            for (const [idx, p] of players.entries())
                this.stoneToPlayerMap[idx + 1] = p;
        }
        else if (cmd === 'h') {
            const n = posInt(parts[1]);
            if (n === null) { this._setCmdOutput('Usage: h <n>  (positive integer)'); return; }
            this.nShowHistory = n;
        }
        else if (cmd === 'w') {
            if (this.gameMode === GameMode.online) { this._setCmdOutput('Cannot withdraw moves in online games'); return; }
            const n = posInt(parts[1]);
            if (n === null) { this._setCmdOutput('Usage: w <n>  (positive integer)'); return; }
            this.engineManager.cancel();
            this.engineManager.sessionId = null;
            for (let i = 0; i < n; i++) this.game.retractMove();
            this.displayPlyNum = Math.min(this.displayPlyNum, this.game.history.length - 1);
        }
        else if (cmd === 'wcd') {
            if (this.gameMode === GameMode.online) { this._setCmdOutput('Cannot withdraw moves in online games'); return; }
            this.engineManager.cancel();
            this.engineManager.sessionId = null;
            const n = this.game.history.length - 1 - this.displayPlyNum;
            for (let i = 0; i < n; i++) this.game.retractMove();
        }
        else if (cmd === 'fw') {
            const n = posInt(parts[1]);
            if (n === null) { this._setCmdOutput('Usage: fw <n>  (positive integer)'); return; }
            this.displayPlyNum = Math.min(this.displayPlyNum + n, this.game.history.length - 1);
        }
        else if (cmd === 'bw') {
            const n = posInt(parts[1]);
            if (n === null) { this._setCmdOutput('Usage: bw <n>  (positive integer)'); return; }
            this.displayPlyNum = Math.max(this.displayPlyNum - n, 0);
        }
        else if (cmd === 're') {
            const n = posInt(parts[1]);
            if (n === null) { this._setCmdOutput('Usage: re <n>  (positive integer)'); return; }
            this.randomEvaled = this.game.randomEvaluate(n);
        }
        else if (cmd === 'new') {
            if (this.gameMode === GameMode.online && !this.onlineGameFinished)
                { this._setCmdOutput('Cannot create new board during active online games'); return; }
            // When an online game has finished, "new" exits online mode and starts a fresh local game.
            this.gameMode = GameMode.local;
            this.onlineGameId = null;
            const fn = PrescribedBoardFns[this.boardTypeForNew];
            this._newGame(fn(...this.boardDimensionForNew[this.boardTypeForNew]));
        }
        else
            this._setCmdOutput(`Unknown command \"${cmd}\"`)
    }

    private _startSelfPlay() {
        this.engineManager.cancel();
        const tick = () => {
            if (!this.selfPlay) return;
            const end = Date.now() + 40;
            while (Date.now() < end) {
                this.game.randomMove();
                if (this.game.lastMove().moveType === MoveType.GAMEOVER) {
                    this.selfPlay = false; break;
                }
            }
            this.displayPlyNum = this.game.history.length - 1;
            this._render();
            if (this.selfPlay) this.selfPlayTimer = requestAnimationFrame(tick);
        };
        this.selfPlayTimer = requestAnimationFrame(tick);
    }

    private _stopSelfPlay() {
        if (this.selfPlayTimer !== null) { cancelAnimationFrame(this.selfPlayTimer); this.selfPlayTimer = null; }
    }

    // ── Online multiplayer ────────────────────────────────────────────────────

    private _setCmdOutput(msg: string) {
        this.cmdOutput.textContent = msg;
    }

    private _isMyOnlineTurn(): boolean {
        if (this.onlinePlayerSlot === null) return false;
        const v = this.game.getView();
        if (v.gameOver) return false;
        return v.stoneToPlayerMap[v.nextPlayer] === this.onlinePlayerSlot;
    }

    private async _createOnlineGame() {
        if (!this.playerName) { this._setCmdOutput('Set your name first: setname <name>'); return; }
        const boardTypeName = PrescribedBoardMap[this.boardTypeForNew][1];
        try {
            const { id, position } = await conn.request<{ id: string; position: number }>('game/create', {
                config: {
                    boardType:        boardTypeName,
                    boardArgs:        [...this.boardDimensionForNew[this.boardTypeForNew]],
                    numStones:        this.numStonesForNew,
                    numPlayers:       this.numPlayersForNew,
                    turnStoneList:    [...this.turnStoneListForNew],
                    stoneToPlayerMap: { ...this.stoneToPlayerMap },
                    forcedPassOnly:   this.forcedPassOnlyForNew,
                },
                playerName: this.playerName,
            }).promise;

            this.onlineGameId     = id;
            this.onlinePosition   = position;
            this.onlinePlayerSlot = null;
            this.onlineMovesSeen  = 0;
            this.onlineGameFinished = false;
            this.createdOnlineGames.set(id, { boardType: boardTypeName, boardArgs: [...this.boardDimensionForNew[this.boardTypeForNew]], numStones: this.numStonesForNew, numPlayers: this.numPlayersForNew, turnStoneList: [...this.turnStoneListForNew], stoneToPlayerMap: { ...this.stoneToPlayerMap }, forcedPassOnly: this.forcedPassOnlyForNew });
            const fn = PrescribedBoardFns[this.boardTypeForNew];
            this._newGame(fn(...this.boardDimensionForNew[this.boardTypeForNew]));
            this._setCmdOutput(`Game created: ${id} - waiting for ${this.numPlayersForNew - 1} more player(s)…`);
            this._render();
        } catch (e: any) { this._setCmdOutput(`Error: ${e.message}`); }
    }

    private async _joinOnlineGame(id: string) {
        if (!this.playerName) { this._setCmdOutput('Set your name first: setname <name>'); return; }
        try {
            const { position, config } = await conn.request<{ position: number; config: OnlineGameConfig; status: string }>(
                'game/join', { id, playerName: this.playerName }).promise;
            const boardEntry = _cmdToBoard.get(config.boardType);
            if (!boardEntry) { this._setCmdOutput(`Unknown board type: ${config.boardType}`); return; }
            const bc = boardEntry.fn(...config.boardArgs);
            this.engineManager.cancel();
            this.engineManager.sessionId = null;
            this.game = new BoardState(
                config.numStones, config.numPlayers,
                config.turnStoneList, config.stoneToPlayerMap,
                config.forcedPassOnly, new Array(bc.N).fill(0), bc,
            );
            this.boardTypeForCurrent = boardEntry.boardType;
            this.boardDimsForCurrent = [...config.boardArgs];
            this.displayPlyNum   = 0;
            this.randomEvaled    = null;
            this.idxShowHistory  = 0;

            this.onlineGameId     = id;
            this.onlinePosition   = position;
            this.onlinePlayerSlot = null;
            this.onlineMovesSeen  = 0;
            this.onlineGameFinished = false;
            this._setCmdOutput(`Joined game: ${id}`);
            this._render();
        } catch (e: any) { this._setCmdOutput(`Error: ${e.message}`); }
    }

    private _applyOnlineState(state: OnlineStateResponse) {
        // Resolve player slot once the game starts
        if (this.onlinePlayerSlot === null && state.status === 'playing') {
            const me = state.players[this.onlinePosition!];
            if (me) {
                this.gameMode         = GameMode.online;
                this.onlinePlayerSlot = me.slot;
                this._setCmdOutput(`Game started! You are player ${me.slot} (${me.name})`);
                this._render();
            }
        }

        // Apply any new moves from the server
        if (state.moves.length > this.onlineMovesSeen) {
            const wasAtLive = this.displayPlyNum === this.game.getView().plyCount;
            for (let i = this.onlineMovesSeen; i < state.moves.length; i++) {
                this.game.makeMove(state.moves[i]);
            }
            this.onlineMovesSeen = state.moves.length;
            if (wasAtLive) this.displayPlyNum = this.game.getView().plyCount;
            this.randomEvaled = null;

            // Notify when it becomes our turn after an opponent move
            if (this.onlinePlayerSlot !== null) {
                const v = this.game.getView();
                if (!v.gameOver) {
                    if (v.stoneToPlayerMap[v.nextPlayer] === this.onlinePlayerSlot)
                        this._setCmdOutput('Your turn!');
                    else
                        this._setCmdOutput("Opponent's turn.")
                }
            }
            this._render();
        }

        // Game-over notification (once)
        if (state.status === 'finished' && !this.onlineGameFinished) {
            this.onlineGameFinished = true;
            this.createdOnlineGames.delete(this.onlineGameId!);
            const v = this.game.getView();
            const winnerText = v.winners.length === 0
                ? 'No winners'
                : v.winners.map(w => `Player ${w}`).join(', ') + ' win!';
            this._setCmdOutput(`Game over! ${winnerText}`);
            this._render();
        }
    }

    private async _resignOnline() {
        if (!this.onlineGameId) return;
        // Resulting state arrives via the game/state push. The server identifies us
        // by our WebSocket connection, so no position is sent.
        try {
            await conn.request('game/resign', { id: this.onlineGameId }).promise;
        } catch (e: any) { this._setCmdOutput(`Resign failed: ${e.message}`); }
    }

    private async _submitOnlineMove(moveIndex: number | null) {
        if (!this.onlineGameId) return;
        // Resulting state arrives via the game/state push. The server identifies us
        // by our WebSocket connection, so no position is sent.
        try {
            await conn.request('game/move', {
                id:        this.onlineGameId,
                moveIndex,
                clientIdx: this.onlineMovesSeen,
            }).promise;
        } catch (e: any) { this._setCmdOutput(`Move rejected: ${e.message}`); }
    }

}
