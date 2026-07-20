import { BoardState, MoveType, STONE_MAP } from '@shared/boardState.js';
import { PlayerInfo, GameConfig, FinishedGame, OnlinePlayerRequest, makeId } from '@shared/types.js';
import type { BoardView, OnlineStateResponse, PendingGame, ScoreRule, KoRule, TurnInfo, ReplayMove } from '@shared/types.js';
import type { BoardConfig } from '@shared/boardConfig.js';
import {
    PrescribedBoard, PrescribedBoardMap, PrescribedBoardFns, computeStarPoints,
} from '@shared/boardConfig.js';
import { ServerConnection, type RequestHandle } from './serverConnection.js';
import {
    SidePanelContent, SidePanelHierarchy, SidePanelBwFw, renderSidePanelChrome, sidePanelParent, childButtons, renderGamePresetSelection,
    currentGameSetupHtml, newGameSetupHtml,
    coloredStoneCircle, fmtTurnList,
} from './sidePanel.js';

// Single persistent WebSocket connection to the main server, shared by the
// EngineManager (AI proxy) and the online-game commands.
const conn = new ServerConnection();


const _cmdToBoard = new Map(
    (Object.entries(PrescribedBoardMap) as [string, [number, string, string, string]][])
        .map(([k, [numArgs, cmd, argStr, desc]]) =>
            [cmd, { boardType: Number(k), numArgs, fn: PrescribedBoardFns[Number(k) as PrescribedBoard], argStr, desc }])
);

// Filename stems (under public/game_presets/) of the GameConfig JSON presets
// loaded at startup into Renderer.presets - see _loadPresets() - each paired
// with a short human-readable description shown in the "Game Presets"
// command-panel section (see _initCommandsPanel()).
const _presetDescriptions = new Map([
    ['go',                 'Traditional 19×19 Go'],
    ['3x3_go_fpo',         'Tiny 3×3 Go, forced-pass-only'],
    ['4x4_go_fpo',         'Tiny 4×4 Go, forced-pass-only'],
    ['5x5_go_fpo',         'Small 5×5 Go, forced-pass-only'],
    ['go_fpo',             'Traditional 19×19 Go, forced-pass-only (must play a legal move if one exists)'],
    ['3_player_go',        'Traditional 19×19 Go for 3 players, one stone color each'],
    ['4_color_go',         '19×19 Go, 2 players each alternating between two of their own stone colors'],
    ['13x13_4_color_go',   '13×13 Go, 2 players each alternating between two of their own stone colors'],
    ['two_ply_go',         '19×19 Go, each player places two stones in a row per turn'],
    ['two_ply_go_fpo',     '19×19 Go, two plies per turn, forced-pass-only'],
    ['13x13_two_ply_go',   '13×13 Go, each player places two stones in a row per turn'],
    ['9x9_go',             'Small 9×9 Go'],
    ['9x9_go_fpo',         'Small 9×9 Go, forced-pass-only'],
    ['7x7x2_twsq_go',      '7×7×2 twisted-square board'],
    ['7x7x2_twsq_go_fpo',  '7×7×2 twisted-square board, forced-pass-only'],
    ['3_coin_go',          "19×19 Go plus a protected, non-friendly 'coin' stone (worth no points) either player may place, up to 3 times each"],
    ['10_coin_go',         "19×19 Go plus a protected, non-friendly 'coin' stone (worth no points) either player may place, up to 10 times each"],
    ['3_friend_go',        "Like 3_coin_go, but the 'coin' stone is also friendly (doesn't block anyone's liberties)"],
    ['10_friend_go',       "Like 10_coin_go, but the 'coin' stone is also friendly (doesn't block anyone's liberties)"],
]);


const COLOR_GRID    = '#000000';
const COLOR_ILLEGAL = '#ca9d44';
const COLOR_BOARD   = '#e5b24c';

// SVG elements must be created via createElementNS with this namespace -
// document.createElement('circle') etc. produce non-rendering HTMLUnknownElements.
const SVG_NS = 'http://www.w3.org/2000/svg';

// ── layout helper ────────────────────────────────────────────────────────────

// Given a board size w×h, compute how to map board coordinates to screen pixels so that
// the board is centred and as large as possible without exceeding 150 px per cell.
// +1.26 adds 1.5 stone diameter of margin so the outermost stones are not flush against the edges.
//
// Returns:
//   originX, originY - screen pixel for board coordinate (0, 0):
//                      sx = originX + bx * cell
//                      sy = originY - by * cell  (board y-up → screen y-down)
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

// ── board SVG drawing ────────────────────────────────────────────────────────

// Render a board state as SVG into `parent` (an already-created/cleared <svg> or
// <g> - this function only appends, never clears `parent` itself; the caller owns
// that lifecycle, same convention as _renderHistoryPanel's histBoards.innerHTML = '').
// legalMoves: if non-null, this is history[ply].legalMoves.captures (captures[stone][loc]) - empty
// nodes illegal for every offered stone are marked with COLOR_ILLEGAL.
// territoryOwner: if non-null, each node with territoryOwner[i] > 0 is marked with a small
// square (side = cell/4, grey-lined with stroke-width = side/6) colored by that stone type.
// dim: if true, grid lines, stones, and the territory overlay are all wrapped in a
// single 50%-opacity <g> (used while a stone-selection popup is up, since the board
// isn't clickable in that state) - the caller draws the (always full-opacity) popup
// circles separately, outside this function's element.
function drawBoardFull(
    parent: SVGElement,
    view: BoardView,
    adj: number[][],
    board: number[],
    config: GameConfig,
    boardW: number, boardH: number,
    legalMoves: (Set<number> | null)[][] | null,
    territoryOwner: number[] | null = null,
    dim = false,
) {
    const { originX, originY, cell, stone_r } = boardLayout(view, boardW, boardH);
    const N = view.N;

    // grid lines and stones/illegal markers are both dimmed together while
    // selecting a stone, so the whole board reads as "not interactive" - the
    // territory overlay is included in the same group, matching how the
    // canvas version's globalAlpha stayed set across all three sections.
    const g = document.createElementNS(SVG_NS, 'g');
    if (dim) g.setAttribute('opacity', '0.5');
    parent.appendChild(g);

    // grid lines
    for (let i = 0; i < N; i++) {
        for (let j = i + 1; j < N; j++) {
            if (!adj[i][j]) continue;
            const [x1, y1] = view.pos[i], [x2, y2] = view.pos[j];
            const line = document.createElementNS(SVG_NS, 'line');
            line.setAttribute('x1', String(originX + x1 * cell));
            line.setAttribute('y1', String(originY - y1 * cell));
            line.setAttribute('x2', String(originX + x2 * cell));
            line.setAttribute('y2', String(originY - y2 * cell));
            line.setAttribute('stroke', COLOR_GRID);
            line.setAttribute('stroke-width', '1');
            g.appendChild(line);
        }
    }

    // star points ("hoshi" board markings, rect boards only - computeStarPoints()
    // returns [] for any other boardType) - drawn before the stones loop below
    // so a stone placed on a star point visually covers the dot, matching real
    // board conventions.
    for (const [x, y] of computeStarPoints(config)) {
        const sx = originX + x * cell, sy = originY - y * cell;
        const c = document.createElementNS(SVG_NS, 'circle');
        c.setAttribute('cx', String(sx));
        c.setAttribute('cy', String(sy));
        c.setAttribute('r', String(cell * 0.09));
        c.setAttribute('fill', COLOR_GRID);
        g.appendChild(c);
    }

    // stones / illegal markers
    for (let i = 0; i < N; i++) {
        const [x, y] = view.pos[i];
        const sx = originX + x * cell, sy = originY - y * cell;
        const stone = board[i];
        if (stone > 0) {
            const c = document.createElementNS(SVG_NS, 'circle');
            c.setAttribute('cx', String(sx));
            c.setAttribute('cy', String(sy));
            c.setAttribute('r', String(stone_r));
            c.setAttribute('fill', STONE_MAP[stone].color);
            c.setAttribute('stroke', '#333');
            c.setAttribute('stroke-width', '1');
            g.appendChild(c);
        } else if (legalMoves !== null && legalMoves.every(row => row[i] === null)) {
            const c = document.createElementNS(SVG_NS, 'circle');
            c.setAttribute('cx', String(sx));
            c.setAttribute('cy', String(sy));
            c.setAttribute('r', String(stone_r));
            c.setAttribute('fill', COLOR_ILLEGAL);
            g.appendChild(c);
        }
    }

    // territory squares
    if (territoryOwner !== null) {
        const side = cell / 4;
        for (let i = 0; i < N; i++) {
            const owner = territoryOwner[i];
            if (owner <= 0) continue;
            const [x, y] = view.pos[i];
            const sx = originX + x * cell, sy = originY - y * cell;
            const r = document.createElementNS(SVG_NS, 'rect');
            r.setAttribute('x', String(sx - side / 2));
            r.setAttribute('y', String(sy - side / 2));
            r.setAttribute('width', String(side));
            r.setAttribute('height', String(side));
            r.setAttribute('fill', STONE_MAP[owner]?.color ?? '#888');
            r.setAttribute('stroke', '#888');
            r.setAttribute('stroke-width', String(side / 6));
            g.appendChild(r);
        }
    }
}

// ── EngineManager ────────────────────────────────────────────────────────────
//
// Manages a sequence of engine move requests without blocking the main thread.
class EngineManager {
    private _handle: RequestHandle | null = null;
    remainingMoves = 0;
    sessionId: string | null = null;

    get running() { return this._handle !== null || this.remainingMoves > 0; }

    // Begin a new sequence of numMoves engine moves. Returns false (no-op) if already running.
    // Caller must call _fireEngineMove() immediately after a successful register().
    register(numMoves: number): boolean {
        if (this.running) return false;
        this.remainingMoves = numMoves;
        return true;
    }

    // Fire one engine request. onMove is called with the resulting move on success;
    // onError is called on failure. Caller chains the next fire() inside onMove.
    fire(
        game_id: string,
        config: GameConfig,
        board: number[],
        moves: ReplayMove[],
        session_id: string | null,
        num_simulations: number,
        temperature: number,
        onMove: (move: number | null) => void,
        onError: (e: any) => void,
    ): void {
        const body = { game_id, config, board, moves, session_id, num_simulations, temperature };
        const handle = conn.request<{ move: number | null; session_id?: string }>('ai/move', { body });
        this._handle = handle;
        handle.promise
            .then(data => {
                this._handle = null;
                if (data.session_id) this.sessionId = data.session_id;
                this.remainingMoves--;
                onMove(data.move);
            })
            .catch(e => {
                this._handle = null;
                console.error('em:', e);
                this.remainingMoves = 0;
                onError(e);
            });
    }

    // Abort the current sequence. If a request is in-flight its eventual response is dropped.
    cancel(): void {
        if (this._handle) { this._handle.cancel(); this._handle = null; }
        this.remainingMoves = 0;
    }
}

// ── Renderer class ───────────────────────────────────────────────────────────

interface ActiveGame {
    bs: BoardState;
    config: GameConfig;
    displayPlyNum: number;
    idxShowHistory: number;
    randomEvaled: Record<number, number> | null;
}

// Response shape of REGISTER/LOGIN/FLOGIN: the finished online games the server
// has recorded this user as an observer of (see _addFinishedGames).
interface LoginResponse {
    name: string;
    finishedGames: { id: string; finishedGame: any }[];
}

// One entry in Renderer's popup queue (see currentPopup/popupQueue) - a
// discriminated union so renderPopup() can render each kind's specific
// content/buttons.
type PopupInfo =
    | { kind: 'invite'; id: string; from: string }
    | { kind: 'create-failed'; message: string }
    | { kind: 'login-prompt' }
    | { kind: 'confirm'; message: string; onYes: () => void; onNo: () => void };

export class Renderer {
    aiEngineReady = false;
    selfPlay   = false;
    autoForced = false;
    showTerritory = false;
    showIllegalMoves = false;
    // True while a click on a multi-stone turn is waiting for the player to
    // pick which offered stone to place (see _onBoardClick/_renderMainBoard).
    selectingStone = false;
    pendingPos: number | null = null;
    newCfg = new GameConfig(PrescribedBoardMap[PrescribedBoard.rectangularBoard][1], [9, 9], 2, 2, [{player: 1, stones: [1, 0], protected: [0, 0], friendly: [0, 0]}, {player: 2, stones: [0, 1], protected: [0, 0], friendly: [0, 0]}], [[null, null], [null, null]], [null, null], {1: new Set([1]), 2: new Set([2])}, true, 'area', [0, 0], 'situational', false, null);
    // Pending online-game player setup, built by tfpro/sol/soe/adde/addl and
    // sent to the server in _createOnlineGame() - the server (not this
    // client) resolves it into actual slot assignments (see
    // OnlinePlayerRequest). Independent of newCfg, which is otherwise just
    // board/rules configuration.
    onlinePlayerRequest = new OnlinePlayerRequest();
    // Transient UI-only state for the Configure Online Players panel's
    // "Invite" flow (not part of onlinePlayerRequest itself) - which slot
    // (fixed mode) or 'random' (random-order mode) currently has its
    // invite-textbox open, and that textbox's current value. The value is
    // field-backed (synced via the input's own 'input' listener) rather than
    // trusted to survive on the DOM node itself, since _renderConfigureOnlinePlayers()
    // rebuilds the whole panel - including this input - on every _render().
    private inviteInputTarget: number | 'random' | null = null;
    private inviteInputValue = '';
    // Dedupes 'localEngine' auto-advance attempts (see _render()) per
    // (activeIdx, plyCount) - a failed attempt leaves plyCount unchanged, so
    // this prevents retrying every single _render() tick in a loop; a fresh
    // key (new game, or plyCount actually advancing) tries again naturally.
    private _lastAutoEngineAttempt: string | null = null;
    // Generic modal-popup mechanism (currently only used for game invites) -
    // popUp is true iff currentPopup is non-null; kept as an explicit field
    // (rather than derived) per spec, and toggles #popup-overlay's
    // visibility/the body.popup-active class that disables the rest of the
    // UI (see renderPopup(), index.html). Additional popups queue rather
    // than interrupting whichever one is currently showing.
    popUp = false;
    private currentPopup: PopupInfo | null = null;
    private popupQueue: PopupInfo[] = [];
    // Loaded at startup from public/game_presets/ (see _loadPresets()); name -> config.
    presets = new Map<string, GameConfig>();
    // Per-board-type dimension memory so 'bt' restores custom dimensions on type switch
    boardDimensionForNew: Record<PrescribedBoard, number[]> = {
        [PrescribedBoard.rectangularBoard]:         [9, 9],
        [PrescribedBoard.rectangularDiagonalBoard]: [9, 9, 3],
        [PrescribedBoard.cubicalBoard]:             [5, 5, 2],
        [PrescribedBoard.hypercubeBoard]:           [5, 5, 2, 2],
        [PrescribedBoard.triangularBoard]:          [13],
        [PrescribedBoard.twistedSquareBoard]:       [4, 4, 3],
        [PrescribedBoard.glueTwistedSquareBoard]:   [4, 4, 3],
    };
    nShowHistory = 10;
    currentSidePanel: SidePanelContent = SidePanelContent.Home;
    sidePanelBwFw: SidePanelBwFw = new SidePanelBwFw(SidePanelContent.Home);
    emNumSims: number = 200;
    emTemperature: number = 0;

    // Online multiplayer state
    // Pending games: created/joined but not yet started. The players map is kept in sync
    // by game/pending-games broadcasts; local slots have type='local'.
    pendingGames = new Map<string, PendingGame>();
    userName: string | null = null;
    activeGames = new Map<string, ActiveGame>();
    // Games (local or online) that have ended - moved here from activeGames by
    // _maybeFinish. Online entries are also synced from the server at login
    // (see _addFinishedGames), so a user's finished-game history survives reconnects.
    finishedGames = new Map<string, ActiveGame>();
    activeIdx: string = '';   // always set before first render (constructor initializes)

    private get _active(): ActiveGame {
        return (this.activeGames.get(this.activeIdx) ?? this.finishedGames.get(this.activeIdx))!;
    }

    // Finds a game (active or finished) by key.
    private _findGame(key: string): ActiveGame | undefined {
        return this.activeGames.get(key) ?? this.finishedGames.get(key);
    }

    // Moves `key` from activeGames to finishedGames the moment its BoardState
    // reports game over. One-way: withdrawing moves is blocked on finished games
    // (see the 'w'/'wcd' commands), so a finished game never needs to move back.
    private _maybeFinish(key: string): void {
        const ag = this.activeGames.get(key);
        if (ag && ag.bs.gameOver()) { this.activeGames.delete(key); this.finishedGames.set(key, ag); }
    }

    private mainSvg:      SVGSVGElement;
    // Current square size (px) of mainSvg - boardLayout()/_stonePopupCircles() need
    // this as a plain number; SVGSVGElement.width isn't usable (undefined in jsdom,
    // an SVGAnimatedLength object rather than a number in real browsers).
    private mainBoardSize = 1;
    private histBoards:   HTMLDivElement;
    private passBtn:       HTMLButtonElement;
    private resignBtn:    HTMLButtonElement;
    private withdrawBtn:  HTMLButtonElement;
    private wcdBtn:       HTMLButtonElement;
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
    private panelDockBtn: HTMLButtonElement;
    private panelFullBtn: HTMLButtonElement;
    private panelHideBtn: HTMLButtonElement;
    private panelHomeBtn:    HTMLButtonElement;
    private panelBackBtn:    HTMLButtonElement;
    private panelForwardBtn: HTMLButtonElement;
    private panelUpBtn:      HTMLButtonElement;
    private panelTitleEl:    HTMLDivElement;
    private homePanel:       HTMLDivElement;
    private currentGameSetupPanel:    HTMLDivElement;
    private currentGameSetupDetails:  HTMLDivElement;
    private currentGameSetupButtons:  HTMLDivElement;
    private newGamePanel:          HTMLDivElement;
    private newGameSetupDetails:   HTMLDivElement;
    private newGameButtons:        HTMLDivElement;
    private gameRecordsPanel:      HTMLDivElement;
    private gamePresetSelectionPanel: HTMLDivElement;
    private activeLocalGamesPanel:    HTMLDivElement;
    private pendingGamesPanel:        HTMLDivElement;
    private activeOnlineGamesPanel:   HTMLDivElement;
    private finishedOnlineGamesPanel: HTMLDivElement;
    private accountPanel: HTMLDivElement;
    private configureOnlinePlayersPanel: HTMLDivElement;
    private popupOverlay: HTMLDivElement;
    // Side-panel layout mode - see _applyPanelMode().
    // Default is overwritten during init() from the screen-width check there;
    // 'locked' here is just the pre-JS/no-JS fallback matching index.html's
    // default markup (#panel-full-btn/#panel-hide-btn visible, #panel-dock-btn
    // hidden), but init() corrects it immediately regardless.
    panelMode: 'hidden' | 'full' | 'locked' = 'locked';
    private selfPlayTimer: number | null = null;
    private engineManager = new EngineManager();

    constructor(game: BoardState) {
        const initCfg = this.newCfg.copy();
        for (let slot = 1; slot <= initCfg.numPlayers; slot++)
            initCfg.players.set(slot, new PlayerInfo('local', ''));
        // Start with a default local game so there is always an active game.
        this._registerGame('L_' + makeId(12), game, initCfg);
        this.mainSvg      = document.getElementById('main-canvas')    as unknown as SVGSVGElement;
        this.histBoards   = document.getElementById('history-boards') as HTMLDivElement;
        this.passBtn       = document.getElementById('pass-btn')        as HTMLButtonElement;
        this.resignBtn    = document.getElementById('resign-btn')      as HTMLButtonElement;
        this.withdrawBtn  = document.getElementById('withdraw-btn')    as HTMLButtonElement;
        this.wcdBtn       = document.getElementById('wcd-btn')         as HTMLButtonElement;
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
        this.panelDockBtn = document.getElementById('panel-dock-btn') as HTMLButtonElement;
        this.panelFullBtn = document.getElementById('panel-full-btn') as HTMLButtonElement;
        this.panelHideBtn = document.getElementById('panel-hide-btn') as HTMLButtonElement;
        this.panelHomeBtn    = document.getElementById('panel-home-btn')  as HTMLButtonElement;
        this.panelBackBtn    = document.getElementById('panel-back-btn')    as HTMLButtonElement;
        this.panelForwardBtn = document.getElementById('panel-forward-btn') as HTMLButtonElement;
        this.panelUpBtn      = document.getElementById('panel-up-btn')    as HTMLButtonElement;
        this.panelTitleEl    = document.getElementById('panel-title')     as HTMLDivElement;
        this.homePanel       = document.getElementById('home-panel')      as HTMLDivElement;
        this.currentGameSetupPanel   = document.getElementById('current-game-setup-panel')   as HTMLDivElement;
        this.currentGameSetupDetails = document.getElementById('current-game-setup-details') as HTMLDivElement;
        this.currentGameSetupButtons = document.getElementById('current-game-setup-buttons') as HTMLDivElement;
        this.newGamePanel          = document.getElementById('new-game-panel')           as HTMLDivElement;
        this.newGameSetupDetails   = document.getElementById('new-game-setup-details')   as HTMLDivElement;
        this.newGameButtons        = document.getElementById('new-game-buttons')         as HTMLDivElement;
        this.gameRecordsPanel      = document.getElementById('game-records-panel')       as HTMLDivElement;
        this.gamePresetSelectionPanel = document.getElementById('game-preset-selection-panel') as HTMLDivElement;
        this.activeLocalGamesPanel    = document.getElementById('active-local-games-panel')    as HTMLDivElement;
        this.pendingGamesPanel        = document.getElementById('pending-games-panel')         as HTMLDivElement;
        this.activeOnlineGamesPanel   = document.getElementById('active-online-games-panel')   as HTMLDivElement;
        this.finishedOnlineGamesPanel = document.getElementById('finished-online-games-panel') as HTMLDivElement;
        this.accountPanel = document.getElementById('account-panel') as HTMLDivElement;
        this.configureOnlinePlayersPanel = document.getElementById('configure-online-players-panel') as HTMLDivElement;
        this.popupOverlay = document.getElementById('popup-overlay') as HTMLDivElement;
    }

    // A docked ('locked') panel isn't usable at 1/3 width on a narrow
    // screen/window. Width, not a "phone vs. laptop" device check, is what
    // actually matters here - also correctly covers a resized desktop window.
    private _screenIsSmall(): boolean {
        return window.innerWidth < 700;
    }

    // Applies `panelMode` to the DOM: the body class driving #main-area/
    // #side-panel's CSS layout (index.html), which of the three mode-switch
    // #panel-mode-bar buttons are visible - each button is simply hidden
    // whenever it would target the state we're already in (contextual
    // controls, not a static tri-state cluster), e.g. in 'hidden' mode both
    // "go full" and "go locked" are shown (the latter only if the screen is
    // large enough), since either is a valid destination from there:
    //   hidden -> [full] [dock (only if screen isn't small)]
    //   full   -> [dock (only if screen isn't small)] [hide]
    //   locked -> [full] [hide]
    // (the dock button is hidden outright, not dimmed/disabled, when the
    // screen is small, per its own design) - and the side panel's own
    // Home/Back/Forward/Up navigation buttons (also in #panel-mode-bar,
    // to the left of the three above), which are only meaningful while the
    // side panel is actually showing something (panelMode 'full' or
    // 'locked'), so they're hidden together with it in 'hidden' mode.
    private _applyPanelMode() {
        document.body.classList.remove('panel-hidden', 'panel-full');
        if (this.panelMode === 'hidden') document.body.classList.add('panel-hidden');
        else if (this.panelMode === 'full') document.body.classList.add('panel-full');

        this.panelDockBtn.hidden = this.panelMode === 'locked' || this._screenIsSmall();
        this.panelFullBtn.hidden = this.panelMode === 'full';
        this.panelHideBtn.hidden = this.panelMode === 'hidden';

        const sidePanelVisible = this.panelMode !== 'hidden';
        this.panelHomeBtn.hidden    = !sidePanelVisible;
        this.panelBackBtn.hidden    = !sidePanelVisible;
        this.panelForwardBtn.hidden = !sidePanelVisible;
        this.panelUpBtn.hidden      = !sidePanelVisible;
    }

    // Re-evaluates screen size (called once at startup and on every resize -
    // see init()): a 'locked' panel that just became unusable falls back to
    // 'hidden' automatically, but the screen becoming large again never
    // force-switches the user's current choice back to 'locked' - it only
    // makes the dock button available again (handled by _applyPanelMode()).
    private _updatePanelModeAvailability() {
        if (this.panelMode === 'locked' && this._screenIsSmall()) this.panelMode = 'hidden';
        this._applyPanelMode();
    }

    // Rebuilds everything tied to a side-panel navigation: the Up/title
    // chrome and content-panel visibility (see sidePanel.ts's
    // renderSidePanelChrome()), then - for whichever of Home/CurrentGameSetup/
    // NewGame/GameRecords is current (the only nodes with a nonempty children
    // list) - rebuilds that node's own children buttons via childButtons()
    // into its own container. Each decides its own container: Home's
    // #home-panel, CurrentGameSetup's bottom #current-game-setup-buttons,
    // NewGame's #new-game-buttons (between its details and the static
    // Start-new-game button), GameRecords' own #game-records-panel (it has
    // no other content, so the panel doubles as the button container).
    private _refreshSidePanel() {
        renderSidePanelChrome(this.currentSidePanel, {
            titleEl:              this.panelTitleEl,
            upBtn:                this.panelUpBtn,
            homePanel:             this.homePanel,
            historyPanel:         this.historyPanel,
            statusPanel:          this.statusPanel,
            commandsPanel:        this.commandsPanel,
            currentGameSetupPanel: this.currentGameSetupPanel,
            newGamePanel:          this.newGamePanel,
            gameRecordsPanel:      this.gameRecordsPanel,
            gamePresetSelectionPanel: this.gamePresetSelectionPanel,
            activeLocalGamesPanel:    this.activeLocalGamesPanel,
            pendingGamesPanel:        this.pendingGamesPanel,
            activeOnlineGamesPanel:   this.activeOnlineGamesPanel,
            finishedOnlineGamesPanel: this.finishedOnlineGamesPanel,
            accountPanel:             this.accountPanel,
            configureOnlinePlayersPanel: this.configureOnlinePlayersPanel,
        });

        const children = SidePanelHierarchy[this.currentSidePanel][1];
        const onNav = (target: SidePanelContent) => this._navigateSidePanel(target);

        this.homePanel.innerHTML = '';
        if (this.currentSidePanel === SidePanelContent.Home)
            for (const btn of childButtons(children, onNav)) this.homePanel.appendChild(btn);

        this.currentGameSetupButtons.innerHTML = '';
        if (this.currentSidePanel === SidePanelContent.CurrentGameSetup)
            for (const btn of childButtons(children, onNav)) this.currentGameSetupButtons.appendChild(btn);

        // The "Select Game Preset" nav button (via childButtons(), like
        // Home/CurrentGameSetup above) and the "Start New Local Game"/"Start
        // New Online Game" action buttons (not SidePanelContent nav targets, so
        // built directly rather than via childButtons()) live in the same
        // #new-game-buttons div, rebuilt together each navigation to New Game.
        this.newGameButtons.innerHTML = '';
        if (this.currentSidePanel === SidePanelContent.NewGame) {
            const navBtnRow = document.createElement('div');
            navBtnRow.className = 'btn-row';
            navBtnRow.append(...childButtons(children, onNav));
            this.newGameButtons.appendChild(navBtnRow);
            const btnRow = document.createElement('div');
            btnRow.className = 'btn-row';
            btnRow.append(this._buildStartLocalGameBtn(), this._buildStartOnlineGameBtn());
            this.newGameButtons.appendChild(btnRow);
        }

        // GameRecords is a pure hub (like Home) - its own content IS its
        // four children's buttons, so #game-records-panel doubles as both
        // the toggled content panel (above) and the button container here.
        this.gameRecordsPanel.innerHTML = '';
        if (this.currentSidePanel === SidePanelContent.GameRecords)
            for (const btn of childButtons(children, onNav)) this.gameRecordsPanel.appendChild(btn);

        // Account has its own content (login form or logged-in view) built by
        // _renderAccountPanel(), not childButtons() - a leaf node, not a hub.
        // Deliberately built here (once per navigation) rather than from
        // _render()'s per-frame dispatch: the login form holds in-progress
        // input values that an unrelated _render() (e.g. a WS event for some
        // other active game, or a window resize) must not wipe out. The
        // Log In/Log Out button handlers call this method directly once the
        // login state actually changes, so the view still updates promptly.
        if (this.currentSidePanel === SidePanelContent.Account) this._renderAccountPanel();

        // Keep the Home/Back/Forward buttons' enabled state in sync - runs
        // after every navigation path (_navigateSidePanel()/_sidePanelBack()/
        // _sidePanelForward() all call this method), same "disabled when
        // there's nowhere to go" convention as the Up button
        // (renderSidePanelChrome, above): Home disables at Home itself,
        // Back/Forward disable at either end of sidePanelBwFw's history.
        this.panelHomeBtn.disabled    = this.currentSidePanel === SidePanelContent.Home;
        this.panelBackBtn.disabled    = this.sidePanelBwFw.currentIdx <= 0;
        this.panelForwardBtn.disabled = this.sidePanelBwFw.currentIdx >= this.sidePanelBwFw.history.length - 1;
    }

    private _navigateSidePanel(target: SidePanelContent) {
        this.currentSidePanel = target;
        // Standard browser back/forward semantics: discard anything past the
        // current position, then append the new entry as the new "current" -
        // see SidePanelBwFw's doc comment (sidePanel.ts).
        const bf = this.sidePanelBwFw;
        bf.history = bf.history.slice(0, bf.currentIdx + 1);
        bf.history.push(target);
        bf.currentIdx = bf.history.length - 1;
        this._refreshSidePanel();
        this._render();
    }

    // Jumps to the Account (login) panel, making the side panel visible
    // first if it's currently hidden - 'full' on a narrow screen (docking
    // isn't usable there, but here we specifically need the panel visible
    // so the user can see the login form, unlike the hide-on-narrow-screen
    // fallback used elsewhere - see _buildStartLocalGameBtn/
    // _buildStartOnlineGameBtn) or 'locked' otherwise. Left alone if the
    // panel is already visible in some mode. _navigateSidePanel() itself
    // never touches panelMode, so this must happen first.
    private _goToLoginPanel() {
        if (this.panelMode === 'hidden') {
            this.panelMode = this._screenIsSmall() ? 'full' : 'locked';
            this._applyPanelMode();
        }
        this._navigateSidePanel(SidePanelContent.Account);
    }

    // Move within sidePanelBwFw's existing history without mutating it -
    // mirrors _navigateSidePanel()'s tail (refresh + render) but skips the
    // truncate+push step, since we're retracing already-visited ground.
    private _sidePanelBack() {
        const bf = this.sidePanelBwFw;
        if (bf.currentIdx <= 0) return;
        bf.currentIdx--;
        this.currentSidePanel = bf.history[bf.currentIdx] as SidePanelContent;
        this._refreshSidePanel();
        this._render();
    }

    private _sidePanelForward() {
        const bf = this.sidePanelBwFw;
        if (bf.currentIdx >= bf.history.length - 1) return;
        bf.currentIdx++;
        this.currentSidePanel = bf.history[bf.currentIdx] as SidePanelContent;
        this._refreshSidePanel();
        this._render();
    }

    // Fetches every preset in _presetDescriptions from public/game_presets/ and
    // stores it in `presets`, keyed by filename stem (also the 'preset <name>'
    // command's <name>). Not awaited by init() - runs in the background so a
    // slow/failed fetch (or, in unit tests, no server at all - see
    // test/renderer/domSetup.ts) never blocks the rest of startup; the
    // commands panel is simply re-rendered once presets actually arrive. Each
    // preset fails independently so one bad file doesn't take out the rest.
    private async _loadPresets(): Promise<void> {
        const entries = await Promise.all([..._presetDescriptions.keys()].map(async name => {
            try {
                const raw = await fetch(`/game_presets/${name}.json`).then(r => r.json());
                return [name, GameConfig.fromJSON(raw)] as const;
            } catch (e) {
                console.warn(`Failed to load game preset '${name}':`, e);
                return null;
            }
        }));
        this.presets = new Map(entries.filter((e): e is readonly [string, GameConfig] => e !== null));
    }

    init() {
        this._initCommandsPanel();
        void this._loadPresets().then(() => this._initCommandsPanel());
        this.mainSvg.addEventListener('click', e => this._onBoardClick(e));
        this.bwEndBtn.addEventListener('click', () => {
            this._active.displayPlyNum = 0;
            this._render();
        });
        this.bw10Btn.addEventListener('click', () => {
            this._active.displayPlyNum = Math.max(this._active.displayPlyNum - 10, 0);
            this._render();
        });
        this.bwBtn.addEventListener('click', () => {
            this._active.displayPlyNum = Math.max(this._active.displayPlyNum - 1, 0);
            this._render();
        });
        this.fwBtn.addEventListener('click', () => {
            this._active.displayPlyNum = Math.min(this._active.displayPlyNum + 1, this._active.bs.situations.length - 1);
            this._render();
        });
        this.fw10Btn.addEventListener('click', () => {
            this._active.displayPlyNum = Math.min(this._active.displayPlyNum + 10, this._active.bs.situations.length - 1);
            this._render();
        });
        this.fwEndBtn.addEventListener('click', () => {
            this._active.displayPlyNum = this._active.bs.situations.length - 1;
            this._render();
        });
        this.passBtn.addEventListener('click', () => {
            const v = this._active.bs.getView();
            if (v.passEnabled && !v.gameOver) this._tryMakeMove(null);
        });
        this.resignBtn.addEventListener('click', () => { void this._resign(); });
        // Same underlying logic as the 'w 1'/'wcd' commands - see
        // _withdrawMove/_withdrawToCurrentDisplay, shared with _parseCommand's
        // 'w'/'wcd' branches so the guards live in exactly one place.
        this.withdrawBtn.addEventListener('click', () => { this._withdrawMove(1); this._render(); });
        this.wcdBtn.addEventListener('click', () => { this._withdrawToCurrentDisplay(); this._render(); });
        this.cmdInput.addEventListener('keydown', e => {
            if (e.key === 'Enter') { this._parseCommand(this.cmdInput.value.trim()); this.cmdInput.value = ''; this._render(); }
        });
        this.panelHomeBtn.addEventListener('click', () => this._navigateSidePanel(SidePanelContent.Home));
        this.panelBackBtn.addEventListener('click', () => this._sidePanelBack());
        this.panelForwardBtn.addEventListener('click', () => this._sidePanelForward());
        this.panelUpBtn.addEventListener('click', () => {
            const parent = sidePanelParent(this.currentSidePanel);
            if (parent !== null) this._navigateSidePanel(parent);
        });
        this._refreshSidePanel();
        document.querySelectorAll<HTMLButtonElement>('.nav-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const action = btn.dataset['action']!;
                const step   = parseInt(btn.dataset['step'] ?? '0');
                const v      = this._active.bs.getView();
                const n      = v.situations.length;
                if      (action === 'prev')  this._active.idxShowHistory = Math.max(0, this._active.idxShowHistory - step);
                else if (action === 'next')  this._active.idxShowHistory = Math.min(this._active.idxShowHistory + step, n - 1);
                else if (action === 'start') this._active.idxShowHistory = 0;
                else if (action === 'end')   this._active.idxShowHistory = n - 1;
                this._render();
            });
        });
        this.panelDockBtn.addEventListener('click', () => { this.panelMode = 'locked'; this._applyPanelMode(); this._render(); });
        this.panelFullBtn.addEventListener('click', () => { this.panelMode = 'full';   this._applyPanelMode(); this._render(); });
        this.panelHideBtn.addEventListener('click', () => { this.panelMode = 'hidden'; this._applyPanelMode(); this._render(); });
        this.panelMode = this._screenIsSmall() ? 'hidden' : 'locked';
        this._updatePanelModeAvailability();

        window.addEventListener('resize', () => { this._updatePanelModeAvailability(); this._render(); });
        this._render();
        conn.request<{ status?: string }>('ai/health').promise.then(data => {
            this.aiEngineReady = data?.status === 'ok';
            this._render();
        }).catch(() => { this.aiEngineReady = false; });

        conn.onEvent('auth/kicked', (msg: { name: string }) => {
            this.userName = null;
            this._setCmdOutput(`Logged out: another connection logged in as ${msg.name} (flogin)`);
            this._render();
        });
        conn.onEvent('game/pending-games', (msg: { id: string; config: any }) => {
            this.pendingGames.set(msg.id, { id: msg.id, config: GameConfig.fromJSON(msg.config) });
            this._render();
        });
        conn.onEvent('game/start', (msg: { id: string; config: any }) => {
            this._activatePendingGame(msg.id, GameConfig.fromJSON(msg.config));
        });
        conn.onEvent('game/move', (msg: { id: string; moveIndex: number | null; stone: number | null }) => {
            this._handleOnlineMove(msg.id, msg.moveIndex, msg.stone);
        });
        conn.onEvent('game/resign', (msg: { id: string; slots: number[] }) => {
            this._handleOnlineResign(msg.id, msg.slots);
        });
        conn.onEvent('game/engine-error', (msg: { id: string; message: string }) => {
            this._setCmdOutput(`Engine error in game ${msg.id}: ${msg.message}`);
        });
        conn.onEvent('game/invite', (msg: { id: string; from: string }) => {
            this.popupQueue.push({ kind: 'invite', id: msg.id, from: msg.from });
            this._advancePopupQueue();
        });
        conn.onEvent('game/invite-failed', (msg: { id: string; message: string }) => {
            this.pendingGames.delete(msg.id);
            this.popupQueue.push({ kind: 'create-failed', message: msg.message });
            this._advancePopupQueue();
            this._render();
        });
        // After a (re)connect, re-subscribe to every active/pending online game so the
        // server re-binds our slot. The reply carries full state for catchup sync.
        // Login state doesn't survive a reconnect (it's tied to the live connection),
        // so the server will reject these until the user logs back in - reset userName
        // to reflect that, and surface it once rather than failing silently.
        conn.onEvent('open', () => {
            this.userName = null;
            this._render();
            let warnedStaleLogin = false;
            const resub = (id: string, position: number) =>
                conn.request<{ state: OnlineStateResponse; config: any }>(
                    'game/subscribe', { id, position })
                    .promise.then(({ state, config: rawConfig }) => {
                        if (state.status === 'playing' || state.status === 'finished') {
                            if (!this._findGame('O_' + id))
                                this._activatePendingGame(id, GameConfig.fromJSON(rawConfig));
                            this._applyOnlineState(id, state);
                        }
                    }).catch(() => {
                        if (warnedStaleLogin) return;
                        warnedStaleLogin = true;
                        this._setCmdOutput('Reconnected - please log in again to resume online games: login <name> <password>');
                    });
            for (const [id, ag] of this.activeGames)
                if (id.startsWith('O_'))
                    for (const [slot, pi] of ag.config.players)
                        if (pi.name === this.userName) resub(id.slice(2), slot);
            for (const [id, pg] of this.pendingGames)
                for (const [slot, pi] of pg.config.players)
                    if (pi.name === this.userName) resub(id, slot);
        });
    }

    private _fireEngineMove(): void {
        const v = this._active.bs.getView();
        if (v.gameOver) { console.warn('em: game is already over'); this.engineManager.cancel(); return; }
        if (this._active.displayPlyNum !== v.plyCount) { console.warn('em: not at live position (navigate to end first)'); this.engineManager.cancel(); return; }
        const moves: ReplayMove[] = this._active.bs.moveInfos().map(m => ({ pos: m.pos, stone: m.stone }));
        // A 'localEngine' slot's own configured emsim/temp take precedence
        // over the global em settings - already concretely populated (never
        // a "0 = default" sentinel needing a fallback), since both the
        // Configure Players "Engine" button and the soe/adde commands default
        // an omitted sim/temp to *this.emNumSims/emTemperature at the moment
        // the slot was configured, not to 0.
        const turnPi = this._active.config.players.get(v.nextTurn.player);
        const numSims = turnPi?.type === 'localEngine' ? turnPi.emsim : this.emNumSims;
        const temp    = turnPi?.type === 'localEngine' ? turnPi.temp  : this.emTemperature;
        this.engineManager.fire(
            this.activeIdx.slice(2),
            this._active.config,
            v.situations[v.plyCount].board,
            moves,
            this.engineManager.sessionId,
            numSims,
            temp,
            (move) => {
                if (!this._active.bs.makeMove(move)) {
                    console.error('em: engine returned an illegal move', move);
                    this.engineManager.cancel();
                } else {
                    this._active.displayPlyNum = this._active.bs.getView().plyCount;
                    this._maybeFinish(this.activeIdx);
                    if (this.engineManager.remainingMoves > 0) this._fireEngineMove();
                }
                this._render();
            },
            (e: any) => {
                this._setCmdOutput(`Engine move failed: ${e?.message ?? 'unknown error'}`);
                this._render();
            },
        );
    }

    private _render() {
        const v = this._active.bs.getView();
        // #main-area (board + control bar) is hidden entirely in 'full' panel
        // mode (see _applyPanelMode()) - skip rebuilding it while invisible;
        // the next _render() after switching back recomputes it against the
        // now-visible, correctly-sized #main-area.
        if (this.panelMode !== 'full') {
            this._renderMainBoard(v);
            this._renderControlBar(v);
        }
        // #side-panel (whichever node is current) is hidden entirely in
        // 'hidden' panel mode (see _applyPanelMode()) - skip rebuilding its
        // current node's contents while invisible, mirroring the #main-area
        // skip above; the next _render() after switching back recomputes it.
        if (this.panelMode !== 'hidden') {
            this._renderHistoryPanel(v);
            if (this.currentSidePanel === SidePanelContent.Status) this._renderStatus(v);
            if (this.currentSidePanel === SidePanelContent.CurrentGameSetup)
                this.currentGameSetupDetails.innerHTML = currentGameSetupHtml(v, this._active.config.players);
            if (this.currentSidePanel === SidePanelContent.NewGame)
                this.newGameSetupDetails.innerHTML = newGameSetupHtml(this.newCfg);
            if (this.currentSidePanel === SidePanelContent.ActiveLocalGames) this._renderActiveLocalGames();
            if (this.currentSidePanel === SidePanelContent.PendingGames) this._renderPendingGames();
            if (this.currentSidePanel === SidePanelContent.ActiveOnlineGames) this._renderActiveOnlineGames();
            if (this.currentSidePanel === SidePanelContent.FinishedOnlineGames) this._renderFinishedOnlineGames();
            if (this.currentSidePanel === SidePanelContent.GamePresetSelection)
                renderGamePresetSelection(this.gamePresetSelectionPanel, [...this.presets.keys()], name => this._selectPreset(name));
            if (this.currentSidePanel === SidePanelContent.ConfigureOnlinePlayers) this._renderConfigureOnlinePlayers();
        }
        this.renderPopup();
        if (this.autoForced && !this.selfPlay && !v.gameOver && this.activeIdx.startsWith('L_')) {
            const legals = this._active.bs.legalPlaceList();
            if (legals.length === 0 || legals.length === 1) {
                if (this.engineManager.running) return;
                if (legals.length === 0) this._active.bs.makeMove(null);
                else this._active.bs.makeMove(legals[0].pos, legals[0].stone);
                this._active.displayPlyNum = this._active.bs.getView().plyCount;
                this._maybeFinish(this.activeIdx);
                requestAnimationFrame(() => this._render());
            }
        }
        // Auto-advance a 'localEngine' slot's turn - see _fireEngineMove()'s
        // per-slot emsim/temp handling and PlayerType's own doc comment
        // (shared/types.ts). Deduped per _lastAutoEngineAttempt so a failed
        // attempt (plyCount unchanged) doesn't retry every _render() tick.
        if (this.activeIdx.startsWith('L_') && !v.gameOver && !this.engineManager.running) {
            const pi = this._active.config.players.get(v.nextTurn.player);
            const attemptKey = `${this.activeIdx}:${v.plyCount}`;
            if (pi?.type === 'localEngine' && this._lastAutoEngineAttempt !== attemptKey) {
                this._lastAutoEngineAttempt = attemptKey;
                if (this.engineManager.register(1)) this._fireEngineMove();
            }
        }
    }

    private _renderMainBoard(v: BoardView) {
        const wrap  = this.mainSvg.parentElement!;
        const style = getComputedStyle(wrap);
        const w = wrap.clientWidth  - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
        const h = wrap.clientHeight - parseFloat(style.paddingTop)  - parseFloat(style.paddingBottom);
        const size = Math.max(Math.min(w, h), 1);
        this.mainBoardSize = size;
        this.mainSvg.setAttribute('width', String(size));
        this.mainSvg.setAttribute('height', String(size));

        while (this.mainSvg.firstChild) this.mainSvg.removeChild(this.mainSvg.firstChild);

        const bg = document.createElementNS(SVG_NS, 'rect');
        bg.setAttribute('x', '0');
        bg.setAttribute('y', '0');
        bg.setAttribute('width', String(size));
        bg.setAttribute('height', String(size));
        bg.setAttribute('fill', COLOR_BOARD);
        if (this.selectingStone) bg.setAttribute('opacity', '0.5');
        this.mainSvg.appendChild(bg);

        drawBoardFull(this.mainSvg, v, this._active.bs.adj, v.situations[this._active.displayPlyNum].board,
                      this._active.config, size, size,
                      this.showIllegalMoves ? v.history[this._active.displayPlyNum].legalMoves.captures : null,
                      this.showTerritory ? v.history[this._active.displayPlyNum].score.territoryOwner : null,
                      this.selectingStone);
        if (this.selectingStone) {
            const popup = document.createElementNS(SVG_NS, 'g');
            for (const { stone, x, y, r } of this._stonePopupCircles(v)) {
                const c = document.createElementNS(SVG_NS, 'circle');
                c.setAttribute('cx', String(x));
                c.setAttribute('cy', String(y));
                c.setAttribute('r', String(r));
                c.setAttribute('fill', STONE_MAP[stone].color);
                c.setAttribute('stroke', '#333');
                c.setAttribute('stroke-width', '1');
                popup.appendChild(c);
            }
            this.mainSvg.appendChild(popup);
        }
    }

    // Computes the centered row of stone-selection popup circles for the
    // stones actually legal to place at pendingPos (see selectingStone) -
    // not just offered this turn, so every circle shown is a real, clickable
    // choice. Shared by rendering (_renderMainBoard) and click hit-testing
    // (_onBoardClick) so they always agree.
    private _stonePopupCircles(v: BoardView): { stone: number; x: number; y: number; r: number }[] {
        const legalStones = this.pendingPos !== null
            ? [...v.history[v.plyCount].legalMoves.legalsForLocation[this.pendingPos]]
            : [];
        const width = this.mainBoardSize, height = this.mainBoardSize;
        const r = width / 24, gap = r;
        const total = legalStones.length * 2 * r + (legalStones.length - 1) * gap;
        const startX = width / 2 - total / 2 + r;
        const y = height / 2;
        return legalStones.map((stone, idx) => ({ stone, x: startX + idx * (2 * r + gap), y, r }));
    }

    private _renderControlBar(v: BoardView) {
        const dpn = this._active.displayPlyNum;
        // Cosmetic preview of every stone color this turn offers - a pie
        // slice per offered stone (#turn-stone's own border supplies the
        // outline, so a 0-stone turn just renders hollow/transparent). The
        // actual choice among multiple offered stones is still made via the
        // popup (_stonePopupCircles) - this is display-only.
        const offeredStones: number[] = [];
        for (let s = 0; s < v.nextTurn.stones.length; s++) if (v.nextTurn.stones[s]) offeredStones.push(s + 1);
        this.turnStone.style.background = offeredStones.length === 0 ? 'transparent' : `conic-gradient(${
            offeredStones.map((stone, i) => `${STONE_MAP[stone].color} ${i / offeredStones.length * 100}% ${(i + 1) / offeredStones.length * 100}%`).join(', ')
        })`;
        this.plyNum.textContent = `${dpn}/${v.plyCount}`;
        this.bwEndBtn.disabled = dpn === 0;
        this.bw10Btn.disabled  = dpn === 0;
        this.bwBtn.disabled    = dpn === 0;
        this.fwBtn.disabled    = dpn === v.plyCount;
        this.fw10Btn.disabled  = dpn === v.plyCount;
        this.fwEndBtn.disabled = dpn === v.plyCount;
        // Withdraws a real move (not just a display-position change like
        // bw/fw), so it's gated on plyCount (actual moves made), not dpn.
        this.withdrawBtn.disabled = v.plyCount === 0;
        // wcd withdraws down to the displayed ply - a no-op once dpn already
        // equals plyCount (nothing ahead of the display position left to cut).
        this.wcdBtn.disabled = dpn === v.plyCount;
        // Disabled while selecting a stone - clicking elsewhere on the board
        // cancels the popup instead (see _onBoardClick); Pass is no longer
        // double-purposed as Cancel.
        this.passBtn.disabled = this.selectingStone
            || dpn !== v.plyCount || !v.passEnabled || !this._isMyTurn();
        this.resignBtn.hidden = this.activeIdx.startsWith('L_');
        this.resignBtn.disabled = this.activeIdx.startsWith('L_') || v.gameOver
            || [...this._active.config.players.entries()].every(([s, pi]) => pi.name !== this.userName || v.resignedPlayers.includes(s));
    }

    private _renderHistoryPanel(v: BoardView) {
        const n = v.situations.length;
        const scroll = Math.max(0, Math.min(this._active.idxShowHistory, n - 1));
        this._active.idxShowHistory = scroll;
        const nAvail = n - scroll;
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
            const canvasWrap = document.createElement('div');
            canvasWrap.className = 'hist-canvas-wrap';
            const svg = document.createElementNS(SVG_NS, 'svg');
            svg.setAttribute('class', 'hist-canvas');
            canvasWrap.appendChild(svg);
            left.append(circle, plyLabel);
            entry.append(left, canvasWrap);

            const t = n - 1 - scroll - idx;
            const he = v.situations[t];
            // A pass/nomove has no chosen stone (see MoveInfo) - show a hollow
            // circle (same border, no fill) rather than guessing a color.
            const heStone = v.moveInfos[he.plyCount - 1]?.stone;
            circle.style.background = heStone != null ? (STONE_MAP[heStone]?.color ?? '#888') : 'transparent';
            plyLabel.textContent = String(he.plyCount);

            // size svg after layout - square, so the board's margin (baked
            // into boardLayout() as a fixed board-unit amount on every side)
            // comes out symmetric rather than stretched to fill a wide,
            // short wrapper (see canvasWrap's centering CSS).
            requestAnimationFrame(() => {
                const rect = canvasWrap.getBoundingClientRect();
                const size = Math.max(1, Math.floor(Math.min(rect.width, rect.height) - 8));
                svg.setAttribute('width', String(size));
                svg.setAttribute('height', String(size));
                const bg = document.createElementNS(SVG_NS, 'rect');
                bg.setAttribute('x', '0');
                bg.setAttribute('y', '0');
                bg.setAttribute('width', String(size));
                bg.setAttribute('height', String(size));
                bg.setAttribute('fill', COLOR_BOARD);
                svg.appendChild(bg);
                drawBoardFull(svg, v, this._active.bs.adj, he.board, this._active.config, size, size, null);
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
            ${row('stt',         'Toggle territory display in the main board area')}
            ${row('simv',        'Toggle illegal move markers on the main board')}
            ${head('New Game Setup')}
            ${row('preset &lt;name&gt;',      'Use the specified preset (see Game Presets, below, for available names)')}
            ${row('fpo',                      'Toggle forced-pass-only for new games')}
            ${row('ascd',                     'Toggle allow-suicide for new games')}
            ${row('bt &lt;name&gt',           'Set board type for new game')}
            ${row('bd &lt;num&gt; &lt;num&gt; …', 'Set board dimension for new game')}
            ${row('ns &lt;n&gt;',             'Set number of stone types for new games')}
            ${row('np &lt;n&gt;',             'Set number of players for new games')}
            ${row('tl &lt;player&gt;-&lt;stone bits&gt; …','Set turn list for new games: which player plays each turn, and which stone(s) they may choose from (numStones-length 0/1 string; the first offered stone is auto-picked - no selection UI yet)')}
            ${row('sprot &lt;0-1 str&gt; …',      'Set protected stones per turn for new games: one numStones-length 0/1 string per turn list entry')}
            ${row('sfriend &lt;0-1 str&gt; …',    'Set friendly stones per turn for new games: one numStones-length 0/1 string per turn list entry')}
            ${row('spm s &lt;stone&gt; p &lt;player&gt; …','Set which player(s) a stone scores for (zero or more; each gets the stone\'s full points). Players are 1-indexed')}
            ${row('spspl &lt;player&gt; s &lt;num|-&gt; …','Set how many times a player may place each stone color (one value per stone, \'-\' = unlimited)')}
            ${row('sgspl &lt;num|-&gt; …','Set how many times each stone color may ever be placed in total, across all players (one value per stone, \'-\' = unlimited)')}
            ${row('sr &lt;rule&gt;',            'Set scoring rule for new games: stone | territoryonly | area | territory')}
            ${row('ko &lt;pos|sit&gt;',          'Set ko rule for new games: positional | situational')}
            ${row('komi &lt;k1&gt; &lt;k2&gt; …', 'Set per-player komi for new games. One value per player, each &gt;= 0')}
            ${row('mpl &lt;num|-&gt;',           "Set maximum number of plies for new games ('-' = unlimited)")}
            ${head('Game Presets')}
            ${[...this.presets.keys()].map(name => row(name, _presetDescriptions.get(name) ?? '')).join('\n            ')}
            ${head('Online Multiplayer')}
            ${row('register &lt;name&gt; &lt;password&gt;', 'Create an account and log in as it')}
            ${row('login &lt;name&gt; &lt;password&gt;',    'Log in to play online games')}
            ${row('flogin &lt;name&gt; &lt;password&gt;',   'Log in, taking over from another connection already logged in as this name')}
            ${row('tfpro',                 'Toggle fixed online player order (sol/soe/soi vs adde/addl/addi)')}
            ${row('sol &lt;num&gt;',      'Mark player slot &lt;num&gt; as local (you) before newo - fixed order only')}
            ${row('soe &lt;num&gt; [sim] [t]', 'Mark player slot &lt;num&gt; as server engine; optional sim count and temperature - fixed order only')}
            ${row('soi &lt;num&gt; &lt;name&gt;', 'Reserve player slot &lt;num&gt; for an invited username, pending their acceptance - fixed order only')}
            ${row('adde [sim] [t]',       'Append a server-engine player to random order; optional sim count and temperature - random order only')}
            ${row('addl',                 'Append yourself (local) to random order - random order only')}
            ${row('addi &lt;name&gt;',    'Append an invited username to random order, pending their acceptance - random order only')}
            ${row('newo',                 'Create online game with current config; prints game ID')}
            ${row('joino &lt;ID&gt;',     'Join an existing online game by ID')}
            ${row('swl &lt;ID&gt;',       'Switch active view to a local game by ID')}
            ${row('swo &lt;ID&gt;',       'Switch active view to an online game by ID')}
            ${row('swf &lt;ID&gt;',       'Switch active view to a finished online game by ID')}
            ${head('Board Types')}
            ${[..._cmdToBoard.entries()].map(([cmd, { argStr, desc }]) => row(`${cmd} ${argStr}`, desc)).join('\n            ')}
        </table>`;
    }

    // Makes `id` the active game and cancels any in-flight engine request for
    // the previous one - the same switching logic as the 'swl'/'swo'/'swf'
    // commands (_parseCommand), reused by the clickable game-record buttons
    // below. Caller is responsible for re-rendering afterward.
    private _switchToGame(id: string) {
        this.engineManager.cancel();
        this.activeIdx = id;
    }

    // Builds one full-width, clickable button per game id into el - clicking
    // one switches to that game, then does the same two mode changes as the
    // "Start New Local Game" button: dock/hide the panel (docking isn't usable on a
    // narrow screen - see _screenIsSmall()'s doc comment) and jump to Status
    // so the switched-to game's state is what the player sees immediately
    // (_navigateSidePanel() triggers its own _render()). Shared by the three
    // ActiveLocalGames/ActiveOnlineGames/FinishedOnlineGames side-panel nodes
    // below (PendingGames is the one Game-Records child that ISN'T
    // clickable - a pending game has no board state yet to switch to).
    private _renderGameButtons(el: HTMLDivElement, ids: string[], label: (id: string) => string) {
        el.innerHTML = '';
        for (const id of ids) {
            const btn = document.createElement('button');
            btn.className = 'panel-child-btn truncate-line';
            btn.innerHTML = label(id);
            btn.addEventListener('click', () => {
                this._switchToGame(id);
                this.panelMode = this._screenIsSmall() ? 'hidden' : 'locked';
                this._applyPanelMode();
                this._navigateSidePanel(SidePanelContent.Status);
            });
            el.appendChild(btn);
        }
    }

    // Turn list + [game ID] - same content shape as currentGameSetupHtml/
    // newGameSetupHtml's own "Turn list:" line (fmtTurnList), so a game
    // record reads consistently with the rest of the UI; truncated with an
    // ellipsis rather than wrapping (see the 'truncate-line' CSS class) since
    // a long turn list would otherwise push the game ID off-screen.
    private _fmtGameRecordLabel(id: string, config: GameConfig): string {
        return `${fmtTurnList(config.turnList, config.players)}${'&emsp;'.repeat(2)}[${id}]`;
    }

    private _renderActiveLocalGames() {
        const ids = [...this.activeGames.keys()].filter(k => k.startsWith('L_'));
        this._renderGameButtons(this.activeLocalGamesPanel, ids,
            id => this._fmtGameRecordLabel(id.slice(2), this.activeGames.get(id)!.config));
    }

    private _renderActiveOnlineGames() {
        const ids = [...this.activeGames.keys()].filter(k => k.startsWith('O_'));
        this._renderGameButtons(this.activeOnlineGamesPanel, ids,
            id => this._fmtGameRecordLabel(id.slice(2), this.activeGames.get(id)!.config));
    }

    private _renderFinishedOnlineGames() {
        const ids = [...this.finishedGames.keys()].filter(k => k.startsWith('O_'));
        this._renderGameButtons(this.finishedOnlineGamesPanel, ids,
            id => this._fmtGameRecordLabel(id.slice(2), this.finishedGames.get(id)!.config));
    }

    // Builds the Account side-panel node's content: a username/password
    // login form (reusing _login(), same as the 'login' command) if signed
    // out, or a "Username: <name>" line plus a Log Out button if signed in.
    // See the call site in _refreshSidePanel() for why this is only invoked
    // on navigation/login-state-change, not on every _render().
    private _renderAccountPanel() {
        this.accountPanel.innerHTML = '';
        if (this.userName) {
            const nameLine = document.createElement('div');
            nameLine.innerHTML = `<b>Username:</b> ${this.userName}`;
            const logoutBtn = document.createElement('button');
            logoutBtn.className = 'panel-child-btn';
            logoutBtn.textContent = 'Log out';
            logoutBtn.addEventListener('click', () => {
                this.userName = null;
                this._setCmdOutput('Logged out');
                this._renderAccountPanel();
                this._render();
            });
            this.accountPanel.append(nameLine, logoutBtn);
            return;
        }

        const form = document.createElement('div');
        form.className = 'account-form';

        const userLabel = document.createElement('div');
        userLabel.innerHTML = '<b>Username</b>';
        const userInput = document.createElement('input');
        userInput.type = 'text';
        userInput.className = 'account-input';
        userInput.autocomplete = 'username';

        const passLabel = document.createElement('div');
        passLabel.innerHTML = '<b>Password</b>';
        const passInput = document.createElement('input');
        passInput.type = 'password';
        passInput.className = 'account-input';
        passInput.autocomplete = 'current-password';

        form.append(userLabel, userInput, passLabel, passInput);

        const loginBtn = document.createElement('button');
        loginBtn.className = 'panel-child-btn';
        loginBtn.textContent = 'Log in';
        loginBtn.addEventListener('click', () => {
            void this._login(userInput.value, passInput.value).then(() => this._renderAccountPanel());
        });

        const registerBtn = document.createElement('button');
        registerBtn.className = 'panel-child-btn';
        registerBtn.textContent = 'Register';
        registerBtn.addEventListener('click', () => {
            void this._register(userInput.value, passInput.value).then(() => this._renderAccountPanel());
        });

        const btnRow = document.createElement('div');
        btnRow.className = 'btn-row';
        btnRow.append(loginBtn, registerBtn);

        this.accountPanel.append(form, btnRow);
    }

    // Same action as the 'newo' command (_parseCommand) - creates an online
    // game from the current newCfg/onlinePlayerRequest; _createOnlineGame()
    // already handles the not-logged-in/error cases (via _setCmdOutput) and
    // calls _render() itself on success. Deliberately does NOT navigate to
    // Status - a created online game may still be "waiting" on other players
    // (see _createOnlineGame()'s status field), so there's no game to show
    // yet; navigate to Pending Games instead once creation actually
    // succeeds. The panel only jumps to Status once the game actually starts
    // (see _activatePendingGame(), triggered by the game/start event) - the
    // panel display mode switches immediately regardless, same as "Start New
    // Game". Shared by #new-game-buttons (_refreshSidePanel()) and the
    // Configure Online Players panel (_renderConfigureOnlinePlayers()).
    // Shared by #new-game-buttons (_refreshSidePanel()) and the Configure
    // Players panel (_renderConfigureOnlinePlayers()) - same pattern as
    // _buildStartOnlineGameBtn() just below.
    private _buildStartLocalGameBtn(): HTMLButtonElement {
        const btn = document.createElement('button');
        btn.className = 'panel-child-btn';
        btn.textContent = 'Start New Local Game';
        btn.addEventListener('click', () => {
            // Deferred to onStarted - _createLocalGame() may show a
            // confirm popup first (invited players in Configure Players),
            // in which case the game (and this panel-mode switch/navigate)
            // only happen once the user actually answers Yes.
            this._startNewGame(() => {
                // Docking isn't usable on a narrow screen (see _screenIsSmall()'s
                // doc comment) - fall back to hiding the panel instead, same
                // choice init() makes for the initial panelMode.
                this.panelMode = this._screenIsSmall() ? 'hidden' : 'locked';
                this._applyPanelMode();
                // Jump to Status so the newly-started game's state is what the
                // player sees immediately - _navigateSidePanel() already
                // triggers its own _render(), so no separate call needed here.
                this._navigateSidePanel(SidePanelContent.Status);
            });
        });
        return btn;
    }

    private _buildStartOnlineGameBtn(): HTMLButtonElement {
        const btn = document.createElement('button');
        btn.className = 'panel-child-btn';
        btn.textContent = 'Start New Online Game';
        btn.addEventListener('click', () => {
            this.panelMode = this._screenIsSmall() ? 'hidden' : 'locked';
            this._applyPanelMode();
            void this._createOnlineGame().then(success => {
                if (success) this._navigateSidePanel(SidePanelContent.PendingGames);
            });
        });
        return btn;
    }

    // Builds the "Configure Online Players" side-panel node's content -
    // a clickable UI for the same onlinePlayerRequest state the
    // tfpro/sol/soe/adde/addl commands mutate (_parseCommand), reusing each
    // command's exact mutation body as a button's click handler, plus a new
    // "Clear" action (delete a fixed slot's assignment) with no command
    // equivalent. Unlike _renderAccountPanel(), this holds no persistent
    // text-input state, so it's safe to rebuild on every _render() (see the
    // call site there) as well as after each of its own button clicks.
    private _renderConfigureOnlinePlayers() {
        const req = this.onlinePlayerRequest;
        // A slot number left over from before numPlayers shrank no longer
        // refers to a real row - drop it rather than render a textbox for a
        // slot that isn't shown anymore.
        if (typeof this.inviteInputTarget === 'number' && this.inviteInputTarget > this.newCfg.numPlayers) {
            this.inviteInputTarget = null;
            this.inviteInputValue = '';
        }
        this.configureOnlinePlayersPanel.innerHTML = '';

        const toggleBtn = document.createElement('button');
        toggleBtn.className = 'panel-child-btn';
        toggleBtn.textContent = req.fixed ? 'Switch to Random Order' : 'Switch to Fixed Order';
        toggleBtn.addEventListener('click', () => {
            req.fixed = !req.fixed;
            // A fixed-mode slot number (or 'random') has no meaning in the
            // other mode - drop any open invite-textbox rather than carry a
            // stale target across the switch.
            this.inviteInputTarget = null;
            this.inviteInputValue = '';
            this._renderConfigureOnlinePlayers();
        });
        this.configureOnlinePlayersPanel.appendChild(toggleBtn);

        const mkBtn = (className: string, label: string, onClick: () => void, disabled = false) => {
            const b = document.createElement('button');
            b.className = className;
            b.textContent = label;
            b.disabled = disabled;
            b.addEventListener('click', () => { onClick(); this._renderConfigureOnlinePlayers(); });
            return b;
        };

        const fmtStatus = (pi: PlayerInfo | undefined) =>
            !pi ? 'Empty' : pi.type === 'local' ? 'Local' : pi.type === 'serverEngine' ? 'Engine' : `${pi.name} (invited)`;

        // Builds the inline textbox+Confirm row shown right below whichever
        // Invite button was clicked - onConfirm gets the trimmed, non-empty
        // name (a no-op confirm click on an empty box does nothing).
        const buildInviteRow = (onConfirm: (name: string) => void) => {
            const row = document.createElement('div');
            row.className = 'colp-invite-row';
            const input = document.createElement('input');
            input.type = 'text';
            input.className = 'account-input';
            input.placeholder = 'Username to invite';
            input.value = this.inviteInputValue;
            input.addEventListener('input', () => { this.inviteInputValue = input.value; });
            const confirmBtn = document.createElement('button');
            confirmBtn.className = 'status-login-btn';
            confirmBtn.textContent = 'Confirm';
            confirmBtn.addEventListener('click', () => {
                const name = this.inviteInputValue.trim();
                this.inviteInputTarget = null;
                this.inviteInputValue = '';
                if (name) onConfirm(name);
                this._renderConfigureOnlinePlayers();
            });
            row.append(input, confirmBtn);
            return row;
        };

        if (req.fixed) {
            // One grid, numPlayers rows x 5 columns (Slot/status/Local/Engine/Invite/Clear)
            // - a plain flow of 6N (+1 per open invite row) children in
            // row-major order, laid out by .colp-grid's grid-template-columns
            // (CSS), rather than one wrapping row div per slot.
            const grid = document.createElement('div');
            grid.className = 'colp-grid';
            for (let slot = 1; slot <= this.newCfg.numPlayers; slot++) {
                const pi = req.fixedOrder.get(slot);
                const slotLabel = document.createElement('span');
                slotLabel.textContent = `Slot ${slot}: ${fmtStatus(pi)}`;

                const localBtn  = mkBtn('status-login-btn', 'Local',  () => req.fixedOrder.set(slot, new PlayerInfo('local', this.userName ?? 'Player')));
                const engineBtn = mkBtn('status-login-btn', 'Engine', () => req.fixedOrder.set(slot, new PlayerInfo('serverEngine', 'Engine', this.emNumSims, this.emTemperature)));
                const inviteBtn = mkBtn('status-login-btn', 'Invite', () => {
                    this.inviteInputTarget = this.inviteInputTarget === slot ? null : slot;
                    this.inviteInputValue = '';
                });
                const clearBtn  = mkBtn('status-login-btn', 'Clear',  () => req.fixedOrder.delete(slot));

                grid.append(slotLabel, localBtn, engineBtn, inviteBtn, clearBtn);
                if (this.inviteInputTarget === slot)
                    grid.appendChild(buildInviteRow(name => req.fixedOrder.set(slot, new PlayerInfo('pendingInvitedOnline', name))));
            }
            this.configureOnlinePlayersPanel.appendChild(grid);
        } else {
            const listLine = document.createElement('div');
            listLine.innerHTML = `List of Players: ${req.randomOrder
                .map(pi => pi.type === 'local' ? 'Local' : pi.type === 'serverEngine' ? 'Engine' : `${pi.name} (invited)`)
                .join('&nbsp;'.repeat(3))}`;
            this.configureOnlinePlayersPanel.appendChild(listLine);

            const atCap = req.randomOrder.length >= this.newCfg.numPlayers;
            const isEmpty = req.randomOrder.length === 0;
            const btnRow = document.createElement('div');
            btnRow.className = 'btn-row';
            btnRow.append(
                mkBtn('panel-child-btn', 'Add Local',  () => req.randomOrder.push(new PlayerInfo('local', this.userName ?? 'Player')), atCap),
                mkBtn('panel-child-btn', 'Add Engine', () => req.randomOrder.push(new PlayerInfo('serverEngine', 'Engine', this.emNumSims, this.emTemperature)), atCap),
                mkBtn('panel-child-btn', 'Invite', () => {
                    this.inviteInputTarget = this.inviteInputTarget === 'random' ? null : 'random';
                    this.inviteInputValue = '';
                }, atCap),
                mkBtn('panel-child-btn', 'Remove Last', () => req.randomOrder.pop(), isEmpty),
            );
            this.configureOnlinePlayersPanel.appendChild(btnRow);
            if (this.inviteInputTarget === 'random')
                this.configureOnlinePlayersPanel.appendChild(
                    buildInviteRow(name => req.randomOrder.push(new PlayerInfo('pendingInvitedOnline', name))));
        }

        const startBtnRow = document.createElement('div');
        startBtnRow.className = 'btn-row';
        startBtnRow.append(this._buildStartLocalGameBtn(), this._buildStartOnlineGameBtn());
        this.configureOnlinePlayersPanel.appendChild(startBtnRow);
    }

    // Rebuilds #popup-overlay from currentPopup and syncs popUp/the
    // body.popup-active class that disables the rest of the UI (see
    // index.html) - same "clear innerHTML, rebuild via createElement"
    // convention as every other panel. Safe to call unconditionally on every
    // _render() (see the call site there): unlike _renderConfigureOnlinePlayers(),
    // there's no persistent text-input state here to lose.
    renderPopup() {
        this.popUp = this.currentPopup !== null;
        document.body.classList.toggle('popup-active', this.popUp);
        this.popupOverlay.hidden = !this.popUp;
        this.popupOverlay.innerHTML = '';
        if (!this.currentPopup) return;

        const box = document.createElement('div');
        box.className = 'popup-box';
        const text = document.createElement('div');
        const btnRow = document.createElement('div');
        btnRow.className = 'btn-row';

        if (this.currentPopup.kind === 'invite') {
            const { id, from } = this.currentPopup;
            text.textContent = `${from} is inviting you to game ${id}`;
            const acceptBtn = document.createElement('button');
            acceptBtn.className = 'panel-child-btn';
            acceptBtn.textContent = 'Accept';
            acceptBtn.addEventListener('click', () => void this._respondToInvite(id, true));
            const refuseBtn = document.createElement('button');
            refuseBtn.className = 'panel-child-btn';
            refuseBtn.textContent = 'Refuse';
            refuseBtn.addEventListener('click', () => void this._respondToInvite(id, false));
            btnRow.append(acceptBtn, refuseBtn);
        } else if (this.currentPopup.kind === 'confirm') {
            const { message, onYes, onNo } = this.currentPopup;
            text.textContent = message;
            const yesBtn = document.createElement('button');
            yesBtn.className = 'panel-child-btn';
            yesBtn.textContent = 'Yes';
            yesBtn.addEventListener('click', () => { onYes(); this._dismissPopup(); });
            const noBtn = document.createElement('button');
            noBtn.className = 'panel-child-btn';
            noBtn.textContent = 'No';
            noBtn.addEventListener('click', () => { onNo(); this._dismissPopup(); });
            btnRow.append(yesBtn, noBtn);
        } else if (this.currentPopup.kind === 'login-prompt') {
            text.textContent = 'Please log in to play online games';
            const loginBtn = document.createElement('button');
            loginBtn.className = 'panel-child-btn';
            loginBtn.textContent = 'Login now';
            loginBtn.addEventListener('click', () => { this._goToLoginPanel(); this._dismissPopup(); });
            const laterBtn = document.createElement('button');
            laterBtn.className = 'panel-child-btn';
            laterBtn.textContent = 'Later';
            laterBtn.addEventListener('click', () => this._dismissPopup());
            btnRow.append(loginBtn, laterBtn);
        } else {
            text.textContent = this.currentPopup.message;
            const okBtn = document.createElement('button');
            okBtn.className = 'panel-child-btn';
            okBtn.textContent = 'Ok';
            okBtn.addEventListener('click', () => this._dismissPopup());
            btnRow.appendChild(okBtn);
        }
        box.append(text, btnRow);
        this.popupOverlay.appendChild(box);
    }

    // Pulls the next queued popup into currentPopup if nothing is currently
    // showing (queueing, not interrupting, per the design - multiple
    // invite/invite-failed events show one at a time), then re-renders.
    private _advancePopupQueue() {
        if (!this.currentPopup && this.popupQueue.length > 0) this.currentPopup = this.popupQueue.shift()!;
        this.renderPopup();
    }

    private _dismissPopup() {
        this.currentPopup = null;
        this._advancePopupQueue();
        this._render();
    }

    private async _respondToInvite(id: string, accept: boolean) {
        try {
            await conn.request('game/invite-respond', { id, accept }).promise;
        } catch (e: any) { this._setCmdOutput(`Error: ${e.message}`); }
        this._dismissPopup();
    }

    // Not clickable (see _renderGameButtons's doc comment) - just a read-only
    // listing, one row per pending game.
    private _renderPendingGames() {
        this.pendingGamesPanel.innerHTML = [...this.pendingGames.values()]
            .map(pg => `<div class="truncate-line">${this._fmtGameRecordLabel(pg.id, pg.config)}</div>`)
            .join('');
    }

    private _renderStatus(v: BoardView) {
        const lm  = v.lastMove;

        let lastMoveStr = '';
        if      (lm.moveType === MoveType.NOMOVE)   lastMoveStr = '—';
        else if (v.gameOver) {
            // v.gameOver (not lm.allPassed - a maxPlies/resignation ending
            // isn't "all passed") means the game has ended, so winners is non-null.
            const winnerNames = v.winners!.map(w => `P${w}`);
            lastMoveStr = v.winners!.length === 1
                ? `Game over, ${winnerNames[0]} wins`
                : `Game over, tied: ${winnerNames.join(', ')}`;
        }
        else if (lm.moveType === MoveType.PLACE)    lastMoveStr = `${coloredStoneCircle(lm.stone!)}@${lm.pos}†${lm.captures.length}`;
        else if (lm.moveType === MoveType.PASS)     lastMoveStr = 'Pass';

        // Renders e.g. "⬤ 3   ⬤ 5" with each circle colored by its stone type.
        // Uses &nbsp; since this is inserted as innerHTML, where plain runs of
        // spaces would otherwise collapse to a single space.
        const fmtCounts = (counts: Record<number, number>) =>
            Object.entries(counts)
                .map(([s, c]) => `${coloredStoneCircle(Number(s))}&nbsp;${c}`)
                .join('&nbsp;&nbsp;&nbsp;');
        const stoneLine     = fmtCounts(v.score.stoneCount);
        const territoryLine = fmtCounts(v.score.territory);
        // Renders e.g. "P1:0  P2:3" (two spaces between players) - one entry
        // per player, unlike stoneLine/territoryLine above which are
        // stone-indexed (captureCount is player-indexed - see ScoreData).
        const captureLine = v.score.captureCount.map((c, i) => `P${i + 1}:${c}`).join('&nbsp;&nbsp;');

        const randomEvaled = this._active.randomEvaled;
        const evalStr = randomEvaled
            ? Object.entries(randomEvaled).map(([p, w]) => `P${p} ${(w as number).toFixed(1)}`).join(' | ')
            : 'None';

        // Renders e.g. "⬤ P1:5  P2:2   ⬤ P2:1" (two spaces between players,
        // three between stones), each stone's circle followed by the number
        // placed so far by every player who has placed at least one - mirrors
        // fmtPlaceLimit's layout, but showing actual running counts
        // (BoardState.playerStonePlaceCnt()) rather than the configured
        // limit. A player with a zero count for a stone is omitted from that
        // stone's entry, and a stone nobody has placed at all is omitted
        // entirely.
        const fmtPlaceCnt = (cnt: number[][]) =>
            cnt
                .map((row, i) => {
                    const entries = row
                        .map((c, j) => c > 0 ? `P${j + 1}:${c}` : null)
                        .filter((s): s is string => s !== null);
                    return entries.length > 0 ? `${coloredStoneCircle(i + 1)}&nbsp;${entries.join('&nbsp;&nbsp;')}` : null;
                })
                .filter((s): s is string => s !== null)
                .join('&nbsp;&nbsp;&nbsp;');
        const nameLine = this.userName
            ? `<div><b>Your Name:</b> ${this.userName}</div>`
            : `<div><b>Please login to play online games</b> <span id="status-login-btn-slot"></span></div>`;
        this.statusPanel.innerHTML = `
            ${nameLine}
            <div><b>Game ID:</b> ${this.activeIdx.slice(2)}</div>
            <div><b>Turn list:</b> ${fmtTurnList(this._active.config.turnList, this._active.config.players)}</div>
            <div><b>To move:</b> ${fmtTurnList([v.turnList[v.plyCount % v.turnList.length]], this._active.config.players)}</div>
            <div><b>Last move:</b> ${lastMoveStr}</div>
            <div><b>Stones:</b> ${stoneLine}</div>
            <div><b>Territory:</b> ${territoryLine}</div>
            <div><b>Captures:</b> ${captureLine}</div>
            <div><b>Ply:</b> ${v.plyCount}</div>
            <div><b>Stones placed:</b> ${fmtPlaceCnt(v.history[v.history.length - 1].playerStonePlaceCnt)}</div>
            <div><b>AI engine:</b> ${this.aiEngineReady ? 'ready' : 'unavailable'}</div>
            <div><b>Engine sims per move:</b> ${this.emNumSims ?? 'default'}</div>
            <div><b>Engine temperature:</b> ${this.emTemperature}</div>
            <div><b>Self play:</b> ${this.selfPlay}</div>
            <div><b>Auto forced:</b> ${this.autoForced}</div>
            <div><b>Show Territory:</b> ${this.showTerritory}</div>
            <div><b>Show Illegal Moves:</b> ${this.showIllegalMoves}</div>
            <div><b>Evaluation:</b> ${evalStr}</div>
        `;

        // The login prompt's button needs a click listener, so it's built
        // via DOM API and swapped in for its placeholder rather than being
        // part of the innerHTML template above (same reason _renderAccountPanel
        // isn't a plain template - see its doc comment).
        if (!this.userName) {
            const loginBtn = document.createElement('button');
            loginBtn.className = 'status-login-btn';
            loginBtn.textContent = 'Log in';
            loginBtn.addEventListener('click', () => this._navigateSidePanel(SidePanelContent.Account));
            this.statusPanel.querySelector('#status-login-btn-slot')?.replaceWith(loginBtn);
        }
    }

    private _onBoardClick(e: MouseEvent) {
        // Local games used to let any click through unconditionally (every
        // slot was 'local'); now a 'localEngine' slot's turn must not be
        // playable by hand - _isMyTurn() already returns false for it (and
        // for a non-turn online slot, where this simply pre-empts what
        // _tryMakeMove()'s own check already silently no-ops on today).
        if (!this._isMyTurn()) return;
        const v    = this._active.bs.getView();
        const rect = this.mainSvg.getBoundingClientRect();
        const mx   = e.clientX - rect.left;
        const my   = e.clientY - rect.top;

        if (this.selectingStone) {
            // The board itself isn't otherwise clickable while selecting -
            // only the popup circles are - but a click that misses every
            // circle cancels the selection instead of being ignored, since
            // Pass is no longer double-purposed as Cancel (see passBtn).
            for (const { stone, x, y, r } of this._stonePopupCircles(v)) {
                if (Math.hypot(mx - x, my - y) < r) {
                    const pos = this.pendingPos!;
                    this.selectingStone = false;
                    this.pendingPos = null;
                    this._tryMakeMove(pos, stone);
                    return;
                }
            }
            this.selectingStone = false;
            this.pendingPos = null;
            this._render();
            return;
        }

        const { originX, originY, cell, stone_r } =
            boardLayout(v, this.mainBoardSize, this.mainBoardSize);

        let bestDist = Infinity, bestId = -1;
        for (let i = 0; i < v.N; i++) {
            const [bx, by] = v.pos[i];
            const sx = originX + bx * cell, sy = originY - by * cell;
            const dist = Math.hypot(mx - sx, my - sy);
            if (dist < bestDist) { bestDist = dist; bestId = i; }
        }
        if (bestId >= 0 && bestDist < stone_r * 1.3) {
            if (this._active.displayPlyNum !== v.plyCount) return;
            const legalStones = [...v.history[v.plyCount].legalMoves.legalsForLocation[bestId]];
            if (legalStones.length === 0) {
                // no legal move at this location for any offered stone - do nothing
            } else if (legalStones.length === 1) {
                this._tryMakeMove(bestId, legalStones[0]);
            } else {
                this.selectingStone = true;
                this.pendingPos = bestId;
                this._render();
            }
        }
    }

    // Shared by the 'new' command and the New Game side-panel node's
    // "Start New Local Game" button (built in _refreshSidePanel(), #new-game-buttons).
    // onStarted (if given) fires once the game actually gets registered -
    // which may happen synchronously (below) or, if onlinePlayerRequest
    // resolves to any invited slots, only once the user confirms the
    // "ignore invited players?" popup below (or never, if they decline) -
    // see _createLocalGame()'s own doc comment.
    private _startNewGame(onStarted?: () => void) {
        const entry = _cmdToBoard.get(this.newCfg.boardType)!;
        this._createLocalGame(entry.fn(...this.newCfg.boardArgs), onStarted);
    }

    // Called by a Select-Game-Preset button click (see renderGamePresetSelection,
    // sidePanel.ts) - same newCfg-overwrite as the 'preset <name>' command
    // (_parseCommand), plus navigating back to New Game to show the result;
    // silently does nothing for an unknown name, since the button list is
    // always built from this.presets' own keys.
    private _selectPreset(name: string) {
        const p = this.presets.get(name);
        if (!p) return;
        this.newCfg = p.copy();
        this.onlinePlayerRequest = new OnlinePlayerRequest();
        this._navigateSidePanel(SidePanelContent.NewGame);
    }

    // Resolves onlinePlayerRequest (Configure Players) into this local game's
    // players, same as online game creation does, but with local-specific
    // normalization: empty slots and discarded invites both become plain
    // 'local'; 'serverEngine' (online-only) becomes 'localEngine' (this
    // game's own client-driven equivalent - see _fireEngineMove()'s
    // auto-advance in _render()); 'local' passes through as-is. If any slot
    // resolves to an invite, confirms via a popup first ("Ignore invited
    // players for new local game?") - the game (and onStarted) only proceed
    // if the user answers Yes; declining leaves the current game untouched.
    private _createLocalGame(bc: BoardConfig, onStarted?: () => void) {
        const config = this.newCfg.copy();
        const request = this.onlinePlayerRequest.copy();  // defensive copy, same pattern as _createOnlineGame()
        let resolved: Map<number, PlayerInfo>;
        try {
            resolved = request.resolve(config.numPlayers);
        } catch (e: any) {
            this._setCmdOutput(`Error: ${e.message}`);
            return;
        }

        const proceed = () => {
            config.players = new Map();
            for (let slot = 1; slot <= config.numPlayers; slot++) {
                const pi = resolved.get(slot);
                if (!pi || pi.type === 'pendingInvitedOnline') config.players.set(slot, new PlayerInfo('local', ''));
                else if (pi.type === 'serverEngine') config.players.set(slot, new PlayerInfo('localEngine', pi.name, pi.emsim, pi.temp));
                else config.players.set(slot, pi);
            }
            const bs = new BoardState(
                config.numStones, config.numPlayers, config.turnList, config.playerStonePlaceLimit, config.globalStonePlaceLimit,
                config.stoneToPlayerMap, config.forcedPassOnly, config.scoreRule, config.komi, config.koRule, config.allowSuicide,
                config.maxPlies, new Array(bc.N).fill(0), bc,
            );
            this._registerGame('L_' + makeId(12), bs, config);
            onStarted?.();
        };

        if (![...resolved.values()].some(pi => pi.type === 'pendingInvitedOnline')) { proceed(); return; }
        this.popupQueue.push({
            kind: 'confirm',
            message: 'Ignore invited players for new local game?',
            onYes: proceed,
            onNo: () => {},
        });
        this._advancePopupQueue();
    }

    private _registerGame(id: string, bs: BoardState, config: GameConfig): void {
        this.engineManager.cancel();
        this.engineManager.sessionId = null;
        this.activeGames.set(id, { bs, config, displayPlyNum: 0, idxShowHistory: 0, randomEvaled: null });
        this.activeIdx = id;
    }

    // Withdraws n real moves (see BoardState.withdrawMove) - shared by the
    // 'w' command and the Withdraw button, so the online/finished-game
    // guards live in exactly one place.
    private _withdrawMove(n: number) {
        if (this.activeIdx.startsWith('O_')) { this._setCmdOutput('Cannot withdraw moves in online games'); return; }
        if (this.finishedGames.has(this.activeIdx)) { this._setCmdOutput('Cannot withdraw moves from a finished game'); return; }
        this.engineManager.cancel();
        this.engineManager.sessionId = null;
        for (let i = 0; i < n; i++) this._active.bs.withdrawMove();
        this._active.displayPlyNum = Math.min(this._active.displayPlyNum, this._active.bs.situations.length - 1);
    }

    // Withdraws down to the currently displayed ply - shared by the 'wcd'
    // command and the WCD button.
    private _withdrawToCurrentDisplay() {
        if (this.activeIdx.startsWith('O_')) { this._setCmdOutput('Cannot withdraw moves in online games'); return; }
        if (this.finishedGames.has(this.activeIdx)) { this._setCmdOutput('Cannot withdraw moves from a finished game'); return; }
        this.engineManager.cancel();
        this.engineManager.sessionId = null;
        const n = this._active.bs.situations.length - 1 - this._active.displayPlyNum;
        for (let i = 0; i < n; i++) this._active.bs.withdrawMove();
    }

    private _parseCommand(raw: string) {
        const parts = raw.trim().split(/\s+/);
        this.cmdOutput.textContent = '';
        if (!parts[0]) return;
        const cmd = parts[0].toLowerCase();
        const posInt = (s: string | undefined) => { const n = Number(s); return Number.isInteger(n) && n > 0 ? n : null; };

        if (cmd === 'register') {
            if (!parts[1] || !parts[2]) { this._setCmdOutput('Usage: register <name> <password>'); return; }
            void this._register(parts[1], parts[2]);
        }
        else if (cmd === 'login') {
            if (!parts[1] || !parts[2]) { this._setCmdOutput('Usage: login <name> <password>'); return; }
            void this._login(parts[1], parts[2]);
        }
        else if (cmd === 'flogin') {
            if (!parts[1] || !parts[2]) { this._setCmdOutput('Usage: flogin <name> <password>'); return; }
            void this._forceLogin(parts[1], parts[2]);
        }
        else if (cmd === 'tfpro') {
            this.onlinePlayerRequest.fixed = !this.onlinePlayerRequest.fixed;
            this._setCmdOutput(`Fixed online player order: ${this.onlinePlayerRequest.fixed}`);
            this._render();
        }
        else if (cmd === 'sol') {
            if (!this.onlinePlayerRequest.fixed) { this._setCmdOutput('sol: only available when fixed order is enabled (see tfpro)'); return; }
            const n = posInt(parts[1]); if (n === null) { this._setCmdOutput('Usage: sol <player-id>'); return; }
            this.onlinePlayerRequest.fixedOrder.set(n, new PlayerInfo('local', this.userName ?? 'Player')); this._render();
        }
        else if (cmd === 'soe') {
            if (!this.onlinePlayerRequest.fixed) { this._setCmdOutput('soe: only available when fixed order is enabled (see tfpro)'); return; }
            const n = posInt(parts[1]); if (n === null) { this._setCmdOutput('Usage: soe <player-id> [emsim] [temp]'); return; }
            const nonNeg = (s: string | undefined) => { const v = Number(s); return s !== undefined && isFinite(v) && v >= 0 ? v : null; };
            const emsim = parts[2] !== undefined ? nonNeg(parts[2]) : this.emNumSims;
            const temp  = parts[3] !== undefined ? nonNeg(parts[3]) : this.emTemperature;
            if (emsim === null || temp === null) { this._setCmdOutput('Usage: soe <player-id> [emsim] [temp]'); return; }
            this.onlinePlayerRequest.fixedOrder.set(n, new PlayerInfo('serverEngine', 'Engine', emsim, temp)); this._render();
        }
        else if (cmd === 'soi') {
            if (!this.onlinePlayerRequest.fixed) { this._setCmdOutput('soi: only available when fixed order is enabled (see tfpro)'); return; }
            const n = posInt(parts[1]); if (n === null || !parts[2]) { this._setCmdOutput('Usage: soi <player-id> <name>'); return; }
            this.onlinePlayerRequest.fixedOrder.set(n, new PlayerInfo('pendingInvitedOnline', parts[2])); this._render();
        }
        else if (cmd === 'adde') {
            if (this.onlinePlayerRequest.fixed) { this._setCmdOutput('adde: only available when fixed order is disabled (see tfpro)'); return; }
            if (this.onlinePlayerRequest.randomOrder.length >= this.newCfg.numPlayers)
                { this._setCmdOutput(`adde: randomOrder is already full (${this.newCfg.numPlayers} players)`); return; }
            const nonNeg = (s: string | undefined) => { const v = Number(s); return s !== undefined && isFinite(v) && v >= 0 ? v : null; };
            const emsim = parts[1] !== undefined ? nonNeg(parts[1]) : this.emNumSims;
            const temp  = parts[2] !== undefined ? nonNeg(parts[2]) : this.emTemperature;
            if (emsim === null || temp === null) { this._setCmdOutput('Usage: adde [emsim] [temp]'); return; }
            this.onlinePlayerRequest.randomOrder.push(new PlayerInfo('serverEngine', 'Engine', emsim, temp)); this._render();
        }
        else if (cmd === 'addl') {
            if (this.onlinePlayerRequest.fixed) { this._setCmdOutput('addl: only available when fixed order is disabled (see tfpro)'); return; }
            if (this.onlinePlayerRequest.randomOrder.length >= this.newCfg.numPlayers)
                { this._setCmdOutput(`addl: randomOrder is already full (${this.newCfg.numPlayers} players)`); return; }
            this.onlinePlayerRequest.randomOrder.push(new PlayerInfo('local', this.userName ?? 'Player')); this._render();
        }
        else if (cmd === 'addi') {
            if (this.onlinePlayerRequest.fixed) { this._setCmdOutput('addi: only available when fixed order is disabled (see tfpro)'); return; }
            if (this.onlinePlayerRequest.randomOrder.length >= this.newCfg.numPlayers)
                { this._setCmdOutput(`addi: randomOrder is already full (${this.newCfg.numPlayers} players)`); return; }
            if (!parts[1]) { this._setCmdOutput('Usage: addi <name>'); return; }
            this.onlinePlayerRequest.randomOrder.push(new PlayerInfo('pendingInvitedOnline', parts[1])); this._render();
        }
        else if (cmd === 'newo') {
            void this._createOnlineGame();
        }
        else if (cmd === 'joino') {
            if (!parts[1]) { this._setCmdOutput('Usage: joino <ID>'); return; }
            void this._joinOnlineGame(parts[1].toUpperCase());
        }
        else if (cmd === 'swl') {
            if (!parts[1]) { this._setCmdOutput('Usage: swl <game-id>'); return; }
            const id = parts[1].startsWith('L_') ? parts[1] : 'L_' + parts[1];
            if (!this.activeGames.has(id)) { this._setCmdOutput(`Local game not found: ${id}`); return; }
            this.engineManager.cancel();
            this.activeIdx = id;
        }
        else if (cmd === 'swo') {
            if (!parts[1]) { this._setCmdOutput('Usage: swo <game-id>'); return; }
            const raw = parts[1].startsWith('O_') ? parts[1].slice(2) : parts[1];
            const id = 'O_' + raw.toUpperCase();
            if (!this.activeGames.has(id)) { this._setCmdOutput(`Online game not found: ${raw.toUpperCase()}`); return; }
            this.engineManager.cancel();
            this.activeIdx = id;
        }
        else if (cmd === 'swf') {
            if (!parts[1]) { this._setCmdOutput('Usage: swf <game-id>'); return; }
            const raw = parts[1].startsWith('O_') ? parts[1].slice(2) : parts[1];
            const id = 'O_' + raw.toUpperCase();
            if (!this.finishedGames.has(id)) { this._setCmdOutput(`Finished online game not found: ${raw.toUpperCase()}`); return; }
            this.engineManager.cancel();
            this.activeIdx = id;
        }
        else if (cmd === 'em') {
            if (this.activeIdx.startsWith('O_')) { this._setCmdOutput('Engine moves are disabled in online mode'); return; }
            const n = posInt(parts[1] ?? '1');
            if (n === null) { this._setCmdOutput('em: n must be a positive integer'); return; }
            if (this.engineManager.register(n)) this._fireEngineMove();
            else console.warn('em: engine move already in progress');
            return;
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
        else if (cmd === 'fpo')  this.newCfg.forcedPassOnly = !this.newCfg.forcedPassOnly;
        else if (cmd === 'ascd') this.newCfg.allowSuicide = !this.newCfg.allowSuicide;
        else if (cmd === 'af')   this.autoForced = !this.autoForced;
        else if (cmd === 'stt')  this.showTerritory = !this.showTerritory;
        else if (cmd === 'simv') this.showIllegalMoves = !this.showIllegalMoves;
        else if (cmd === 'bt') {
            if (!parts[1]) { this._setCmdOutput('Usage: bt <board-type>'); return; }
            if (!_cmdToBoard.has(parts[1])) { this._setCmdOutput(`Unknown board type: ${parts[1]}`); return; }
            const entry = _cmdToBoard.get(parts[1])!;
            this.newCfg.boardType = parts[1];
            this.newCfg.boardArgs = [...this.boardDimensionForNew[entry.boardType as PrescribedBoard]];
        }
        else if (cmd === 'bd') {
            if (!parts[1]) { this._setCmdOutput('Usage: bd <num> <num> …'); return; }
            const nums = parts.slice(1).map(Number);
            if (nums.some(n => !Number.isInteger(n) || n <= 0)) { this._setCmdOutput('bd: all arguments must be positive integers'); return; }
            const boardTypeEnum = _cmdToBoard.get(this.newCfg.boardType)!.boardType as PrescribedBoard;
            for (const [idx, val] of nums.entries())
                if (idx < PrescribedBoardMap[boardTypeEnum][0]) {
                    this.boardDimensionForNew[boardTypeEnum][idx] = val;
                    this.newCfg.boardArgs[idx] = val;
                }
        }
        else if (cmd === 'ns') {
            const n = Number(parts[1]);
            if (!parts[1] || !Number.isInteger(n) || n < 1 || n > 8) { this._setCmdOutput('Usage: ns <n>  (1–8)'); return; }
            const oldTurnList = this.newCfg.turnList;
            const oldPlayerStonePlaceLimit = this.newCfg.playerStonePlaceLimit;
            const oldGlobalStonePlaceLimit = this.newCfg.globalStonePlaceLimit;
            this.newCfg.numStones = n;
            // protected/friendly are resized per surviving turn-index (truncate/
            // zero-extend to the new numStones); entries beyond the old turnList's
            // length start fully unprotected/unfriendly, same as new stone/player
            // defaults below.
            this.newCfg.turnList = Array.from({ length: n }, (_, i) => {
                const oldProtected = oldTurnList[i]?.protected ?? [];
                const protectedStones = Array.from({ length: n }, (_, j) => oldProtected[j] ?? 0);
                const oldFriendly = oldTurnList[i]?.friendly ?? [];
                const friendlyStones = Array.from({ length: n }, (_, j) => oldFriendly[j] ?? 0);
                // Only stone i+1 offered, same single-stone default as before -
                // multi-stone turns are only ever set explicitly, via tl.
                const offeredStones = Array.from({ length: n }, (_, j) => j === i ? 1 : 0);
                return { player: i % this.newCfg.numPlayers + 1, stones: offeredStones, protected: protectedStones, friendly: friendlyStones };
            });
            this.newCfg.stoneToPlayerMap = Object.fromEntries(Array.from({ length: n }, (_, i) => [i + 1, new Set([i % this.newCfg.numPlayers + 1])]));
            // playerStonePlaceLimit's outer (per-stone) axis is resized to n;
            // each surviving stone's inner (per-player) axis keeps its length
            // unchanged (numPlayers isn't touched by ns), null-extended for any
            // newly-added stone row.
            this.newCfg.playerStonePlaceLimit = Array.from({ length: n }, (_, i) => {
                const old = oldPlayerStonePlaceLimit[i] ?? [];
                return Array.from({ length: this.newCfg.numPlayers }, (_, j) => old[j] ?? null);
            });
            // globalStonePlaceLimit is indexed by stone too, so it resizes here
            // (not np, which leaves it untouched).
            this.newCfg.globalStonePlaceLimit = Array.from({ length: n }, (_, i) => oldGlobalStonePlaceLimit[i] ?? null);
        }
        else if (cmd === 'np') {
            const n = Number(parts[1]);
            if (!parts[1] || !Number.isInteger(n) || n < 1 || n > 8) { this._setCmdOutput('Usage: np <n>  (1–8)'); return; }
            this.newCfg.numPlayers = n;
            this.newCfg.stoneToPlayerMap = Object.fromEntries(Array.from({ length: this.newCfg.numStones }, (_, i) => [i + 1, new Set([i % n + 1])]));
            // Keep turnList's player assignments in sync with the new player
            // count (same round-robin as stoneToPlayerMap above), keeping each
            // entry's stone and protected/friendly lists unchanged (numStones is
            // untouched by np, so neither needs resizing here).
            this.newCfg.turnList = this.newCfg.turnList.map((t, i) => ({ player: i % n + 1, stones: t.stones, protected: t.protected, friendly: t.friendly }));
            // Keep komi in sync with the new player count: truncate if shorter,
            // zero-extend (no komi for the new players) if longer.
            this.newCfg.komi = Array.from({ length: n }, (_, i) => this.newCfg.komi[i] ?? 0);
            // Resize each stone's inner (per-player) axis of playerStonePlaceLimit
            // the same way, null-extended (numStones is untouched by np).
            this.newCfg.playerStonePlaceLimit = this.newCfg.playerStonePlaceLimit.map(row => Array.from({ length: n }, (_, j) => row[j] ?? null));
        }
        else if (cmd === 'tl') {
            if (parts.length < 2) { this._setCmdOutput('Usage: tl <player>-<stone bits> <player>-<stone bits> …'); return; }
            const oldTurnList = this.newCfg.turnList;
            const entries: TurnInfo[] = [];
            for (const [i, part] of parts.slice(1).entries()) {
                const pieces = part.split('-');
                if (pieces.length !== 2) { this._setCmdOutput('tl: each entry must be <player>-<stone bits>'); return; }
                const player = Number(pieces[0]);
                const stoneBits = pieces[1];
                if (!Number.isInteger(player) || player < 1 || player > this.newCfg.numPlayers)
                    { this._setCmdOutput(`tl: player must be an integer between 1 and ${this.newCfg.numPlayers}`); return; }
                if (stoneBits.length !== this.newCfg.numStones || !/^[01]+$/.test(stoneBits))
                    { this._setCmdOutput(`tl: stone bits must be a ${this.newCfg.numStones}-character string of 0s and 1s`); return; }
                const stones = stoneBits.split('').map(Number);
                if (!stones.some(s => s === 1))
                    { this._setCmdOutput('tl: each entry must offer at least one stone'); return; }
                // Carry over protected/friendly settings by turn-index; new entries
                // beyond the previous turnList's length start fully unprotected/unfriendly.
                const protectedStones = oldTurnList[i]?.protected ?? new Array(this.newCfg.numStones).fill(0);
                const friendlyStones = oldTurnList[i]?.friendly ?? new Array(this.newCfg.numStones).fill(0);
                entries.push({ player, stones, protected: protectedStones, friendly: friendlyStones });
            }
            this.newCfg.turnList = entries;
        }
        else if (cmd === 'sprot') {
            if (parts.length < 2) { this._setCmdOutput('Usage: sprot <0-1 str> <0-1 str> …'); return; }
            const strs = parts.slice(1);
            if (strs.length !== this.newCfg.turnList.length)
                { this._setCmdOutput(`sprot: expected ${this.newCfg.turnList.length} value(s) (one per turn), got ${strs.length}`); return; }
            if (!strs.every(s => s.length === this.newCfg.numStones && /^[01]+$/.test(s)))
                { this._setCmdOutput(`sprot: each value must be a ${this.newCfg.numStones}-character string of 0s and 1s`); return; }
            this.newCfg.turnList = this.newCfg.turnList.map((t, i) => ({ ...t, protected: strs[i].split('').map(Number) }));
        }
        else if (cmd === 'sfriend') {
            if (parts.length < 2) { this._setCmdOutput('Usage: sfriend <0-1 str> <0-1 str> …'); return; }
            const strs = parts.slice(1);
            if (strs.length !== this.newCfg.turnList.length)
                { this._setCmdOutput(`sfriend: expected ${this.newCfg.turnList.length} value(s) (one per turn), got ${strs.length}`); return; }
            if (!strs.every(s => s.length === this.newCfg.numStones && /^[01]+$/.test(s)))
                { this._setCmdOutput(`sfriend: each value must be a ${this.newCfg.numStones}-character string of 0s and 1s`); return; }
            this.newCfg.turnList = this.newCfg.turnList.map((t, i) => ({ ...t, friendly: strs[i].split('').map(Number) }));
        }
        else if (cmd === 'spm') {
            if (parts.length < 4 || parts[1] !== 's' || parts[3] !== 'p')
                { this._setCmdOutput('Usage: spm s <stone> p <player> <player> …'); return; }
            const stone = Number(parts[2]);
            if (!Number.isInteger(stone) || stone < 1 || stone > this.newCfg.numStones)
                { this._setCmdOutput(`spm: stone must be an integer between 1 and ${this.newCfg.numStones}`); return; }
            const players = parts.slice(4).map(Number);
            if (!players.every(p => Number.isInteger(p) && p >= 1 && p <= this.newCfg.numPlayers))
                { this._setCmdOutput(`spm: each player must be an integer between 1 and ${this.newCfg.numPlayers}`); return; }
            this.newCfg.stoneToPlayerMap[stone] = new Set(players);
        }
        else if (cmd === 'spspl') {
            if (parts.length < 3 || parts[2] !== 's')
                { this._setCmdOutput('Usage: spspl <player-id> s <num|-> <num|-> …'); return; }
            const player = Number(parts[1]);
            if (!Number.isInteger(player) || player < 1 || player > this.newCfg.numPlayers)
                { this._setCmdOutput(`spspl: player must be an integer between 1 and ${this.newCfg.numPlayers}`); return; }
            const toks = parts.slice(3);
            if (toks.length !== this.newCfg.numStones)
                { this._setCmdOutput(`spspl: expected ${this.newCfg.numStones} value(s) (one per stone), got ${toks.length}`); return; }
            const limits: (number | null)[] = [];
            for (const tok of toks) {
                if (tok === '-') { limits.push(null); continue; }
                const n = Number(tok);
                if (!Number.isInteger(n) || n < 0) { this._setCmdOutput(`spspl: each value must be a non-negative integer or '-'`); return; }
                limits.push(n);
            }
            limits.forEach((lim, i) => { this.newCfg.playerStonePlaceLimit[i][player - 1] = lim; });
        }
        else if (cmd === 'sgspl') {
            const toks = parts.slice(1);
            if (toks.length !== this.newCfg.numStones)
                { this._setCmdOutput(`sgspl: expected ${this.newCfg.numStones} value(s) (one per stone), got ${toks.length}`); return; }
            const limits: (number | null)[] = [];
            for (const tok of toks) {
                if (tok === '-') { limits.push(null); continue; }
                const n = Number(tok);
                if (!Number.isInteger(n) || n < 0) { this._setCmdOutput(`sgspl: each value must be a non-negative integer or '-'`); return; }
                limits.push(n);
            }
            this.newCfg.globalStonePlaceLimit = limits;
        }
        else if (cmd === 'sr') {
            const rules: ScoreRule[] = ['stone', 'territoryonly', 'area', 'territory'];
            if (!rules.includes(parts[1] as ScoreRule))
                { this._setCmdOutput(`Usage: sr <rule>  (${rules.join(' | ')})`); return; }
            this.newCfg.scoreRule = parts[1] as ScoreRule;
        }
        else if (cmd === 'ko') {
            const koRules: Record<string, KoRule> = { pos: 'positional', sit: 'situational' };
            const rule = parts[1] ? koRules[parts[1]] : undefined;
            if (!rule) { this._setCmdOutput('Usage: ko <pos|sit>'); return; }
            this.newCfg.koRule = rule;
        }
        else if (cmd === 'komi') {
            if (parts.length < 2) { this._setCmdOutput('Usage: komi <k1> <k2> …'); return; }
            const values = parts.slice(1).map(Number);
            if (values.some(v => !Number.isFinite(v)))
                { this._setCmdOutput('komi: each value must be a number'); return; }
            if (values.some(v => v < 0))
                { this._setCmdOutput('komi: each value must be >= 0'); return; }
            if (values.length !== this.newCfg.numPlayers)
                { this._setCmdOutput(`komi: expected ${this.newCfg.numPlayers} value(s) (one per player), got ${values.length}`); return; }
            this.newCfg.komi = values;
        }
        else if (cmd === 'mpl') {
            if (!parts[1]) { this._setCmdOutput('Usage: mpl <num|->'); return; }
            if (parts[1] === '-') { this.newCfg.maxPlies = null; }
            else {
                const n = Number(parts[1]);
                if (!Number.isInteger(n) || n < 1) { this._setCmdOutput(`mpl: value must be a positive integer or '-'`); return; }
                this.newCfg.maxPlies = n;
            }
        }
        else if (cmd === 'preset') {
            if (!parts[1]) { this._setCmdOutput('Usage: preset <name>'); return; }
            const p = this.presets.get(parts[1]);
            if (!p) { this._setCmdOutput(`Unknown preset: ${parts[1]} (known: ${[...this.presets.keys()].join(', ')})`); return; }
            this.newCfg = p.copy();
            this.onlinePlayerRequest = new OnlinePlayerRequest();
        }
        else if (cmd === 'h') {
            const n = posInt(parts[1]);
            if (n === null) { this._setCmdOutput('Usage: h <n>  (positive integer)'); return; }
            this.nShowHistory = n;
        }
        else if (cmd === 'w') {
            const n = posInt(parts[1]);
            if (n === null) { this._setCmdOutput('Usage: w <n>  (positive integer)'); return; }
            this._withdrawMove(n);
        }
        else if (cmd === 'wcd') {
            this._withdrawToCurrentDisplay();
        }
        else if (cmd === 'fw') {
            const n = posInt(parts[1]);
            if (n === null) { this._setCmdOutput('Usage: fw <n>  (positive integer)'); return; }
            this._active.displayPlyNum = Math.min(this._active.displayPlyNum + n, this._active.bs.situations.length - 1);
        }
        else if (cmd === 'bw') {
            const n = posInt(parts[1]);
            if (n === null) { this._setCmdOutput('Usage: bw <n>  (positive integer)'); return; }
            this._active.displayPlyNum = Math.max(this._active.displayPlyNum - n, 0);
        }
        else if (cmd === 're') {
            const n = posInt(parts[1]);
            if (n === null) { this._setCmdOutput('Usage: re <n>  (positive integer)'); return; }
            this._active.randomEvaled = this._active.bs.randomEvaluate(n);
        }
        else if (cmd === 'new') {
            this._startNewGame();
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
                this._active.bs.randomMove();
                if (this._active.bs.gameOver()) {
                    this.selfPlay = false; break;
                }
            }
            this._active.displayPlyNum = this._active.bs.situations.length - 1;
            this._maybeFinish(this.activeIdx);
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

    private _isMyTurn(): boolean {
        const ag = this._active;
        const v = ag.bs.getView();
        if (v.gameOver) return false;
        const pi = ag.config.players.get(v.nextTurn.player);
        return pi?.type === 'local' || (pi?.type === 'client' && pi.name === this.userName);
    }

    private _tryMakeMove(moveIndex: number | null, stone?: number): void {
        if (this.activeIdx.startsWith('O_')) {
            if (this._isMyTurn()) void this._submitOnlineMove(moveIndex, stone);
        } else {
            this.engineManager.cancel();
            this._active.bs.makeMove(moveIndex, stone);
            this._active.displayPlyNum = this._active.bs.getView().plyCount;
            this._maybeFinish(this.activeIdx);
            this._render();
        }
    }

    private async _register(name: string, password: string) {
        try {
            const data = await conn.request<LoginResponse>('REGISTER', { name, password }).promise;
            this.userName = data.name;
            this._addFinishedGames(data.finishedGames);
            this._setCmdOutput(`Registered and logged in as: ${data.name}`);
        } catch (e: any) { this._setCmdOutput(`Registration failed: ${e.message}`); }
        this._render();
    }

    private async _login(name: string, password: string) {
        try {
            const data = await conn.request<LoginResponse>('LOGIN', { name, password }).promise;
            this.userName = data.name;
            this._addFinishedGames(data.finishedGames);
            this._setCmdOutput(`Logged in as: ${data.name}`);
        } catch (e: any) { this._setCmdOutput(`Login failed: ${e.message}`); }
        this._render();
    }

    // Like _login, but takes over from (closes) any other connection already
    // logged in as this username, instead of failing with a conflict.
    private async _forceLogin(name: string, password: string) {
        try {
            const data = await conn.request<LoginResponse>('FLOGIN', { name, password }).promise;
            this.userName = data.name;
            this._addFinishedGames(data.finishedGames);
            this._setCmdOutput(`Logged in as: ${data.name} (took over from other connection)`);
        } catch (e: any) { this._setCmdOutput(`Login failed: ${e.message}`); }
        this._render();
    }

    // Reconstructs each finished online game (sent by the server at login) into a
    // full BoardState via the shared replay logic, and stores it in finishedGames -
    // the same reconstruction path the server itself uses to rebuild finishedGames
    // at startup (see BoardState.fromFinishedGame()).
    private _addFinishedGames(entries: { id: string; finishedGame: any }[]): void {
        for (const { id, finishedGame: raw } of entries) {
            try {
                const fg = FinishedGame.fromJSON(raw);
                const boardEntry = _cmdToBoard.get(fg.config.boardType);
                if (!boardEntry) continue;
                const bc = boardEntry.fn(...fg.config.boardArgs);
                const bs = BoardState.fromFinishedGame(fg, bc);
                this.finishedGames.set('O_' + id, {
                    bs, config: fg.config, displayPlyNum: bs.getView().plyCount,
                    idxShowHistory: 0, randomEvaled: null,
                });
            } catch (e) { console.error('Failed to reconstruct finished game', id, e); }
        }
    }

    // Returns true iff the game/create request succeeded, so callers that
    // care (e.g. the New Game side-panel node's "Start New Online Game"
    // button) can react to success without this method needing to know
    // anything about the side panel itself - the 'newo' command (_parseCommand)
    // just ignores the return value.
    private async _createOnlineGame(): Promise<boolean> {
        if (!this.userName) {
            this.popupQueue.push({ kind: 'login-prompt' });
            this._advancePopupQueue();
            return false;
        }
        const config = this.newCfg.copy();
        const request = this.onlinePlayerRequest.copy();
        const renameLocal = (pi: PlayerInfo) => { if (pi.type === 'local') pi.name = this.userName!; };
        for (const pi of request.fixedOrder.values()) renameLocal(pi);
        for (const pi of request.randomOrder) renameLocal(pi);
        try {
            const { id, status } = await conn.request<{ id: string; status: 'waiting' | 'playing' }>(
                'game/create', { config, onlinePlayerRequest: request }).promise;
            this.onlinePlayerRequest = new OnlinePlayerRequest();
            this._setCmdOutput(status === 'waiting' ? `Game created: ${id}` : `Game started: ${id}`);
            this._render();
            return true;
        } catch (e: any) {
            this.popupQueue.push({ kind: 'create-failed', message: e.message });
            this._advancePopupQueue();
            return false;
        }
    }

    private async _joinOnlineGame(id: string) {
        if (!this.userName) {
            this.popupQueue.push({ kind: 'login-prompt' });
            this._advancePopupQueue();
            return;
        }
        try {
            await conn.request('game/join', { id }).promise;
            this._setCmdOutput(`Joined game: ${id} - waiting for the game to start…`);
            this._render();
        } catch (e: any) { this._setCmdOutput(`Error: ${e.message}`); }
    }

    // Promote a pending game to active once it starts.
    private _activatePendingGame(id: string, config: GameConfig) {
        const boardEntry = _cmdToBoard.get(config.boardType);
        if (!boardEntry) { this._setCmdOutput(`Unknown board type: ${config.boardType}`); return; }
        const bc = boardEntry.fn(...config.boardArgs);
        const bs = new BoardState(
            config.numStones, config.numPlayers,
            config.turnList, config.playerStonePlaceLimit, config.globalStonePlaceLimit, config.stoneToPlayerMap,
            config.forcedPassOnly, config.scoreRule, config.komi, config.koRule, config.allowSuicide,
            config.maxPlies, new Array(bc.N).fill(0), bc,
        );
        this.pendingGames.delete(id);
        this._registerGame('O_' + id, bs, config);
        const localEntries = [...this._active.config.players.entries()].filter(([, pi]) => pi.name === this.userName);
        this._setCmdOutput(`Game started! You are player(s) ${localEntries.map(([s, pi]) => `${s} (${pi.name})`).join(', ')}`);
        // This is the actual "an online game started" moment (as opposed to
        // creating/joining one, which may still be waiting on other players)
        // - jump the side panel to Status now; _navigateSidePanel() already
        // triggers its own _render(), so no separate call needed here.
        this._navigateSidePanel(SidePanelContent.Status);
    }


    private _applyOnlineState(id: string, state: OnlineStateResponse) {
        const ag = this._findGame('O_' + id)!;
        const isActive = 'O_' + id === this.activeIdx;

        const wasGameOver = ag.bs.gameOver();
        // Sync resigned players before replaying moves so auto-passes succeed.
        for (const player of state.resignedPlayers) ag.bs.resign(player);

        // Apply any new moves from the server.
        const plyCount = ag.bs.getView().plyCount;
        if (state.moves.length > plyCount) {
            const wasAtLive = ag.displayPlyNum === plyCount;
            for (let i = plyCount; i < state.moves.length; i++)
                ag.bs.makeMove(state.moves[i].pos, state.moves[i].stone ?? undefined);
            if (wasAtLive) {
                ag.displayPlyNum = ag.bs.getView().plyCount;
            }
            ag.randomEvaled = null;

            if (isActive) this._notifyTurn(ag, wasGameOver);
        }
        this._maybeFinish('O_' + id);

        if (isActive) this._render();
    }

    private _handleOnlineMove(id: string, moveIndex: number | null, stone: number | null) {
        const ag = this._findGame('O_' + id);
        if (!ag) return;
        const wasGameOver = ag.bs.gameOver();
        const wasAtLive = ag.displayPlyNum === ag.bs.getView().plyCount;
        ag.bs.makeMove(moveIndex, stone ?? undefined);
        ag.bs.advanceResigned();
        if (wasAtLive) ag.displayPlyNum = ag.bs.getView().plyCount;
        ag.randomEvaled = null;
        this._maybeFinish('O_' + id);
        const isActive = 'O_' + id === this.activeIdx;
        if (isActive) {
            this._notifyTurn(ag, wasGameOver);
            this._render();
        }
    }

    private _handleOnlineResign(id: string, slots: number[]) {
        const ag = this._findGame('O_' + id);
        if (!ag) return;
        const wasGameOver = ag.bs.gameOver();
        for (const slot of slots) ag.bs.resign(slot);
        ag.bs.advanceResigned();
        this._maybeFinish('O_' + id);
        const isActive = 'O_' + id === this.activeIdx;
        if (isActive) {
            this._notifyTurn(ag, wasGameOver);
            this._render();
        }
    }

    private _notifyTurn(ag: ActiveGame, wasGameOver: boolean) {
        const v = ag.bs.getView();
        if (v.gameOver) {
            if (!wasGameOver) {
                // v.gameOver is true here, so winners is non-null.
                const winnerText = v.winners!.length === 0 ? 'No winners'
                    : v.winners!.length === 1 ? `Player ${v.winners![0]} wins!`
                    : v.winners!.map(w => `Player ${w}`).join(', ') + ' tied.';
                this._setCmdOutput(`Game over! ${winnerText}`);
            }
        } else {
            if (ag.config.players.get(v.nextTurn.player)?.name === this.userName)
                this._setCmdOutput('Your turn!');
            else {
                const slot = v.nextTurn.player;
                const p = ag.config.players.get(slot);
                this._setCmdOutput(p ? `${p.name} [${slot}]'s turn.` : "Opponent's turn.");
            }
        }
    }

    private async _resign() {
        if (this.activeIdx.startsWith('L_')) return;
        try {
            await conn.request('game/resign', { id: this.activeIdx.slice(2) }).promise;
        } catch (e: any) { this._setCmdOutput(`Resign failed: ${e.message}`); }
    }

    private async _submitOnlineMove(moveIndex: number | null, stone?: number) {
        if (this.activeIdx.startsWith('L_')) return;
        try {
            await conn.request('game/move', {
                id:        this.activeIdx.slice(2),
                moveIndex,
                stone:     stone ?? null,
                clientIdx: this._active.bs.getView().plyCount,
            }).promise;
        } catch (e: any) { this._setCmdOutput(`Move rejected: ${e.message}`); }
    }

}
