import { BoardState, MoveType, STONE_MAP } from '@shared/boardState.js';
import {
    PlayerInfo, OnlinePlayerRequest, makeId,
} from '@shared/types.js';
import { GameConfig, FinishedGame } from '@shared/gameConfig.js';
import type {
    BoardView, OnlineStateResponse, PendingGame, ScoreRule, KoRule, TurnInfo, ReplayMove, ChatMessage,
    BoardConfig,
} from '@shared/types.js';
import { PrescribedBoardMap, computeStarPoints } from '@shared/boardConfig.js';
import { parseCleg, unparseCleg } from '@shared/clegParser.js';
import { typecheckClegAsBoard, buildBoardFromCleg } from '@shared/clegEval.js';
import type { ClegProgram } from '@shared/clegBase.js';
import { ServerConnection, type RequestHandle } from './serverConnection.js';
import {
    SidePanelContent, SidePanelHierarchy, SidePanelBwFw, renderSidePanelChrome, sidePanelParent, childButtons,
    renderGamePresetSelection, currentGameSetupHtml, newGameSetupHtml,
    coloredStoneCircle, fmtTurnList,
} from './sidePanel.js';
import {
    type Viewport, QUAT_IDENTITY, defaultViewport, computeAlpha, computePerspectiveScale,
    quatToMat3, quatConjugate, applyOrbitDrag, applyRoll, projectPoint,
} from './camera.js';

// Single persistent WebSocket connection to the main server, shared by the
// EngineManager (AI proxy) and the online-game commands.
const conn = new ServerConnection();

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
    ['3_coin_go',          "19×19 Go plus a protected, non-friendly 'coin' stone (worth no points) either "
                          + 'player may place, up to 3 times each'],
    ['10_coin_go',         "19×19 Go plus a protected, non-friendly 'coin' stone (worth no points) either "
                          + 'player may place, up to 10 times each'],
    ['3_friend_go',        "Like 3_coin_go, but the 'coin' stone is also friendly (doesn't block anyone's liberties)"],
    ['10_friend_go',       "Like 10_coin_go, but the 'coin' stone is also friendly (doesn't block anyone's liberties)"],
]);

// Filename stems (under public/board_presets/) of the board-only (boardType/boardArgs/
// boardModifiers) presets loaded at startup into Renderer.boardConfigs - see _loadBoardConfigs().
// The "Select Board Preset" side panel (renderGamePresetSelection, sidePanel.ts) displays these raw
// names directly (underscores replaced with spaces), not a separate description - there's no
// human-readable text shown anywhere for these beyond the name itself.
const _boardConfigNames = [
    'rect_3_3', 'rect_9_9', 'rect_13_13', 'rect_19_19', 'twsq_7_7_2',
    'twsq_3_3_2_es_3_prod_lin_4', 'regpoly_5_es_5_prod_lin_6', 'cublat_2_2_2_quadform_9',
    'cublat_3_3_3_sel_(deg_gt_3)_quadform_6', 'cublat_3_3_3_quadform_5_nice_drop_0.1',
    'cublat_9_9_9_nice_drop_0.2', 'regpoly_13_prod_regpoly_13', 'star_5_es_6_prod_line_5',
    'regpoly_5_prod_line_2_gcent_triform_7', 'regpoly_5_gcent_prod_line_2_quadform_7',
    'dodeca_gcent_triform_6', 'twsq_3_3_4_quadocta', 'octa_triform_4_quadform_4_triform_4',
    'sier_3_5', 'hcub_2_6_6_6_6', 'cpentflake_4', 'cpolyflake_8_3', 'menger_3_3_0101',
    'menger_4_2_011', 'rect_19_19_nis_(rrmp_0.1_(all))_nis_(conve_quad_(conva_node_(all)))',
    'rect_5_5_fractaldrop_3_0.05', 'tri_4_fractaldrop_3_0.05',
    'cublat_2_2_2_es_1_rect_form_7_(tri)_(quad)_scale_0.75',
    'cublat_3_3_3_es_1_rect_nice_form_4_quad_tri', 'hexpipe_6',
    'cublat_4_4_4_cub_0010_quadform_4', 'biTemple_13_13_9_3', 'twsqCluster4D_4_4_2',
    'goDesk_19_19_5_2_6_2', 'ring_5_12.5', 'shell_6_7.5', 'roundTable_9.5_5_3_2',
    'snubsqtri_4_4_4', 'trunc_trunc_cublat_3_3_3', 'truncated_24_cell',
    'truncated_centralized_rect_6_6', 'tetrahedron_centering_9', 'diamondCubic_10', 'decorated_rect_13_13_0.3',
    'rect_7_7_diag_ortho_3_0.5', 'rect_7_7_bishop_ortho_3_0.5', 'rect_6_6_knight_ortho_4_0.5', 'cuboid_5_5_5_diag_ortho_3_0.5', 'soccer_ball', 'heart_9.5', 'teardrop_24.5', 'racket_24.5_10_1.5',
];


// A color theme names every themeable color: the three SVG board colors (drawn directly via
// setAttribute, so switching themes must reassign the module-level COLOR_GRID/COLOR_ILLEGAL/
// COLOR_BOARD variables below) plus the surrounding UI chrome, which index.html defines entirely
// as CSS custom properties (--color-*) on :root - switching a theme just overwrites those on
// document.documentElement.style. Player stone colors (STONE_MAP, shared/types.ts) are deliberately
// NOT themed - they're shared game data, not purely visual chrome.
interface ColorTheme {
    grid: string;
    illegal: string;
    board: string;
    css: Record<string, string>;
}

// One entry per theme name usable with the 'ctheme' command - 'wooden' matches index.html's own
// :root defaults exactly, so applying it is a visual no-op.
const COLOR_THEMES: Record<string, ColorTheme> = {
    wooden: {
        grid: '#000000',
        illegal: '#ba9347',
        board: '#e5b24c',
        css: {
            '--color-bg-main': '#997750',
            '--color-bg-panel': '#b0895c',
            '--color-bg-surface': '#c29765',
            '--color-bg-surface-hover': '#d6a76f',
            '--color-bg-cmdarea': '#1a1208',
            '--color-bg-overlay': 'rgba(0, 0, 0, 0.5)',
            '--color-accent': '#5c442e',
            '--color-accent-hover': '#7a5827',
            '--color-btn-disabled': '#b5a99e',
            '--color-border': '#5c442e',
            '--color-ctrl-border': '#4a3625',
            // Same value as --color-bg-panel - #side-panel's own border should be invisible in
            // 'wooden', unlike 'default' below. var(...) rather than repeating the literal hex:
            // custom properties resolve lazily at use time (not at applyColorTheme's own
            // setProperty call), so this always tracks --color-bg-panel's current value exactly.
            '--color-panel-border': 'var(--color-bg-panel)',
            // Same reasoning as --color-panel-border just above, but #ctrl-bar has no background
            // of its own - it sits directly on body's, so this tracks --color-bg-main instead.
            '--color-ctrlbar-border': 'var(--color-bg-main)',
            '--color-icon-filter': 'none',
            '--color-text': '#000',
            '--color-text-inverse': '#fff',
            '--color-text-muted': '#aaa',
            '--color-text-placeholder': '#533d29',
            '--color-divider': '#8a6830',
            '--color-heading': '#1a0a00',
            '--color-outline': '#333',
        },
    },
    // Minimal white/black theme: every background var is white, every text var is black; the two
    // hover vars (--color-accent-hover backs .panel-mode-btn's hover, --color-bg-surface-hover
    // backs .panel-child-btn/.status-login-btn/.nav-btn's hover - both are buttons whose own
    // background is now white) get a light gray, a conventional hover shade for a white button.
    // --color-border/--color-divider/--color-outline aren't a background or text color (borders,
    // a table-header rule, and a stone-dot border, respectively) - kept visible against the new
    // white backgrounds (--color-border black - it's what frames every side-panel button/input,
    // and wants to read clearly rather than recede - --color-divider/--color-outline a lighter
    // gray, for their own more minor dividing-line/stone-dot roles), now that --color-border
    // (unlike --color-accent, which
    // still backs several buttons' own white background) is free to be a real border color.
    default: {
        grid: '#000000',
        illegal: '#ba9347',
        board: '#e5b24c',
        css: {
            '--color-bg-main': '#ffffff',
            '--color-bg-panel': '#e4e4e4',
            '--color-bg-surface': '#ffffff',
            '--color-bg-surface-hover': '#e6e6e6',
            '--color-bg-cmdarea': '#ffffff',
            '--color-bg-overlay': 'rgba(255, 255, 255, 0.5)',
            '--color-accent': '#ffffff',
            '--color-accent-hover': '#e6e6e6',
            '--color-btn-disabled': '#d9d9d9',
            '--color-border': '#000',
            '--color-ctrl-border': '#000',
            '--color-panel-border': '#000',
            '--color-ctrlbar-border': '#000',
            // .panel-mode-btn's nav icons (public/icons/*.svg) are baked white - invisible against
            // this theme's own white --color-accent button background without this.
            '--color-icon-filter': 'invert(1)',
            '--color-text': '#000',
            '--color-text-inverse': '#000',
            '--color-text-muted': '#000',
            '--color-text-placeholder': '#000',
            '--color-divider': '#ccc',
            '--color-heading': '#000',
            '--color-outline': '#ccc',
        },
    },
};

let COLOR_GRID    = COLOR_THEMES.wooden!.grid;
let COLOR_ILLEGAL = COLOR_THEMES.wooden!.illegal;
let COLOR_BOARD   = COLOR_THEMES.wooden!.board;

// Applies a COLOR_THEMES entry: reassigns the module-level SVG board color variables (picked up by
// the next drawBoardFull() call - every command handler triggers one via _render()) and overwrites
// index.html's --color-* custom properties on documentElement, which restyles the CSS chrome
// immediately (no redraw needed for that half).
function applyColorTheme(theme: ColorTheme): void {
    COLOR_GRID = theme.grid;
    COLOR_ILLEGAL = theme.illegal;
    COLOR_BOARD = theme.board;
    for (const [prop, value] of Object.entries(theme.css))
        document.documentElement.style.setProperty(prop, value);
}

// SVG elements must be created via createElementNS with this namespace -
// document.createElement('circle') etc. produce non-rendering HTMLUnknownElements.
const SVG_NS = 'http://www.w3.org/2000/svg';

// Cumulative pixel movement below which a mainSvg mousedown->mouseup is treated as a plain click
// (place a stone) rather than a camera-orbit drag - see Renderer._onBoardMouseDown().
const DRAG_THRESHOLD_PX = 4;

// Stone radius as a fraction of cell (pixels per board-coordinate unit) - see boardLayout()'s
// stone_r. Also, since cell is pixels-per-natural-unit, this same factor IS the stone's radius
// in natural (un-scaled) board units - used by drawBoardFull() to offset a stone's depth by its
// own radius without mixing pixel-space and natural-unit-space quantities.
const STONE_RADIUS_FACTOR = 0.42;

// ── layout helper ────────────────────────────────────────────────────────────

// Given a board size w×h, compute how to map board coordinates to screen pixels.
// viewport.scale is a render-area-independent ratio (see its own doc comment, src/camera.ts);
// multiplying by min(w, h) converts it to actual pixels for this call's own box, so the same
// Viewport renders at the correct relative size in both the main board and the smaller
// history-panel thumbnails.
//
// viewport.quat rotates each node's projected (x, y, z) point before the scale is applied below,
// so the board is sized/positioned around its actual on-screen (rotated) extent - see
// src/camera.ts. Also returns `pos` (the already-rotated points, x/y used here, z still unused
// downstream), `rotMat` (the 3x3 matrix itself, reused for ad-hoc points like star points), and
// `dmax` (see computeAlpha()'s own doc comment) so callers never need to redo this projection or
// dmax computation themselves.
//
// Returns:
//   originX, originY - screen pixel for board coordinate (0, 0), i.e. w/2, h/2:
//                      sx = originX + bx * cell
//                      sy = originY - by * cell  (board y-up → screen y-down)
//   cell             - pixels per board-coordinate unit (= viewport.scale * min(w, h))
//   stone_r          - stone radius in pixels (= STONE_RADIUS_FACTOR * cell)
//   pos              - every node's rotated (x, y, z) point, in the same order as view.emb.pos
//   rotMat           - the 3x3 matrix actually applied to get pos - see projectPoint()
//   dmax             - the board's own max raw-point distance from the origin (rotation- and
//                      focus-invariant - see computeAlpha()'s own doc comment)
function boardLayout(view: BoardView, w: number, h: number, viewport: Viewport) {
    const rawPos = view.emb.pos.map(p => projectPoint(viewport.projMat, p));
    // dmax must be measured from the RAW (untranslated, unrotated) points - it's the board's own
    // fixed size scale, unrelated to where the camera currently looks (see Focus's doc comment,
    // src/camera.ts) or is oriented. 0 for an empty board (Math.max() of nothing is -Infinity, not
    // a usable dmax).
    const dmax = rawPos.length > 0 ? Math.max(...rawPos.map(p => Math.hypot(p[0], p[1], p[2]))) : 0;
    // Subtracting focus*dmax from every point BEFORE rotating recenters the scene on the focus
    // point - the camera's rotation is always centered on/facing world (0, 0, 0), so translating
    // the focus point there first makes the camera continue orbiting around and looking at it,
    // exactly as it always did around the true origin when focus is [0, 0, 0].
    const focusPos = rawPos.map(p => p.map((v, k) => v - viewport.focus[k] * dmax));
    // viewport.quat is the camera's own orientation IN WORLD SPACE (applyOrbitDrag derives the
    // camera's world-space right/up axes via quatRotateVector(viewport.quat, ...) - see
    // src/camera.ts) - rendering a world point into the camera's view needs the INVERSE of that
    // rotation, not viewport.quat itself, hence the conjugate here.
    const rotMat = quatToMat3(quatConjugate(viewport.quat));
    const pos = focusPos.map(p => projectPoint(rotMat, p));
    const cell = viewport.scale * Math.min(w, h);
    const stone_r = cell * STONE_RADIUS_FACTOR;
    const originX = w / 2;
    const originY = h / 2;
    return { originX, originY, cell, stone_r, pos, rotMat, dmax };
}

// Computes a game's initial viewport.scale, using the default camera (identity rotation, focus at
// the origin, default distToFocus/aperture). For every node, takes the larger of its rotated
// |x|/|y| (perspective-scaled) plus a margin equal to its own perspective scale; 1/(2*maxExtent)
// is the ratio at which the largest such extent touches half of a unit-sized box, *0.9 for a small
// margin. Render-area-independent (see boardLayout()'s own doc comment) - no w×h needed.
function computeInitialScale(view: BoardView): number {
    const viewport = defaultViewport(view.emb.embDim);
    const rawPos = view.emb.pos.map(p => projectPoint(viewport.projMat, p));
    const dmax = rawPos.length > 0 ? Math.max(...rawPos.map(p => Math.hypot(p[0], p[1], p[2]))) : 0;
    const focusPos = rawPos.map(p => p.map((v, k) => v - viewport.focus[k] * dmax));
    const rotMat = quatToMat3(quatConjugate(viewport.quat));
    const pos = focusPos.map(p => projectPoint(rotMat, p));

    let maxExtent = 0;
    for (const [x, y, z] of pos) {
        const scale = computePerspectiveScale(z, viewport, dmax);
        if (scale === null) continue;
        const extent = Math.max(Math.abs(x * scale), Math.abs(y * scale)) + scale;
        maxExtent = Math.max(maxExtent, extent);
    }
    return maxExtent > 0 ? (1 / (2 * maxExtent)) * 0.9 : 1;
}

// ── board SVG drawing ────────────────────────────────────────────────────────

// One line connecting two adjacent nodes' screen positions. alpha1/alpha2 are each endpoint's own
// fade alpha (see computeAlpha(), src/camera.ts) - when they differ, a plain stroke opacity can't
// express that, so a per-line <linearGradient> (added to `defs`, referenced via url(#gradientId))
// fades smoothly between them instead; when they're equal, a flat opacity is enough and no
// gradient/defs entry is created at all.
function drawGridLine(
    g: SVGElement, defs: SVGDefsElement,
    x1: number, y1: number, x2: number, y2: number, alpha1: number, alpha2: number, gradientId: string,
): void {
    const line = document.createElementNS(SVG_NS, 'line');
    line.setAttribute('x1', String(x1));
    line.setAttribute('y1', String(y1));
    line.setAttribute('x2', String(x2));
    line.setAttribute('y2', String(y2));
    line.setAttribute('stroke-width', '1');
    if (alpha1 === alpha2) {
        line.setAttribute('stroke', COLOR_GRID);
        if (alpha1 !== 1) line.setAttribute('opacity', String(alpha1));
    } else {
        const grad = document.createElementNS(SVG_NS, 'linearGradient');
        grad.setAttribute('id', gradientId);
        grad.setAttribute('gradientUnits', 'userSpaceOnUse');
        grad.setAttribute('x1', String(x1));
        grad.setAttribute('y1', String(y1));
        grad.setAttribute('x2', String(x2));
        grad.setAttribute('y2', String(y2));
        const stop1 = document.createElementNS(SVG_NS, 'stop');
        stop1.setAttribute('offset', '0%');
        stop1.setAttribute('stop-color', COLOR_GRID);
        stop1.setAttribute('stop-opacity', String(alpha1));
        const stop2 = document.createElementNS(SVG_NS, 'stop');
        stop2.setAttribute('offset', '100%');
        stop2.setAttribute('stop-color', COLOR_GRID);
        stop2.setAttribute('stop-opacity', String(alpha2));
        grad.append(stop1, stop2);
        defs.appendChild(grad);
        line.setAttribute('stroke', `url(#${gradientId})`);
    }
    g.appendChild(line);
}

// A "hoshi" star-point dot.
function drawStarPoint(g: SVGElement, sx: number, sy: number, r: number, alpha: number): void {
    const c = document.createElementNS(SVG_NS, 'circle');
    c.setAttribute('cx', String(sx));
    c.setAttribute('cy', String(sy));
    c.setAttribute('r', String(r));
    c.setAttribute('fill', COLOR_GRID);
    if (alpha !== 1) c.setAttribute('opacity', String(alpha));
    g.appendChild(c);
}

// A stone circle - also used for an illegal-move marker (same shape, just a different
// color/no stroke), since the two are visually identical apart from those two properties.
function drawStone(
    g: SVGElement, sx: number, sy: number, r: number, color: string, stroke: string | null, alpha: number,
): void {
    const c = document.createElementNS(SVG_NS, 'circle');
    c.setAttribute('cx', String(sx));
    c.setAttribute('cy', String(sy));
    c.setAttribute('r', String(r));
    c.setAttribute('fill', color);
    if (stroke !== null) {
        c.setAttribute('stroke', stroke);
        c.setAttribute('stroke-width', '1');
    }
    if (alpha !== 1) c.setAttribute('opacity', String(alpha));
    g.appendChild(c);
}

// A territory-ownership marker square.
function drawTerritorySquare(
    g: SVGElement, sx: number, sy: number, side: number, color: string, alpha: number,
): void {
    const r = document.createElementNS(SVG_NS, 'rect');
    r.setAttribute('x', String(sx - side / 2));
    r.setAttribute('y', String(sy - side / 2));
    r.setAttribute('width', String(side));
    r.setAttribute('height', String(side));
    r.setAttribute('fill', color);
    r.setAttribute('stroke', '#888');
    r.setAttribute('stroke-width', String(side / 6));
    if (alpha !== 1) r.setAttribute('opacity', String(alpha));
    g.appendChild(r);
}

// Everything drawBoardFull can draw, tagged with the depth (z, after camera rotation - larger is
// nearer, see boardLayout()) it should be painter's-algorithm-sorted by, plus the exact argument
// tuple its drawing function above takes. A grid line's depth is its FAR endpoint's (the smaller of
// its two endpoints' z), not the midpoint - so the whole line reliably sits behind both endpoints'
// own stones, rather than a near stone at one end poking through the middle of "its own" line. A
// stone's depth is offset by +0.1*radius (its own screen radius) from its center's raw depth, to
// account for the circle representing a 3D sphere (whose near surface bulges toward the camera)
// rather than a flat disc sitting exactly at its center.
type DrawItem =
    | { kind: 'gridLine'; depth: number; args: [number, number, number, number, number, number, string] }
    | { kind: 'starPoint'; depth: number; args: [number, number, number, number] }
    | { kind: 'stone'; depth: number; args: [number, number, number, string, string | null, number] }
    | { kind: 'territorySquare'; depth: number; args: [number, number, number, string, number] };

function drawItem(g: SVGElement, defs: SVGDefsElement, item: DrawItem): void {
    switch (item.kind) {
        case 'gridLine': return drawGridLine(g, defs, ...item.args);
        case 'starPoint': return drawStarPoint(g, ...item.args);
        case 'stone': return drawStone(g, ...item.args);
        case 'territorySquare': return drawTerritorySquare(g, ...item.args);
    }
}

// Render a board state as SVG into `parent` (an already-created/cleared <svg> or
// <g> - this function only appends, never clears `parent` itself; the caller owns
// that lifecycle, same convention as _renderHistoryPanel's histBoards.innerHTML = '').
// legalMoves: if non-null, this is history[ply].legalMoves.captures (captures[stone][loc]) - empty
// nodes illegal for every offered stone are marked with COLOR_ILLEGAL.
// territoryOwner: if non-null, each node with territoryOwner[i] > 0 is marked with a small
// square (side = cell/4, grey-lined with stroke-width = side/6) colored by that stone type.
// showNodes: if true, every node (regardless of board shape, unlike the rect-only star points
// below) gets the same small dot a star point does.
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
    showNodes = false,
    dim = false,
    viewport: Viewport = defaultViewport(view.emb.embDim),
    nextGradientId: () => number,
) {
    const { originX, originY, cell, stone_r, pos, rotMat, dmax } = boardLayout(view, boardW, boardH, viewport);
    const N = view.N;
    const alphaOf = (depth: number) => computeAlpha(depth, dmax, viewport.fadecfg);
    const scaleOf = (depth: number) => computePerspectiveScale(depth, viewport, dmax);

    // grid lines and stones/illegal markers are both dimmed together while
    // selecting a stone, so the whole board reads as "not interactive" - the
    // territory overlay is included in the same group, matching how the
    // canvas version's globalAlpha stayed set across all three sections.
    const g = document.createElementNS(SVG_NS, 'g');
    if (dim) g.setAttribute('opacity', '0.5');
    parent.appendChild(g);
    // Holds the <linearGradient> defs drawGridLine() creates for lines whose two endpoints fade to
    // different alphas - id'd via nextGradientId() (never reused across calls, see its own doc
    // comment - Renderer.nextGradientId); stale ones from the previous render are already gone,
    // since the caller clears `parent`'s children before calling this.
    const defs = document.createElementNS(SVG_NS, 'defs') as unknown as SVGDefsElement;
    g.appendChild(defs);

    // Every grid line/star point/stone/illegal-marker/territory-square gets collected here first
    // (with its screen-space draw args and its depth), rather than drawn immediately - so the
    // whole board can be painter's-algorithm-sorted and drawn back-to-front in one pass below,
    // instead of each shape category being internally sorted (or not) independently. Larger depth
    // is nearer the camera (see boardLayout()'s pos, the (x, y, z) point after rotMat).
    const items: DrawItem[] = [];
    const screenX = (x: number) => originX + x * cell, screenY = (y: number) => originY - y * cell;

    // grid lines - depth is the FAR endpoint's (the smaller of the two), so the whole line sits
    // behind both endpoints' own stones rather than poking through the middle of either. Alpha is
    // computed separately per endpoint (its own actual depth, not the line's sort-key depth above),
    // since the two ends can genuinely fade to different amounts - see drawGridLine()'s own comment.
    // Each endpoint gets its own perspective scale (see computePerspectiveScale(), src/camera.ts);
    // if either is behind the camera (null), the whole line is skipped rather than drawn with a
    // degenerate/inverted endpoint.
    for (let i = 0; i < N; i++) {
        for (let j = i + 1; j < N; j++) {
            if (!adj[i][j]) continue;
            const [x1, y1, z1] = pos[i], [x2, y2, z2] = pos[j];
            const scale1 = scaleOf(z1), scale2 = scaleOf(z2);
            if (scale1 === null || scale2 === null) continue;
            items.push({
                kind: 'gridLine', depth: Math.min(z1, z2),
                args: [
                    screenX(x1 * scale1), screenY(y1 * scale1), screenX(x2 * scale2), screenY(y2 * scale2),
                    alphaOf(z1), alphaOf(z2), `gradient-${nextGradientId()}`,
                ],
            });
        }
    }

    // star points ("hoshi" board markings, rect boards only - computeStarPoints() returns [] for
    // any other boardType). computeStarPoints() returns raw (un-projected) natural-space
    // coordinates, same as node positions before projection - so they need the exact same pipeline
    // boardLayout() itself runs to produce `pos` (viewport.projMat, then the focus translation,
    // then the camera rotMat) to stay aligned with the actual (possibly focus-shifted/rotated)
    // projected grid. Skipped entirely (like every other item type) if behind the camera.
    for (const starPoint of computeStarPoints(config)) {
        const raw = projectPoint(viewport.projMat, starPoint);
        const focusPoint = raw.map((v, k) => v - viewport.focus[k] * dmax);
        const [x, y, z] = projectPoint(rotMat, focusPoint);
        const scale = scaleOf(z);
        if (scale === null) continue;
        items.push({
            kind: 'starPoint', depth: z,
            args: [screenX(x * scale), screenY(y * scale), cell * 0.09 * scale, alphaOf(z)],
        });
    }

    // per-node markers ("snode" command, Renderer.showNodes) - a small dot at every node,
    // regardless of board shape (unlike the rect-only star points above) - reuses the exact same
    // 'starPoint' draw shape, since that's already exactly this: a small COLOR_GRID-filled circle.
    if (showNodes) {
        for (let i = 0; i < N; i++) {
            const [x, y, z] = pos[i];
            const scale = scaleOf(z);
            if (scale === null) continue;
            items.push({
                kind: 'starPoint', depth: z,
                args: [screenX(x * scale), screenY(y * scale), cell * 0.09 * scale, alphaOf(z)],
            });
        }
    }

    // stones / illegal markers - depth offset by +0.1*radius (see DrawItem's own doc comment).
    // STONE_RADIUS_FACTOR, not stone_r, since depth/z is in natural board units, not pixels -
    // stone_r (a pixel quantity, = cell * STONE_RADIUS_FACTOR) would be the wrong scale entirely.
    for (let i = 0; i < N; i++) {
        const [x, y, z] = pos[i];
        const depth = z + 0.1 * STONE_RADIUS_FACTOR;
        const scale = scaleOf(z);
        if (scale === null) continue;
        const alpha = alphaOf(depth);
        const stone = board[i];
        if (stone > 0) {
            items.push({
                kind: 'stone', depth,
                args: [screenX(x * scale), screenY(y * scale), stone_r * scale, STONE_MAP[stone].color, '#333', alpha],
            });
        } else if (legalMoves !== null && legalMoves.every(row => row[i] === null)) {
            items.push({
                kind: 'stone', depth,
                args: [screenX(x * scale), screenY(y * scale), stone_r * scale, COLOR_ILLEGAL, null, alpha],
            });
        }
    }

    // territory squares
    if (territoryOwner !== null) {
        const side = cell / 4;
        for (let i = 0; i < N; i++) {
            const owner = territoryOwner[i];
            if (owner <= 0) continue;
            const [x, y, z] = pos[i];
            const scale = scaleOf(z);
            if (scale === null) continue;
            items.push({
                kind: 'territorySquare', depth: z,
                args: [
                    screenX(x * scale), screenY(y * scale), side * scale,
                    STONE_MAP[owner]?.color ?? '#888', alphaOf(z),
                ],
            });
        }
    }

    // SVG has no z-index; later elements in document order simply render on top of earlier ones -
    // so drawing back-to-front (ascending depth) is what makes nearer shapes correctly paint over
    // farther ones where they visually overlap after camera rotation (painter's algorithm).
    items.sort((a, b) => a.depth - b.depth);
    for (const item of items) drawItem(g, defs, item);
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
    // Orbiting camera orientation + fade-out settings for this game's board (see src/camera.ts) -
    // ephemeral UI state, same as displayPlyNum, not persisted/restored beyond this in-memory
    // ActiveGame.
    viewport: Viewport;
    // When true, _onBoardPointerDown ignores drags (no camera orbit) - toggled by the
    // #lock-rotation-btn control-bar button.
    rotationLocked: boolean;
    // Chat log for this game - append-only, oldest first, same order _refreshChatLog() renders
    // in (newest at the bottom, auto-scrolled into view). Local games update this directly on
    // Send; online games only ever push here from a server game/chatmessage broadcast (see
    // conn.onEvent('game/chatmessage', ...) in init()), never optimistically on send.
    chat: ChatMessage[];
}

// Response shape of REGISTER/LOGIN/FLOGIN: the finished online games the server
// has recorded this user as an observer of (see _addFinishedGames).
interface LoginResponse {
    name: string;
    finishedGames: { id: string; finishedGame: any; chat: ChatMessage[] }[];
}

// One entry in Renderer's popup queue (see currentPopup/popupQueue) - a
// discriminated union so renderPopup() can render each kind's specific
// content/buttons.
type PopupInfo =
    | { kind: 'invite'; id: string; from: string }
    | { kind: 'withdraw-request'; id: string; from: string; numWithdrawn: number }
    | { kind: 'create-failed'; message: string }
    | { kind: 'login-prompt' }
    | { kind: 'confirm'; message: string; onYes: () => void; onNo: () => void }
    | { kind: 'edit-board' };

export class Renderer {
    aiEngineReady = false;
    selfPlay   = false;
    autoForced = false;
    showTerritory = false;
    showIllegalMoves = false;
    showNodes = false;
    colorTheme = 'default';
    // True while a click on a multi-stone turn is waiting for the player to
    // pick which offered stone to place (see _onBoardClick/_renderMainBoard).
    selectingStone = false;
    pendingPos: number | null = null;
    newCfg = new GameConfig(
        parseCleg('rectB(9, 9);'),
        2, 2,
        [
            {player: 1, stones: [1, 0], protected: [0, 0], friendly: [0, 0]},
            {player: 2, stones: [0, 1], protected: [0, 0], friendly: [0, 0]},
        ],
        [[null, null], [null, null]],
        [null, null],
        {1: new Set([1]), 2: new Set([2])},
        true, 'area', [0, 0], 'situational', false, null,
    );
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
    // Transient UI-only state for the 'edit-board' popup (the 'board' command) - the textarea's
    // current text (field-backed the same way inviteInputValue is above, since renderPopup()
    // rebuilds the whole popup, textarea included, on every call) and the error to show above the
    // Ok button, if the last Ok click's parseCleg/typecheckClegAsBoard call threw. Seeded from
    // newCfg.boardDescr (via unparseCleg) when 'board' opens the popup; written back to
    // newCfg.boardDescr only once Ok's parse+typecheck succeeds (see _applyBoardEdit()).
    private _boardDescrText = '';
    private _boardDescrError: string | null = null;
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
    // Loaded at startup from public/board_presets/ (see _loadBoardConfigs()); name -> a
    // boardDescr-only preset, applied via GameConfig.adoptBoardDescr(). On disk each preset stores
    // { boardDescr: string } (cleg source text) - _loadBoardConfigs() parses it, so this map always
    // holds the parsed ClegProgram, matching what GameConfig.boardDescr itself needs.
    boardConfigs = new Map<string, { boardDescr: ClegProgram }>();
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
    // The pointerId currently being tracked as a board drag/click gesture, or null if none - see
    // _onBoardPointerDown.
    private _activePointerId: number | null = null;
    // Tears down the currently-tracked pointer's drag/click listeners without firing a click -
    // set/cleared by _onBoardPointerDown, called when a second pointer interrupts the gesture.
    private _abortBoardDrag: (() => void) | null = null;
    // Next id drawBoardFull() hands drawGridLine() for a faded grid line's <linearGradient> (see
    // its own comment) - SVG ids are unique per-document, not per-<svg>, and drawBoardFull() is
    // called separately for the main board and each history-panel thumbnail, so this must never
    // reset per call. Grows monotonically for the life of the Renderer; harmless (old gradients
    // are discarded, not accumulated, each render).
    private nextGradientId = 0;
    private histBoards:   HTMLDivElement;
    private passBtn:       HTMLButtonElement;
    private resignBtn:    HTMLButtonElement;
    private withdrawBtn:  HTMLButtonElement;
    private wcdBtn:       HTMLButtonElement;
    private resetViewportBtn: HTMLButtonElement;
    private lockRotationBtn:  HTMLButtonElement;
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
    private chatPanel:     HTMLDivElement;
    private configureViewportPanel: HTMLDivElement;
    private commandsPanel: HTMLDivElement;
    private commandReferenceGamePanel:              HTMLDivElement;
    private commandReferenceDisplayPanel:           HTMLDivElement;
    private commandReferenceNewGameSetupPanel:      HTMLDivElement;
    private commandReferenceGamePresetsPanel:       HTMLDivElement;
    private commandReferenceOnlineMultiplayerPanel: HTMLDivElement;
    private clegReferencePanel:                     HTMLDivElement;
    private commandReferenceBoardTypesPanel:        HTMLDivElement;
    private commandReferenceBoardModifiersPanel:    HTMLDivElement;
    private commandReferenceSelectorsPanel:         HTMLDivElement;
    private commandReferenceLocalReplaceSelectorsPanel: HTMLDivElement;
    private commandReferenceFormSelectorsPanel:     HTMLDivElement;
    private commandReferenceBuiltinFunctionsPanel:  HTMLDivElement;
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
    private boardPresetSelectionPanel: HTMLDivElement;
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
        applyColorTheme(COLOR_THEMES[this.colorTheme]!);
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
        this.resetViewportBtn = document.getElementById('reset-viewport-btn') as HTMLButtonElement;
        this.lockRotationBtn  = document.getElementById('lock-rotation-btn')  as HTMLButtonElement;
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
        this.chatPanel     = document.getElementById('chat-panel')      as HTMLDivElement;
        this.configureViewportPanel = document.getElementById('configure-viewport-panel') as HTMLDivElement;
        this.commandsPanel = document.getElementById('commands-panel')  as HTMLDivElement;
        this.commandReferenceGamePanel =
            document.getElementById('cmdref-game-panel') as HTMLDivElement;
        this.commandReferenceDisplayPanel =
            document.getElementById('cmdref-display-panel') as HTMLDivElement;
        this.commandReferenceNewGameSetupPanel =
            document.getElementById('cmdref-new-game-setup-panel') as HTMLDivElement;
        this.commandReferenceGamePresetsPanel =
            document.getElementById('cmdref-game-presets-panel') as HTMLDivElement;
        this.commandReferenceOnlineMultiplayerPanel =
            document.getElementById('cmdref-online-multiplayer-panel') as HTMLDivElement;
        this.clegReferencePanel =
            document.getElementById('cleg-reference-panel') as HTMLDivElement;
        this.commandReferenceBoardTypesPanel =
            document.getElementById('cmdref-board-types-panel') as HTMLDivElement;
        this.commandReferenceBoardModifiersPanel =
            document.getElementById('cmdref-board-modifiers-panel') as HTMLDivElement;
        this.commandReferenceSelectorsPanel =
            document.getElementById('cmdref-selectors-panel') as HTMLDivElement;
        this.commandReferenceLocalReplaceSelectorsPanel =
            document.getElementById('cmdref-local-replace-selectors-panel') as HTMLDivElement;
        this.commandReferenceFormSelectorsPanel =
            document.getElementById('cmdref-form-selectors-panel') as HTMLDivElement;
        this.commandReferenceBuiltinFunctionsPanel =
            document.getElementById('cmdref-builtin-functions-panel') as HTMLDivElement;
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
        this.boardPresetSelectionPanel =
            document.getElementById('board-preset-selection-panel') as HTMLDivElement;
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
            chatPanel:             this.chatPanel,
            configureViewportPanel: this.configureViewportPanel,
            commandsPanel:        this.commandsPanel,
            commandReferenceGamePanel:              this.commandReferenceGamePanel,
            commandReferenceDisplayPanel:           this.commandReferenceDisplayPanel,
            commandReferenceNewGameSetupPanel:      this.commandReferenceNewGameSetupPanel,
            commandReferenceGamePresetsPanel:       this.commandReferenceGamePresetsPanel,
            commandReferenceOnlineMultiplayerPanel: this.commandReferenceOnlineMultiplayerPanel,
            clegReferencePanel:                     this.clegReferencePanel,
            commandReferenceBoardTypesPanel:        this.commandReferenceBoardTypesPanel,
            commandReferenceBoardModifiersPanel:    this.commandReferenceBoardModifiersPanel,
            commandReferenceSelectorsPanel:         this.commandReferenceSelectorsPanel,
            commandReferenceLocalReplaceSelectorsPanel: this.commandReferenceLocalReplaceSelectorsPanel,
            commandReferenceFormSelectorsPanel:     this.commandReferenceFormSelectorsPanel,
            commandReferenceBuiltinFunctionsPanel:  this.commandReferenceBuiltinFunctionsPanel,
            currentGameSetupPanel: this.currentGameSetupPanel,
            newGamePanel:          this.newGamePanel,
            gameRecordsPanel:      this.gameRecordsPanel,
            gamePresetSelectionPanel: this.gamePresetSelectionPanel,
            boardPresetSelectionPanel: this.boardPresetSelectionPanel,
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

        // Three rows: Game Preset/Board Preset nav buttons (via childButtons(), like
        // Home/CurrentGameSetup above); Configure Players (also childButtons()) alongside
        // Configure Modifiers (not a SidePanelContent nav target, so built directly like the
        // Start-new-game buttons below); Start New Local Game/Start New Online Game (likewise built
        // directly). All three rows live in the same #new-game-buttons div, rebuilt together each
        // navigation to New Game. children here is SidePanelHierarchy[NewGame][1] - see its own doc
        // comment - [GamePresetSelection, BoardPresetSelection, ConfigureOnlinePlayers], in that order.
        this.newGameButtons.innerHTML = '';
        if (this.currentSidePanel === SidePanelContent.NewGame) {
            const presetBtnRow = document.createElement('div');
            presetBtnRow.className = 'btn-row';
            presetBtnRow.append(...childButtons(children.slice(0, 2), onNav));
            this.newGameButtons.appendChild(presetBtnRow);
            const configureBtnRow = document.createElement('div');
            configureBtnRow.className = 'btn-row';
            configureBtnRow.append(...childButtons(children.slice(2), onNav), this._buildConfigureBoardBtn());
            this.newGameButtons.appendChild(configureBtnRow);
            const startBtnRow = document.createElement('div');
            startBtnRow.className = 'btn-row';
            startBtnRow.append(this._buildStartLocalGameBtn(), this._buildStartOnlineGameBtn());
            this.newGameButtons.appendChild(startBtnRow);
        }

        // GameRecords is a pure hub (like Home) - its own content IS its
        // four children's buttons, so #game-records-panel doubles as both
        // the toggled content panel (above) and the button container here.
        this.gameRecordsPanel.innerHTML = '';
        if (this.currentSidePanel === SidePanelContent.GameRecords)
            for (const btn of childButtons(children, onNav)) this.gameRecordsPanel.appendChild(btn);

        // CommandReference is likewise a pure hub - one child button per former section of the old
        // single command-reference table, each now its own leaf page (see _initCommandsPanel()).
        this.commandsPanel.innerHTML = '';
        if (this.currentSidePanel === SidePanelContent.CommandReference)
            for (const btn of childButtons(children, onNav)) this.commandsPanel.appendChild(btn);

        // ClegReference is a sibling hub of its own, a direct child of Home (not nested under
        // CommandReference - see SidePanelHierarchy's own comment on why).
        this.clegReferencePanel.innerHTML = '';
        if (this.currentSidePanel === SidePanelContent.ClegReference)
            for (const btn of childButtons(children, onNav)) this.clegReferencePanel.appendChild(btn);

        // Account has its own content (login form or logged-in view) built by
        // _renderAccountPanel(), not childButtons() - a leaf node, not a hub.
        // Deliberately built here (once per navigation) rather than from
        // _render()'s per-frame dispatch: the login form holds in-progress
        // input values that an unrelated _render() (e.g. a WS event for some
        // other active game, or a window resize) must not wipe out. The
        // Log In/Log Out button handlers call this method directly once the
        // login state actually changes, so the view still updates promptly.
        if (this.currentSidePanel === SidePanelContent.Account) this._renderAccountPanel();

        // Like Account just above, Chat holds persistent input state (the in-progress,
        // not-yet-sent message in the textarea) that an unrelated _render() (e.g. self-play's
        // requestAnimationFrame loop, a window resize, or any other game event) must not wipe
        // out - so it's built once here, on navigation, not from _render()'s per-frame dispatch.
        if (this.currentSidePanel === SidePanelContent.Chat) this._renderChatPanel();

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

    // Fetches every preset in _boardConfigNames from public/board_presets/ - each *.cleg
    // file holds nothing but plain cleg SOURCE TEXT (readable/editable on disk as-is, unlike the
    // parsed-AST shape GameConfig.boardDescr itself uses), parsed here via parseCleg into the
    // ClegProgram `boardConfigs` actually stores - same "not awaited by init(), each fails
    // independently" convention as _loadPresets() above; a parse error is caught and dropped
    // exactly like a fetch failure, since a board-only preset has none of GameConfig's other
    // required fields anyway.
    private async _loadBoardConfigs(): Promise<void> {
        const entries = await Promise.all(_boardConfigNames.map(async name => {
            try {
                const source = await fetch(`/board_presets/${name}.cleg`).then(r => r.text());
                return [name, { boardDescr: parseCleg(source) }] as const;
            } catch (e) {
                console.warn(`Failed to load board config '${name}':`, e);
                return null;
            }
        }));
        this.boardConfigs = new Map(entries.filter(
            (e): e is readonly [string, { boardDescr: ClegProgram }] => e !== null,
        ));
    }

    init() {
        this._initCommandsPanel();
        void this._loadPresets().then(() => this._initCommandsPanel());
        void this._loadBoardConfigs();
        this.mainSvg.addEventListener('pointerdown', e => this._onBoardPointerDown(e));
        // Camera roll (left/right) and scale (up/down, same 1.02 multiply/divide as the status
        // panel's own Scale textbox - see src/camera.ts) - global so they work regardless of which
        // side-panel node is focused, but skipped while a text input (cmdInput, the projMat cell
        // editor, etc.) has focus, or while any popup (e.g. the 'mod' command's edit-modifiers
        // textarea) is showing, so they don't hijack arrow-key input meant for those instead.
        document.addEventListener('keydown', e => {
            if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) return;
            if (this.popUp) return;
            if (document.activeElement instanceof HTMLInputElement) return;
            if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
                this._active.viewport.quat = applyRoll(this._active.viewport.quat, e.key === 'ArrowLeft' ? 1 : -1);
                this._render();
            } else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
                e.preventDefault();
                this._active.viewport.scale *= e.key === 'ArrowUp' ? 1.02 : 1 / 1.02;
                this._render();
            }
        });
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
        this.resetViewportBtn.addEventListener('click', () => {
            this._active.viewport.quat = QUAT_IDENTITY;
            this._active.viewport.focus = [0, 0, 0];
            this._render();
        });
        this.lockRotationBtn.addEventListener('click', () => {
            this._active.rotationLocked = !this._active.rotationLocked;
            this._render();
        });
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
        this.panelDockBtn.addEventListener('click', () => {
            this.panelMode = 'locked'; this._applyPanelMode(); this._render();
        });
        this.panelFullBtn.addEventListener('click', () => {
            this.panelMode = 'full'; this._applyPanelMode(); this._render();
        });
        this.panelHideBtn.addEventListener('click', () => {
            this.panelMode = 'hidden'; this._applyPanelMode(); this._render();
        });
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
        conn.onEvent('game/chatmessage', (msg: { id: string; player: number; time: number; content: string }) => {
            this._handleChatMessage(msg.id, msg.player, msg.time, msg.content);
        });
        conn.onEvent('game/engine-error', (msg: { id: string; message: string }) => {
            this._setCmdOutput(`Engine error in game ${msg.id}: ${msg.message}`);
        });
        conn.onEvent('game/invite', (msg: { id: string; from: string }) => {
            this.popupQueue.push({ kind: 'invite', id: msg.id, from: msg.from });
            this._advancePopupQueue();
        });
        conn.onEvent('game/invite-failed', (msg: { id: string; message: string }) => {
            this._handleInviteFailed(msg.id, msg.message);
        });
        conn.onEvent('game/withdraw-proposed', (msg: { id: string; from: string; numWithdrawn: number }) => {
            this.popupQueue.push({ kind: 'withdraw-request', id: msg.id, from: msg.from, numWithdrawn: msg.numWithdrawn });
            this._advancePopupQueue();
        });
        conn.onEvent('game/withdraw-failed', (msg: { id: string; message: string }) => {
            this._handleWithdrawFailed(msg.id, msg.message);
        });
        conn.onEvent('game/withdraw', (msg: { id: string; toPly: number; numWithdrawn: number }) => {
            this._handleOnlineWithdraw(msg.id, msg.toPly);
        });
        // After a (re)connect, re-subscribe to every active/pending online game so the
        // server re-binds our slot. The reply carries full state for catchup sync.
        // Login state doesn't survive a reconnect (it's tied to the live connection),
        // so the server will reject these until the user logs back in - reset userName
        // to reflect that, and surface it once rather than failing silently.
        conn.onEvent('open', () => {
            const previousUserName = this.userName;
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
                        if (pi.name === previousUserName) resub(id.slice(2), slot);
            for (const [id, pg] of this.pendingGames)
                for (const [slot, pi] of pg.config.players)
                    if (pi.name === previousUserName) resub(id, slot);
        });
    }

    private _fireEngineMove(): void {
        const v = this._active.bs.getView();
        if (v.gameOver) { console.warn('em: game is already over'); this.engineManager.cancel(); return; }
        if (this._active.displayPlyNum !== v.plyCount) {
            console.warn('em: not at live position (navigate to end first)'); this.engineManager.cancel(); return;
        }
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
            if (this.currentSidePanel === SidePanelContent.History) this._renderHistoryPanel(v);
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
                renderGamePresetSelection(
                    this.gamePresetSelectionPanel, [...this.presets.keys()], name => this._selectPreset(name),
                );
            if (this.currentSidePanel === SidePanelContent.BoardPresetSelection)
                renderGamePresetSelection(
                    this.boardPresetSelectionPanel, [...this.boardConfigs.keys()],
                    name => this._selectBoardConfig(name),
                );
            if (this.currentSidePanel === SidePanelContent.ConfigureOnlinePlayers) this._renderConfigureOnlinePlayers();
            if (this.currentSidePanel === SidePanelContent.ConfigureViewport) this._renderConfigureViewport();
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
        // 0 means viewport.scale hasn't been computed yet (see its own doc comment, src/camera.ts).
        if (this._active.viewport.scale <= 0) {
            this._active.viewport.scale = computeInitialScale(v);
        }

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
                      this.showNodes,
                      this.selectingStone, this._active.viewport, () => this.nextGradientId++);
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
            offeredStones.map((stone, i) =>
                `${STONE_MAP[stone].color} ${i / offeredStones.length * 100}% ${(i + 1) / offeredStones.length * 100}%`,
            ).join(', ')
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
        this.lockRotationBtn.textContent = this._active.rotationLocked ? 'Unlock' : 'Lock';
        // Disabled while selecting a stone - clicking elsewhere on the board
        // cancels the popup instead (see _onBoardClick).
        this.passBtn.disabled = this.selectingStone
            || dpn !== v.plyCount || !v.passEnabled || !this._isMyTurn();
        this.resignBtn.hidden = this.activeIdx.startsWith('L_');
        this.resignBtn.disabled = this.activeIdx.startsWith('L_') || v.gameOver
            || [...this._active.config.players.entries()].every(
                ([s, pi]) => pi.name !== this.userName || v.resignedPlayers.includes(s),
            );
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
                drawBoardFull(
                    svg, v, this._active.bs.adj, he.board, this._active.config, size, size, null,
                    null, false, false, this._active.viewport, () => this.nextGradientId++,
                );
            });
        }
    }

    private _initCommandsPanel() {
        const row = (cmd: string, desc: string) =>
            `<tr><td>${cmd}</td><td>${desc}</td></tr>`;
        const table = (rows: string) =>
            `<table><colgroup><col style="width:40%"><col style="width:60%"></colgroup>${rows}</table>`;

        this.commandReferenceGamePanel.innerHTML = table(`
            ${row('new',              'Start new local game')}
            ${row('em [&lt;n&gt;]',  'Engine move (optional n consecutive moves)')}
            ${row('cem',             'Cancel current engine move')}
            ${row('temp &lt;f&gt;',  'Set engine temperature (0 = argmax visits)')}
            ${row('s',           'Toggle self-play (random moves)')}
            ${row('af',          'Toggle auto-forced: auto-execute forced moves')}
            ${row('w &lt;n&gt;', 'Withdraw n moves')}
            ${row('wcd',         'Withdraw moves until ply equals display position')}
            ${row('re &lt;n&gt;','Random evaluation over n playouts')}
        `);

        this.commandReferenceDisplayPanel.innerHTML = table(`
            ${row('fw &lt;n&gt;','Step display forward n plies')}
            ${row('bw &lt;n&gt;','Step display backward n plies')}
            ${row('h &lt;n&gt;', 'Show n entries in the history panel')}
            ${row('stt',         'Toggle territory display in the main board area')}
            ${row('simv',        'Toggle illegal move markers on the main board')}
            ${row('tlv',         'Toggle lock view: lock/unlock camera rotation on the main board')}
            ${row('snode',       'Toggle node markers: draw a small dot at every node on the main board')}
            ${row('ctheme <name>', 'Switch color theme (known: wooden, default)')}
            ${row('rsv',
                'Reset view: reset the main board camera to its default orientation and focus (0 0 0)')}
            ${row('focus &lt;x&gt; &lt;y&gt; &lt;z&gt;',
                'Set the point (in units of dmax along each render axis) the camera looks at/orbits '
                + 'around, instead of the origin')}
            ${row('dtf &lt;num&gt;',
                'Set the camera\'s distance from the focus point (in units of dmax); must be &gt; 0')}
            ${row('aperture &lt;num&gt;',
                'Set the camera\'s field of view in degrees; must be between 0 and 120')}
            ${row('scale &lt;num&gt;', 'Set the board\'s render-area-independent size ratio; must be &gt; 0')}
        `);

        this.commandReferenceNewGameSetupPanel.innerHTML = table(`
            ${row('preset &lt;name&gt;',      'Use the specified preset (see the Game Presets page for available names)')}
            ${row('fpo',                      'Toggle forced-pass-only for new games')}
            ${row('ascd',                     'Toggle allow-suicide for new games')}
            ${row('board', 'Open a text box (pre-filled with the new-game config\'s current board '
                + 'description) to freely edit the whole cleg program at once - see the Prescribed '
                + 'Boards/Board Modifiers pages for available construction functions; Ok re-parses and '
                + 'type-checks it (must produce an egr) and, if valid, replaces the new-game board '
                + 'description - otherwise the box stays open with an error')}
            ${row('ns &lt;n&gt;',             'Set number of stone types for new games')}
            ${row('np &lt;n&gt;',             'Set number of players for new games')}
            ${row('tl &lt;player&gt;-&lt;stone bits&gt; …',
                'Set turn list for new games: which player plays each turn, and which stone(s) they may '
                + 'choose from (numStones-length 0/1 string; the first offered stone is auto-picked - no '
                + 'selection UI yet)')}
            ${row('sprot &lt;0-1 str&gt; …',
                'Set protected stones per turn for new games: one numStones-length 0/1 string per turn list entry')}
            ${row('sfriend &lt;0-1 str&gt; …',
                'Set friendly stones per turn for new games: one numStones-length 0/1 string per turn list entry')}
            ${row('spm s &lt;stone&gt; p &lt;player&gt; …',
                'Set which player(s) a stone scores for (zero or more; each gets the stone\'s full points). '
                + 'Players are 1-indexed')}
            ${row('spspl &lt;player&gt; s &lt;num|-&gt; …',
                'Set how many times a player may place each stone color (one value per stone, \'-\' = unlimited)')}
            ${row('sgspl &lt;num|-&gt; …',
                'Set how many times each stone color may ever be placed in total, across all players (one '
                + 'value per stone, \'-\' = unlimited)')}
            ${row('sr &lt;rule&gt;',            'Set scoring rule for new games: stone | territoryonly | area | territory')}
            ${row('ko &lt;pos|sit&gt;',          'Set ko rule for new games: positional | situational')}
            ${row('komi &lt;k1&gt; &lt;k2&gt; …', 'Set per-player komi for new games. One value per player, each &gt;= 0')}
            ${row('mpl &lt;num|-&gt;',           "Set maximum number of plies for new games ('-' = unlimited)")}
        `);

        this.commandReferenceGamePresetsPanel.innerHTML = table(
            [...this.presets.keys()].map(name => row(name, _presetDescriptions.get(name) ?? '')).join('\n            '),
        );

        this.commandReferenceOnlineMultiplayerPanel.innerHTML = table(`
            ${row('register &lt;name&gt; &lt;password&gt;', 'Create an account and log in as it')}
            ${row('login &lt;name&gt; &lt;password&gt;',    'Log in to play online games')}
            ${row('flogin &lt;name&gt; &lt;password&gt;',
                'Log in, taking over from another connection already logged in as this name')}
            ${row('tfpro', 'Toggle fixed online player order (sol/soe/soi vs adde/addl/addi)')}
            ${row('sol &lt;num&gt;', 'Mark player slot &lt;num&gt; as local (you) before newo - fixed order only')}
            ${row('soe &lt;num&gt; [sim] [t]',
                'Mark player slot &lt;num&gt; as server engine; optional sim count and temperature - fixed order only')}
            ${row('soi &lt;num&gt; &lt;name&gt;',
                'Reserve player slot &lt;num&gt; for an invited username, pending their acceptance - fixed order only')}
            ${row('adde [sim] [t]',
                'Append a server-engine player to random order; optional sim count and temperature - random order only')}
            ${row('addl',                 'Append yourself (local) to random order - random order only')}
            ${row('addi &lt;name&gt;',
                'Append an invited username to random order, pending their acceptance - random order only')}
            ${row('newo',                 'Create online game with current config; prints game ID')}
            ${row('joino &lt;ID&gt;',     'Join an existing online game by ID')}
            ${row('swl &lt;ID&gt;',       'Switch active view to a local game by ID')}
            ${row('swo &lt;ID&gt;',       'Switch active view to an online game by ID')}
            ${row('swf &lt;ID&gt;',       'Switch active view to a finished online game by ID')}
        `);

        this.commandReferenceBoardTypesPanel.innerHTML = table(
            Object.values(PrescribedBoardMap)
                .map(([, clegName, argStr, desc]) => row(`${clegName}${argStr}`, desc))
                .join('\n            '),
        );

        this.commandReferenceBoardModifiersPanel.innerHTML = table(`
            ${row('modify([mods…], egr)',
                'Applies every mod in the array, in order, to the given board (egr) - each function '
                + 'below builds one mod value; wrap the ones you want in modify([...], ...) to '
                + 'actually transform a board with them, '
                + 'e.g. modify([rectify(), scale(0.5)], rectB(9, 9))')}
            ${row('rectify()',
                'Rectify: place a node at each edge midpoint, connected via the convex-hull vertex figure '
                + 'around each original node')}
            ${row('truncate()',
                'Truncate: place two nodes per edge, one near each endpoint, connected to each other and '
                + '(via the convex-hull vertex figure) to the other near-points around their own original '
                + 'node - the near-point fraction is chosen per-node so the gap left at the shrunk edge '
                + "matches the scale of that node's own new vertex polygon")}
            ${row('edgeSplit(splitN)',
                'EdgeSplit: split every edge into splitN sub-edges')}
            ${row('mergeClose(dist)',
                'MergeClose: merge every pair of nodes closer than dist into one node')}
            ${row('triangleForm(w, sel?)',
                'TriangleForm: replace every triangle (3 mutually-adjacent nodes) - or, if a triangle '
                + 'selector sel (a string - see the Selectors page) is given, only the ones it selects '
                + '- with a side-length-w triangular board, gluing new corners to the old vertices and '
                + 'gluing shared triangle edges together (w=1 collapses each triangle to a point; w=2 '
                + "is a no-op); an unselected triangle is left untouched, as if it weren't there at all")}
            ${row('quadForm(w, sel?)',
                'QuadForm: replace every quad (4-cycle with no diagonal edges) - or, if a quad '
                + 'selector sel is given, only the ones it selects - with a w-by-w grid, gluing new '
                + 'corners to the old vertices and gluing shared quad edges together (w=1 collapses '
                + "each quad to a point; w=2 is a no-op); an unselected quad is left untouched, as if it weren't there at all")}
            ${row('quadDiagForm(w, sel?)',
                'QuadDiagForm: same as quadForm, but each replaced quad becomes a diagonally-oriented '
                + 'square lattice of side w (w*w + (w-1)*(w-1) nodes: a w-by-w corner grid plus a '
                + '(w-1)-by-(w-1) center grid, each center connected only to its own 4 diagonal '
                + 'corners) instead of a plain w-by-w grid')}
            ${row('quadKnightForm(w, sel?)',
                'QuadKnightForm: same as quadForm (same w-by-w node grid, no extra nodes), but two '
                + "grid nodes are connected iff they're a knight's move apart instead of axis-adjacent")}
            ${row('quadBishopForm(w, sel?)',
                'QuadBishopForm: same as quadForm (same w-by-w node grid, no extra nodes), but two '
                + "grid nodes are connected iff they're diagonally adjacent instead of axis-adjacent")}
            ${row('form(w, [FormSel…])',
                'Form: generalizes triangleForm/quadForm (see their own entries above) to one or more '
                + 'FormSel values (each built by mkFormSel - see Form Selectors) at once, all sharing '
                + 'this one w - unlike calling triangleForm/quadForm separately, a triangle and a quad '
                + 'sharing an edge still glue seamlessly here')}
            ${row('localReplace([LRS…])',
                'LocalReplace: replaces every face named by the given LRS values (each built by '
                + 'mkLRS - see Local Replacement Selectors for what each shape does) with that '
                + 'shape\'s own local piece - independently per LRS value, the same way form '
                + "generalizes triangleForm/quadForm (but without form's own w or edge-gluing). "
                + 'quadCentralize/quadCentering/quadOctarize/simpCentralize/simpCentering/'
                + 'triCentralize/triCentering below are one-step shortcuts building this same kind of '
                + "mod directly for their own single shape, without needing localReplace/mkLRS at all")}
            ${row('quadCentralize(sel?)',
                'One-step shortcut for localReplace([mkLRS("quadCentralize", sel)])')}
            ${row('quadCentering(sel?)',
                'One-step shortcut for localReplace([mkLRS("quadCentering", sel)])')}
            ${row('quadOctarize(sel?)',
                'One-step shortcut for localReplace([mkLRS("quadOctarize", sel)])')}
            ${row('simpCentralize(n, sel?)',
                'One-step shortcut for localReplace([mkLRS("simpCentralize", sel)]), sel\'s own arity fixed to n')}
            ${row('simpCentering(n, sel?)',
                'One-step shortcut for localReplace([mkLRS("simpCentering", sel)]), sel\'s own arity fixed to n')}
            ${row('triCentralize(sel?)', 'One-step shortcut for simpCentralize(2, sel)')}
            ${row('triCentering(sel?)', 'One-step shortcut for simpCentering(2, sel)')}
            ${row('globalCentralize()',
                'GlobalCentralize: add one new node at the barycenter of every existing node, connected '
                + 'to all of them')}
            ${row('scale(num)', 'Scale: multiply every node\'s natural-dimension position by num')}
            ${row('nis(sel)',
                'NodeInducedSubgraph: keep only the nodes the given node selector sel (a string, see '
                + 'the Selectors page) selects, dropping every other node and any edge touching one')}
            ${row('eis(sel)',
                'EdgeInducedSubgraph: keep only the edges the given edge selector sel selects, and '
                + 'only the nodes touched by at least one of them - unlike nis, a node with no '
                + 'selected incident edge is dropped even if adjacent to a surviving node via some '
                + 'other, non-selected edge')}
        `);

        this.commandReferenceSelectorsPanel.innerHTML = table(`
            ${row('(union SEL...)', 'Set union of one or more operands, all the same kind')}
            ${row('(inter SEL...)', 'Set intersection of one or more operands, all the same kind')}
            ${row('(diff SEL SEL)', 'Set difference (left minus right) - both operands the same kind')}
            ${row('(compl SEL)',
                'Complement, within all objects of whichever kind SEL selects from (node/edge/simp N/quad)')}
            ${row('(more [&lt;num&gt;] SEL)',
                'Nodes/edges only: expands SEL outward by num steps (default 1 if omitted), repeating the '
                + 'one-step expansion that many times: one step adds, for nodes, every node one edge away '
                + 'from the current selection; for edges, every edge sharing a node with a currently '
                + 'selected edge - either way SEL\'s own result stays included too, and 0 steps is a no-op')}
            ${row('(all &lt;node|edge|simp N|tri|quad&gt;)',
                'Every object of the given kind - "simp N" is every complete (N+1)-node subgraph '
                + '(clique); "tri" is sugar for "simp 2"')}
            ${row('(none &lt;node|edge|simp N|tri|quad&gt;)', 'No objects of the given kind')}
            ${row('(deg &lt;eq|gt|lt&gt; &lt;num&gt;)', 'Nodes only: whose degree is =/&gt;/&lt; num')}
            ${row('(conva &lt;node|edge|simp N|tri|quad&gt; SEL)',
                'Converts SEL (of whichever kind it itself turns out to be) into the given result kind: a '
                + '"to" object is selected iff ALL of its associated "from" objects are selected - two '
                + 'objects are associated iff one\'s own node set is completely contained in the '
                + 'other\'s (vacuously true for a "to" object with no associated "from" objects at '
                + 'all). Converting a kind to itself is a no-op (this includes simp M -&gt; simp M); '
                + 'simp &lt;-&gt; quad (of any arity, including tri) has no defined association and is '
                + 'rejected, but simp M &lt;-&gt; simp N for M != N is allowed')}
            ${row('(conve &lt;node|edge|simp N|tri|quad&gt; SEL)',
                'Same as conva, but a "to" object is selected iff AT LEAST ONE of its associated '
                + '"from" objects is selected (vacuously false if it has none)')}
            ${row('(rrmn &lt;num&gt; SEL)',
                'Randomly removes exactly num (a nonnegative integer) items from SEL, uniformly at random')}
            ${row('(rrmp &lt;num&gt; SEL)',
                'Randomly removes a portion of SEL: num (a nonnegative fraction) times SEL\'s own '
                + 'size, rounded down')}
        `);

        this.commandReferenceLocalReplaceSelectorsPanel.innerHTML = table(`
            ${row('"quadCentralize"',
                'Every quad (or, if sel is given, only the ones it selects) gets a new hub node '
                + "connected to all 4 of its own corners, its own 4-cycle edges kept")}
            ${row('"quadCentering"',
                'Same as quadCentralize, but the quad\'s own 4 original edges are DROPPED rather than '
                + "kept - its corners end up connected only through the new hub, not to each other "
                + 'directly')}
            ${row('"quadOctarize"',
                'Every quad (or, if sel is given, only the ones it selects) is replaced with an '
                + 'octahedron: two new apex nodes, one on each side along a new embedding dimension, '
                + "each connected to that quad's own 4 corners")}
            ${row('"simpCentralize"',
                'Every n-simplex (n+1 mutually-adjacent nodes) sel selects gets a new hub node '
                + 'connected to all n+1 of its own corners, its own original edges kept - sel is '
                + "required here (unlike every other type string): its own arity (via a simp N "
                + "selector) is what tells this which n to use, there being no separate n argument")}
            ${row('"simpCentering"',
                'Same as simpCentralize, but the simplex\'s own original edges are DROPPED rather than '
                + "kept - its corners end up connected only through the new hub, not to each other "
                + 'directly')}
        `);

        this.commandReferenceFormSelectorsPanel.innerHTML = table(`
            ${row('"triForm"',
                'Every triangle (simp 2) - or, if sel is given, only the ones it selects')}
            ${row('"quadForm"',
                'Every quad - or, if sel is given, only the ones it selects')}
            ${row('"quadDiagForm"',
                'Every quad - or, if sel is given, only the ones it selects - built as a '
                + 'diagonally-oriented square lattice rather than a plain grid')}
            ${row('"quadKnightForm"',
                "Every quad - or, if sel is given, only the ones it selects - connected knight's-move "
                + 'style instead of a plain grid')}
            ${row('"quadBishopForm"',
                'Every quad - or, if sel is given, only the ones it selects - connected diagonally '
                + 'instead of a plain grid')}
        `);

        this.commandReferenceBuiltinFunctionsPanel.innerHTML = table(`
            ${row('abs(x)', 'Absolute value of x')}
            ${row('sqrt(x)', 'Square root of x - x must be nonnegative')}
            ${row('pow(a, b)', 'a raised to the power b')}
            ${row('range(stop)/range(start, stop)/range(start, stop, step)',
                'A number[] with the same semantics as Python\'s range() - start defaults to 0, step '
                + 'defaults to 1, stop is exclusive; every argument must be an integer and step must '
                + 'not be zero')}
            ${row('nil(TYPE)', 'An empty array of the given element type - e.g. nil(number) is an '
                + 'empty number[] - needed anywhere an empty [...]/{...} literal can\'t infer its own '
                + 'element type on its own, such as msUnion([]) or msInter([])')}
            ${row('len(arr|set)', 'Number of elements in an array or set')}
            ${row('has(arr|set, e)',
                'Whether an array or set contains e - only defined for number/string/bool/edge/simp/'
                + 'quad elements, the same kinds a set can hold (see Selectors)')}
            ${row('toSet(arr)',
                'Builds a set from an array, dropping duplicate elements - only defined for '
                + 'number/string/bool/edge/simp/quad elements, the same kinds a set can hold')}
            ${row('randRmN(set, num)',
                'Randomly (uniformly) removes exactly num (a nonnegative integer) elements from a set '
                + 'of any element type - same semantics as the (rrmn ...) selector operator (see '
                + 'Selectors), but usable on any set, not just node/edge/simp/quad selections')}
            ${row('randRmP(set, num)',
                'Randomly removes a portion of a set: num (a nonnegative fraction) times the set\'s own '
                + 'size, rounded down - same semantics as the (rrmp ...) selector operator')}
            ${row('randTakeN(set, num)',
                'Randomly (uniformly) keeps exactly num (a nonnegative integer) elements from a set, '
                + 'discarding the rest - the take-instead-of-remove counterpart of randRmN')}
            ${row('randTakeP(set, num)',
                'Randomly keeps a portion of a set: num (a nonnegative fraction) times the set\'s own '
                + 'size, rounded down - the take-instead-of-remove counterpart of randRmP')}
            ${row('mkEdge(a, b)', 'Builds an edge value between node indices a and b')}
            ${row('mkTri(a, b, c)', 'Builds a triangle (simp) value from three node indices - sugar for mkSimp(a, b, c)')}
            ${row('mkSimp(a, b, c, ...)',
                'Builds a simp value (a complete-subgraph clique) from 3 or more node indices - its '
                + 'arity is however many indices are given, minus 1, but (like every simp value) '
                + "isn't tracked in its own type (type simp), just its actual node list")}
            ${row('mkQuad(a, b, c, d)',
                'Builds a quad value from four node indices, which must already be in cycle order')}
            ${row('prod(egr, egr)', 'The graph (tensor) product of two boards')}
            ${row('subHcublatB(bounds, cond)',
                'A board over the integer lattice points inside an N-dimensional hyperrectangle - '
                + 'bounds is an N-length array of [lo, hi] pairs (rounded inward to the nearest '
                + 'integer point), and cond(point) - point being that candidate\'s own N-length '
                + 'coordinate array - decides which of those points actually become nodes. Surviving '
                + 'nodes keep plain grid adjacency (connected iff exactly one coordinate differs by '
                + '1) and are re-centered around the surviving nodes\' own bounding box')}
            ${row('mkSel(X)',
                'Builds a selector value (type sel) from X - a string (parsed as a selector, kind '
                + 'inferred from the text itself - see Selectors) or a set of number/edge/simp/quad '
                + '(kind read off the set\'s own element type). For "every object of kind K" pass the '
                + 'string "(all K)"')}
            ${row('mkLRS(typeStr, sel?)',
                'Builds an lrs value naming which local shape to replace a face with - see Local '
                + 'Replacement Selectors for the type strings typeStr accepts. Pass the result to '
                + 'localReplace (see Board Modifiers) to build the actual mod')}
            ${row('mkFormSel(typeStr, sel?)',
                'Builds a formsel value naming which lattice kind to build over a face - see Form '
                + 'Selectors for the type strings typeStr accepts. Pass the result to form (see Board '
                + 'Modifiers) to build the actual mod')}
            ${row('selectNode(X, egr)',
                'Evaluates a node selector X (a sel, string, or set) against a real board, returning '
                + 'the exact set of nodes it selects (a number{}) - unlike nis, this runs immediately '
                + 'against a board instead of building a mod to apply later')}
            ${row('selectEdge(X, egr)', 'Same as selectNode, but for an edge selector, returning an edge{}')}
            ${row('selectTriangle(X, egr)', 'Same as selectNode, but for a triangle (simp 2) selector, returning a simp{}')}
            ${row('selectSimp(X, egr)',
                'Same as selectTriangle, but for a simp selector of any arity, not just 2 - X\'s own '
                + 'text/value decides which; returns a simp{} either way, since (unlike '
                + 'selectTriangle) the arity was never fixed at the type level to begin with')}
            ${row('selectQuad(X, egr)', 'Same as selectNode, but for a quad selector, returning a quad{}')}
            ${row('multiProd([egr...], msel)',
                'The N-ary Cartesian product of the given boards, restricted to the subgraph the '
                + 'multiselector msel denotes - a fixed indexing over the full (unrestricted) product '
                + 'space is used throughout, so msUnion/msInter/msDiff operands - built against '
                + 'independently-restricted boards - combine meaningfully, and unused nodes are '
                + 'dropped from the final result')}
            ${row('msAll()', 'Multiselector: every node of the full product, unrestricted')}
            ${row('msBase(num, X)',
                'Multiselector: every full-product node whose num-th coordinate is kept by X - a node '
                + 'or edge selector (sel or set; a node selector keeps the node-induced subgraph of '
                + 'board num, an edge selector its edge-induced subgraph) - every other coordinate '
                + 'left unrestricted')}
            ${row('msUnion([msel...])', 'Multiselector union of zero or more operands (zero is the empty set)')}
            ${row('msInter([msel...])',
                'Multiselector intersection of zero or more operands (zero is the universal set - same as msAll())')}
            ${row('msDiff(msel, msel)', 'Multiselector difference (left minus right)')}
        `);
    }

    // Makes `id` the active game and cancels any in-flight engine request for
    // the previous one - the same switching logic as the 'swl'/'swo'/'swf'
    // commands (_parseCommand), reused by the clickable game-record buttons
    // below. Caller is responsible for re-rendering afterward.
    private _switchToGame(id: string) {
        this.engineManager.cancel();
        this._cancelSelfPlay();
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
    private _fmtGameRecordLabel(id: string, config: GameConfig, winners: number[] | null = null): string {
        return `${fmtTurnList(config.turnList, config.players, winners)}${'&emsp;'.repeat(2)}[${id}]`;
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
        this._renderGameButtons(this.finishedOnlineGamesPanel, ids, id => {
            const ag = this.finishedGames.get(id)!;
            return this._fmtGameRecordLabel(id.slice(2), ag.config, ag.bs.getView().winners);
        });
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
            nameLine.className = 'account-name-line';
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
        userLabel.className = 'account-label';
        userLabel.innerHTML = '<b>Username</b>';
        const userInput = document.createElement('input');
        userInput.type = 'text';
        userInput.className = 'account-input';
        userInput.autocomplete = 'username';

        const passLabel = document.createElement('div');
        passLabel.className = 'account-label';
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

    // Builds the Chat panel's static structure once per navigation into it - see
    // _refreshSidePanel()'s call site for why this must NOT run on every _render() tick.
    // Delegates the actual message list to _refreshChatLog(), also called on its own whenever
    // this._active.chat changes without a full re-navigation (a local send, or an incoming
    // game/chatmessage broadcast for the active game).
    private _renderChatPanel(): void {
        this.chatPanel.innerHTML = '';

        // Same "Game ID"/"Turn list" lines as the Status panel's own template - Game ID and turn
        // list never change while a game is active, so (unlike the log below) this needs no
        // separate refresh mechanism.
        const info = document.createElement('div');
        info.className = 'chat-game-info';
        info.innerHTML = `
            <div><b>Game ID:</b> ${this.activeIdx.slice(2)}</div>
            <div><b>Turn list:</b> ${fmtTurnList(this._active.config.turnList, this._active.config.players)}</div>
        `;
        this.chatPanel.appendChild(info);

        const log = document.createElement('div');
        log.id = 'chat-log';
        this.chatPanel.appendChild(log);

        const row = document.createElement('div');
        row.id = 'chat-input-row';
        row.className = 'colp-invite-row';

        const textarea = document.createElement('textarea');
        textarea.id = 'chat-input';
        textarea.className = 'account-input';
        textarea.rows = 4;
        textarea.placeholder = 'Type a message…';
        // Plain Enter sends (matches the cmd-input console's own Enter-to-submit convention);
        // Shift+Enter still inserts a newline for a genuinely multi-line message.
        textarea.addEventListener('keydown', e => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this._sendChat(textarea);
            }
        });

        const sendBtn = document.createElement('button');
        sendBtn.className = 'status-login-btn';
        sendBtn.textContent = 'Send';
        sendBtn.addEventListener('click', () => this._sendChat(textarea));

        row.append(textarea, sendBtn);
        this.chatPanel.appendChild(row);

        this._refreshChatLog();
    }

    // Rebuilds just #chat-log's contents from this._active.chat - safe to call often (local send,
    // incoming game/chatmessage broadcast) since it never touches #chat-input-row/the textarea,
    // unlike _renderChatPanel() above. No-ops if the Chat panel hasn't been navigated to yet this
    // session (no #chat-log in the DOM).
    // Chat-specific player label - unlike fmtPlayerString (sidePanel.ts), drops the "(Pn)" slot
    // suffix and wraps just the name/symbol in parens, e.g. "(alice)"/"(⌂)"/"(⚙)".
    private _chatPlayerLabel(playerNum: number): string {
        const pi = this._active.config.players.get(playerNum);
        if (!pi) return `(P${playerNum})`;
        if (pi.type === 'local') return '(⌂)';
        if (pi.type === 'serverEngine' || pi.type === 'localEngine') return '(⚙)';
        return `(${pi.name})`;
    }

    private _refreshChatLog(): void {
        const log = this.chatPanel.querySelector<HTMLDivElement>('#chat-log');
        if (!log) return;
        log.innerHTML = '';
        // Oldest first (chronological, matches this._active.chat's own storage order) - newest
        // ends up at the bottom, then the scrollTop assignment below brings it into view.
        for (const { player, content } of this._active.chat) {
            const entry = document.createElement('div');
            entry.className = 'chat-entry';
            // _chatPlayerLabel's return value is plain text (Unicode symbols or a bare name),
            // never HTML markup, so the whole entry (formatted player + ':' + the
            // 100%-untrusted, free-typed `content`) can go through one .textContent assignment -
            // it can never be parsed as markup.
            entry.textContent = `${this._chatPlayerLabel(player)}: ${content}`;
            log.appendChild(entry);
        }
        log.scrollTop = log.scrollHeight;
    }

    // Local games update `chat` directly; online games send to the server and wait for the
    // game/chatmessage broadcast to update state (see conn.onEvent('game/chatmessage', ...) in
    // init()) - mirrors _resign()/_submitOnlineMove()'s existing 'L_' prefix check elsewhere in
    // this file.
    private _sendChat(textarea: HTMLTextAreaElement): void {
        const content = textarea.value.trim();
        if (!content) return;

        if (this.activeIdx.startsWith('L_')) {
            const players = this._active.config.players;
            const localSlot = [...players.entries()]
                .sort(([a], [b]) => a - b)
                .find(([, pi]) => pi.type === 'local')?.[0];
            // Fallback: an all-AI self-play game being spectated has no 'local' slot at all -
            // attribute the message to whoever's turn it is.
            const player = localSlot ?? this._active.bs.getView().nextTurn.player;
            this._active.chat.push({ player, time: Date.now(), content });
            textarea.value = '';
            this._refreshChatLog();
            return;
        }

        const id = this.activeIdx.slice(2);
        conn.request('game/sendchat', { id, content }).promise
            .then(() => { textarea.value = ''; })
            .catch((e: any) => this._setCmdOutput(`Chat failed: ${e.message}`));
    }

    // Creates a local game via _startNewGame() and, once it actually starts, sets the panel mode
    // and navigates to Status - same shape as _buildStartOnlineGameBtn() below (see its own comment
    // for why an online game's own equivalent navigates to Pending Games instead). Shared by
    // #new-game-buttons (_refreshSidePanel()) and the Configure Players panel
    // (_renderConfigureOnlinePlayers()).
    private _buildStartLocalGameBtn(): HTMLButtonElement {
        const btn = document.createElement('button');
        btn.className = 'panel-child-btn';
        btn.textContent = 'New Local Game';
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
    private _buildStartOnlineGameBtn(): HTMLButtonElement {
        const btn = document.createElement('button');
        btn.className = 'panel-child-btn';
        btn.textContent = 'New Online Game';
        btn.addEventListener('click', () => {
            this.panelMode = this._screenIsSmall() ? 'hidden' : 'locked';
            this._applyPanelMode();
            void this._createOnlineGame().then(success => {
                if (success) this._navigateSidePanel(SidePanelContent.PendingGames);
            });
        });
        return btn;
    }

    // New Game panel's own "Configure Board" button - not a SidePanelContent nav target (like
    // ConfigureOnlinePlayers's own button, built via childButtons()), so built directly here, same
    // as _buildStartLocalGameBtn/_buildStartOnlineGameBtn above; does exactly what the 'board'
    // command does (_openBoardEditor()).
    private _buildConfigureBoardBtn(): HTMLButtonElement {
        const btn = document.createElement('button');
        btn.className = 'panel-child-btn';
        btn.textContent = 'Configure Board';
        btn.addEventListener('click', () => this._openBoardEditor());
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
            // One bordered box per slot - a label line ("Slot <n>: <status>") above a button row
            // (Local/Engine/Invite/Clear), plus an optional third line (the invite textbox) while
            // that slot's Invite is open.
            for (let slot = 1; slot <= this.newCfg.numPlayers; slot++) {
                const pi = req.fixedOrder.get(slot);
                const box = document.createElement('div');
                box.className = 'colp-slot';

                const slotLabel = document.createElement('div');
                slotLabel.className = 'colp-slot-label';
                slotLabel.innerHTML = `<b>Slot ${slot}:</b> ${fmtStatus(pi)}`;

                const localBtn  = mkBtn('status-login-btn', 'Local',
                    () => req.fixedOrder.set(slot, new PlayerInfo('local', this.userName ?? 'Player')));
                const engineBtn = mkBtn('status-login-btn', 'Engine', () => req.fixedOrder.set(
                    slot, new PlayerInfo('serverEngine', 'Engine', this.emNumSims, this.emTemperature),
                ));
                const inviteBtn = mkBtn('status-login-btn', 'Invite', () => {
                    this.inviteInputTarget = this.inviteInputTarget === slot ? null : slot;
                    this.inviteInputValue = '';
                });
                const clearBtn  = mkBtn('status-login-btn', 'Clear',  () => req.fixedOrder.delete(slot));

                const slotBtnRow = document.createElement('div');
                slotBtnRow.className = 'btn-row';
                slotBtnRow.append(localBtn, engineBtn, inviteBtn, clearBtn);

                box.append(slotLabel, slotBtnRow);
                if (this.inviteInputTarget === slot)
                    box.appendChild(buildInviteRow(
                        name => req.fixedOrder.set(slot, new PlayerInfo('pendingInvitedOnline', name)),
                    ));
                this.configureOnlinePlayersPanel.appendChild(box);
            }
        } else {
            const listLine = document.createElement('div');
            listLine.className = 'colp-list-line';
            listLine.innerHTML = `<b>List of Players:</b> ${req.randomOrder
                .map(pi => pi.type === 'local' ? 'Local' : pi.type === 'serverEngine' ? 'Engine' : `${pi.name} (invited)`)
                .join('&nbsp;'.repeat(3))}`;
            this.configureOnlinePlayersPanel.appendChild(listLine);

            const atCap = req.randomOrder.length >= this.newCfg.numPlayers;
            const isEmpty = req.randomOrder.length === 0;
            const btnRow = document.createElement('div');
            btnRow.className = 'btn-row';
            btnRow.append(
                mkBtn('panel-child-btn', 'Add Local',
                    () => req.randomOrder.push(new PlayerInfo('local', this.userName ?? 'Player')), atCap),
                mkBtn('panel-child-btn', 'Add Engine', () => req.randomOrder.push(
                    new PlayerInfo('serverEngine', 'Engine', this.emNumSims, this.emTemperature),
                ), atCap),
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
        } else if (this.currentPopup.kind === 'withdraw-request') {
            const { id, from, numWithdrawn } = this.currentPopup;
            text.textContent = `${from} wants to withdraw ${numWithdrawn} move(s). Agree?`;
            const agreeBtn = document.createElement('button');
            agreeBtn.className = 'panel-child-btn';
            agreeBtn.textContent = 'Agree';
            agreeBtn.addEventListener('click', () => void this._respondToWithdraw(id, true));
            const declineBtn = document.createElement('button');
            declineBtn.className = 'panel-child-btn';
            declineBtn.textContent = 'Decline';
            declineBtn.addEventListener('click', () => void this._respondToWithdraw(id, false));
            btnRow.append(agreeBtn, declineBtn);
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
        } else if (this.currentPopup.kind === 'edit-board') {
            // Own layout (label/textarea/error stacked above the Ok button) rather than the shared
            // text+btnRow pair every other popup kind uses below - appends directly and returns.
            box.classList.add('mod-edit-box');
            text.textContent = 'Board description:';
            const textarea = document.createElement('textarea');
            // account-input matches the chat textbox's own look (#chat-input); mod-edit-textarea
            // layers this popup's own sizing on top (see index.html).
            textarea.className = 'account-input mod-edit-textarea';
            textarea.rows = 12;
            // No soft-wrap - a long line scrolls horizontally (see .mod-edit-textarea's own
            // overflow-x) instead of wrapping and obscuring cleg's own indentation.
            textarea.wrap = 'off';
            textarea.value = this._boardDescrText;
            // Field-backed the same way inviteInputValue is (see its own doc comment) - not
            // re-rendered on every keystroke, so the textarea keeps focus/cursor position while typing.
            textarea.addEventListener('input', () => { this._boardDescrText = textarea.value; });
            box.append(text, textarea);
            if (this._boardDescrError !== null) {
                const errorDiv = document.createElement('div');
                errorDiv.className = 'mod-edit-error';
                errorDiv.textContent = this._boardDescrError;
                box.appendChild(errorDiv);
            }
            const okBtn = document.createElement('button');
            okBtn.className = 'panel-child-btn';
            okBtn.textContent = 'Ok';
            okBtn.addEventListener('click', () => this._applyBoardEdit());
            const cancelBtn = document.createElement('button');
            cancelBtn.className = 'panel-child-btn';
            cancelBtn.textContent = 'Cancel';
            cancelBtn.addEventListener('click', () => this._dismissPopup());
            btnRow.append(okBtn, cancelBtn);
            box.appendChild(btnRow);
            this.popupOverlay.appendChild(box);
            return;
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

    // Opens the edit-board popup (the 'board' command, and the New Game panel's own "Configure
    // Board" button - see _buildConfigureBoardBtn()) - seeded from the current newCfg.boardDescr
    // via unparseCleg, same as _boardDescrText's own doc comment describes.
    private _openBoardEditor() {
        this._boardDescrText = unparseCleg(this.newCfg.boardDescr);
        this._boardDescrError = null;
        this.popupQueue.push({ kind: 'edit-board' });
        this._advancePopupQueue();
    }

    // The 'edit-board' popup's Ok button: parses the whole textarea via parseCleg, then requires it
    // typecheck with an `egr` result via typecheckClegAsBoard (shared/clegEval.ts) - a board description
    // that doesn't actually describe a board is rejected here rather than later, when a game
    // actually starts. On success, adopts the result into newCfg.boardDescr and closes the popup;
    // on failure, keeps the popup open and shows the error above the Ok button instead, leaving
    // _boardDescrText as the user left it.
    private _applyBoardEdit() {
        let program;
        try {
            program = parseCleg(this._boardDescrText);
            typecheckClegAsBoard(program);
        } catch (e) {
            this._boardDescrError = e instanceof Error ? e.message : String(e);
            this._render();
            return;
        }
        this.newCfg.boardDescr = program;
        this._dismissPopup();
    }

    // Drops any 'invite' popup for game `id` - currently showing or still queued (relevant when
    // this fires for someone ELSE'S decline while this client still has its own unanswered invite
    // popup for the same game) - removes the game from pendingGames, and queues the "invite
    // failed" popup in its place. Called both from the game/invite-failed broadcast handler
    // (everyone except whoever declined) and directly by _respondToInvite() below (the decliner
    // self-reports the same outcome, since the server never echoes this event back to them).
    private _handleInviteFailed(id: string, message: string) {
        if (this.currentPopup?.kind === 'invite' && this.currentPopup.id === id) this.currentPopup = null;
        this.popupQueue = this.popupQueue.filter(p => !(p.kind === 'invite' && p.id === id));
        this.pendingGames.delete(id);
        this.popupQueue.push({ kind: 'create-failed', message });
        this._advancePopupQueue();
        this._render();
    }

    private async _respondToInvite(id: string, accept: boolean) {
        if (!accept) {
            // Self-report the decline outcome immediately, without waiting on the round trip
            // below - the server never echoes game/invite-failed back to the decliner (see
            // wsServer.ts's 'game/invite-respond' case), so the client simulates the same
            // handling every OTHER observer gets from that broadcast.
            this._handleInviteFailed(id, `You declined the invite to game ${id}`);
            try {
                await conn.request('game/invite-respond', { id, accept: false }).promise;
            } catch (e: any) { this._setCmdOutput(`Error: ${e.message}`); }
            return;
        }
        try {
            await conn.request('game/invite-respond', { id, accept: true }).promise;
        } catch (e: any) { this._setCmdOutput(`Error: ${e.message}`); }
        this._dismissPopup();
    }

    // Drops any 'withdraw-request' popup for game `id` (currently showing or still queued) and
    // shows the failure message in its place - called from the game/withdraw-failed broadcast
    // handler, reaching every voter except whoever declined (see respondToWithdraw()'s own
    // comment on the server side).
    private _handleWithdrawFailed(id: string, message: string) {
        if (this.currentPopup?.kind === 'withdraw-request' && this.currentPopup.id === id) this.currentPopup = null;
        this.popupQueue = this.popupQueue.filter(p => !(p.kind === 'withdraw-request' && p.id === id));
        this.popupQueue.push({ kind: 'create-failed', message });
        this._advancePopupQueue();
        this._render();
        // Ack the failure back to the server, the same way the decliner's own _respondToWithdraw
        // does - the request is already known-doomed, and this client would otherwise never send
        // a response at all (its popup is gone, replaced by the message above), leaving it stuck
        // in unresponded forever and the game locked (see respondToWithdraw()'s own comment).
        // A no-op 403/404 for the original requestor and anyone who already responded.
        void conn.request('game/withdraw-respond', { id, accept: false }).promise.catch(() => {});
    }

    private async _respondToWithdraw(id: string, accept: boolean) {
        if (!accept) this._dismissPopup();
        try {
            await conn.request('game/withdraw-respond', { id, accept }).promise;
        } catch (e: any) { this._setCmdOutput(`Error: ${e.message}`); }
        if (accept) this._dismissPopup();
    }

    // Sends a withdraw proposal to the server (toPly omitted = "my own last move", the Withdraw
    // button; toPly given = an explicit target, the WCD button). Surfaces a "cannot start" error
    // via the same generic create-failed popup used elsewhere for one-shot request failures.
    private async _requestWithdraw(toPly?: number) {
        const id = this.activeIdx.slice(2);
        try {
            const result = await conn.request<{ status: string }>(
                'game/withdraw-request', toPly === undefined ? { id } : { id, toPly },
            ).promise;
            if (result.status === 'pending') {
                this._setCmdOutput('Withdrawal request sent - waiting for other players.');
            }
        } catch (e: any) {
            this.popupQueue.push({ kind: 'create-failed', message: e.message });
            this._advancePopupQueue();
        }
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
        else if (lm.moveType === MoveType.PLACE)
            lastMoveStr = `${coloredStoneCircle(lm.stone!)}@${lm.pos}†${lm.captures.length}`;
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
            <div><b>Game chat:</b> <button id="status-chat-btn" class="status-login-btn">View Chat</button></div>
            <div><b>AI engine:</b> ${this.aiEngineReady ? 'ready' : 'unavailable'}</div>
            <div><b>Engine sims per move:</b> ${this.emNumSims ?? 'default'}</div>
            <div><b>Engine temperature:</b> ${this.emTemperature}</div>
            <div><b>Self play:</b> ${this.selfPlay}</div>
            <div><b>Auto forced:</b> ${this.autoForced}</div>
            <div><b>Show Territory:</b> ${this.showTerritory}</div>
            <div><b>Show Illegal Moves:</b> ${this.showIllegalMoves}</div>
            <div><b>Evaluation:</b> ${evalStr}</div>
            <div><b>Viewport:</b> <button id="status-configure-viewport-btn" class="status-login-btn">Configure</button></div>
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

        // Whole innerHTML is torn down and rebuilt every call - listeners need re-attaching each
        // time, same reason the login button's own is re-attached above.
        this.statusPanel.querySelector('#status-chat-btn')
            ?.addEventListener('click', () => this._navigateSidePanel(SidePanelContent.Chat));
        this.statusPanel.querySelector('#status-configure-viewport-btn')
            ?.addEventListener('click', () => this._navigateSidePanel(SidePanelContent.ConfigureViewport));
    }

    // The projection matrix / fading / focus / scale / distance / aperture editors, reached via
    // Status's own "Viewport: Configure" button (_renderStatus above) - split out of Status itself
    // since this content doesn't fit "one line per field" the rest of Status uses, and dominated the
    // panel. Assumes an active game exists (guaranteed here - see _render()'s own doc comment).
    private _renderConfigureViewport() {
        this.configureViewportPanel.innerHTML = `
            <div><b>Projection matrix:</b></div>
            <div id="viewport-projmat-slot"></div>
            <div><b>Fading:</b> <span id="viewport-fading-mode-slot"></span></div>
            <div id="viewport-fading-fields-slot"></div>
            <div><b>Focus:</b>
                <span id="viewport-focus-row"><span id="viewport-focusx-slot"></span><span id="viewport-focusy-slot"></span><span id="viewport-focusz-slot"></span></span>
            </div>
            <div><b>Scale:</b> <span id="viewport-scale-slot"></span></div>
            <div><b>Distance:</b> <span id="viewport-distToFocus-slot"></span></div>
            <div><b>Aperture:</b> <span id="viewport-aperture-slot"></span></div>
        `;

        // Projection matrix editor: one textbox per entry (2 rows x embDim columns), built via
        // DOM API since each box needs its own Enter-key listener. Pressing Enter parses the box's
        // value and writes it straight into the live Viewport.projMat, mutating in place
        // (boardLayout() always reads the current array, so this takes effect on the very next
        // render), then re-renders so the board redraws with the new projection.
        const projMat = this._active.viewport.projMat;
        const projMatEl = document.createElement('div');
        for (let r = 0; r < projMat.length; r++) {
            const rowEl = document.createElement('div');
            rowEl.className = 'status-projmat-row';
            for (let c = 0; c < projMat[r].length; c++) {
                const box = document.createElement('input');
                box.type = 'text';
                box.className = 'status-projmat-input';
                box.dataset.r = String(r);
                box.dataset.c = String(c);
                box.value = String(projMat[r][c]);
                // _render() rebuilds this whole panel (fresh innerHTML + fresh projMat boxes),
                // which destroys this box and creates a new one at the same (r,c) - without
                // re-focusing it explicitly, the textbox would lose focus on every edit.
                const refocus = () => this.configureViewportPanel
                    .querySelector<HTMLInputElement>(`.status-projmat-input[data-r="${r}"][data-c="${c}"]`)
                    ?.focus();
                box.addEventListener('keydown', e => {
                    if (e.key === 'Enter') {
                        const val = Number(box.value);
                        if (!Number.isFinite(val)) { box.value = String(projMat[r][c]); return; }
                        projMat[r][c] = val;
                        this._render();
                        refocus();
                    } else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
                        e.preventDefault();
                        const delta = e.key === 'ArrowUp' ? 0.05 : -0.05;
                        // Rounded to 2dp - repeated +-0.05 steps otherwise accumulate ugly
                        // floating-point noise (e.g. 0.1 + 0.05 = 0.15000000000000002).
                        projMat[r][c] = Math.round((projMat[r][c] + delta) * 100) / 100;
                        box.value = String(projMat[r][c]);
                        this._render();
                        refocus();
                    }
                });
                rowEl.appendChild(box);
            }
            projMatEl.appendChild(rowEl);
        }
        this.configureViewportPanel.querySelector('#viewport-projmat-slot')?.replaceWith(projMatEl);

        // Single-value editor core: same textbox styling, same Enter-to-commit behavior as the
        // projection-matrix editor above. ArrowUp/ArrowDown nudges via `step` (added by default;
        // scale below instead multiplies/divides, so it passes its own step). Returns the box
        // itself rather than slotting it into the template - shared by makeScalarBox (below, for
        // the fixed template-slot fields) and the fading editor (built entirely via DOM API, since
        // its own field set varies with fadecfg.kind - see the fading block below).
        const createScalarBox = (
            id: string, get: () => number, set: (v: number) => void,
            step: (current: number, direction: 1 | -1) => number =
                (current, direction) => Math.round((current + direction * 0.05) * 100) / 100,
        ): HTMLInputElement => {
            const box = document.createElement('input');
            box.type = 'text';
            box.id = id;
            box.className = 'status-projmat-input';
            box.value = String(get());
            const refocus = () => this.configureViewportPanel.querySelector<HTMLInputElement>(`#${box.id}`)?.focus();
            box.addEventListener('keydown', e => {
                if (e.key === 'Enter') {
                    const val = Number(box.value);
                    if (!Number.isFinite(val)) { box.value = String(get()); return; }
                    set(val);
                    this._render();
                    refocus();
                } else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
                    e.preventDefault();
                    set(step(get(), e.key === 'ArrowUp' ? 1 : -1));
                    box.value = String(get());
                    this._render();
                    refocus();
                }
            });
            return box;
        };
        const makeScalarBox = (
            slotId: string, get: () => number, set: (v: number) => void,
            step?: (current: number, direction: 1 | -1) => number,
        ) => {
            this.configureViewportPanel.querySelector(`#${slotId}`)
                ?.replaceWith(createScalarBox(`${slotId}-input`, get, set, step));
        };

        // Fading mode row: Clamp/Slice buttons - clicking one switches straight to that mode
        // (fresh defaults), with the mode already active disabled, same convention as
        // .panel-mode-btn's own disabled-on-current state (e.g. panelHomeBtn).
        const fadecfg = this._active.viewport.fadecfg;
        const modeEl = document.createElement('span');
        const setFadingMode = (kind: 'clamp' | 'slice') => {
            this._active.viewport.fadecfg = kind === 'clamp'
                ? { kind: 'clamp', init: 0.0, rate: 0.8 }
                : { kind: 'slice', z: 0, solidThick: 0.2, falloffThick: 0.2 };
            this._render();
        };
        const clampBtn = document.createElement('button');
        clampBtn.className = 'status-login-btn';
        clampBtn.textContent = 'Clamp';
        clampBtn.disabled = fadecfg.kind === 'clamp';
        clampBtn.addEventListener('click', () => setFadingMode('clamp'));
        const sliceBtn = document.createElement('button');
        sliceBtn.className = 'status-login-btn';
        sliceBtn.textContent = 'Slice';
        sliceBtn.disabled = fadecfg.kind === 'slice';
        sliceBtn.addEventListener('click', () => setFadingMode('slice'));
        modeEl.appendChild(clampBtn);
        modeEl.appendChild(sliceBtn);
        this.configureViewportPanel.querySelector('#viewport-fading-mode-slot')?.replaceWith(modeEl);

        // Fading field rows: one "Label: [box]" row per fadecfg field (clamp: Init/Rate; slice:
        // z/solid/falloff), indented like the projection-matrix rows (same class/margin) to read as
        // children of the Fading row above - unlike every other status/viewport line, the label
        // here is plain text, not bold (matches every other row's own trailing colon, but not its
        // bolding).
        const fieldsEl = document.createElement('div');
        const addFadingField = (label: string, get: () => number, set: (v: number) => void) => {
            const rowEl = document.createElement('div');
            rowEl.className = 'status-projmat-row';
            const labelEl = document.createElement('span');
            labelEl.textContent = `${label}: `;
            rowEl.appendChild(labelEl);
            rowEl.appendChild(createScalarBox(`viewport-fading-${label}-input`, get, set));
            fieldsEl.appendChild(rowEl);
        };
        if (fadecfg.kind === 'clamp') {
            addFadingField('Init', () => fadecfg.init, v => { fadecfg.init = v; });
            addFadingField('Rate', () => fadecfg.rate, v => { fadecfg.rate = v; });
        } else {
            addFadingField('z', () => fadecfg.z, v => { fadecfg.z = v; });
            addFadingField('solid', () => fadecfg.solidThick, v => { fadecfg.solidThick = v; });
            addFadingField('falloff', () => fadecfg.falloffThick, v => { fadecfg.falloffThick = v; });
        }
        this.configureViewportPanel.querySelector('#viewport-fading-fields-slot')?.replaceWith(fieldsEl);

        const viewport = this._active.viewport;
        makeScalarBox('viewport-focusx-slot', () => viewport.focus[0], v => { viewport.focus[0] = v; });
        makeScalarBox('viewport-focusy-slot', () => viewport.focus[1], v => { viewport.focus[1] = v; });
        makeScalarBox('viewport-focusz-slot', () => viewport.focus[2], v => { viewport.focus[2] = v; });
        makeScalarBox('viewport-aperture-slot', () => viewport.aperture, v => { viewport.aperture = v; });
        // Multiplies/divides by 1.02 per arrow-key tap instead of the default fixed +-0.05 step -
        // scale spans a much wider, non-linear range (see its own doc comment, src/camera.ts), and
        // no rounding, since scale can be far smaller than the default step's 2dp precision.
        const multiplicativeStep = (current: number, direction: 1 | -1) => current * (direction === 1 ? 1.02 : 1 / 1.02);
        makeScalarBox('viewport-scale-slot', () => viewport.scale, v => { viewport.scale = v; }, multiplicativeStep);
        // distToFocus (like scale) must stay positive and spans a similarly wide range - same
        // multiplicative step as scale, not the default fixed +-0.05.
        makeScalarBox(
            'viewport-distToFocus-slot', () => viewport.distToFocus, v => { viewport.distToFocus = v; },
            multiplicativeStep,
        );
    }

    // Wired to mainSvg's 'pointerdown' (init()) - disambiguates a plain click (place a stone, via
    // _onBoardClick) from a drag (orbit the camera, via applyOrbitDrag/src/camera.ts). Pointer
    // Events unify mouse/touch/pen into one model, so this single handler replaces what used to be
    // separate mouse and touch implementations - notably, touch-sourced *touch* events (touchstart/
    // touchmove/touchend) were found to silently stop being dispatched by the browser partway
    // through a drag on some devices once a synchronous re-render occurred mid-gesture, while the
    // underlying pointer events for that same physical touch kept flowing normally throughout -
    // pointer events are the browser's own more fundamental stream, not a layer that can be dropped
    // independently. setPointerCapture guarantees this element keeps receiving this pointer's
    // move/up/cancel events for the rest of the gesture regardless of where it physically travels,
    // so there's no need for window-level listeners or manual "still-active gesture" bookkeeping
    // the way the old touch implementation needed.
    private _onBoardPointerDown(e: PointerEvent) {
        if (this._activePointerId !== null) {
            // A second pointer went down while we were already tracking one as a drag/click - a
            // multi-touch gesture (e.g. a pinch-zoom), not a single-finger drag. Abort our own
            // tracking entirely (no click, no further rotation) rather than just ignoring this
            // second pointer, so the first finger's continued movement during the pinch doesn't
            // keep rotating the camera - see touch-action: pinch-zoom (index.html), which is what
            // lets the browser handle the pinch natively once we get out of its way.
            this._abortBoardDrag?.();
            return;
        }
        this._activePointerId = e.pointerId;
        this.mainSvg.setPointerCapture(e.pointerId);
        const start = { x: e.clientX, y: e.clientY };
        let last = { x: e.clientX, y: e.clientY };
        let moved = false;
        const onMove = (ev: PointerEvent) => {
            if (ev.pointerId !== e.pointerId) return;
            const dx = ev.clientX - last.x, dy = ev.clientY - last.y;
            last = { x: ev.clientX, y: ev.clientY };
            if (!moved && Math.hypot(ev.clientX - start.x, ev.clientY - start.y) < DRAG_THRESHOLD_PX) return;
            // Still counts as a drag (so onUp below skips _onBoardClick) even while locked - a
            // locked drag should do nothing at all, not fall back to placing a stone wherever the
            // pointer happened to lift, which is what skipping this "moved = true" would cause.
            moved = true;
            if (this._active.rotationLocked) return;
            this._active.viewport.quat = applyOrbitDrag(this._active.viewport.quat, dx, dy);
            // Only the main board itself needs to move every tick of a drag - control bar/history
            // panel/side panel state doesn't depend on camera orientation, so a full _render() here
            // would waste work rebuilding all of that on every pointermove.
            this._renderMainBoard(this._active.bs.getView());
        };
        const cleanup = () => {
            this.mainSvg.removeEventListener('pointermove', onMove);
            this.mainSvg.removeEventListener('pointerup', onUp);
            this.mainSvg.removeEventListener('pointercancel', onCancel);
            if (this._abortBoardDrag === cleanup) this._abortBoardDrag = null;
            this._activePointerId = null;
            // Drag ticks only re-rendered the main board (see onMove) - catch up the history
            // panel/control bar/side panel whenever this pointer interaction ends (release,
            // cancel, or getting aborted by a new pointerdown).
            this._render();
        };
        this._abortBoardDrag = cleanup;
        const onCancel = (ev: PointerEvent) => {
            if (ev.pointerId !== e.pointerId) return;
            cleanup();
        };
        const onUp = (ev: PointerEvent) => {
            if (ev.pointerId !== e.pointerId) return;
            cleanup();
            if (!moved) this._onBoardClick(ev);
        };
        this.mainSvg.addEventListener('pointermove', onMove);
        this.mainSvg.addEventListener('pointerup', onUp);
        this.mainSvg.addEventListener('pointercancel', onCancel);
    }

    // clientX/clientY only (not the full MouseEvent) - so this is callable from either a real
    // pointerup (PointerEvent) or a synthesized point.
    private _onBoardClick(e: { clientX: number; clientY: number }) {
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
            // circle cancels the selection instead of being ignored.
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

        const { originX, originY, cell, stone_r, pos: vpos, dmax } =
            boardLayout(v, this.mainBoardSize, this.mainBoardSize, this._active.viewport);
        const board = v.situations[v.plyCount].board;

        let bestDist = Infinity, bestId = -1, bestScale = 1;
        for (let i = 0; i < v.N; i++) {
            // Faded-out (mostly-transparent) locations don't participate in hit-testing at all -
            // matches what's visually legible on the board, see computeAlpha()/src/camera.ts.
            if (computeAlpha(vpos[i][2], dmax, this._active.viewport.fadecfg) < 0.5) continue;
            // Already-occupied locations can't be played on, so they don't participate either.
            if (board[i] > 0) continue;
            // Same z drawBoardFull() scales a stone at this location by - kept consistent so
            // hit-testing matches what's actually drawn.
            const scale = computePerspectiveScale(vpos[i][2], this._active.viewport, dmax);
            // Behind the camera (or exactly at it) - not rendered, so not clickable either.
            if (scale === null) continue;
            const [bx, by] = vpos[i];
            const sx = originX + bx * scale * cell, sy = originY - by * scale * cell;
            const dist = Math.hypot(mx - sx, my - sy);
            if (dist < bestDist) { bestDist = dist; bestId = i; bestScale = scale; }
        }
        if (bestId >= 0 && bestDist < stone_r * bestScale * 1.3) {
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
        let bc: BoardConfig;
        try {
            bc = buildBoardFromCleg(this.newCfg.boardDescr);
        } catch (e) {
            this._setCmdOutput(e instanceof Error ? e.message : String(e));
            return;
        }
        this._createLocalGame(bc, onStarted);
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

    // Called by a Select-Board-Preset button click (see renderGamePresetSelection,
    // sidePanel.ts) - applies just the board-only field (boardDescr) onto newCfg in place via
    // GameConfig.adoptBoardDescr(), leaving every other field (turnList, players, scoring rules,
    // etc.) untouched, then navigates back to New Game to show the result; silently does nothing
    // for an unknown name, since the button list is always built from this.boardConfigs' own keys.
    private _selectBoardConfig(name: string) {
        const bc = this.boardConfigs.get(name);
        if (!bc) return;
        this.newCfg.adoptBoardDescr(bc.boardDescr);
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
                else if (pi.type === 'serverEngine')
                    config.players.set(slot, new PlayerInfo('localEngine', pi.name, pi.emsim, pi.temp));
                else config.players.set(slot, pi);
            }
            const bs = new BoardState(
                config.numStones, config.numPlayers, config.turnList,
                config.playerStonePlaceLimit, config.globalStonePlaceLimit,
                config.stoneToPlayerMap, config.forcedPassOnly, config.scoreRule, config.komi, config.koRule,
                config.allowSuicide, config.maxPlies, new Array(bc.N).fill(0), bc,
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

    private _registerGame(id: string, bs: BoardState, config: GameConfig, chat: ChatMessage[] = []): void {
        this.engineManager.cancel();
        this._cancelSelfPlay();
        this.engineManager.sessionId = null;
        this.activeGames.set(id, {
            bs, config, displayPlyNum: 0, idxShowHistory: 0, randomEvaled: null,
            viewport: defaultViewport(bs.emb.embDim), rotationLocked: false, chat,
        });
        this.activeIdx = id;
    }

    // Withdraws n real moves (see BoardState.withdrawMove) - shared by the
    // 'w' command and the Withdraw button, so the online/finished-game
    // guards live in exactly one place.
    private _withdrawMove(n: number) {
        if (this.finishedGames.has(this.activeIdx)) {
            this._setCmdOutput('Cannot withdraw moves from a finished game'); return;
        }
        if (this.activeIdx.startsWith('O_')) { void this._requestWithdraw(); return; }
        this.engineManager.cancel();
        this.engineManager.sessionId = null;
        for (let i = 0; i < n; i++) this._active.bs.withdrawMove();
        this._active.displayPlyNum = Math.min(this._active.displayPlyNum, this._active.bs.situations.length - 1);
    }

    // Withdraws down to the currently displayed ply - shared by the 'wcd'
    // command and the WCD button.
    private _withdrawToCurrentDisplay() {
        if (this.finishedGames.has(this.activeIdx)) {
            this._setCmdOutput('Cannot withdraw moves from a finished game'); return;
        }
        if (this.activeIdx.startsWith('O_')) { void this._requestWithdraw(this._active.displayPlyNum); return; }
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
            if (!this.onlinePlayerRequest.fixed) {
                this._setCmdOutput('sol: only available when fixed order is enabled (see tfpro)'); return;
            }
            const n = posInt(parts[1]); if (n === null) { this._setCmdOutput('Usage: sol <player-id>'); return; }
            this.onlinePlayerRequest.fixedOrder.set(n, new PlayerInfo('local', this.userName ?? 'Player')); this._render();
        }
        else if (cmd === 'soe') {
            if (!this.onlinePlayerRequest.fixed) {
                this._setCmdOutput('soe: only available when fixed order is enabled (see tfpro)'); return;
            }
            const n = posInt(parts[1]); if (n === null) { this._setCmdOutput('Usage: soe <player-id> [emsim] [temp]'); return; }
            const nonNeg = (s: string | undefined) => {
                const v = Number(s); return s !== undefined && isFinite(v) && v >= 0 ? v : null;
            };
            const emsim = parts[2] !== undefined ? nonNeg(parts[2]) : this.emNumSims;
            const temp  = parts[3] !== undefined ? nonNeg(parts[3]) : this.emTemperature;
            if (emsim === null || temp === null) { this._setCmdOutput('Usage: soe <player-id> [emsim] [temp]'); return; }
            this.onlinePlayerRequest.fixedOrder.set(n, new PlayerInfo('serverEngine', 'Engine', emsim, temp));
            this._render();
        }
        else if (cmd === 'soi') {
            if (!this.onlinePlayerRequest.fixed) {
                this._setCmdOutput('soi: only available when fixed order is enabled (see tfpro)'); return;
            }
            const n = posInt(parts[1]);
            if (n === null || !parts[2]) { this._setCmdOutput('Usage: soi <player-id> <name>'); return; }
            this.onlinePlayerRequest.fixedOrder.set(n, new PlayerInfo('pendingInvitedOnline', parts[2])); this._render();
        }
        else if (cmd === 'adde') {
            if (this.onlinePlayerRequest.fixed) {
                this._setCmdOutput('adde: only available when fixed order is disabled (see tfpro)'); return;
            }
            if (this.onlinePlayerRequest.randomOrder.length >= this.newCfg.numPlayers) {
                this._setCmdOutput(`adde: randomOrder is already full (${this.newCfg.numPlayers} players)`); return;
            }
            const nonNeg = (s: string | undefined) => {
                const v = Number(s); return s !== undefined && isFinite(v) && v >= 0 ? v : null;
            };
            const emsim = parts[1] !== undefined ? nonNeg(parts[1]) : this.emNumSims;
            const temp  = parts[2] !== undefined ? nonNeg(parts[2]) : this.emTemperature;
            if (emsim === null || temp === null) { this._setCmdOutput('Usage: adde [emsim] [temp]'); return; }
            this.onlinePlayerRequest.randomOrder.push(new PlayerInfo('serverEngine', 'Engine', emsim, temp));
            this._render();
        }
        else if (cmd === 'addl') {
            if (this.onlinePlayerRequest.fixed) {
                this._setCmdOutput('addl: only available when fixed order is disabled (see tfpro)'); return;
            }
            if (this.onlinePlayerRequest.randomOrder.length >= this.newCfg.numPlayers) {
                this._setCmdOutput(`addl: randomOrder is already full (${this.newCfg.numPlayers} players)`); return;
            }
            this.onlinePlayerRequest.randomOrder.push(new PlayerInfo('local', this.userName ?? 'Player'));
            this._render();
        }
        else if (cmd === 'addi') {
            if (this.onlinePlayerRequest.fixed) {
                this._setCmdOutput('addi: only available when fixed order is disabled (see tfpro)'); return;
            }
            if (this.onlinePlayerRequest.randomOrder.length >= this.newCfg.numPlayers) {
                this._setCmdOutput(`addi: randomOrder is already full (${this.newCfg.numPlayers} players)`); return;
            }
            if (!parts[1]) { this._setCmdOutput('Usage: addi <name>'); return; }
            this.onlinePlayerRequest.randomOrder.push(new PlayerInfo('pendingInvitedOnline', parts[1]));
            this._render();
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
            this._cancelSelfPlay();
            this.activeIdx = id;
        }
        else if (cmd === 'swo') {
            if (!parts[1]) { this._setCmdOutput('Usage: swo <game-id>'); return; }
            const raw = parts[1].startsWith('O_') ? parts[1].slice(2) : parts[1];
            const id = 'O_' + raw.toUpperCase();
            if (!this.activeGames.has(id)) { this._setCmdOutput(`Online game not found: ${raw.toUpperCase()}`); return; }
            this.engineManager.cancel();
            this._cancelSelfPlay();
            this.activeIdx = id;
        }
        else if (cmd === 'swf') {
            if (!parts[1]) { this._setCmdOutput('Usage: swf <game-id>'); return; }
            const raw = parts[1].startsWith('O_') ? parts[1].slice(2) : parts[1];
            const id = 'O_' + raw.toUpperCase();
            if (!this.finishedGames.has(id)) {
                this._setCmdOutput(`Finished online game not found: ${raw.toUpperCase()}`); return;
            }
            this.engineManager.cancel();
            this._cancelSelfPlay();
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
            if (this.activeIdx.startsWith('O_')) { this._setCmdOutput('Self play is disabled in online mode'); return; }
            this.selfPlay = !this.selfPlay;
            if (this.selfPlay) this._startSelfPlay();
            else this._stopSelfPlay();
        }
        else if (cmd === 'fpo')  this.newCfg.forcedPassOnly = !this.newCfg.forcedPassOnly;
        else if (cmd === 'ascd') this.newCfg.allowSuicide = !this.newCfg.allowSuicide;
        else if (cmd === 'af')   this.autoForced = !this.autoForced;
        else if (cmd === 'stt')  this.showTerritory = !this.showTerritory;
        else if (cmd === 'simv') this.showIllegalMoves = !this.showIllegalMoves;
        else if (cmd === 'tlv')  this._active.rotationLocked = !this._active.rotationLocked;
        else if (cmd === 'snode') this.showNodes = !this.showNodes;
        else if (cmd === 'ctheme') {
            if (!parts[1]) { this._setCmdOutput('Usage: ctheme <name>'); return; }
            const theme = COLOR_THEMES[parts[1]];
            if (!theme) {
                this._setCmdOutput(`Unknown color theme: ${parts[1]} (known: ${Object.keys(COLOR_THEMES).join(', ')})`);
                return;
            }
            this.colorTheme = parts[1];
            applyColorTheme(theme);
        }
        else if (cmd === 'rsv') {
            this._active.viewport.quat = QUAT_IDENTITY;
            this._active.viewport.focus = [0, 0, 0];
        }
        else if (cmd === 'focus') {
            const nums = parts.slice(1, 4).map(Number);
            if (nums.length !== 3 || nums.some(n => !Number.isFinite(n))) {
                this._setCmdOutput('Usage: focus <num> <num> <num>');
                return;
            }
            this._active.viewport.focus = nums as [number, number, number];
        }
        else if (cmd === 'dtf') {
            const v = parseFloat(parts[1]);
            if (!Number.isFinite(v) || v <= 0) {
                this._setCmdOutput('Usage: dtf <num>  (must be > 0)');
                return;
            }
            this._active.viewport.distToFocus = v;
        }
        else if (cmd === 'aperture') {
            const v = parseFloat(parts[1]);
            if (!Number.isFinite(v) || v <= 0 || v >= 120) {
                this._setCmdOutput('Usage: aperture <num>  (degrees, 0 < aperture < 120)');
                return;
            }
            this._active.viewport.aperture = v;
        }
        else if (cmd === 'scale') {
            const v = parseFloat(parts[1]);
            if (!Number.isFinite(v) || v <= 0) {
                this._setCmdOutput('Usage: scale <num>  (must be > 0)');
                return;
            }
            this._active.viewport.scale = v;
        }
        else if (cmd === 'board') {
            this._openBoardEditor();
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
                return {
                    player: i % this.newCfg.numPlayers + 1,
                    stones: offeredStones, protected: protectedStones, friendly: friendlyStones,
                };
            });
            this.newCfg.stoneToPlayerMap = Object.fromEntries(
                Array.from({ length: n }, (_, i) => [i + 1, new Set([i % this.newCfg.numPlayers + 1])]),
            );
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
            this.newCfg.stoneToPlayerMap = Object.fromEntries(
                Array.from({ length: this.newCfg.numStones }, (_, i) => [i + 1, new Set([i % n + 1])]),
            );
            // Keep turnList's player assignments in sync with the new player
            // count (same round-robin as stoneToPlayerMap above), keeping each
            // entry's stone and protected/friendly lists unchanged (numStones is
            // untouched by np, so neither needs resizing here).
            this.newCfg.turnList = this.newCfg.turnList.map((t, i) => ({
                player: i % n + 1, stones: t.stones, protected: t.protected, friendly: t.friendly,
            }));
            // Keep komi in sync with the new player count: truncate if shorter,
            // zero-extend (no komi for the new players) if longer.
            this.newCfg.komi = Array.from({ length: n }, (_, i) => this.newCfg.komi[i] ?? 0);
            // Resize each stone's inner (per-player) axis of playerStonePlaceLimit
            // the same way, null-extended (numStones is untouched by np).
            this.newCfg.playerStonePlaceLimit = this.newCfg.playerStonePlaceLimit.map(
                row => Array.from({ length: n }, (_, j) => row[j] ?? null),
            );
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
                if (!Number.isInteger(player) || player < 1 || player > this.newCfg.numPlayers) {
                    this._setCmdOutput(`tl: player must be an integer between 1 and ${this.newCfg.numPlayers}`);
                    return;
                }
                if (stoneBits.length !== this.newCfg.numStones || !/^[01]+$/.test(stoneBits)) {
                    this._setCmdOutput(
                        `tl: stone bits must be a ${this.newCfg.numStones}-character string of 0s and 1s`,
                    );
                    return;
                }
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
            if (strs.length !== this.newCfg.turnList.length) {
                this._setCmdOutput(
                    `sprot: expected ${this.newCfg.turnList.length} value(s) (one per turn), got ${strs.length}`,
                );
                return;
            }
            if (!strs.every(s => s.length === this.newCfg.numStones && /^[01]+$/.test(s))) {
                this._setCmdOutput(
                    `sprot: each value must be a ${this.newCfg.numStones}-character string of 0s and 1s`,
                );
                return;
            }
            this.newCfg.turnList = this.newCfg.turnList.map((t, i) => ({ ...t, protected: strs[i].split('').map(Number) }));
        }
        else if (cmd === 'sfriend') {
            if (parts.length < 2) { this._setCmdOutput('Usage: sfriend <0-1 str> <0-1 str> …'); return; }
            const strs = parts.slice(1);
            if (strs.length !== this.newCfg.turnList.length) {
                this._setCmdOutput(
                    `sfriend: expected ${this.newCfg.turnList.length} value(s) (one per turn), got ${strs.length}`,
                );
                return;
            }
            if (!strs.every(s => s.length === this.newCfg.numStones && /^[01]+$/.test(s))) {
                this._setCmdOutput(
                    `sfriend: each value must be a ${this.newCfg.numStones}-character string of 0s and 1s`,
                );
                return;
            }
            this.newCfg.turnList = this.newCfg.turnList.map((t, i) => ({ ...t, friendly: strs[i].split('').map(Number) }));
        }
        else if (cmd === 'spm') {
            if (parts.length < 4 || parts[1] !== 's' || parts[3] !== 'p') {
                this._setCmdOutput('Usage: spm s <stone> p <player> <player> …'); return;
            }
            const stone = Number(parts[2]);
            if (!Number.isInteger(stone) || stone < 1 || stone > this.newCfg.numStones) {
                this._setCmdOutput(`spm: stone must be an integer between 1 and ${this.newCfg.numStones}`); return;
            }
            const players = parts.slice(4).map(Number);
            if (!players.every(p => Number.isInteger(p) && p >= 1 && p <= this.newCfg.numPlayers)) {
                this._setCmdOutput(`spm: each player must be an integer between 1 and ${this.newCfg.numPlayers}`);
                return;
            }
            this.newCfg.stoneToPlayerMap[stone] = new Set(players);
        }
        else if (cmd === 'spspl') {
            if (parts.length < 3 || parts[2] !== 's') {
                this._setCmdOutput('Usage: spspl <player-id> s <num|-> <num|-> …'); return;
            }
            const player = Number(parts[1]);
            if (!Number.isInteger(player) || player < 1 || player > this.newCfg.numPlayers) {
                this._setCmdOutput(`spspl: player must be an integer between 1 and ${this.newCfg.numPlayers}`);
                return;
            }
            const toks = parts.slice(3);
            if (toks.length !== this.newCfg.numStones) {
                this._setCmdOutput(
                    `spspl: expected ${this.newCfg.numStones} value(s) (one per stone), got ${toks.length}`,
                );
                return;
            }
            const limits: (number | null)[] = [];
            for (const tok of toks) {
                if (tok === '-') { limits.push(null); continue; }
                const n = Number(tok);
                if (!Number.isInteger(n) || n < 0) {
                    this._setCmdOutput(`spspl: each value must be a non-negative integer or '-'`); return;
                }
                limits.push(n);
            }
            limits.forEach((lim, i) => { this.newCfg.playerStonePlaceLimit[i][player - 1] = lim; });
        }
        else if (cmd === 'sgspl') {
            const toks = parts.slice(1);
            if (toks.length !== this.newCfg.numStones) {
                this._setCmdOutput(
                    `sgspl: expected ${this.newCfg.numStones} value(s) (one per stone), got ${toks.length}`,
                );
                return;
            }
            const limits: (number | null)[] = [];
            for (const tok of toks) {
                if (tok === '-') { limits.push(null); continue; }
                const n = Number(tok);
                if (!Number.isInteger(n) || n < 0) {
                    this._setCmdOutput(`sgspl: each value must be a non-negative integer or '-'`); return;
                }
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
            if (values.length !== this.newCfg.numPlayers) {
                this._setCmdOutput(
                    `komi: expected ${this.newCfg.numPlayers} value(s) (one per player), got ${values.length}`,
                );
                return;
            }
            this.newCfg.komi = values;
        }
        else if (cmd === 'mpl') {
            if (!parts[1]) { this._setCmdOutput('Usage: mpl <num|->'); return; }
            if (parts[1] === '-') { this.newCfg.maxPlies = null; }
            else {
                const n = Number(parts[1]);
                if (!Number.isInteger(n) || n < 1) {
                    this._setCmdOutput(`mpl: value must be a positive integer or '-'`); return;
                }
                this.newCfg.maxPlies = n;
            }
        }
        else if (cmd === 'preset') {
            if (!parts[1]) { this._setCmdOutput('Usage: preset <name>'); return; }
            const p = this.presets.get(parts[1]);
            if (!p) {
                this._setCmdOutput(`Unknown preset: ${parts[1]} (known: ${[...this.presets.keys()].join(', ')})`);
                return;
            }
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
            if (this.selfPlay) {
                // Still running - only the board itself and the ply/turn info in the control bar
                // change each tick; history/side panel catch up in one full _render() once
                // self-play actually stops (below, and _stopSelfPlay()).
                const v = this._active.bs.getView();
                this._renderMainBoard(v);
                this._renderControlBar(v);
                this.selfPlayTimer = requestAnimationFrame(tick);
            } else {
                this._render();
            }
        };
        this.selfPlayTimer = requestAnimationFrame(tick);
    }

    private _stopSelfPlay() {
        if (this.selfPlayTimer !== null) { cancelAnimationFrame(this.selfPlayTimer); this.selfPlayTimer = null; }
        this._render();
    }

    // Stops self-play if it's running - called everywhere activeIdx switches, alongside the
    // existing engineManager.cancel() call. Self-play's tick() operates on whatever this._active
    // currently is, not a captured reference to the game it started on, so switching away while
    // it's running would otherwise keep mutating whichever game gets switched to.
    private _cancelSelfPlay(): void {
        if (!this.selfPlay) return;
        this.selfPlay = false;
        this._stopSelfPlay();
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
    private _addFinishedGames(entries: { id: string; finishedGame: any; chat: ChatMessage[] }[]): void {
        for (const { id, finishedGame: raw, chat } of entries) {
            try {
                const fg = FinishedGame.fromJSON(raw);
                const bc = buildBoardFromCleg(fg.config.boardDescr);
                const bs = BoardState.fromFinishedGame(fg, bc);
                this.finishedGames.set('O_' + id, {
                    bs, config: fg.config, displayPlyNum: bs.getView().plyCount,
                    idxShowHistory: 0, randomEvaled: null, viewport: defaultViewport(bs.emb.embDim),
                    rotationLocked: false, chat,
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
    private _activatePendingGame(id: string, config: GameConfig, chat: ChatMessage[] = []) {
        let bc: BoardConfig;
        try {
            bc = buildBoardFromCleg(config.boardDescr);
        } catch (e) {
            this._setCmdOutput(e instanceof Error ? e.message : String(e));
            return;
        }
        const bs = new BoardState(
            config.numStones, config.numPlayers,
            config.turnList, config.playerStonePlaceLimit, config.globalStonePlaceLimit, config.stoneToPlayerMap,
            config.forcedPassOnly, config.scoreRule, config.komi, config.koRule, config.allowSuicide,
            config.maxPlies, new Array(bc.N).fill(0), bc,
        );
        this.pendingGames.delete(id);
        this._registerGame('O_' + id, bs, config, chat);
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
        } else if (state.moves.length < plyCount) {
            // A withdrawal happened while this client was disconnected - rewind to match.
            ag.bs.withdrawTo(state.moves.length);
            ag.bs.advanceResigned();
            ag.displayPlyNum = Math.min(ag.displayPlyNum, ag.bs.situations.length - 1);
            ag.randomEvaled = null;
            if (isActive) this._notifyTurn(ag, wasGameOver);
        }
        // Full authoritative resync from the server's chat log - unlike moves (which need
        // incremental apply to preserve displayPlyNum/viewport bookkeeping), chat has no such
        // per-message side effects, so a plain overwrite is simplest and correct for both a
        // freshly-seeded game and one catching up on a reconnect.
        ag.chat = state.chat;
        this._maybeFinish('O_' + id);

        if (isActive) {
            if (this.currentSidePanel === SidePanelContent.Chat) this._refreshChatLog();
            this._render();
        }
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

    private _handleOnlineWithdraw(id: string, toPly: number) {
        const ag = this._findGame('O_' + id);
        if (!ag) return;
        const wasGameOver = ag.bs.gameOver();
        ag.bs.withdrawTo(toPly);
        ag.bs.advanceResigned();
        ag.displayPlyNum = Math.min(ag.displayPlyNum, ag.bs.situations.length - 1);
        ag.randomEvaled = null;
        const isActive = 'O_' + id === this.activeIdx;
        if (isActive) {
            this._notifyTurn(ag, wasGameOver);
            this._render();
        }
    }

    private _handleChatMessage(id: string, player: number, time: number, content: string): void {
        // _findGame checks both activeGames and finishedGames - chat isn't gated on the game
        // still being in progress (see sendChat()'s own comment, server-side).
        const ag = this._findGame('O_' + id);
        if (!ag) return;
        ag.chat.push({ player, time, content });
        const isActive = 'O_' + id === this.activeIdx;
        // Narrower than _handleOnlineMove/_handleOnlineResign's isActive-only gate: also require
        // the Chat panel to actually be showing, so a chat message for the active game never
        // touches #chat-log (or risks the textarea) while some other side-panel node is current.
        if (isActive && this.currentSidePanel === SidePanelContent.Chat) this._refreshChatLog();
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
