// Side-panel content nodes, navigable as a tree (see SidePanelHierarchy) -
// Home/Up buttons plus one full-width button per child. Replaces the old flat
// Renderer.activeTab: 'history'|'status'|'commands' switch with a hierarchy
// that isn't hardcoded to one flat level.

import type { BoardView, PlayerInfo, TurnInfo } from '@shared/types.js';
import type { GameConfig } from '@shared/gameConfig.js';
import { unparseCleg } from '@shared/clegParser.js';
import { STONE_MAP } from '@shared/boardState.js';

// Minimal HTML-escaping for text embedded in an innerHTML template - needed for cleg source text
// (unparseCleg), which can contain '<'/'>' (comparison operators) or '&' that would otherwise be
// misread as markup.
function escapeHtml(s: string): string {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

export enum SidePanelContent {
    Home                = 'home',
    History             = 'history',
    Status              = 'status',
    Chat                = 'chat',
    CommandReference    = 'commandReference',
    CommandReferenceGame              = 'commandReferenceGame',
    CommandReferenceDisplay           = 'commandReferenceDisplay',
    CommandReferenceNewGameSetup      = 'commandReferenceNewGameSetup',
    CommandReferenceGamePresets       = 'commandReferenceGamePresets',
    CommandReferenceOnlineMultiplayer = 'commandReferenceOnlineMultiplayer',
    ClegReference                     = 'clegReference',
    CommandReferenceBoardTypes        = 'commandReferenceBoardTypes',
    CommandReferenceBoardModifiers    = 'commandReferenceBoardModifiers',
    CommandReferenceSelectors         = 'commandReferenceSelectors',
    CommandReferenceLocalReplaceSelectors = 'commandReferenceLocalReplaceSelectors',
    CommandReferenceBuiltinFunctions  = 'commandReferenceBuiltinFunctions',
    CurrentGameSetup    = 'currentGameSetup',
    NewGame             = 'newGame',
    GameRecords         = 'gameRecords',
    GamePresetSelection = 'gamePresetSelection',
    BoardPresetSelection = 'boardPresetSelection',
    ActiveLocalGames    = 'activeLocalGames',
    PendingGames        = 'pendingGames',
    ActiveOnlineGames   = 'activeOnlineGames',
    FinishedOnlineGames = 'finishedOnlineGames',
    Account             = 'account',
    ConfigureOnlinePlayers = 'configureOnlinePlayers',
    ConfigureViewport   = 'configureViewport',
}

// [parent, children] per node - parent is null only for Home (VOID: no
// parent, so the Up button is disabled there). Children are listed in the
// order their buttons should appear. Mirrors the PrescribedBoard/
// PrescribedBoardMap Record<Enum, [...]> pattern (shared/boardConfig.ts).
export const SidePanelHierarchy: Record<SidePanelContent, [SidePanelContent | null, SidePanelContent[]]> = {
    [SidePanelContent.Home]:             [null, [
        SidePanelContent.Status, SidePanelContent.CurrentGameSetup, SidePanelContent.NewGame,
        SidePanelContent.GameRecords, SidePanelContent.CommandReference, SidePanelContent.ClegReference,
        SidePanelContent.Account,
    ]],
    // History's nav button isn't rendered in the generic #home-panel slot -
    // Renderer._refreshSidePanel() renders CurrentGameSetup's children (via
    // childButtons(), below) into a small container at the bottom of
    // #current-game-setup-panel instead, right below that panel's own
    // content.
    [SidePanelContent.History]:          [SidePanelContent.CurrentGameSetup, []],
    [SidePanelContent.Status]:           [SidePanelContent.Home, []],
    // Chat's nav button lives inside Status's own template (Renderer._renderStatus()), not the
    // generic childButtons() mechanism - same reason History's parent is CurrentGameSetup rather
    // than Home.
    [SidePanelContent.Chat]:              [SidePanelContent.Status, []],
    // ConfigureViewport's own "Configure" button likewise lives inside Status's own template
    // (Renderer._renderStatus()), same reason as Chat's above.
    [SidePanelContent.ConfigureViewport]: [SidePanelContent.Status, []],
    // CommandReference is a pure hub (like Home/GameRecords) - one child per former section of the
    // old single command-reference table, each now its own leaf page (see
    // Renderer._initCommandsPanel()). ClegReference (the cleg-language reference) is a sibling of
    // this node, not a child of it - see Home's own children list above.
    [SidePanelContent.CommandReference]: [SidePanelContent.Home, [
        SidePanelContent.CommandReferenceGame, SidePanelContent.CommandReferenceDisplay,
        SidePanelContent.CommandReferenceNewGameSetup, SidePanelContent.CommandReferenceGamePresets,
        SidePanelContent.CommandReferenceOnlineMultiplayer,
    ]],
    [SidePanelContent.CommandReferenceGame]:              [SidePanelContent.CommandReference, []],
    [SidePanelContent.CommandReferenceDisplay]:           [SidePanelContent.CommandReference, []],
    [SidePanelContent.CommandReferenceNewGameSetup]:      [SidePanelContent.CommandReference, []],
    [SidePanelContent.CommandReferenceGamePresets]:       [SidePanelContent.CommandReference, []],
    [SidePanelContent.CommandReferenceOnlineMultiplayer]: [SidePanelContent.CommandReference, []],
    // ClegReference is itself a pure hub (like CommandReference above), and a direct child of Home
    // - every cleg-language reference page (prescribed board constructors, board modifiers, the
    // selector grammar, and every other builtin function) lives under it, one leaf page each.
    [SidePanelContent.ClegReference]: [SidePanelContent.Home, [
        SidePanelContent.CommandReferenceBoardTypes, SidePanelContent.CommandReferenceBoardModifiers,
        SidePanelContent.CommandReferenceSelectors, SidePanelContent.CommandReferenceLocalReplaceSelectors,
        SidePanelContent.CommandReferenceBuiltinFunctions,
    ]],
    [SidePanelContent.CommandReferenceBoardTypes]:        [SidePanelContent.ClegReference, []],
    [SidePanelContent.CommandReferenceBoardModifiers]:    [SidePanelContent.ClegReference, []],
    [SidePanelContent.CommandReferenceSelectors]:         [SidePanelContent.ClegReference, []],
    [SidePanelContent.CommandReferenceLocalReplaceSelectors]: [SidePanelContent.ClegReference, []],
    [SidePanelContent.CommandReferenceBuiltinFunctions]:  [SidePanelContent.ClegReference, []],
    [SidePanelContent.CurrentGameSetup]: [SidePanelContent.Home, [SidePanelContent.History]],
    // GamePresetSelection/BoardPresetSelection's nav buttons are likewise rendered by
    // Renderer._refreshSidePanel() - into #new-game-buttons, alongside the
    // Start-new-game action button - not the generic #home-panel slot.
    [SidePanelContent.NewGame]:          [SidePanelContent.Home, [
        SidePanelContent.GamePresetSelection, SidePanelContent.BoardPresetSelection,
        SidePanelContent.ConfigureOnlinePlayers,
    ]],
    // GameRecords is a pure hub (like Home) - its own buttons go into the
    // generic #game-records-panel slot itself (see Renderer._refreshSidePanel(),
    // same pattern as #home-panel), since it has no other content of its own.
    [SidePanelContent.GameRecords]:      [SidePanelContent.Home, [
        SidePanelContent.PendingGames, SidePanelContent.ActiveLocalGames,
        SidePanelContent.ActiveOnlineGames, SidePanelContent.FinishedOnlineGames,
    ]],
    [SidePanelContent.GamePresetSelection]: [SidePanelContent.NewGame, []],
    [SidePanelContent.BoardPresetSelection]: [SidePanelContent.NewGame, []],
    [SidePanelContent.ActiveLocalGames]:    [SidePanelContent.GameRecords, []],
    [SidePanelContent.PendingGames]:        [SidePanelContent.GameRecords, []],
    [SidePanelContent.ActiveOnlineGames]:   [SidePanelContent.GameRecords, []],
    [SidePanelContent.FinishedOnlineGames]: [SidePanelContent.GameRecords, []],
    [SidePanelContent.Account]:             [SidePanelContent.Home, []],
    [SidePanelContent.ConfigureOnlinePlayers]: [SidePanelContent.NewGame, []],
};

// Display title per node - shown below the Home/Up row when that node is
// current, and (unless overridden by SidePanelButtonLabel below) also
// reused as the label of whichever button navigates to it (e.g.
// CommandReference's enum key has no space; its title does).
export const SidePanelTitle: Record<SidePanelContent, string> = {
    [SidePanelContent.Home]:             'Home',
    [SidePanelContent.History]:          'History Of Current Game',
    [SidePanelContent.Status]:           'Status',
    [SidePanelContent.Chat]:             'Chat',
    [SidePanelContent.CommandReference]: 'Command Reference',
    [SidePanelContent.CommandReferenceGame]:              'Game',
    [SidePanelContent.CommandReferenceDisplay]:           'Display',
    [SidePanelContent.CommandReferenceNewGameSetup]:      'New Game Setup',
    [SidePanelContent.CommandReferenceGamePresets]:       'Game Presets',
    [SidePanelContent.CommandReferenceOnlineMultiplayer]: 'Online Multiplayer',
    [SidePanelContent.ClegReference]:                     'Cleg Reference',
    [SidePanelContent.CommandReferenceBoardTypes]:        'Prescribed Boards',
    [SidePanelContent.CommandReferenceBoardModifiers]:    'Board Modifiers',
    [SidePanelContent.CommandReferenceSelectors]:         'Selectors',
    [SidePanelContent.CommandReferenceLocalReplaceSelectors]: 'Local Replacement Selectors',
    [SidePanelContent.CommandReferenceBuiltinFunctions]:  'Builtin Functions',
    [SidePanelContent.CurrentGameSetup]: 'Current Game Info',
    [SidePanelContent.NewGame]:          'New Game',
    [SidePanelContent.GameRecords]:      'Game Records',
    [SidePanelContent.GamePresetSelection]: 'Select Game Preset',
    [SidePanelContent.BoardPresetSelection]: 'Select Board Preset',
    [SidePanelContent.ActiveLocalGames]:    'Active Local Games',
    [SidePanelContent.PendingGames]:        'Pending Games',
    [SidePanelContent.ActiveOnlineGames]:   'Active Online Games',
    [SidePanelContent.FinishedOnlineGames]: 'Finished Online Games',
    [SidePanelContent.Account]:             'Account',
    [SidePanelContent.ConfigureOnlinePlayers]: 'Configure Players',
    [SidePanelContent.ConfigureViewport]:      'Configure Viewport',
};

// Button-label overrides for nodes whose nav-button text should read
// differently from their own page title (SidePanelTitle above) - e.g.
// History's title is the fuller "History Of Current Game", but the button
// embedded in Current Game Info that leads there reads "Game History"
// instead. Consulted by childButtons() below; falls back to SidePanelTitle
// for any node not listed here.
export const SidePanelButtonLabel: Partial<Record<SidePanelContent, string>> = {
    [SidePanelContent.History]: 'Game History',
    [SidePanelContent.GamePresetSelection]: 'Game Preset',
    [SidePanelContent.BoardPresetSelection]: 'Board Preset',
};

// Parent of `current`, or null if it's Home (VOID - no parent to go up to).
export function sidePanelParent(current: SidePanelContent): SidePanelContent | null {
    return SidePanelHierarchy[current][0];
}

// Browser-style back/forward navigation state for the side panel: the
// sequence of SidePanelContent values (stored as plain strings) the user has
// navigated to via Renderer._navigateSidePanel(), and currentIdx - where in
// that sequence they currently are. Renderer._sidePanelBack()/_sidePanelForward()
// move currentIdx without touching history; _navigateSidePanel() itself
// truncates anything past currentIdx and appends the new entry - standard
// back/forward semantics, so navigating somewhere new after going back
// discards the now-stale forward entries.
export class SidePanelBwFw {
    history: string[];
    currentIdx: number;
    constructor(initial: SidePanelContent) {
        this.history = [initial];
        this.currentIdx = 0;
    }
}

// DOM refs the side panel's nav chrome needs - owned/passed in by Renderer,
// which holds all element refs elsewhere in this codebase too.
export interface SidePanelElements {
    titleEl:              HTMLDivElement;
    upBtn:                 HTMLButtonElement;
    homePanel:             HTMLDivElement;
    historyPanel:          HTMLDivElement;
    statusPanel:           HTMLDivElement;
    chatPanel:             HTMLDivElement;
    commandsPanel:         HTMLDivElement;
    commandReferenceGamePanel:              HTMLDivElement;
    commandReferenceDisplayPanel:           HTMLDivElement;
    commandReferenceNewGameSetupPanel:      HTMLDivElement;
    commandReferenceGamePresetsPanel:       HTMLDivElement;
    commandReferenceOnlineMultiplayerPanel: HTMLDivElement;
    clegReferencePanel:                     HTMLDivElement;
    commandReferenceBoardTypesPanel:        HTMLDivElement;
    commandReferenceBoardModifiersPanel:    HTMLDivElement;
    commandReferenceSelectorsPanel:         HTMLDivElement;
    commandReferenceLocalReplaceSelectorsPanel: HTMLDivElement;
    commandReferenceBuiltinFunctionsPanel:  HTMLDivElement;
    currentGameSetupPanel: HTMLDivElement;
    newGamePanel:          HTMLDivElement;
    gameRecordsPanel:      HTMLDivElement;
    gamePresetSelectionPanel: HTMLDivElement;
    boardPresetSelectionPanel: HTMLDivElement;
    activeLocalGamesPanel:    HTMLDivElement;
    pendingGamesPanel:        HTMLDivElement;
    activeOnlineGamesPanel:   HTMLDivElement;
    finishedOnlineGamesPanel: HTMLDivElement;
    accountPanel:             HTMLDivElement;
    configureOnlinePlayersPanel: HTMLDivElement;
    configureViewportPanel:  HTMLDivElement;
}

// Rebuilds the nav chrome (title, Up-button disabled state) and toggles
// which content panel is visible - same "just rebuild, no diffing"
// convention as _renderHistoryPanel/_initCommandsPanel in renderer.ts. Does
// NOT render any children buttons - each node with a nonempty children list
// (Home/CurrentGameSetup/NewGame/GameRecords/CommandReference/ClegReference) does that itself, via
// childButtons() below, into its own panel/container (see Renderer._refreshSidePanel()).
export function renderSidePanelChrome(current: SidePanelContent, els: SidePanelElements): void {
    const [parent] = SidePanelHierarchy[current];

    els.titleEl.textContent = SidePanelTitle[current];
    els.upBtn.disabled = parent === null;

    els.homePanel.style.display                = current === SidePanelContent.Home                ? 'block' : 'none';
    els.historyPanel.style.display             = current === SidePanelContent.History             ? 'flex'  : 'none';
    els.statusPanel.style.display              = current === SidePanelContent.Status              ? 'block' : 'none';
    els.chatPanel.style.display                 = current === SidePanelContent.Chat                ? 'flex'  : 'none';
    els.commandsPanel.style.display            = current === SidePanelContent.CommandReference    ? 'block' : 'none';
    els.commandReferenceGamePanel.style.display =
        current === SidePanelContent.CommandReferenceGame ? 'block' : 'none';
    els.commandReferenceDisplayPanel.style.display =
        current === SidePanelContent.CommandReferenceDisplay ? 'block' : 'none';
    els.commandReferenceNewGameSetupPanel.style.display =
        current === SidePanelContent.CommandReferenceNewGameSetup ? 'block' : 'none';
    els.commandReferenceGamePresetsPanel.style.display =
        current === SidePanelContent.CommandReferenceGamePresets ? 'block' : 'none';
    els.commandReferenceOnlineMultiplayerPanel.style.display =
        current === SidePanelContent.CommandReferenceOnlineMultiplayer ? 'block' : 'none';
    els.clegReferencePanel.style.display =
        current === SidePanelContent.ClegReference ? 'block' : 'none';
    els.commandReferenceBoardTypesPanel.style.display =
        current === SidePanelContent.CommandReferenceBoardTypes ? 'block' : 'none';
    els.commandReferenceBoardModifiersPanel.style.display =
        current === SidePanelContent.CommandReferenceBoardModifiers ? 'block' : 'none';
    els.commandReferenceSelectorsPanel.style.display =
        current === SidePanelContent.CommandReferenceSelectors ? 'block' : 'none';
    els.commandReferenceLocalReplaceSelectorsPanel.style.display =
        current === SidePanelContent.CommandReferenceLocalReplaceSelectors ? 'block' : 'none';
    els.commandReferenceBuiltinFunctionsPanel.style.display =
        current === SidePanelContent.CommandReferenceBuiltinFunctions ? 'block' : 'none';
    els.currentGameSetupPanel.style.display    = current === SidePanelContent.CurrentGameSetup    ? 'block' : 'none';
    els.newGamePanel.style.display             = current === SidePanelContent.NewGame             ? 'block' : 'none';
    els.gameRecordsPanel.style.display         = current === SidePanelContent.GameRecords         ? 'block' : 'none';
    els.gamePresetSelectionPanel.style.display = current === SidePanelContent.GamePresetSelection ? 'flex' : 'none';
    els.boardPresetSelectionPanel.style.display =
        current === SidePanelContent.BoardPresetSelection ? 'flex' : 'none';
    els.activeLocalGamesPanel.style.display    = current === SidePanelContent.ActiveLocalGames    ? 'block' : 'none';
    els.pendingGamesPanel.style.display        = current === SidePanelContent.PendingGames        ? 'block' : 'none';
    els.activeOnlineGamesPanel.style.display   = current === SidePanelContent.ActiveOnlineGames   ? 'block' : 'none';
    els.finishedOnlineGamesPanel.style.display = current === SidePanelContent.FinishedOnlineGames ? 'block' : 'none';
    els.accountPanel.style.display             = current === SidePanelContent.Account             ? 'block' : 'none';
    els.configureOnlinePlayersPanel.style.display = current === SidePanelContent.ConfigureOnlinePlayers ? 'flex' : 'none';
    els.configureViewportPanel.style.display   = current === SidePanelContent.ConfigureViewport   ? 'block' : 'none';
}

// Builds one full-width nav button per entry in `children` (dataset.child =
// the SidePanelContent value, for tests/CSS to target) - a plain
// constructor with no side effects beyond the returned elements themselves
// (doesn't touch any existing DOM). Called independently by whichever node
// actually has children (see Renderer._refreshSidePanel()), each deciding
// for itself where the resulting buttons get appended.
export function childButtons(
    children: SidePanelContent[],
    onNavigate: (target: SidePanelContent) => void,
): HTMLButtonElement[] {
    return children.map(child => {
        const btn = document.createElement('button');
        btn.className = 'panel-child-btn';
        btn.dataset['child'] = child;
        btn.textContent = SidePanelButtonLabel[child] ?? SidePanelTitle[child];
        btn.addEventListener('click', () => onNavigate(child));
        return btn;
    });
}

// Renders one full-width button per known preset name into el (rebuilt
// fresh each call, matching the "just rebuild innerHTML" convention used
// elsewhere) - clicking one calls onSelect(name); the caller (Renderer)
// decides what selecting a preset actually does (set newCfg, navigate back
// to New Game - see Renderer._selectPreset()).
export function renderGamePresetSelection(
    el: HTMLDivElement, presetNames: string[], onSelect: (name: string) => void,
): void {
    el.innerHTML = '';
    for (const name of presetNames) {
        const btn = document.createElement('button');
        btn.className = 'panel-child-btn';
        // Display only - underscores are replaced with spaces for readability, but onSelect still
        // gets the raw name (the actual filename stem/map key, e.g. for presets.get()).
        btn.textContent = name.replace(/_/g, ' ');
        btn.addEventListener('click', () => onSelect(name));
        el.appendChild(btn);
    }
}

// ── Status/Current Game Info/New Game formatting helpers ──────────────────────
// Pure functions (no `this`), shared by Renderer._renderStatus (renderer.ts)
// and this file's currentGameSetupHtml/newGameSetupHtml below - those
// three are separate side-panel nodes that all format the same GameConfig-
// shaped fields.

// Renders e.g. "⬤"/"●" colored by stone type p.
export const coloredStoneCircle = (p: number) => `<span style="color:${STONE_MAP[p]?.color ?? '#888'}">⬤</span>`;
export const coloredStoneDot    = (p: number) => `<span style="color:${STONE_MAP[p]?.color ?? '#888'}">●</span>`;

// Renders a single player's identity for use in a stone-circle line: a
// name for a human ('client') player, ⌂ for a local-only slot, ⚙ for a
// serverEngine slot.
export const fmtPlayerString = (players: Map<number, PlayerInfo>, playerNum: number) => {
    const pi = players.get(playerNum);
    if (!pi) return `(P${playerNum})`;
    if (pi.type === 'local') return `⌂ (P${playerNum})`;
    if (pi.type === 'serverEngine' || pi.type === 'localEngine') return `⚙ (P${playerNum})`;
    return `${pi.name} (P${playerNum})`;
};
// Renders e.g. "⬤ alice ⌂" (single space between players sharing a
// stone) "     ⬤ ⌂" (five spaces between adjacent stone entries), each
// circle colored by its stone type, followed by every player it scores
// for; just the bare stone circle when it's mapped to no players
// (scores for no one).
export const fmtMap = (map: Record<number, Set<number>>, players: Map<number, PlayerInfo>) =>
    Object.entries(map)
        .map(([s, ps]) => `${coloredStoneCircle(Number(s))}${
            ps.size > 0 ? `&nbsp;${[...ps].map(p => fmtPlayerString(players, p)).join(' ')}` : ''
        }`)
        .join('&nbsp;'.repeat(5));
// Renders e.g. "⬤⬤ alice     ⬤ ⌂" (five spaces between adjacent turns), each
// turn showing one large circle per offered stone (the player picks
// among all of them at move time via the stone-selection popup - see
// selectingStone), then who plays it - same convention as fmtMap, but
// sourced from turnList's own player field (turn order/ownership)
// rather than stoneToPlayerMap (scoring). winners (finished-game record entries only -
// null everywhere else) marks each winning player with ♚.
export const fmtTurnList = (
    turnList: TurnInfo[], players: Map<number, PlayerInfo>, winners: number[] | null = null,
) =>
    turnList
        .map(({ player, stones, protected: prot, friendly }) => {
            const offeredStones = stones
                .map((s, i) => s === 1 ? i + 1 : -1)
                .filter(s => s >= 0);
            const stoneIcons = offeredStones.map(coloredStoneCircle).join('');
            const protectedStones = prot
                .map((p, i) => p === 1 ? i + 1 : -1)
                .filter(s => s >= 0);
            const protSuffix = protectedStones.length > 0
                ? `&nbsp;🔒${protectedStones.map(coloredStoneDot).join('')}`
                : '';
            const friendlyStones = friendly
                .map((f, i) => f === 1 ? i + 1 : -1)
                .filter(s => s >= 0);
            const friendSuffix = friendlyStones.length > 0
                ? `&nbsp;🤝${friendlyStones.map(coloredStoneDot).join('')}`
                : '';
            const crownSuffix = winners?.includes(player) ? ' &#9818;' : '';
            return `${stoneIcons}&nbsp;${fmtPlayerString(players, player)}${crownSuffix}${protSuffix}${friendSuffix}`;
        })
        .join('&nbsp;'.repeat(5));
// Renders e.g. "⬤ P1:5  P2:2   ⬤ P2:3" (two spaces between players,
// three between stones), each stone's circle followed by every player
// who has a finite placement limit for that color - see
// GameConfig.playerStonePlaceLimit's doc comment in types.ts. A player
// with no limit (null/unlimited) is omitted from that stone's entry,
// and a stone nobody has a limit on at all is omitted entirely.
export const fmtPlaceLimit = (limit: (number | null)[][]) =>
    limit
        .map((row, i) => {
            const entries = row
                .map((lim, j) => lim !== null ? `P${j + 1}:${lim}` : null)
                .filter((s): s is string => s !== null);
            return entries.length > 0 ? `${coloredStoneCircle(i + 1)}&nbsp;${entries.join('&nbsp;&nbsp;')}` : null;
        })
        .filter((s): s is string => s !== null)
        .join('&nbsp;&nbsp;&nbsp;');
// Renders e.g. "⬤ 5   ⬤ ∞" (three spaces between stones), each
// stone's circle followed by its total-across-all-players placement
// limit ('∞' for null/unlimited) - see GameConfig.globalStonePlaceLimit's
// doc comment in types.ts.
export const fmtGlobalLimit = (limit: (number | null)[]) =>
    limit
        .map((lim, i) => `${coloredStoneCircle(i + 1)}&nbsp;${lim === null ? '∞' : lim}`)
        .join('&nbsp;&nbsp;&nbsp;');

// Pure: HTML for the "Current Game Info" side-panel node's content - the
// active game's live rules, sourced from its BoardView v (already-resolved
// playerStonePlaceLimit/globalStonePlaceLimit grids etc.) rather than the
// original GameConfig object, which may hold those fields unresolved/empty -
// plus players (player identity has no BoardView representation, since it's
// UI-session info rather than board/rules state). Caller assigns the result
// to a container's innerHTML.
export function currentGameSetupHtml(v: BoardView, players: Map<number, PlayerInfo>): string {
    return `
        <div><b>Type of stones:</b> ${v.numStones}</div>
        <div><b>Number of players:</b> ${v.numPlayers}</div>
        <div><b>Turn list:</b> ${fmtTurnList(v.turnList, players)}</div>
        <div><b>Stone to player map:</b> ${fmtMap(v.stoneToPlayerMap, players)}</div>
        <div><b>Player stone placement limit:</b> ${fmtPlaceLimit(v.playerStonePlaceLimit)}</div>
        <div><b>Global stone placement limit:</b> ${fmtGlobalLimit(v.globalStonePlaceLimit)}</div>
        <div><b>Forced pass only:</b> ${v.forcedPassOnly}</div>
        <div><b>Allow suicide:</b> ${v.allowSuicide}</div>
        <div><b>Score rule:</b> ${v.scoreRule}</div>
        <div><b>Ko rule:</b> ${v.koRule}</div>
        <div><b>Komi:</b> ${v.komi.join(', ')}</div>
        <div><b>Max plies:</b> ${v.maxPlies === null ? '∞' : v.maxPlies}</div>
    `;
}

// Pure: HTML for the "New Game" side-panel node's read-only detail rows from
// cfg (Renderer.newCfg, the not-yet-submitted config the ns/np/etc. commands
// mutate) - the preset-selection/Start-new-game buttons live in
// #new-game-buttons, built separately by Renderer._refreshSidePanel(), not
// part of this HTML. Caller assigns the result to a container's innerHTML.
export function newGameSetupHtml(cfg: GameConfig): string {
    const boardDescrText = unparseCleg(cfg.boardDescr);
    // Sized to the actual line count (capped, same rationale as the Configure Board popup's own
    // fixed rows="12") rather than a fixed height - most board descriptions are only 1-3 lines.
    const boardDescrRows = Math.min(Math.max(boardDescrText.split('\n').length, 1), 12);
    return `
        <div><b>Board description:</b></div>
        <textarea class="account-input board-descr-view" readonly wrap="off" rows="${boardDescrRows}">${escapeHtml(boardDescrText)}</textarea>
        <div><b>Type of stones:</b> ${cfg.numStones}</div>
        <div><b>Number of players:</b> ${cfg.numPlayers}</div>
        <div><b>Turn list:</b> ${fmtTurnList(cfg.turnList, cfg.players)}</div>
        <div><b>Stone to player map:</b> ${fmtMap(cfg.stoneToPlayerMap, cfg.players)}</div>
        <div><b>Player stone placement limit:</b> ${fmtPlaceLimit(cfg.playerStonePlaceLimit)}</div>
        <div><b>Global stone placement limit:</b> ${fmtGlobalLimit(cfg.globalStonePlaceLimit)}</div>
        <div><b>Forced pass only:</b> ${cfg.forcedPassOnly}</div>
        <div><b>Allow suicide:</b> ${cfg.allowSuicide}</div>
        <div><b>Score rule:</b> ${cfg.scoreRule}</div>
        <div><b>Ko rule:</b> ${cfg.koRule}</div>
        <div><b>Komi:</b> ${cfg.komi.join(', ')}</div>
        <div><b>Max plies:</b> ${cfg.maxPlies === null ? '∞' : cfg.maxPlies}</div>
    `;
}
