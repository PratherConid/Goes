// Shared BoardState/GameConfig construction for the rule tests (not a test file itself).
//
// Both constructors take the same long list of positional rule parameters, nearly all of which any
// given test wants at the same neutral setting (no placement limits, stone i scores for player i,
// no komi, area scoring, situational ko, no suicide, no ply cap). These builders default all of
// them, so a test spells out only the rule it is actually about.
import { BoardState } from '../shared/boardState.ts';
import { GameConfig } from '../shared/gameConfig.ts';
import { parseCleg } from '../shared/clegParser.ts';
import type { BoardConfig, TurnInfo, ScoreRule, KoRule } from '../shared/types.ts';

const zeros = (n: number) => new Array(n).fill(0);

// A turnList of `numPlayers` turns where player i moves in turn i, offered stone i alone.
export function soloStoneTurns(numPlayers: number): TurnInfo[] {
    return Array.from({ length: numPlayers }, (_, i) => ({
        player: i + 1,
        stones: zeros(numPlayers).map((_, k) => (k === i ? 1 : 0)),
        protected: zeros(numPlayers), friendly: zeros(numPlayers),
    }));
}

// A turnList of `numPlayers` turns where player i moves in turn i, offered every stone - so
// captures/liberties never constrain which stone a test may place.
export function allStonesTurns(numPlayers: number, numStones = numPlayers): TurnInfo[] {
    return Array.from({ length: numPlayers }, (_, i) => ({
        player: i + 1,
        stones: new Array(numStones).fill(1),
        protected: zeros(numStones), friendly: zeros(numStones),
    }));
}

// Every rule parameter BoardState and GameConfig share. numStones defaults to the offered-stone
// count of the first turn and numPlayers to the highest player appearing in the turn list, both of
// which need overriding for a game whose turn list doesn't mention every stone/player.
export interface RuleOptions {
    numStones?: number;
    numPlayers?: number;
    playerStonePlaceLimit?: (number | null)[][];    // [numStones][numPlayers], default all null
    globalStonePlaceLimit?: (number | null)[];      // [numStones], default all null
    stoneToPlayerMap?: Record<number, Set<number>>; // default: stone i scores for player i alone
    forcedPassOnly?: boolean;
    scoreRule?: ScoreRule;
    komi?: number[];                                // [numPlayers], default all 0
    koRule?: KoRule;
    allowSuicide?: boolean;
    maxPlies?: number | null;
}

// RuleOptions with every default filled in, in BoardState/GameConfig constructor order.
function resolveRules(turnList: TurnInfo[], opts: RuleOptions) {
    const numStones = opts.numStones ?? turnList[0].stones.length;
    const numPlayers = opts.numPlayers ?? Math.max(...turnList.map(t => t.player));
    return [
        numStones, numPlayers, turnList,
        opts.playerStonePlaceLimit ?? Array.from({ length: numStones }, () => new Array(numPlayers).fill(null)),
        opts.globalStonePlaceLimit ?? new Array(numStones).fill(null),
        opts.stoneToPlayerMap
            ?? Object.fromEntries(Array.from({ length: numStones }, (_, i) => [i + 1, new Set([i + 1])])),
        opts.forcedPassOnly ?? false,
        opts.scoreRule ?? 'area',
        opts.komi ?? zeros(numPlayers),
        opts.koRule ?? 'situational',
        opts.allowSuicide ?? false,
        opts.maxPlies ?? null,
    ] as const;
}

// `board` defaults to an empty board of bc's own size.
export function makeBoardState(
    bc: BoardConfig, turnList: TurnInfo[], opts: RuleOptions & { board?: number[] } = {},
): BoardState {
    return new BoardState(...resolveRules(turnList, opts), opts.board ?? new Array(bc.N).fill(0), bc);
}

export function makeGameConfig(boardDescr: string, turnList: TurnInfo[], opts: RuleOptions = {}): GameConfig {
    return new GameConfig(parseCleg(boardDescr), ...resolveRules(turnList, opts));
}
