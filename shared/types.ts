export const enum MoveType {
    NOMOVE   = 0,
    PLACE    = 1,
    PASS     = 2,
    GAMEOVER = 3,
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
    captures: number[];
    passedPlayers: Set<number>;
}

export interface HistoryEntry {
    board: number[];
    nextPlayer: number;
    plyCount: number;
    lenTurnList: number;
}

export type PlayerType = 'local' | 'server' | 'serverEngine' | 'client';

export class PlayerInfo {
    type: PlayerType;
    name: string;
    socket: WebSocket | null;   // set on server (ws package); null on client
    emsim: number;  // AI simulations per move (0 = server default); for serverEngine slots
    temp: number;   // AI temperature (0 = server default); for serverEngine slots

    constructor(type: PlayerType, name: string, socket: WebSocket | null = null,
                emsim = 0, temp = 0) {
        this.type   = type;
        this.name   = name;
        this.socket = socket;
        this.emsim  = emsim;
        this.temp   = temp;
    }
}

export class GameConfig {
    boardType: string;
    boardArgs: number[];
    numStones: number;
    numPlayers: number;
    turnStoneList: number[];
    stoneToPlayerMap: Record<number, number>;
    forcedPassOnly: boolean;
    players: Map<number, PlayerInfo>;  // slot → player; empty slots are pending/unassigned

    constructor(
        boardType: string,
        boardArgs: number[],
        numStones: number,
        numPlayers: number,
        turnStoneList: number[],
        stoneToPlayerMap: Record<number, number>,
        forcedPassOnly: boolean,
        players: Map<number, PlayerInfo> = new Map(),
    ) {
        this.boardType        = boardType;
        this.boardArgs        = boardArgs;
        this.numStones        = numStones;
        this.numPlayers       = numPlayers;
        this.turnStoneList    = turnStoneList;
        this.stoneToPlayerMap = stoneToPlayerMap;
        this.forcedPassOnly   = forcedPassOnly;
        this.players          = players;
    }

    copy(): GameConfig {
        return new GameConfig(
            this.boardType,
            [...this.boardArgs],
            this.numStones,
            this.numPlayers,
            [...this.turnStoneList],
            { ...this.stoneToPlayerMap },
            this.forcedPassOnly,
            new Map([...this.players.entries()].map(
                ([s, pi]) => [s, new PlayerInfo(pi.type, pi.name, null, pi.emsim, pi.temp)]
            )),
        );
    }

    // Serialise players Map as an array so JSON.stringify works.
    toJSON() {
        return {
            boardType: this.boardType, boardArgs: this.boardArgs,
            numStones: this.numStones, numPlayers: this.numPlayers,
            turnStoneList: this.turnStoneList, stoneToPlayerMap: this.stoneToPlayerMap,
            forcedPassOnly: this.forcedPassOnly,
            players: [...this.players.entries()].map(([slot, pi]) =>
                ({ slot, type: pi.type, name: pi.name, emsim: pi.emsim, temp: pi.temp })),
        };
    }

    static fromJSON(raw: any): GameConfig {
        const players = new Map<number, PlayerInfo>(
            ((raw.players ?? []) as { slot: number; type: PlayerType; name: string; emsim: number; temp: number }[])
                .map(p => [p.slot, new PlayerInfo(p.type, p.name, null, p.emsim ?? 0, p.temp ?? 0)])
        );
        return new GameConfig(
            raw.boardType, raw.boardArgs, raw.numStones, raw.numPlayers,
            raw.turnStoneList, raw.stoneToPlayerMap, raw.forcedPassOnly, players,
        );
    }
}

export interface PendingGame {
    id: string;
    config: GameConfig;   // always set; config.players tracks assigned slots
}

export interface OnlineStateResponse {
    status: 'waiting' | 'playing' | 'finished';
    moves: (number | null)[];
    winners: number[];
    resignedPlayers: number[];
}

export interface BoardView {
    N: number;
    pos: number[][];           // N×2
    boardDimension: [[number, number], [number, number]];
    numStones: number;
    numPlayers: number;
    turnStoneList: number[];
    stoneToPlayerMap: Record<number, number>;
    forcedPassOnly: boolean;
    nextPlayer: number;
    lastMove: MoveInfo;
    stoneCount: Record<number, number>;
    winners: number[];
    resignedPlayers: number[];
    plyCount: number;
    history: HistoryEntry[];
    legalMoveHistory: (Set<number> | null)[][];
    gameOver: boolean;
    passEnabled: boolean;
}

const _ID_CHARS = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
export function makeId(len: number): string {
    let id = '';
    for (let i = 0; i < len; i++) id += _ID_CHARS[Math.floor(Math.random() * _ID_CHARS.length)];
    return id;
}
