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

export type PlayerType = 'local' | 'server' | 'client';

export class PlayerInfo {
    type: PlayerType;
    name: string;
    socket: WebSocket | null;   // set on server (ws package); null on client

    constructor(type: PlayerType, name: string, socket: WebSocket | null = null) {
        this.type   = type;
        this.name   = name;
        this.socket = socket;
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

    constructor(
        boardType: string,
        boardArgs: number[],
        numStones: number,
        numPlayers: number,
        turnStoneList: number[],
        stoneToPlayerMap: Record<number, number>,
        forcedPassOnly: boolean,
    ) {
        this.boardType       = boardType;
        this.boardArgs       = boardArgs;
        this.numStones       = numStones;
        this.numPlayers      = numPlayers;
        this.turnStoneList   = turnStoneList;
        this.stoneToPlayerMap = stoneToPlayerMap;
        this.forcedPassOnly  = forcedPassOnly;
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
        );
    }
}

export interface PendingGame {
    id: string;
    config?: GameConfig;                // always set server-side; set client-side only by creator
    players: Map<number, PlayerInfo>;   // key = slot; socket always null on client
    pendingSlots: number[];             // pre-shuffled unassigned slots (server); [] on client
}

export interface OnlineStateResponse {
    status: 'waiting' | 'playing' | 'finished';
    players: ({ name: string; slot: number } | null)[];
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
