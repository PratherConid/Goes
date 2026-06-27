export const enum MoveType {
    NOMOVE   = 0,
    ILLEGAL  = 1,
    PLACE    = 2,
    PASS     = 3,
    GAMEOVER = 4,
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

export interface OnlineGameConfig {
    boardType: string;
    boardArgs: number[];
    numStones: number;
    numPlayers: number;
    turnStoneList: number[];
    stoneToPlayerMap: Record<number, number>;
    forcedPassOnly: boolean;
}

export interface OnlineStateResponse {
    status: 'waiting' | 'playing' | 'finished';
    numPlayersRequired: number;
    numJoined: number;
    players: ({ name: string; slot: number } | null)[];
    moves: (number | null)[];
    currentStone: number | null;
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
    legalMoves: (Set<number> | null)[];
    legalMoveHistory: (Set<number> | null)[][];
    gameOver: boolean;
    passEnabled: boolean;
}
