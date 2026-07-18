import { BoardState } from '@shared/boardState.js';
import { rectangularBoard } from '@shared/boardConfig.js';
import { GameConfig } from '@shared/types.js';
import { Renderer } from './renderer.js';

async function main() {
    const raw = await fetch('/game_presets/9x9_go_fpo.json').then(r => r.json());
    const cfg = GameConfig.fromJSON(raw);
    const bc  = rectangularBoard(cfg.boardArgs[0], cfg.boardArgs[1]);
    const game = new BoardState(
        cfg.numStones, cfg.numPlayers, cfg.turnList, cfg.playerStonePlaceLimit, cfg.globalStonePlaceLimit,
        cfg.stoneToPlayerMap, cfg.forcedPassOnly, cfg.scoreRule, cfg.komi, cfg.koRule, cfg.allowSuicide,
        cfg.maxPlies, new Array(bc.N).fill(0), bc,
    );
    const renderer = new Renderer(game);
    renderer.init();
}

void main();
