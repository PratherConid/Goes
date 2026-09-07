import { BoardState } from '@shared/boardState.js';
import { buildBoardFromCleg } from '@shared/clegEval.js';
import { GameConfig } from '@shared/gameConfig.js';
import { loadFont, BUNDLED_FONT_NAMES } from '@shared/glyphRaster.js';
import { Renderer } from './renderer.js';

// Loads every bundled font (public/fonts/<name>.ttf) and registers it with shared/glyphRaster.ts's
// own loadFont() - see that file's own "font registry" section for why this has to happen before
// any cleg program using strB(...) can be evaluated (cleg evaluation is fully synchronous, so the
// actual async loading has to be finished up front, at startup, rather than on demand).
async function loadBundledFonts(): Promise<void> {
    await Promise.all(BUNDLED_FONT_NAMES.map(async name => {
        const data = await fetch(`/fonts/${name}.ttf`).then(r => r.arrayBuffer());
        loadFont(name, data);
    }));
}

async function main() {
    const [raw] = await Promise.all([
        fetch('/game_presets/9x9_go_fpo.json').then(r => r.json()),
        loadBundledFonts(),
    ]);
    const cfg = GameConfig.fromJSON(raw);
    const bc  = buildBoardFromCleg(cfg.boardDescr);
    const game = new BoardState(
        cfg.numStones, cfg.numPlayers, cfg.turnList, cfg.playerStonePlaceLimit, cfg.globalStonePlaceLimit,
        cfg.stoneToPlayerMap, cfg.forcedPassOnly, cfg.scoreRule, cfg.komi, cfg.koRule, cfg.allowSuicide,
        cfg.maxPlies, new Array(bc.N).fill(0), bc,
    );
    const renderer = new Renderer(game);
    renderer.init();
}

void main();
