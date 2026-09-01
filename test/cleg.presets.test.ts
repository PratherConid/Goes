// Regression/smoke tests over every real preset file: every public/board_presets/*.cleg file
// parses and evaluates to a stable board (node/edge count locked in as a golden value below), and
// every public/game_presets/*.json file parses via GameConfig.fromJSON() (which itself parses its
// own boardDescr via parseCleg() - see shared/gameConfig.ts).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseCleg } from '../shared/clegParser.ts';
import { buildBoardFromCleg } from '../shared/clegEval.ts';
import { GameConfig } from '../shared/gameConfig.ts';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const boardPresetsDir = path.join(__dirname, '..', 'public', 'board_presets');
const gamePresetsDir = path.join(__dirname, '..', 'public', 'game_presets');

// shared/selector.ts's randomlyRemove() is the one Math.random() call site any preset can reach
// (via a (rrmp ...)/(rrmn ...) selector, or the randRmN/randRmP cleg builtins) - seeded here so the
// handful of presets using it (fractaldrop-built ones, nice_drop, rrmp) evaluate to the exact same
// board every run, the same substitute-a-seeded-PRNG technique used earlier this session to verify
// randomized presets structurally.
function mulberry32(seed: number): () => number {
    return () => {
        seed |= 0;
        seed = (seed + 0x6D2B79F5) | 0;
        let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function edgeCount(adj: number[][]): number {
    let count = 0;
    for (let i = 0; i < adj.length; i++)
        for (let j = i + 1; j < adj.length; j++)
            if (adj[i]![j]) count++;
    return count;
}

// Golden N/edge-count values, one per public/board_presets/*.cleg file - captured from
// buildBoardFromCleg()'s current, verified-correct behavior (cross-checked this session against an
// independent C++ port's own evaluation of the same files), under mulberry32(42) for the presets
// whose own selectors call randomlyRemove(). A failure here means either a real regression or a
// deliberate preset-content change - update the golden value in the latter case, don't delete the
// assertion; add a new entry (and re-run to capture its own golden value) for a new preset file.
const BOARD_PRESET_GOLDEN: Record<string, { N: number; edges: number }> = {
    'biTemple_13_13_9_3.cleg': { N: 450, edges: 864 },
    'cpentflake_4.cleg': { N: 560, edges: 820 },
    'cpolyflake_8_3.cleg': { N: 392, edges: 560 },
    'cublat_2_2_2_es_1_rect_form_7_(tri)_(quad)_scale_0.75.cleg': { N: 362, edges: 864 },
    'cublat_2_2_2_quadform_9.cleg': { N: 386, edges: 768 },
    'cublat_3_3_3_es_1_rect_nice_form_4_quad_tri.cleg': { N: 502, edges: 1296 },
    'cublat_3_3_3_quadform_5_nice_drop_0.1.cleg': { N: 450, edges: 861 },
    'cublat_3_3_3_sel_(deg_gt_3)_quadform_6.cleg': { N: 331, edges: 630 },
    'cublat_4_4_4_cub_0010_quadform_4.cleg': { N: 584, edges: 1224 },
    'cublat_9_9_9_nice_drop_0.2.cleg': { N: 569, edges: 1201 },
    'diamondCubic_10.cleg': { N: 385, edges: 660 },
    'dodeca_gcent_triform_6.cleg': { N: 401, edges: 1150 },
    'goDesk_19_19_5_2_6_2.cleg': { N: 1300, edges: 2604 },
    'hcub_2_6_6_6_6.cleg': { N: 528, edges: 1120 },
    'hexpipe_6.cleg': { N: 656, edges: 1260 },
    'menger_3_3_0101.cleg': { N: 776, edges: 1776 },
    'menger_4_2_011.cleg': { N: 688, edges: 1272 },
    'octa_triform_4_quadform_4_triform_4.cleg': { N: 254, edges: 564 },
    'rect_13_13.cleg': { N: 169, edges: 312 },
    'rect_19_19.cleg': { N: 361, edges: 684 },
    'rect_19_19_nis_(rrmp_0.1_(all))_nis_(conve_quad_(conva_node_(all))).cleg': { N: 315, edges: 537 },
    'rect_3_3.cleg': { N: 9, edges: 12 },
    'rect_5_5_fractaldrop_3_0.05.cleg': { N: 636, edges: 1142 },
    'rect_7_7_diag_non_diag_3_0.5.cleg': { N: 241, edges: 488 },
    'rect_9_9.cleg': { N: 81, edges: 144 },
    'regpoly_13_prod_regpoly_13.cleg': { N: 169, edges: 338 },
    'regpoly_5_es_5_prod_lin_6.cleg': { N: 150, edges: 275 },
    'regpoly_5_gcent_prod_line_2_quadform_7.cleg': { N: 392, edges: 756 },
    'regpoly_5_prod_line_2_gcent_triform_7.cleg': { N: 286, edges: 825 },
    'ring_5_12.5.cleg': { N: 420, edges: 772 },
    'roundTable_9.5_5_3_2.cleg': { N: 457, edges: 884 },
    'shell_6_7.5.cleg': { N: 896, edges: 1830 },
    'sier_3_5.cleg': { N: 514, edges: 1536 },
    'snubsqtri_4_4_4.cleg': { N: 286, edges: 645 },
    'star_5_es_6_prod_line_5.cleg': { N: 155, edges: 274 },
    'tetrahedron_centering_9.cleg': { N: 341, edges: 704 },
    'tri_4_fractaldrop_3_0.05.cleg': { N: 275, edges: 689 },
    'trunc_trunc_cublat_3_3_3.cleg': { N: 420, edges: 810 },
    'truncated_24_cell.cleg': { N: 192, edges: 384 },
    'truncated_centralized_rect_6_6.cleg': { N: 320, edges: 480 },
    'twsqCluster4D_4_4_2.cleg': { N: 960, edges: 1408 },
    'twsq_3_3_2_es_3_prod_lin_4.cleg': { N: 528, edges: 972 },
    'twsq_3_3_4_quadocta.cleg': { N: 306, edges: 876 },
    'twsq_7_7_2.cleg': { N: 196, edges: 280 },
};

test('every public/board_presets/*.cleg file parses and evaluates to its golden node/edge count', () => {
    const files = fs.readdirSync(boardPresetsDir).filter(f => f.endsWith('.cleg')).sort();
    assert.deepEqual(
        files, Object.keys(BOARD_PRESET_GOLDEN).sort(),
        'BOARD_PRESET_GOLDEN must have exactly one entry per public/board_presets/*.cleg file - ' +
        'update it when presets are added/removed/renamed');

    const originalRandom = Math.random;
    try {
        for (const file of files) {
            const source = fs.readFileSync(path.join(boardPresetsDir, file), 'utf8');
            const program = parseCleg(source);
            Math.random = mulberry32(42);
            const bc = buildBoardFromCleg(program);
            const golden = BOARD_PRESET_GOLDEN[file]!;
            assert.equal(bc.N, golden.N, `${file}: node count`);
            assert.equal(edgeCount(bc.adj), golden.edges, `${file}: edge count`);
        }
    } finally {
        Math.random = originalRandom;
    }
});

test('every public/game_presets/*.json file parses via GameConfig.fromJSON()', () => {
    const files = fs.readdirSync(gamePresetsDir).filter(f => f.endsWith('.json'));
    assert.ok(files.length > 0, 'expected at least one game preset file');
    for (const file of files) {
        const raw = JSON.parse(fs.readFileSync(path.join(gamePresetsDir, file), 'utf8'));
        const config = GameConfig.fromJSON(raw);
        assert.ok(config.numStones > 0, `${file}: numStones`);
        assert.ok(config.numPlayers > 0, `${file}: numPlayers`);
        assert.ok(config.turnList.length > 0, `${file}: turnList`);
    }
});
