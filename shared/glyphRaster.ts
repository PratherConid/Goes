/**
 * Rasterizes a single character of an already-loaded TrueType/OpenType font into a boolean grid -
 * pure TypeScript, no <canvas>/DOM dependency, so this runs identically in the browser and under
 * plain Node (e.g. a board-preset generator or a test). Uses opentype.js to parse the font and
 * extract the glyph's own vector outline (a list of line/quadratic/cubic-bezier path commands), then
 * rasterizes that outline itself via a nonzero-winding-number point-in-polygon test - see
 * rasterizeGlyph's own doc comment below for the grid-mapping/sampling details.
 */
// opentype.js resolves to two genuinely different module shapes depending on who's loading it, and
// neither a default import nor a named import works under both:
//   - Node's own module resolution (used by the tsx-based test suite) has no "exports" map to go on
//     in opentype.js's own package.json, so it falls back to the CJS "main" field and
//     cjs-module-lexer - which doesn't reliably detect this particular bundle's own named exports
//     (`import { parse } from 'opentype.js'` fails at runtime: "does not provide an export named
//     'parse'"), but DOES expose the whole CJS module.exports object as the default import.
//   - vite/rollup (bundler moduleResolution) instead picks opentype.js's own real ESM build
//     (dist/opentype.mjs, its package.json's "module" field) - a true ES module with `parse` etc. as
//     genuine named exports and NO default export at all (a default import fails to build: "'default'
//     is not exported by ... opentype.mjs").
// A namespace import works syntactically either way, but its own shape still differs (the CJS
// interop case puts everything under `.default`; the real-ESM case has `parse` etc. directly on the
// namespace, no `.default`) - so this line handles both shapes explicitly instead of relying on
// either resolver's own default-export behavior. Types are unaffected either way (erased at compile
// time, so a type-only import never hits this at runtime).
import * as opentypeNs from 'opentype.js';
import type { Path, PathCommand } from 'opentype.js';
import { assert } from './types.js';

const opentype = (opentypeNs as unknown as { default?: typeof opentypeNs }).default ?? opentypeNs;

// ── font registry ────────────────────────────────────────────────────────────
//
// stringBoard()/strB (shared/boardConfig.ts, shared/clegEval.ts) resolve a font by plain string name
// at cleg-eval time, but cleg evaluation is fully synchronous end to end and shared/ files never do
// I/O themselves - so the actual font bytes have to be loaded (an inherently async operation, at
// least in the browser) and registered here BEFORE any cleg program that might use them runs.
// loadFont() below does the registering; loading the bytes themselves is each entry point's own job
// (src/main.ts fetches public/fonts/<name>.ttf; server/src/index.ts fs.readFileSync's the same
// files) - see BUNDLED_FONT_NAMES just below for the fixed list both of them load at startup.

/** The fixed set of font names both app entry points (src/main.ts, server/src/index.ts) load from
 * public/fonts/<name>.ttf and register via loadFont() at their own startup - also the exact set of
 * `fontName` values strB()/stringBoard() can succeed with once that's done. */
export const BUNDLED_FONT_NAMES = [
    'Roboto-Regular', 'OpenSans-Regular', 'Montserrat-Regular', 'Lato-Regular',
    'Roboto-Bold', 'OpenSans-Bold', 'Montserrat-Bold', 'Lato-Bold',
];

const fontRegistry = new Map<string, ArrayBuffer>();

/** Registers `data` (already-loaded font bytes) under `name`, so a later rasterizeGlyph/stringBoard
 * call using that name can find it via getFont() below. Call once per font, before evaluating any
 * cleg program that might reference it (see this section's own top comment) - silently overwrites
 * if called again with the same `name`. */
export function loadFont(name: string, data: ArrayBuffer): void {
    fontRegistry.set(name, data);
}

/** Returns the bytes registered under `name` via loadFont() above - throws if nothing was ever
 * registered under that name. */
export function getFont(name: string): ArrayBuffer {
    const data = fontRegistry.get(name);
    if (!data)
        throw new Error(
            `glyphRaster: no font loaded under '${name}' - loaded fonts: ` +
            (fontRegistry.size === 0 ? '(none)' : [...fontRegistry.keys()].join(', ')));
    return data;
}

// A single (already curve-flattened) line segment, in font units - the glyph's own unscaled
// coordinate space, y-up, origin at the typesetting baseline (see rasterizeGlyph's own use of
// glyph.path, font.ascender/descender, and glyph.advanceWidth below, all in that same space).
interface Segment { x0: number; y0: number; x1: number; y1: number; }

// How many line segments a single quadratic/cubic bezier curve is flattened into - fine enough that
// the resulting polygon is visually indistinguishable from the true curve at any grid resolution
// this function is realistically called with.
const CURVE_STEPS = 12;

// Flattens `path`'s own M/L/Q/C/Z commands into a flat list of line segments, one list per glyph
// (a glyph's path may have several closed contours - e.g. "o"'s outer ring and inner counter - all
// concatenated here; contour boundaries don't need to be tracked separately since the winding-number
// test below only cares about the full segment list, not which contour each one came from).
function flattenPath(path: Path): Segment[] {
    const segments: Segment[] = [];
    let startX = 0, startY = 0, curX = 0, curY = 0;

    function line(x1: number, y1: number) {
        segments.push({ x0: curX, y0: curY, x1, y1 });
        curX = x1; curY = y1;
    }
    function quadratic(cx: number, cy: number, x1: number, y1: number) {
        const x0 = curX, y0 = curY;
        for (let s = 1; s <= CURVE_STEPS; s++) {
            const t = s / CURVE_STEPS, mt = 1 - t;
            line(mt*mt*x0 + 2*mt*t*cx + t*t*x1, mt*mt*y0 + 2*mt*t*cy + t*t*y1);
        }
    }
    function cubic(c1x: number, c1y: number, c2x: number, c2y: number, x1: number, y1: number) {
        const x0 = curX, y0 = curY;
        for (let s = 1; s <= CURVE_STEPS; s++) {
            const t = s / CURVE_STEPS, mt = 1 - t;
            line(
                mt*mt*mt*x0 + 3*mt*mt*t*c1x + 3*mt*t*t*c2x + t*t*t*x1,
                mt*mt*mt*y0 + 3*mt*mt*t*c1y + 3*mt*t*t*c2y + t*t*t*y1);
        }
    }

    for (const cmd of path.commands as PathCommand[]) {
        switch (cmd.type) {
            case 'M': startX = curX = cmd.x; startY = curY = cmd.y; break;
            case 'L': line(cmd.x, cmd.y); break;
            case 'Q': quadratic(cmd.x1, cmd.y1, cmd.x, cmd.y); break;
            case 'C': cubic(cmd.x1, cmd.y1, cmd.x2, cmd.y2, cmd.x, cmd.y); break;
            case 'Z': line(startX, startY); break;
        }
    }
    return segments;
}

// The standard nonzero-winding-number point-in-polygon test (Dan Sunday's formulation): sums a
// signed crossing count of a rightward ray from (px, py) against every segment, without needing to
// construct the ray explicitly. Matches TrueType's own nonzero fill rule (a contour's clockwise/
// counterclockwise direction distinguishes a solid region from a hole - e.g. "o"'s inner counter is
// wound opposite to its outer ring), unlike a simpler even-odd rule which happens to agree for most
// ordinary letterforms but isn't what the format itself specifies.
function windingNumber(segments: Segment[], px: number, py: number): number {
    let winding = 0;
    for (const { x0, y0, x1, y1 } of segments) {
        if (y0 <= py) {
            if (y1 > py && (x1 - x0) * (py - y0) - (px - x0) * (y1 - y0) > 0) winding++;
        } else {
            if (y1 <= py && (x1 - x0) * (py - y0) - (px - x0) * (y1 - y0) < 0) winding--;
        }
    }
    return winding;
}

// Each output cell is sampled at this many points per axis (SUPERSAMPLE^2 total), marked on iff at
// least half fall inside the glyph - a single center-point sample per cell can make a thin stroke
// vanish entirely between two adjacent samples; supersampling avoids that without needing true
// (fractional) coverage/anti-aliasing, which a boolean-per-cell result has no way to represent anyway.
const SUPERSAMPLE = 4;

/**
 * Rasterizes `char` (a single character) from `fontData` (raw TrueType/OpenType file bytes - e.g.
 * from fs.readFileSync in Node, or fetch(...).then(r => r.arrayBuffer()) in the browser; loading the
 * bytes isn't this function's own concern, matching every other shared/ file's avoidance of direct
 * I/O) into a boolean grid, `grid[row][col]`, `row` 0 at the TOP - `true` where the glyph's own
 * filled outline covers that cell (per its nonzero winding rule).
 *
 * `hScale`/`vScale` are cells-per-em (1 em = font.unitsPerEm font units), independently per axis -
 * NOT a fixed cols/rows target - so that composing several characters (e.g. spelling out a word)
 * keeps their true relative proportions instead of every glyph being stretched to the same box:
 *   - `rows` comes from the font's own ascender-to-descender span (a fixed per-FONT quantity, not
 *     per-glyph) at `vScale`, so every character rasterized from the same font at the same `vScale`
 *     gets the exact same row count - required for stacking/placing them on a shared baseline.
 *   - `cols` comes from THIS glyph's own advance width (the horizontal space it actually occupies
 *     when typeset next to another character) at `hScale`, so a narrow "i" and a wide "m" come out
 *     proportionally narrow/wide rather than both filling an identical box.
 * The grid's own origin (row 0/col 0) is the character's own typesetting origin (baseline, left
 * edge of its advance box) - NOT its own ink bounding box - again so multiple glyphs from one call
 * site line up correctly if placed side by side at that shared scale, rather than each being
 * independently centered/stretched to its own ink extent. A glyph with no visible outline (e.g. a
 * space) returns an all-false grid of the size its advance width/the font's line height still work
 * out to, rather than throwing, since "nothing to draw" is a legitimate result.
 */
export function rasterizeGlyph(fontData: ArrayBuffer, char: string, hScale: number, vScale: number): boolean[][] {
    assert(char.length === 1, `rasterizeGlyph: expected a single character, got '${char}'`);
    assert(hScale > 0, `rasterizeGlyph: hScale must be positive, got ${hScale}`);
    assert(vScale > 0, `rasterizeGlyph: vScale must be positive, got ${vScale}`);

    const font = opentype.parse(fontData);
    const glyph = font.charToGlyph(char);
    const unitsPerEm = font.unitsPerEm;

    const rows = Math.max(1, Math.round(vScale * (font.ascender - font.descender) / unitsPerEm));
    const cols = Math.max(1, Math.round(hScale * (glyph.advanceWidth ?? 0) / unitsPerEm));
    const grid: boolean[][] = Array.from({ length: rows }, () => new Array(cols).fill(false));
    if (glyph.path.commands.length === 0) return grid; // no outline (e.g. space) - nothing to draw

    // glyph.path (the raw, unscaled outline, in the same font-unit/y-up coordinate space as
    // font.ascender/descender and glyph.advanceWidth above) - NOT glyph.getPath(...), which negates y
    // along the way (it targets <canvas>'s own y-down convention) and so would disagree with them.
    const segments = flattenPath(glyph.path);

    for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
            let inside = 0;
            for (let sy = 0; sy < SUPERSAMPLE; sy++) {
                // row 0 is the TOP (font-y = ascender); font-y decreases by one em/vScale per row.
                const rowFrac = row + (sy + 0.5) / SUPERSAMPLE;
                const py = font.ascender - rowFrac * unitsPerEm / vScale;
                for (let sx = 0; sx < SUPERSAMPLE; sx++) {
                    // col 0 is the glyph's own left side (font-x = 0); font-x increases by one
                    // em/hScale per column.
                    const colFrac = col + (sx + 0.5) / SUPERSAMPLE;
                    const px = colFrac * unitsPerEm / hScale;
                    if (windingNumber(segments, px, py) !== 0) inside++;
                }
            }
            grid[row][col] = inside * 2 >= SUPERSAMPLE * SUPERSAMPLE;
        }
    }
    return grid;
}

/**
 * Renders `str` (one or more characters) into a single combined boolean grid, `grid[row][col]` -
 * each character's own rasterizeGlyph() bitmap, placed at its own cumulative column offset, one
 * after another left to right. Since every character from one font at one `vScale` shares the same
 * row count and the same typesetting-origin convention (see rasterizeGlyph's own doc comment),
 * concatenating them this way reproduces ordinary left-to-right typesetting, including each
 * character's own natural spacing (already folded into its own column count) with no extra gap
 * logic needed here. shared/boardConfig.ts's own stringBoard() builds directly on this.
 */
export function rasterizeString(fontData: ArrayBuffer, str: string, hScale: number, vScale: number): boolean[][] {
    assert(str.length > 0, 'rasterizeString: str must be nonempty');
    const glyphs = [...str].map(ch => rasterizeGlyph(fontData, ch, hScale, vScale));
    const rows = glyphs[0].length;
    const totalCols = glyphs.reduce((sum, g) => sum + g[0].length, 0);

    const grid: boolean[][] = Array.from({ length: rows }, () => new Array(totalCols).fill(false));
    let colOffset = 0;
    for (const glyph of glyphs) {
        const cols = glyph[0].length;
        for (let row = 0; row < rows; row++)
            for (let col = 0; col < cols; col++)
                grid[row][colOffset + col] = glyph[row][col];
        colOffset += cols;
    }
    return grid;
}
