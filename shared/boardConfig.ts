import type { GameConfig } from './types.js';

export interface BoardConfig {
    pos: number[][];  // N×2 array of node positions
    adj: number[][];  // N×N symmetric adjacency matrix, entries 0/1
    N: number;
    boardDimension: [[number, number], [number, number]];  // [[xmin,ymin],[xmax,ymax]]
}

function assert(cond: boolean, msg: string): asserts cond {
    if (!cond) throw new Error(`Assertion failed: ${msg}`);
}

function make(pos: number[][], adj: number[][]): BoardConfig {
    const N = pos.length;
    assert(adj.length === N && (N === 0 || adj[0].length === N), 'adj dimensions must match pos length');
    for (let i = 0; i < N; i++)
        for (let j = 0; j < N; j++)
            assert(adj[i][j] === adj[j][i], `adj must be symmetric: [${i}][${j}]`);
    const xs = pos.map(p => p[0]), ys = pos.map(p => p[1]);
    const boardDimension: [[number, number], [number, number]] = [
        [Math.min(...xs), Math.min(...ys)],
        [Math.max(...xs), Math.max(...ys)],
    ];
    return { pos, adj, N, boardDimension };
}

function zeroAdj(N: number): number[][] {
    return Array.from({ length: N }, () => new Array<number>(N).fill(0));
}

/** Glue pairs of nodes in `quot` together. The position of the new node is the average of its predecessors. */
export function quotientBoard(bc: BoardConfig, quot: [number, number][]): BoardConfig {
    const N = bc.N;
    for (const [a, b] of quot)
        assert(a >= 0 && a < N && b >= 0 && b < N, `quot indices [${a}, ${b}] out of bounds for N=${N}`);
    // Union-Find to compute equivalence classes
    const parent = Array.from({ length: N }, (_, i) => i);
    function find(x: number): number {
        while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    }
    for (const [a, b] of quot) {
        const pa = find(a), pb = find(b);
        if (pa !== pb) parent[pa] = pb;
    }
    const roots = Array.from({ length: N }, (_, i) => find(i));
    const uniqueRoots = [...new Set(roots)].sort((a, b) => a - b);
    const rootToNew = new Map(uniqueRoots.map((r, i) => [r, i]));
    const newN = uniqueRoots.length;
    const nodeToNew = roots.map(r => rootToNew.get(r)!);

    // New positions: average of class members
    const newPos = Array.from({ length: newN }, () => [0, 0]);
    const cnt = new Array<number>(newN).fill(0);
    for (let i = 0; i < N; i++) {
        const ni = nodeToNew[i];
        newPos[ni][0] += bc.pos[i][0];
        newPos[ni][1] += bc.pos[i][1];
        cnt[ni]++;
    }
    for (let ni = 0; ni < newN; ni++) {
        newPos[ni][0] /= cnt[ni];
        newPos[ni][1] /= cnt[ni];
    }

    // New adjacency: adjacent if any pair across the two classes was adjacent
    const newAdj = zeroAdj(newN);
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            if (!bc.adj[i][j]) continue;
            const ni = nodeToNew[i], nj = nodeToNew[j];
            if (ni !== nj) newAdj[ni][nj] = 1;
        }
    }
    return make(newPos, newAdj);
}

/** A rectangular board with width `w` and height `h`. Each node is identified by (col, row) where 0 ≤ col < w, 0 ≤ row < h. */
export function rectangularBoard(w: number, h: number): BoardConfig {
    assert(w > 0 && h > 0, `w and h must be positive, got w=${w} h=${h}`);
    const pos: number[][] = [];
    for (let r = 0; r < h; r++)
        for (let c = 0; c < w; c++)
            pos.push([c - (w - 1) / 2, r - (h - 1) / 2]);
    const adj = zeroAdj(w * h);
    for (let r = 0; r < h; r++)
        for (let c = 0; c < w; c++)
            for (const [dr, dc] of [[0,1],[1,0],[0,-1],[-1,0]]) {
                const nr = r + dr, nc = c + dc;
                if (nr < 0 || nr >= h || nc < 0 || nc >= w) continue;
                adj[r*w+c][nr*w+nc] = 1;
            }
    return make(pos, adj);
}

/**
 * Traditional Go board star points ("hoshi") for a rectangular board,
 * derived from `config.boardType`/`boardArgs` - [] for any non-'rect'
 * board. Corner points sit at the 3-3 point (boards whose smaller edge is
 * 9 or 11) or the 4-4 point (smaller edge > 11), edge points sit at the
 * midpoint of an odd, >=19-length edge whose cross edge is >=9, and a
 * single center point appears when both edges are odd and >=5 - together
 * reproducing the real 9x9 (4 corners + center), 13x13 (4 corners +
 * center), and 19x19 (4 corner + 4 edge + center) star-point layouts.
 * Returned as [x, y] pairs in the same board-coordinate space as
 * BoardConfig.pos (see rectangularBoard() above), ready for the same
 * originX + x*cell / originY - y*cell screen transform.
 */
export function computeStarPoints(config: GameConfig): number[][] {
    if (config.boardType !== 'rect') return [];
    const [w, h] = config.boardArgs;
    const toBoard = (c: number, r: number): number[] => [c - (w - 1) / 2, r - (h - 1) / 2];
    const points: number[][] = [];
    // Same inset (distance from an edge) for every star point, corner or edge-midpoint alike -
    // keeps e.g. a 19x9 board's edge-midpoint points on the same row as its corner points instead
    // of assuming the 19x19 board's 4-4 line regardless of h.
    const inset = Math.min(w, h) <= 11 ? 3 : 4;

    if (w >= 9 && h >= 9) {
        for (const c of [inset - 1, w - inset])
            for (const r of [inset - 1, h - inset])
                points.push(toBoard(c, r));
    }
    if (w % 2 === 1 && w >= 19 && h >= 9) {
        const c = (w - 1) / 2;
        points.push(toBoard(c, inset - 1), toBoard(c, h - inset));
    }
    if (h % 2 === 1 && h >= 19 && w >= 9) {
        const r = (h - 1) / 2;
        points.push(toBoard(inset - 1, r), toBoard(w - inset, r));
    }
    if (w % 2 === 1 && h % 2 === 1 && w >= 5 && h >= 5)
        points.push(toBoard((w - 1) / 2, (h - 1) / 2));

    return points;
}

/** A rectangular board with width `w` and height `h` where diagonally adjacent nodes are also connected, but only at every `m`-th square. */
export function rectangularDiagonalBoard(w: number, h: number, m: number): BoardConfig {
    assert(w > 0 && h > 0 && m > 0, `w, h, and m must be positive, got w=${w} h=${h} m=${m}`);
    const pos: number[][] = [];
    for (let r = 0; r < h; r++)
        for (let c = 0; c < w; c++)
            pos.push([c - (w - 1) / 2, r - (h - 1) / 2]);
    const adj = zeroAdj(w * h);
    const dirs: [number, number][] = [[0,1],[1,0],[0,-1],[-1,0],[1,1],[-1,1]];
    for (let r = 0; r < h; r++) {
        for (let c = 0; c < w; c++) {
            for (const [dr, dc] of dirs) {
                const nr = r + dr, nc = c + dc;
                if (nr < 0 || nr >= h || nc < 0 || nc >= w) continue;
                if (Math.abs(dr) === 1 && Math.abs(dc) === 1 && m > 1 &&
                  ((r + Math.floor((dr - 1) / 2)) % m !== 0 || c % m !== 0)) continue;
                adj[r*w+c][nr*w+nc] = 1;
                adj[nr*w+nc][r*w+c] = 1;
            }
        }
    }
    return make(pos, adj);
}

/** A cubical board with width `w`, height `h` and depth `d`. Each node is identified by (col, row, slice) where 0 ≤ col < w, 0 ≤ row < h, 0 ≤ slice < d. */
export function cubicalBoard(w: number, h: number, d: number): BoardConfig {
    assert(w > 0 && h > 0 && d > 0, `w, h, and d must be positive, got w=${w} h=${h} d=${d}`);
    const pos: number[][] = [];
    const scale = d > 1 ? 1 / 1.2 : 1;
    for (let s = 0; s < d; s++)
        for (let r = 0; r < h; r++)
            for (let c = 0; c < w; c++)
                pos.push([
                    (d * (c - (w-1)/2) + s * 0.8) * scale,
                    (d * (r - (h-1)/2) + s * 0.8) * scale,
                ]);
    const N = w * h * d;
    const adj = zeroAdj(N);
    const idx = (r: number, c: number, s: number) => s * h * w + r * w + c;
    for (let s = 0; s < d; s++)
        for (let r = 0; r < h; r++)
            for (let c = 0; c < w; c++)
                for (const [dr, dc, ds] of [[0,1,0],[1,0,0],[0,-1,0],[-1,0,0],[0,0,1],[0,0,-1]]) {
                    const nr = r+dr, nc = c+dc, ns = s+ds;
                    if (nr<0||nr>=h||nc<0||nc>=w||ns<0||ns>=d) continue;
                    adj[idx(r,c,s)][idx(nr,nc,ns)] = 1;
                }
    return make(pos, adj);
}

/** A hypercubical board with width `w`, height `h`, depth `d` and hyperdepth `t`. Each node is identified by (col, row, slice, hyperslice) where 0 ≤ col < w, 0 ≤ row < h, 0 ≤ slice < d, 0 ≤ hyperslice < t. */
export function hypercubeBoard(w: number, h: number, d: number, t: number): BoardConfig {
    assert(w > 0 && h > 0 && d > 0 && t > 0, `w, h, d, and t must be positive, got w=${w} h=${h} d=${d} t=${t}`);
    const pos: number[][] = [];
    for (let s = 0; s < t; s++)
        for (let u = 0; u < d; u++)
            for (let r = 0; r < h; r++)
                for (let c = 0; c < w; c++)
                    pos.push([
                        c - (w-1)/2 + u * (w+1) - (d-1)*(w+1)/2,
                        r - (h-1)/2 + s * (h+1) - (t-1)*(h+1)/2,
                    ]);
    const N = w * h * d * t;
    const adj = zeroAdj(N);
    const idx = (r: number, c: number, u: number, s: number) =>
        ((s * d + u) * h + r) * w + c;
    for (let s = 0; s < t; s++)
        for (let u = 0; u < d; u++)
            for (let r = 0; r < h; r++)
                for (let c = 0; c < w; c++)
                    for (const [dr,dc,du,ds] of [[0,1,0,0],[1,0,0,0],[0,-1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,-1,0],[0,0,0,1],[0,0,0,-1]]) {
                        const nr=r+dr, nc=c+dc, nu=u+du, ns=s+ds;
                        if (nr<0||nr>=h||nc<0||nc>=w||nu<0||nu>=d||ns<0||ns>=t) continue;
                        adj[idx(r,c,u,s)][idx(nr,nc,nu,ns)] = 1;
                    }
    return make(pos, adj);
}

/** A triangular board with side length `w`. */
export function triangularBoard(w: number): BoardConfig {
    assert(w > 0, `w must be positive, got w=${w}`);
    const rowDist = Math.sqrt(3) / 2;
    const pos: number[][] = [];
    for (let i = 0; i < w; i++) {
        for (let j = 0; j <= i; j++) {
            pos.push([j - i/2, rowDist * (i + 1 - w / 3)]);
        }
    }
    const N = w * (w + 1) / 2;
    const adj = zeroAdj(N);
    const idx = (i: number, j: number) => i * (i + 1) / 2 + j;
    for (let i = 0; i < w; i++)
        for (let j = 0; j <= i; j++)
            for (const [di, dj] of [[1,0],[1,1],[0,1],[-1,0],[-1,-1],[0,-1]]) {
                const ni = i+di, nj = j+dj;
                if (ni < 0 || ni >= w || nj < 0 || nj > ni) continue;
                adj[idx(i,j)][idx(ni,nj)] = 1;
            }
    return make(pos, adj);
}

/**
 * A triangular-lattice board arranged in a hexagon shape, with `d` layers of triangles surrounding
 * the central point (side length d+1, in hex terms) - the shape used by boards like Havannah/Y.
 * Not tiled by hexagons - see hexagonalBoard (TODO) for that. Cells are indexed by axial
 * coordinates (q, r) with max(|q|, |r|, |q+r|) <= d (hex distance from the center), laid out on the
 * same triangular lattice as `triangularBoard` (unit edge length, rowDist row spacing); each cell
 * connects to its up to six axial neighbors.
 */
export function triangularHexBoard(d: number): BoardConfig {
    assert(d >= 0, `d must be non-negative, got d=${d}`);
    const rowDist = Math.sqrt(3) / 2;
    const coords: [number, number][] = [];
    for (let q = -d; q <= d; q++)
        for (let r = Math.max(-d, -d - q); r <= Math.min(d, d - q); r++)
            coords.push([q, r]);
    const N = coords.length;
    const idx = new Map<string, number>();
    coords.forEach(([q, r], i) => idx.set(`${q},${r}`, i));
    const pos = coords.map(([q, r]) => [q + r / 2, rowDist * r]);
    const adj = zeroAdj(N);
    const dirs: [number, number][] = [[1,0],[1,-1],[0,-1],[-1,0],[-1,1],[0,1]];
    for (let i = 0; i < N; i++) {
        const [q, r] = coords[i];
        for (const [dq, dr] of dirs) {
            const ni = idx.get(`${q+dq},${r+dr}`);
            if (ni !== undefined) adj[i][ni] = 1;
        }
    }
    return make(pos, adj);
}

/**
 * A board actually tiled by regular hexagons: a central hexagonal cell surrounded by `d` further
 * layers of hexagonal cells (honeycomb topology - degree 3 in the interior, degree 2 on the
 * boundary). Built by carving the honeycomb lattice out of the same triangular lattice
 * `triangularHexBoard` uses, rather than laying out hexagons directly: a triangular lattice is
 * 3-colorable (color(q, r) = (q - r) mod 3) into three interlocking triangular sublattices, and the
 * points of any one color class are exactly the *centers* of the hexagonal faces formed by the
 * other two colors' points - so "erasing" one color class turns the triangular lattice into a
 * honeycomb lattice, with the erased points marking where each hexagonal face used to be.
 * `centers` enumerates that color-0 sublattice - itself a triangular lattice, spanned by (q, r) =
 * a*(1,1) + b*(2,-1) (both color 0, at a 60° angle, so (a, b) is a fresh axial coordinate system
 * for it) - restricted to hex-distance <= d in (a, b), i.e. the center cell plus d surrounding
 * rings of cells. Each kept center then contributes its six triangular-lattice neighbors as that
 * hexagon's six corners; two corners are connected iff they're triangular-lattice-adjacent, which
 * reproduces exactly the honeycomb edges (every triangular-lattice edge between two non-color-0
 * points borders exactly two color-0 points, i.e. two hexagonal faces) without needing a separate
 * erase/dedupe pass over edges.
 */
export function hexBoard(d: number): BoardConfig {
    assert(d >= 0, `d must be non-negative, got d=${d}`);
    const rowDist = Math.sqrt(3) / 2;
    const dirs: [number, number][] = [[1,0],[1,-1],[0,-1],[-1,0],[-1,1],[0,1]];

    const centers: [number, number][] = [];
    for (let a = -d; a <= d; a++)
        for (let b = Math.max(-d, -d - a); b <= Math.min(d, d - a); b++)
            centers.push([a + 2 * b, a - b]);

    const vertices = new Map<string, [number, number]>();
    for (const [q, r] of centers)
        for (const [dq, dr] of dirs)
            vertices.set(`${q+dq},${r+dr}`, [q + dq, r + dr]);
    const coords = [...vertices.values()];
    const N = coords.length;
    const idx = new Map<string, number>();
    coords.forEach(([q, r], i) => idx.set(`${q},${r}`, i));
    const pos = coords.map(([q, r]) => [q + r / 2, rowDist * r]);
    const adj = zeroAdj(N);
    for (let i = 0; i < N; i++) {
        const [q, r] = coords[i];
        for (const [dq, dr] of dirs) {
            const ni = idx.get(`${q+dq},${r+dr}`);
            if (ni !== undefined) adj[i][ni] = 1;
        }
    }
    return make(pos, adj);
}

/**
 * A trihexagonal ("hexdel") tiling: hexagons and triangles alternate, 2 of each around every
 * vertex (degree 4 in the interior) - `d` layers of hexagons, connected by triangles, surrounding
 * a central hexagon. Built the same way as `hexBoard` - as the triangular lattice with certain
 * nodes removed - but with a coarser removed sublattice: instead of `hexBoard`'s 1-of-3 coloring
 * (which erases every triangular face along with 2/3 of the nodes, leaving pure honeycomb), this
 * removes the 1-of-4 sublattice where both axial coordinates are even (`centers`, spanned by
 * (2, 0)/(0, 2) - double the original lattice spacing - restricted to hex-distance <= d in its own
 * halved (a, b) = (q/2, r/2) coordinates). Erasing only 1/4 of the nodes leaves each surviving node
 * with 4 of its 6 original neighbors (2 got erased), which works out to exactly 2 hexagon-bordering
 * and 2 triangle-bordering edges per node - so, just like `hexBoard`, simply connecting
 * triangular-lattice-adjacent survivors reproduces both the hexagons (surrounding each erased node)
 * and the triangles (the untouched elementary triangles of the original lattice) with no separate
 * pass needed for either.
 */
export function trihexBoard(d: number): BoardConfig {
    assert(d >= 0, `d must be non-negative, got d=${d}`);
    const rowDist = Math.sqrt(3) / 2;
    const dirs: [number, number][] = [[1,0],[1,-1],[0,-1],[-1,0],[-1,1],[0,1]];

    const centers: [number, number][] = [];
    for (let a = -d; a <= d; a++)
        for (let b = Math.max(-d, -d - a); b <= Math.min(d, d - a); b++)
            centers.push([2 * a, 2 * b]);

    const vertices = new Map<string, [number, number]>();
    for (const [q, r] of centers)
        for (const [dq, dr] of dirs)
            vertices.set(`${q+dq},${r+dr}`, [q + dq, r + dr]);
    const coords = [...vertices.values()];
    const N = coords.length;
    const idx = new Map<string, number>();
    coords.forEach(([q, r], i) => idx.set(`${q},${r}`, i));
    const pos = coords.map(([q, r]) => [q + r / 2, rowDist * r]);
    const adj = zeroAdj(N);
    for (let i = 0; i < N; i++) {
        const [q, r] = coords[i];
        for (const [dq, dr] of dirs) {
            const ni = idx.get(`${q+dq},${r+dr}`);
            if (ni !== undefined) adj[i][ni] = 1;
        }
    }
    return make(pos, adj);
}

/** Auxiliary function for `twistedSquareBoard` and `glueTwistedSquareBoard`. Not used by the renderer directly. */
function tiltedDisconnectedSquareBoard(w: number, h: number, g: number, gap: number) {
    const rm = Math.SQRT2 / 2;
    const sqWidth = (g - 1) * Math.SQRT2 + gap;
    const pos: number[][] = [];
    for (let rb = 0; rb < h; rb++)
        for (let cb = 0; cb < w; cb++) {
            const bx = (cb - (w-1)/2) * sqWidth;
            const by = (rb - (h-1)/2) * sqWidth;
            for (let r = 0; r < g; r++)
                for (let c = 0; c < g; c++) {
                    const lx = c - (g-1)/2, ly = r - (g-1)/2;
                    pos.push([bx + rm*lx - rm*ly, by + rm*lx + rm*ly]);
                }
        }
    const N = w * h * g * g;
    const adj = zeroAdj(N);
    const bIdx = (rb: number, cb: number) => (rb * w + cb) * g * g;
    for (let rb = 0; rb < h; rb++)
        for (let cb = 0; cb < w; cb++) {
            const b = bIdx(rb, cb);
            // Edges within the squares
        for (let r = 0; r < g; r++)
                for (let c = 0; c < g; c++)
                    for (const [dr, dc] of [[0,1],[1,0],[0,-1],[-1,0]]) {
                        const nr = r+dr, nc = c+dc;
                        if (nr<0||nr>=g||nc<0||nc>=g) continue;
                        adj[b+r*g+c][b+nr*g+nc] = 1;
                    }
        }
    // Connections between the squares
    const interConn: [number, number][] = [];
    for (let rb = 0; rb < h; rb++)
        for (let cb = 0; cb < w; cb++) {
            const b = bIdx(rb, cb);
            for (const [dr, dc] of [[0,1],[1,0],[0,-1],[-1,0]]) {
                const nrb = rb+dr, ncb = cb+dc;
                if (nrb<0||nrb>=h||ncb<0||ncb>=w) continue;
                const nb = bIdx(nrb, ncb);
                const selfIdx  = ((dr - dc + 1) >> 1) * g * (g-1) + ((dr + dc + 1) >> 1) * (g-1);
                const otherIdx = g*g - 1 - selfIdx;
                interConn.push([b + selfIdx, nb + otherIdx]);
            }
        }
    return { pos, adj, interConn, N };
}

/**
 * A board of `w × h` squares each rotated 45°, arranged in a rectangle. The squares have
 * the usual square topology. The closest nodes of two adjacent squares are glued together.
 */
export function glueTwistedSquareBoard(w: number, h: number, g: number): BoardConfig {
    assert(w > 0 && h > 0 && g > 0, `w, h, and g must be positive, got w=${w} h=${h} g=${g}`);
    const { pos, adj, interConn } = tiltedDisconnectedSquareBoard(w, h, g, 0.0);
    const bc = make(pos, adj);
    return quotientBoard(bc, interConn);
}

/**
 * A board of `w × h` squares each rotated 45°, arranged in a rectangle. The squares have
 * the usual square topology. The closest nodes of two adjacent squares are connected.
 */
export function twistedSquareBoard(w: number, h: number, g: number): BoardConfig {
    assert(w > 0 && h > 0 && g > 0, `w, h, and g must be positive, got w=${w} h=${h} g=${g}`);
    const { pos, adj, interConn } = tiltedDisconnectedSquareBoard(w, h, g, 1.0);
    for (const [i, j] of interConn) { adj[i][j] = 1; adj[j][i] = 1; }
    return make(pos, adj);
}

/**
 * A `w × h` grid of `g × g` squares, each rotated ±30° in a checkerboard pattern, arranged as a
 * snub square tiling. Adjacent squares are glued or triangle-connected at their nearest corners.
 */
export function snubSquareBoard(w: number, h: number, g: number): BoardConfig {
    assert(w > 0 && h > 0 && g > 0, `w, h, and g must be positive, got w=${w} h=${h} g=${g}`);

    const spacing = (g - 1) * (0.5 + Math.sqrt(3) / 2);
    const pos: number[][] = [];
    for (let rb = 0; rb < h; rb++)
        for (let cb = 0; cb < w; cb++) {
            const bx = (cb - (w-1)/2) * spacing;
            const by = (rb - (h-1)/2) * spacing;
            const angle = ((rb + cb) % 2 === 0 ? -1 : 1) * Math.PI / 6;
            const ca = Math.cos(angle), sa = Math.sin(angle);
            for (let r = 0; r < g; r++)
                for (let c = 0; c < g; c++) {
                    const lx = c - (g-1)/2, ly = r - (g-1)/2;
                    pos.push([bx + ca*lx - sa*ly, by + sa*lx + ca*ly]);
                }
        }
    const N = w * h * g * g;
    const adj = zeroAdj(N);
    const bIdx = (rb: number, cb: number) => (rb * w + cb) * g * g;
    const cornerRC: Record<'NW'|'NE'|'SW'|'SE', [number, number]> =
        { NW: [0,0], NE: [0,g-1], SW: [g-1,0], SE: [g-1,g-1] };
    const corner = (name: 'NW'|'NE'|'SW'|'SE'): [number, number] => cornerRC[name];

    // Edges within each big cell (ordinary rectangular grid)
    for (let rb = 0; rb < h; rb++)
        for (let cb = 0; cb < w; cb++) {
            const b = bIdx(rb, cb);
            for (let r = 0; r < g; r++)
                for (let c = 0; c < g; c++)
                    for (const [dr, dc] of [[0,1],[1,0],[0,-1],[-1,0]]) {
                        const nr = r+dr, nc = c+dc;
                        if (nr<0||nr>=g||nc<0||nc>=g) continue;
                        adj[b+r*g+c][b+nr*g+nc] = 1;
                    }
        }

    // Inter-cell connections, keyed by self cell's checkerboard parity then by "dr,dc" (only
    // forward directions, so each neighboring pair of big cells is handled once): glue is a single
    // coincident corner pair (merged via quotientBoard below); tri is three equal-distance pairs
    // (wired as new edges into adj instead).
    type Corner = 'NW'|'NE'|'SW'|'SE';
    const CONN: Record<number, Record<string, { glue?: [Corner,Corner], tri?: [Corner,Corner] }>> = {
        0: {
            '0,1':  { glue: ['SE','SW'], tri: ['NE', 'NW'] },
            '1,0':  { glue: ['SW','NW'], tri: ['SE', 'NE'] },
        },
        1: {
            '0,1':  { glue: ['NE','NW'], tri: ['SE', 'SW'] },
            '1,0':  { glue: ['SE','NE'], tri: ['SW', 'NW'] },
        },
    };
    const interConn: [number, number][] = [];
    for (let rb = 0; rb < h; rb++)
        for (let cb = 0; cb < w; cb++) {
            const b = bIdx(rb, cb);
            const entry = CONN[(rb + cb) % 2];
            for (const [dr, dc] of [[0,1],[1,0]] as [number, number][]) {
                const nrb = rb+dr, ncb = cb+dc;
                if (nrb<0||nrb>=h||ncb<0||ncb>=w) continue;
                const nb = bIdx(nrb, ncb);
                const conn = entry[`${dr},${dc}`];
                if (conn.glue) {
                    const [sr,sc] = corner(conn.glue[0]), [or_,oc] = corner(conn.glue[1]);
                    interConn.push([b + sr*g + sc, nb + or_*g + oc]);
                }
                if (conn.tri) {
                    const [sr,sc] = corner(conn.tri[0]), [or_,oc] = corner(conn.tri[1]);
                    const i = b + sr*g + sc, j = nb + or_*g + oc;
                    adj[i][j] = 1; adj[j][i] = 1;
                }
            }
        }

    const bc = make(pos, adj);
    return quotientBoard(bc, interConn);
}

/**
 * Triangle-inflated variant of snubSquareBoard: same w×h grid of g×g squares (rotated ±30° in the
 * same checkerboard pattern), but every square-to-square gap is filled by an actual side-length-g
 * triangular sub-board (same construction as triangularBoard(g)) instead of a single glued corner
 * plus a bare edge - squares only ever touch via triangles, and every connection is a whole-edge glue.
 */
export function snubSquareTriBoard(w: number, h: number, g: number): BoardConfig {
    assert(w > 0 && h > 0 && g > 0, `w, h, and g must be positive, got w=${w} h=${h} g=${g}`);

    const spacing = (g - 1) * (0.5 + Math.sqrt(3) / 2);
    const nTri = g * (g + 1) / 2;
    const triIdx = (i: number, j: number) => i * (i + 1) / 2 + j;

    const sqIdx = (x: number, y: number) => (y * w + x) * g * g;
    const sqN = w * h * g * g;
    const hCount = (w - 1) * h;
    const vCount = w * (h - 1);
    const hBase = (x: number, y: number) => sqN + (y * (w - 1) + x) * nTri;
    const vBase = (x: number, y: number) => sqN + hCount * nTri + (y * w + x) * nTri;
    const N = sqN + (hCount + vCount) * nTri;

    // Positions: squares laid out exactly as in snubSquareBoard first, then h-triangles, then
    // v-triangles. Each triangle's (i,0)/(i,i) nodes are placed at the exact real position of the
    // square corner they glue to (already pushed above, read back via sqIdx - never
    // recomputed independently), and every other node (i,j), 0<j<i, is placed by linear
    // interpolation between them at fraction j/i: triangularBoard's own template has row i's nodes
    // collinear and evenly spaced ((i,0)=[-i/2,Y], (i,j)=[j-i/2,Y], (i,i)=[i/2,Y], same Y), and the
    // square-to-triangle glue is an exact rigid (distance-preserving) fit - the gap between a
    // triangle's (i,0)/(i,i) glue targets is exactly i, matching the template's own (i,0)-(i,i)
    // distance for every i, not just the endpoints - so this interpolation reproduces the triangle's
    // true real position exactly, unlike a naive unrotated placeholder.
    const pos: number[][] = [];
    for (let y = 0; y < h; y++)
        for (let x = 0; x < w; x++) {
            const bx = (x - (w-1)/2) * spacing, by = (y - (h-1)/2) * spacing;
            const angle = ((x + y) % 2 === 0 ? -1 : 1) * Math.PI / 6;
            const ca = Math.cos(angle), sa = Math.sin(angle);
            for (let r = 0; r < g; r++)
                for (let c = 0; c < g; c++) {
                    const lx = c - (g-1)/2, ly = r - (g-1)/2;
                    pos.push([bx + ca*lx - sa*ly, by + sa*lx + ca*ly]);
                }
        }
    for (let y = 0; y < h; y++)
        for (let x = 0; x < w - 1; x++) {
            const p = (x + y) % 2;
            for (let i = 0; i < g; i++) {
                const selfR = p === 0 ? g-1-i : i;
                const left = pos[sqIdx(x, y) + selfR*g + (g-1)];
                const right = pos[sqIdx(x+1, y) + selfR*g + 0];
                for (let j = 0; j <= i; j++) {
                    const t = i === 0 ? 0 : j / i;
                    pos.push([left[0] + (right[0]-left[0])*t, left[1] + (right[1]-left[1])*t]);
                }
            }
        }
    for (let y = 0; y < h - 1; y++)
        for (let x = 0; x < w; x++) {
            const p = (x + y) % 2;
            for (let i = 0; i < g; i++) {
                const selfC = p === 0 ? i : g-1-i;
                const left = pos[sqIdx(x, y) + (g-1)*g + selfC];
                const right = pos[sqIdx(x, y+1) + 0*g + selfC];
                for (let j = 0; j <= i; j++) {
                    const t = i === 0 ? 0 : j / i;
                    pos.push([left[0] + (right[0]-left[0])*t, left[1] + (right[1]-left[1])*t]);
                }
            }
        }

    const adj = zeroAdj(N);

    // Intra-square edges (ordinary rectangular grid; no direct square-to-square connections here).
    for (let y = 0; y < h; y++)
        for (let x = 0; x < w; x++) {
            const b = sqIdx(x, y);
            for (let r = 0; r < g; r++)
                for (let c = 0; c < g; c++)
                    for (const [dr, dc] of [[0,1],[1,0],[0,-1],[-1,0]]) {
                        const nr = r+dr, nc = c+dc;
                        if (nr<0||nr>=g||nc<0||nc>=g) continue;
                        adj[b+r*g+c][b+nr*g+nc] = 1;
                    }
        }

    // Intra-triangle edges (mirrors triangularBoard's own edge loop).
    const triDirs: [number, number][] = [[1,0],[1,1],[0,1],[-1,0],[-1,-1],[0,-1]];
    const addTriEdges = (base: number) => {
        for (let i = 0; i < g; i++)
            for (let j = 0; j <= i; j++)
                for (const [di, dj] of triDirs) {
                    const ni = i+di, nj = j+dj;
                    if (ni < 0 || ni >= g || nj < 0 || nj > ni) continue;
                    adj[base+triIdx(i,j)][base+triIdx(ni,nj)] = 1;
                }
    };
    for (let y = 0; y < h; y++)
        for (let x = 0; x < w - 1; x++) addTriEdges(hBase(x, y));
    for (let y = 0; y < h - 1; y++)
        for (let x = 0; x < w; x++) addTriEdges(vBase(x, y));

    // Gluing: every square-triangle and triangle-triangle edge, merged via a single quotientBoard call.
    const interConn: [number, number][] = [];
    for (let y = 0; y < h; y++)
        for (let x = 0; x < w - 1; x++) {
            const p = (x + y) % 2;
            const base = hBase(x, y);
            for (let i = 0; i < g; i++) {
                const selfR = p === 0 ? g-1-i : i;
                interConn.push([base + triIdx(i,0), sqIdx(x, y) + selfR*g + (g-1)]);
                interConn.push([base + triIdx(i,i), sqIdx(x+1, y) + selfR*g + 0]);
            }
            if (p === 1 && y + 1 <= h - 1) {
                const nbase = hBase(x, y+1);
                for (let j = 0; j < g; j++)
                    interConn.push([base + triIdx(g-1,j), nbase + triIdx(g-1,j)]);
            }
        }
    for (let y = 0; y < h - 1; y++)
        for (let x = 0; x < w; x++) {
            const p = (x + y) % 2;
            const base = vBase(x, y);
            for (let i = 0; i < g; i++) {
                const selfC = p === 0 ? i : g-1-i;
                interConn.push([base + triIdx(i,0), sqIdx(x, y) + (g-1)*g + selfC]);
                interConn.push([base + triIdx(i,i), sqIdx(x, y+1) + 0*g + selfC]);
            }
            if (p === 0 && x + 1 <= w - 1) {
                const nbase = vBase(x+1, y);
                for (let j = 0; j < g; j++)
                    interConn.push([base + triIdx(g-1,j), nbase + triIdx(g-1,j)]);
            }
        }

    const bc = make(pos, adj);
    return quotientBoard(bc, interConn);
}


export enum PrescribedBoard {
    rectangularBoard,
    rectangularDiagonalBoard,
    cubicalBoard,
    hypercubeBoard,
    triangularBoard,
    triangularHexBoard,
    hexBoard,
    trihexBoard,
    snubSquareBoard,
    snubSquareTriBoard,
    twistedSquareBoard,
    glueTwistedSquareBoard
}

export const PrescribedBoardMap: Record<PrescribedBoard, [number, string, string, string]> = {
    [PrescribedBoard.rectangularBoard]:         [2, "rect",  "&lt;w&gt; &lt;h&gt;",                         "Rectangular board"],
    [PrescribedBoard.rectangularDiagonalBoard]: [3, "rectd", "&lt;w&gt; &lt;h&gt; &lt;m&gt;",               "Rectangular + diagonal connections every m squares"],
    [PrescribedBoard.cubicalBoard]:             [3, "cub",   "&lt;w&gt; &lt;h&gt; &lt;d&gt;",               "Cubical board"],
    [PrescribedBoard.hypercubeBoard]:           [4, "hcub",  "&lt;w&gt; &lt;h&gt; &lt;d&gt; &lt;t&gt;",    "Hypercubical board"],
    [PrescribedBoard.triangularBoard]:          [1, "tri",   "&lt;w&gt;",                                    "Triangular board of side w"],
    [PrescribedBoard.triangularHexBoard]:       [1, "trihex", "&lt;d&gt;",                                   "Triangular-lattice board in a hexagon shape, with d layers of triangles around the center"],
    [PrescribedBoard.hexBoard]:                 [1, "hex",   "&lt;d&gt;",                                    "Hexagon-tiled board with d layers of hexagons around a center hexagon"],
    [PrescribedBoard.trihexBoard]:               [1, "hexdel", "&lt;d&gt;",                                   "Trihexagonal (hexdel) board, d layers of hexagons connected by triangles around a center hexagon"],
    [PrescribedBoard.snubSquareBoard]:          [3, "snubsq", "&lt;w&gt; &lt;h&gt; &lt;g&gt;",              "Snub square board (g\xD7g squares)"],
    [PrescribedBoard.snubSquareTriBoard]:       [3, "snubsqtri", "&lt;w&gt; &lt;h&gt; &lt;g&gt;",           "Snub square board with the connecting triangles as g\xD7g triangular boards too"],
    [PrescribedBoard.twistedSquareBoard]:       [3, "twsq",  "&lt;w&gt; &lt;h&gt; &lt;g&gt;",               "Twisted-square board (g\xD7g squares)"],
    [PrescribedBoard.glueTwistedSquareBoard]:   [3, "gtsq",  "&lt;w&gt; &lt;h&gt; &lt;g&gt;",               "Glued-twisted-square board (g\xD7g squares)"],
};

export const PrescribedBoardFns: Record<PrescribedBoard, (...args: number[]) => BoardConfig> = {
    [PrescribedBoard.rectangularBoard]:         (...a) => rectangularBoard(a[0], a[1]),
    [PrescribedBoard.rectangularDiagonalBoard]: (...a) => rectangularDiagonalBoard(a[0], a[1], a[2]),
    [PrescribedBoard.cubicalBoard]:             (...a) => cubicalBoard(a[0], a[1], a[2]),
    [PrescribedBoard.hypercubeBoard]:           (...a) => hypercubeBoard(a[0], a[1], a[2], a[3]),
    [PrescribedBoard.triangularBoard]:          (...a) => triangularBoard(a[0]),
    [PrescribedBoard.triangularHexBoard]:       (...a) => triangularHexBoard(a[0]),
    [PrescribedBoard.hexBoard]:                 (...a) => hexBoard(a[0]),
    [PrescribedBoard.trihexBoard]:               (...a) => trihexBoard(a[0]),
    [PrescribedBoard.snubSquareBoard]:          (...a) => snubSquareBoard(a[0], a[1], a[2]),
    [PrescribedBoard.snubSquareTriBoard]:       (...a) => snubSquareTriBoard(a[0], a[1], a[2]),
    [PrescribedBoard.twistedSquareBoard]:       (...a) => twistedSquareBoard(a[0], a[1], a[2]),
    [PrescribedBoard.glueTwistedSquareBoard]:   (...a) => glueTwistedSquareBoard(a[0], a[1], a[2]),
};