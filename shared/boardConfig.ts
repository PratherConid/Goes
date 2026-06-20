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
                if (dr === 1 && dc === 1 && m > 1 && (r % m !== 0 || c % m !== 0)) continue;
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
                // corner index formula matches Python's _tilted_disconnected_square_board
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


export enum PrescribedBoard {
    rectangularBoard,
    rectangularDiagonalBoard,
    cubicalBoard,
    hypercubeBoard,
    triangularBoard,
    twistedSquareBoard,
    glueTwistedSquareBoard
}

export const PrescribedBoardMap: Record<PrescribedBoard, [number, string, string, string]> = {
    [PrescribedBoard.rectangularBoard]:         [2, "rect",  "&lt;w&gt; &lt;h&gt;",                         "Rectangular board"],
    [PrescribedBoard.rectangularDiagonalBoard]: [3, "rectd", "&lt;w&gt; &lt;h&gt; &lt;m&gt;",               "Rectangular + diagonal connections every m squares"],
    [PrescribedBoard.cubicalBoard]:             [3, "cub",   "&lt;w&gt; &lt;h&gt; &lt;d&gt;",               "Cubical board"],
    [PrescribedBoard.hypercubeBoard]:           [4, "hcub",  "&lt;w&gt; &lt;h&gt; &lt;d&gt; &lt;t&gt;",    "Hypercubical board"],
    [PrescribedBoard.triangularBoard]:          [1, "tri",   "&lt;w&gt;",                                    "Triangular board of side w"],
    [PrescribedBoard.twistedSquareBoard]:       [3, "twsq",  "&lt;w&gt; &lt;h&gt; &lt;g&gt;",               "Twisted-square board (g\xD7g squares)"],
    [PrescribedBoard.glueTwistedSquareBoard]:   [3, "gtsq",  "&lt;w&gt; &lt;h&gt; &lt;g&gt;",               "Glued-twisted-square board (g\xD7g squares)"],
};

export const PrescribedBoardFns: Record<PrescribedBoard, (...args: number[]) => BoardConfig> = {
    [PrescribedBoard.rectangularBoard]:         (...a) => rectangularBoard(a[0], a[1]),
    [PrescribedBoard.rectangularDiagonalBoard]: (...a) => rectangularDiagonalBoard(a[0], a[1], a[2]),
    [PrescribedBoard.cubicalBoard]:             (...a) => cubicalBoard(a[0], a[1], a[2]),
    [PrescribedBoard.hypercubeBoard]:           (...a) => hypercubeBoard(a[0], a[1], a[2], a[3]),
    [PrescribedBoard.triangularBoard]:          (...a) => triangularBoard(a[0]),
    [PrescribedBoard.twistedSquareBoard]:       (...a) => twistedSquareBoard(a[0], a[1], a[2]),
    [PrescribedBoard.glueTwistedSquareBoard]:   (...a) => glueTwistedSquareBoard(a[0], a[1], a[2]),
};