// The FractalDescr/nodeEdgeMergeFlakeRec recursive core, and each shape's own static *FractalDescr()
// builder - split out of boardConfig.ts (see git history) as a self-contained unit distinct from
// that file's many one-off, non-recursive board constructors. The actual `BoardConfig`-returning
// functions (dodecahedronBoard/dodecahedronFlake/etc.) stay in boardConfig.ts, which calls
// buildFractal()/the `*FractalDescr()` functions exported below - keeping this file free of
// `BoardConfig`/`Embedding`/positioning concerns entirely (every value here is either a leaf-space
// position, used only for the one-time glue search, or a plain index/adjacency structure) - and,
// since its own dependencies (below) are both more general-purpose than boardConfig.ts itself, free
// of any dependency on boardConfig.ts at all.
import { assert } from './types.js';
import { zeroAdj, mergeBoards } from './topology.js';

/**
 * Exhaustively searches for the "edge glue" relationship a `*FractalDescr()` function needs (see
 * FractalDescr's own doc comment for what an entry means): for each base edge `(i, j)`, the unique
 * pair of other base edges - `(m1, m2)` within the copy at `i`, `(n1, n2)` within the copy at `j` -
 * whose `S_i`/`S_j`-transformed endpoints coincide two-for-two. There is no closed-form shortcut for
 * this relation in general (see nodeEdgeMergeFlakeRec()'s own doc comment for what `S_i` means) -
 * this is exactly the kind of search that originally found each shape's own relation by hand,
 * generalized to run (and be cached) once per shape instead of hand-transcribing its output.
 */
function computeFlakeGlue(
    verts: number[][], edges: [number, number][], r: number, c: number,
): [number, number, number, number, number, number][] {
    const dist = (a: number[], b: number[]) => Math.hypot(...a.map((x, k) => x - b[k]));
    const transform = (i: number, m: number) => verts[m].map((x, k) => r * x + c * verts[i][k]);

    const glue: [number, number, number, number, number, number][] = [];
    for (const [i, j] of edges) {
        const matches: [number, number, number, number][] = [];
        for (const [m1, m2] of edges)
            for (const [ma, mb] of [[m1, m2], [m2, m1]] as [number, number][])
                for (const [n1, n2] of edges)
                    for (const [na, nb] of [[n1, n2], [n2, n1]] as [number, number][])
                        if (dist(transform(i, ma), transform(j, na)) < 1e-9
                            && dist(transform(i, mb), transform(j, nb)) < 1e-9)
                            matches.push([ma, mb, na, nb]);

        // Dedupe representations that differ only by swapping (ma,mb)<->(na,nb) together.
        const seen = new Set<string>();
        const canon: [number, number, number, number][] = [];
        for (const [ma, mb, na, nb] of matches) {
            const key = `${ma},${mb},${na},${nb}`, mirrorKey = `${mb},${ma},${nb},${na}`;
            if (seen.has(key) || seen.has(mirrorKey)) continue;
            seen.add(key);
            canon.push([ma, mb, na, nb]);
        }
        assert(canon.length === 1,
            `computeFlakeGlue: expected exactly one glue relation for edge (${i},${j}), found ${canon.length}`);
        glue.push([i, j, ...canon[0]]);
    }
    return glue;
}

/**
 * Exhaustively searches for the "node glue" relationship a `*FractalDescr()` function needs (its own
 * non-multiple-of-4 case for regularPolygonFractalDescr() - see FractalDescr's own doc comment for
 * what an entry means): for each base edge `(i, j)`, the unique pair of vertices `(m, p)` - `m`
 * within the copy at `i`, `p` within the copy at `j` - whose `S_i`/`S_j`-transformed positions
 * coincide. Unlike computeFlakeGlue()'s own edge-to-edge (2-point) search, this is a single-point
 * search: node-merge copies share exactly one point per base edge, not a whole growing edge, so
 * there is no second point to match and no chain to track (see nodeEdgeMergeFlakeRec()'s own doc
 * comment).
 */
function computeNodeGlue(
    verts: number[][], edges: [number, number][], r: number, c: number,
): [number, number, number, number][] {
    const dist = (a: number[], b: number[]) => Math.hypot(...a.map((x, k) => x - b[k]));
    const transform = (i: number, m: number) => verts[m].map((x, k) => r * x + c * verts[i][k]);

    const glue: [number, number, number, number][] = [];
    for (const [i, j] of edges) {
        const matches: [number, number][] = [];
        for (let m = 0; m < verts.length; m++)
            for (let p = 0; p < verts.length; p++)
                if (dist(transform(i, m), transform(j, p)) < 1e-9) matches.push([m, p]);
        assert(matches.length === 1,
            `computeNodeGlue: expected exactly one node-glue relation for edge (${i},${j}), found ${matches.length}`);
        glue.push([i, j, ...matches[0]]);
    }
    return glue;
}

/**
 * One sub-copy of a "flake" fractal, attached at recursion level n>1 via the map `S(x) = scale*x +
 * shift*parentScale` (see nodeEdgeMergeFlakeRec()'s own doc comment for the full derivation of that
 * `parentScale` factor). `scale` plays the same role the old shared `r` coefficient used to (how much
 * this copy shrinks relative to its immediate parent), and `shift` the same role `c*leafPos[i]` used
 * to (how far it sits from the parent's own origin, here measured in units of the parent's own scale
 * rather than tied to any particular leaf vertex) - but both are now picked independently per
 * sub-copy instead of being forced to a single shared ratio, so sub-copies may differ in size and
 * position with no relationship required between them.
 */
interface SubDescr {
    scale: number;
    shift: number[];
}

/**
 * Static description of a "flake" fractal shape, independent of recursion order - everything
 * nodeEdgeMergeFlakeRec() needs to build any order, bundled so each exported `*Flake()` function only
 * has to build/cache one of these per shape (see buildFractal()) rather than threading every piece
 * through by hand.
 *
 * `leafPos`/`leafConn` are the base graph itself (the n=1 case): unit-edge leaf-vertex positions and
 * their connectivity. `subDescr` has one entry per sub-copy attached at n>1 (see SubDescr's own doc
 * comment) - the FIRST `leafPos.length` entries correspond 1:1 to leaf vertices, in order (`subDescr`
 * index `i` is "the copy attached near leaf vertex `i`"; `nodeEdgeMergeFlakeRec()`'s own `corners`
 * output is indexed this same way, only ever `leafPos.length` long). `subDescr` MAY have further
 * entries beyond that (e.g. regularPolygonFractalDescr()'s own optional central copy) - these are
 * purely auxiliary internal structure with no leaf vertex of their own, so they're never exposed as
 * one of this shape's own attachment points, only ever referenced as one side of an `edgeGlueMap`/
 * `nodeGlueMap` entry (below).
 *
 * `edgeGlueMap`/`nodeGlueMap` key every glued pair `(P, Q)` - a pair of `subDescr` indices, not
 * necessarily a `leafConn` edge (an auxiliary entry like a central copy has no `leafConn` edges of its
 * own at all, only glue-map entries) - to EITHER a whole, growing shared edge OR a single shared
 * point, never both: `edgeGlueMap`'s `[C, D, E, F]` means sub-copy `P`'s own leaf-vertex edge `{C, D}`
 * coincides with sub-copy `Q`'s own leaf-vertex edge `{E, F}`; `nodeGlueMap`'s `[m, p]` means sub-copy
 * `P`'s own leaf-vertex `m` coincides with sub-copy `Q`'s own leaf-vertex `p`. Both found by
 * exhaustive search (computeFlakeGlue()/computeNodeGlue()) over `leafPos`/`leafConn`, since there is
 * no general closed-form shortcut for either relation.
 *
 * `edgeLevelUpMap` keys a leaf-vertex pair `(A, B)` (NOT a `subDescr`-index pair like `edgeGlueMap`/
 * `nodeGlueMap` - always two valid indices into `leafPos`) to an ordered list of `(subIdx, a, b)`
 * triples: this shape's own chain for edge `(A, B)`, at ANY recursion order, is the concatenation -
 * in list order - of `subDescr[subIdx]`'s own chain for its OWN edge `(a, b)` (oriented to start at
 * `a`), one list entry at a time. This is pure output-chain plumbing (see nodeEdgeMergeFlakeRec()'s
 * own doc comment for what a chain is/is used for) - it says nothing about which pairs actually merge
 * (that's `edgeGlueMap`'s job) or produce an edge in `adj` (that's `leafConn`/the base case). Every
 * shape with a growing shared edge between ADJACENT sub-copies `P`/`Q` (dodeca/icosa/octahedron/
 * `regularPolygonFlake`'s own 4n-gon case) sets this via `growingEdgeLevelUpMap(edgeGlueMap)`: for
 * each `edgeGlueMap` key `(P, Q)` (which, for exactly these shapes, doubles as a valid leaf-vertex
 * pair, since `subDescr` index equals leaf-vertex index one-to-one), the concatenation
 * `[[P, P, Q], [Q, P, Q]]` - `subDescr[P]`'s own `(P, Q)`-chain then `subDescr[Q]`'s own `(P, Q)`-
 * chain - reproducing the shared-edge-doubles-per-level behavior this map replaced (see git history).
 * `centralPentagonFractalDescr()` is the one shape that needs a genuinely different map: its own
 * "corner `i` to corner `i+1`" chain exists (each entry a length-2^n concatenation, same shape as
 * `growingEdgeLevelUpMap`'s, just using the leaf-adjacent indices `i`, `i+1` in place of `P`, `Q`)
 * even though, unlike the growing-edge shapes above, it is never itself an `edgeGlueMap` key - it goes
 * entirely unused by the plain (non-central) pentagon flake, which merges its own adjacent copies by a
 * single node instead (see regularPolygonFractalDescr()'s own doc comment) - only becoming load-
 * bearing once a central copy is added, which glues to it via `edgeGlueMap`.
 *
 * `nodeLevelUpMap` keys a leaf-vertex index `vtx` (as a string - NOT a `subDescr` index like
 * `edgeGlueMap`/`nodeGlueMap`'s own keys) to a `[subIdx, subflakeNode]` pair: this shape's own corner
 * `vtx`, at recursion order `n > 1`, is `subDescr[subIdx]`'s own corner `subflakeNode` (recursively -
 * see nodeEdgeMergeFlakeRec()'s own doc comment for exactly where this is consumed). This is what lets
 * a shape's RECURSION-STEP topology (which `subDescr` slot attaches where, and via which of that
 * slot's own corners) be picked independently of its LEAF board's own topology (`leafPos`/`leafConn`)
 * - every shape here still sets it via identityNodeLevelUpMap() (leaf vertex `vtx` maps to `[vtx,
 * vtx]`: `subDescr` slot `vtx`'s own corner `vtx`), the "chase the same index every level" convention
 * this map replaced (see git history), but a future shape's own recursion step need not follow leaf
 * vertex `vtx`'s own numbering at all.
 *
 * `globalScale`: the overall board at recursion order `n` is built at scale `globalScale ** (n - 1)`
 * (see buildFractal()) - chosen so that leaf-level (deepest, order-1) copies always come out unit
 * edge length, however deep `n` goes.
 */
interface FractalDescr {
    leafPos: number[][];
    leafConn: [number, number][];
    subDescr: SubDescr[];
    edgeGlueMap: Map<string, [number, number, number, number]>;
    nodeGlueMap: Map<string, [number, number]>;
    edgeLevelUpMap: Map<string, [number, number, number][]>;
    nodeLevelUpMap: Map<string, [number, number]>;
    globalScale: number;
}

/**
 * Derives the standard `nodeLevelUpMap` every shape here uses: leaf vertex `vtx` maps to `[vtx,
 * vtx]` - `subDescr` slot `vtx`'s own corner `vtx` - see FractalDescr's own doc comment.
 */
function identityNodeLevelUpMap(leafPos: number[][]): Map<string, [number, number]> {
    return new Map(leafPos.map((_, vtx): [string, [number, number]] => [`${vtx}`, [vtx, vtx]]));
}

/**
 * Derives the standard `edgeLevelUpMap` for a shape whose growing shared edges are exactly its
 * `edgeGlueMap` entries between ADJACENT sub-copies (dodeca/icosa/octahedron/regularPolygonFlake's own
 * 4n-gon case) - see FractalDescr's own doc comment for the `[[P, P, Q], [Q, P, Q]]` derivation.
 */
function growingEdgeLevelUpMap(
    edgeGlueMap: Map<string, [number, number, number, number]>,
): Map<string, [number, number, number][]> {
    return new Map(
        [...edgeGlueMap.keys()].map(key => {
            const [P, Q] = key.split(',').map(Number);
            return [key, [[P, P, Q], [Q, P, Q]]];
        }),
    );
}

/**
 * Recursive core shared by every "flake" fractal board built this way: n=1 is the base graph itself
 * (`descr.leafPos`/`descr.leafConn`, unit edge length); n>1 recurses into one order-(n-1) copy per
 * `descr.subDescr` entry, positioned by the map `S_i(x) = subDescr[i].scale*x +
 * subDescr[i].shift*scale` - `S_i` takes THIS call's own leaf-space positions (`x` ranging over
 * `descr.leafPos`, the same fixed set at every recursion depth) and produces the positions of the
 * shrunk copy attached via slot `i`, e.g. `S_i(leafPos[j])` is where that copy's own version of leaf
 * vertex `j` ends up. No rotation, ever, at any slot or level.
 *
 * `scale`/`offset` are this call's own accumulated affine transform `T(x) = scale*x + offset`,
 * mapping `leafPos` into this call's own actual `pos` (`pos[j] = scale*leafPos[j] + offset` at the
 * base case). A child built from slot `i` gets `childScale = scale*subDescr[i].scale` and
 * `childOffset = offset + scale*subDescr[i].shift` (composing `S_i` after `T`, i.e. `T_child =
 * T_parent . S_i`) - note this is `offset + scale*shift`, not `T_parent(shift)`: that's exactly what
 * lets each slot's `scale`/`shift` be picked completely independently, with no relationship required
 * between them. (The older, less general version of this shared construction, before `scale`/`shift`
 * were split out into `SubDescr`, used a single `S_i(x) = r*x + c*corners[i]` for every slot, which
 * only composes this same simple way when `r + c = 1` for every slot - see git history for that
 * derivation. `scale`/`shift` sidestep needing any such shared constraint at all.)
 *
 * `descr.edgeGlueMap`/`descr.nodeGlueMap`/`descr.edgeLevelUpMap`/`descr.nodeLevelUpMap` are computed
 * once by the caller (per shape, cached in the `descr` itself) and threaded through unchanged rather
 * than recomputed on every recursive call - see FractalDescr's own doc comment for exactly what each
 * map's keys/values mean. `corners` (returned alongside `pos`/`adj`/`edgeChains` below) is built from
 * `nodeLevelUpMap` at `n > 1` (`nodeLevelUpMap.get('vtx') = [subIdx, subflakeNode]` means corner `vtx`
 * is `subs[subIdx]`'s own corner `subflakeNode`) and trivially (`pos.map((_, i) => i)`) at the `n = 1`
 * base case.
 *
 * Alongside `pos`/`adj`/`corners`, also returns `edgeChains`: for every `edgeLevelUpMap` key `(A, B)`
 * (node-merge edges never get an entry - there is no "chain", just one point), the ordered list of
 * node indices lying along that edge at THIS call's own recursion depth - length `2^n` (base case:
 * literally `[A, B]`, since an `edgeLevelUpMap` key is always a valid leaf-vertex pair - see
 * FractalDescr's own doc comment). This is pure bookkeeping data for an *outer* caller's own merge
 * step - concatenating node lists here never adds an edge between them; it says nothing about this
 * call's own `adj`.
 *
 * Two things consume it, mirroring the fact that a sub-flake's own edges incident to its own
 * attachment slot are exactly the ones an adjacent copy might need to glue against:
 *  - The structural merge, built as one `merges` list across every `edgeGlueMap`/`nodeGlueMap` entry
 *    and resolved by a single mergeBoards() call (see that function's own doc comment): unlike a
 *    simplex-style flake (sierpinskiRec, sharing one point per adjacent pair), an `edgeGlueMap` entry
 *    shares every node along a whole edge, so it pairs up `subs[P]`'s own chain for its own `(C, D)`
 *    with `subs[Q]`'s own chain for `(E, F)`, **position by position over the whole chain** (length
 *    `2^(n-1)`, since `subs[P]`/`subs[Q]` are themselves order-`(n-1)`) - not just their 2 endpoints.
 *    A `nodeGlueMap` entry instead contributes exactly one merge pair: `subs[P]`'s own corner `m` and
 *    `subs[Q]`'s own corner `p` - no chain, so no growth with `n`.
 *  - This call's own returned `chain(A, B)`, for an even-outer caller: built fresh from
 *    `edgeLevelUpMap.get('A,B')`'s own segment list, concatenating each `(subIdx, a, b)` segment's
 *    `subs[subIdx]`'s own chain for `(a, b)` (remapped into this call's own final node-index space,
 *    oriented to start at `a`) in list order - see FractalDescr's own doc comment for why this is
 *    usually (but not always, see `centralPentagonFractalDescr()`) the same `(P, Q)` pair twice over,
 *    matching "shared edge length doubles per level".
 */
interface SubFlakeResult {
    pos: number[][];
    adj: number[][];
    corners: number[];
    edgeChains: Map<string, number[]>;
}

function nodeEdgeMergeFlakeRec(
    n: number, scale: number, offset: number[], descr: FractalDescr,
): SubFlakeResult {
    const { leafPos, leafConn, subDescr, edgeGlueMap, nodeGlueMap, edgeLevelUpMap, nodeLevelUpMap } = descr;
    if (n === 1) {
        const pos = leafPos.map(p => p.map((v, d) => scale * v + offset[d]));
        const adj = zeroAdj(leafPos.length);
        for (const [a, b] of leafConn) { adj[a][b] = 1; adj[b][a] = 1; }
        const edgeChains = new Map<string, number[]>();
        for (const key of edgeLevelUpMap.keys()) {
            const [a, b] = key.split(',').map(Number);
            edgeChains.set(key, [a, b]);
        }
        return { pos, adj, corners: pos.map((_, i) => i), edgeChains };
    }

    const subs = subDescr.map(sd =>
        nodeEdgeMergeFlakeRec(n - 1, scale * sd.scale, offset.map((o, d) => o + scale * sd.shift[d]), descr));

    const merges: [[number, number], [number, number]][] = [];
    for (const [key, eg] of edgeGlueMap) {
        const [P, Q] = key.split(',').map(Number);
        const [selfC, selfD, otherE, otherF] = eg;
        const selfLo = Math.min(selfC, selfD), selfHi = Math.max(selfC, selfD);
        let chainSelf = subs[P].edgeChains.get(`${selfLo},${selfHi}`)!;
        if (selfC > selfD) chainSelf = [...chainSelf].reverse(); // orient to start at selfC
        const otherLo = Math.min(otherE, otherF), otherHi = Math.max(otherE, otherF);
        let chainOther = subs[Q].edgeChains.get(`${otherLo},${otherHi}`)!;
        if (otherE > otherF) chainOther = [...chainOther].reverse(); // orient to start at otherE
        for (let i = 0; i < chainSelf.length; i++) merges.push([[P, chainSelf[i]], [Q, chainOther[i]]]);
    }
    for (const [key, ng] of nodeGlueMap) {
        const [P, Q] = key.split(',').map(Number);
        const [m, p] = ng;
        merges.push([[P, subs[P].corners[m]], [Q, subs[Q].corners[p]]]);
    }

    const combined = mergeBoards(subs, merges);
    // Only the first leafPos.length subDescr entries correspond to actual leaf-vertex attachment
    // points (see FractalDescr's own doc comment) - any further entries (e.g. an auxiliary central
    // copy) are purely internal structure, not exposed as one of this call's own `corners`.
    // nodeLevelUpMap decouples which subDescr slot/corner each leaf vertex chases (see FractalDescr's
    // own doc comment) from leaf vertex `vtx`'s own numbering - every shape here still chases slot
    // `vtx`'s own corner `vtx`, but not because that's hardcoded here.
    const cornersOut = leafPos.map((_, vtx) => {
        const [subIdx, subflakeNode] = nodeLevelUpMap.get(`${vtx}`)!;
        return combined.maps[subIdx][subs[subIdx].corners[subflakeNode]];
    });

    const edgeChains = new Map<string, number[]>();
    for (const [key, segments] of edgeLevelUpMap) {
        const chain: number[] = [];
        for (const [subIdx, a, b] of segments) {
            const lo = Math.min(a, b), hi = Math.max(a, b);
            let seg = subs[subIdx].edgeChains.get(`${lo},${hi}`)!;
            if (a > b) seg = [...seg].reverse(); // orient to start at a
            chain.push(...seg.map(idx => combined.maps[subIdx][idx]));
        }
        edgeChains.set(key, chain);
    }

    return { pos: combined.pos, adj: combined.adj, corners: cornersOut, edgeChains };
}

/** Builds a "flake" fractal board of recursion order `n` from its static `descr` - see FractalDescr's own doc comment. */
export function buildFractal(n: number, descr: FractalDescr): { pos: number[][]; adj: number[][] } {
    const offset = new Array(descr.leafPos[0].length).fill(0);
    const built = nodeEdgeMergeFlakeRec(n, descr.globalScale ** (n - 1), offset, descr);
    return { pos: built.pos, adj: built.adj };
}

let dodecahedronFractalDescrCache: FractalDescr | null = null;

/**
 * The static description dodecahedronBoard() and dodecahedronFlake() both build on: 20 unit-edge
 * regular-dodecahedron vertices (`leafPos`) and their 30 edges (`leafConn`). Vertices form 4 groups
 * of the classic "three mutually orthogonal golden rectangles" construction (`phi` = golden ratio): 8
 * cube corners `(sa, sb, sc)`, plus 4+4+4 more at `(0, sb/phi, sc*phi)`, `(sa/phi, sb*phi, 0)`,
 * `(sa*phi, 0, sc/phi)` - each coordinate independently `+-1`. At that raw scale, edge length is
 * `2/phi`; every coordinate below is pre-multiplied by `phi/2` so edges come out exactly 1.
 *
 * Connectivity (worked out by checking which vertex pairs land exactly 1 apart, then cross-checked
 * against the dodecahedron's known 30-edge, degree-3-per-vertex structure): each cube vertex
 * `(sa, sb, sc)` connects to exactly one vertex in each of the other 3 groups - `(0, sb/phi,
 * sc*phi)`, `(sa/phi, sb*phi, 0)`, `(sa*phi, 0, sc/phi)` - and each of those 12 non-cube vertices'
 * third edge (beyond its 2 cube-vertex edges) goes to its own sign-flipped partner within the same
 * group, e.g. `(0, sb/phi, sc*phi)` - `(0, -sb/phi, sc*phi)`.
 *
 * `edgeGlueMap` is found by computeFlakeGlue() over all 30x30 candidate base-edge pairs (verified
 * unique and consistent for all 30 base edges); `subDescr[i]` is `{ scale: r, shift: c*leafPos[i] }`
 * for every `i` - i.e. every sub-copy uses the exact same `r`, `c` here (unlike SubDescr's own doc
 * comment's general case), fixed by two requirements: (1) leaf-level (deepest) copies must come out
 * unit edge length; (2) each `edgeGlueMap` entry's two named edges must actually coincide once
 * transformed. Requirement (2) forces `c/r = phi^2` (verified numerically: the unique ratio producing
 * a consistent edge match across all 30 base edges - a plain per-vertex single-point join, by
 * contrast, only needs c=r, at any scale, which is a fundamentally different and simpler construction
 * that does NOT reproduce a shared edge - see this function's git history for that false start).
 * Requirement (1), combined with `r + c = 1` (see nodeEdgeMergeFlakeRec()'s own doc comment for why
 * every shape here satisfies that), then fixes `r = 1/(2+phi)` exactly - this also reproduces a
 * levels-grow-by-`(2+phi)` size relation for free (`globalScale = 1/r = 2+phi` - verified numerically
 * that circumradius(n+1)/circumradius(n) == 2+phi exactly, at both n=1->2 and n=2->3), cached since
 * this is all fixed, shape-level data that never changes.
 */
export function dodecahedronFractalDescr(): FractalDescr {
    if (dodecahedronFractalDescrCache) return dodecahedronFractalDescrCache;

    const phi = (1 + Math.sqrt(5)) / 2;
    const scale = phi / 2; // normalizes edge length (2/phi at the raw scale above) to exactly 1
    const s = (bit: number) => (bit === 0 ? 1 : -1); // 0/1 sign-bit -> +-1

    const xIdx = (sa: number, sb: number, sc: number) => sa * 4 + sb * 2 + sc;
    const yIdx = (sb: number, sc: number) => 8 + sb * 2 + sc;
    const zIdx = (sa: number, sb: number) => 12 + sa * 2 + sb;
    const wIdx = (sa: number, sc: number) => 16 + sa * 2 + sc;

    const leafPos: number[][] = new Array(20);
    for (let sa = 0; sa < 2; sa++)
        for (let sb = 0; sb < 2; sb++)
            for (let sc = 0; sc < 2; sc++)
                leafPos[xIdx(sa, sb, sc)] = [s(sa) * scale, s(sb) * scale, s(sc) * scale];
    for (let sb = 0; sb < 2; sb++)
        for (let sc = 0; sc < 2; sc++)
            leafPos[yIdx(sb, sc)] = [0, (s(sb) / phi) * scale, s(sc) * phi * scale];
    for (let sa = 0; sa < 2; sa++)
        for (let sb = 0; sb < 2; sb++)
            leafPos[zIdx(sa, sb)] = [(s(sa) / phi) * scale, s(sb) * phi * scale, 0];
    for (let sa = 0; sa < 2; sa++)
        for (let sc = 0; sc < 2; sc++)
            leafPos[wIdx(sa, sc)] = [s(sa) * phi * scale, 0, (s(sc) / phi) * scale];

    const leafConn: [number, number][] = [];
    for (let sa = 0; sa < 2; sa++)
        for (let sb = 0; sb < 2; sb++)
            for (let sc = 0; sc < 2; sc++) {
                const x = xIdx(sa, sb, sc);
                leafConn.push([x, yIdx(sb, sc)], [x, zIdx(sa, sb)], [x, wIdx(sa, sc)]);
            }
    for (let sc = 0; sc < 2; sc++) leafConn.push([yIdx(0, sc), yIdx(1, sc)]);
    for (let sb = 0; sb < 2; sb++) leafConn.push([zIdx(0, sb), zIdx(1, sb)]);
    for (let sa = 0; sa < 2; sa++) leafConn.push([wIdx(sa, 0), wIdx(sa, 1)]);

    const r = 1 / (2 + phi);
    const c = phi * phi * r;
    const glue = computeFlakeGlue(leafPos, leafConn, r, c);
    const edgeGlueMap = new Map<string, [number, number, number, number]>(
        glue.map(([ga, gb, gc, gd, ge, gf]) => [`${ga},${gb}`, [gc, gd, ge, gf]]),
    );
    const subDescr: SubDescr[] = leafPos.map(v => ({ scale: r, shift: v.map(x => x * c) }));

    dodecahedronFractalDescrCache = {
        leafPos, leafConn, subDescr, edgeGlueMap, nodeGlueMap: new Map(),
        edgeLevelUpMap: growingEdgeLevelUpMap(edgeGlueMap), nodeLevelUpMap: identityNodeLevelUpMap(leafPos),
        globalScale: 1 / r,
    };
    return dodecahedronFractalDescrCache;
}

let icosahedronFractalDescrCache: FractalDescr | null = null;

/**
 * The static description icosahedronBoard() and icosahedronFlake() both build on: 12 unit-edge
 * regular-icosahedron vertices (`leafPos`) and their 30 edges (`leafConn`). Vertices form 3 groups of
 * 4, each the set of cyclic-coordinate permutations of `(0, +-1, +-phi)` (`phi` = golden ratio)
 * sharing one fixed-zero axis: `A(sp, sq) = (0, sp, sq*phi)`, `B(sp, sq) = (sp, sq*phi, 0)`,
 * `C(sp, sq) = (sq*phi, 0, sp)`, each coordinate independently `+-1`. At that raw scale, edge length
 * is 2; every coordinate below is pre-multiplied by 1/2 so edges come out exactly 1.
 *
 * Connectivity (worked out by checking which vertex pairs land exactly the minimum distance apart,
 * then cross-checked against the icosahedron's known 30-edge, degree-5-per-vertex structure - this
 * one is easy to get backwards by hand, so every relation below was independently re-derived
 * algebraically, not just pattern-matched from a couple of examples): within each group, `(sp, sq)`
 * connects to its own sign-flipped-`sp` partner `(-sp, sq)`. Across groups, the three relations cycle
 * A -> B -> C -> A, each keyed off the *sending* group's own `sp`: `A(sp, sq)` connects to both
 * `B(+-1, sp)`; `B(sp, sq)` connects to both `C(+-1, sp)`; `C(sp, sq)` connects to both `A(+-1, sp)`.
 *
 * `edgeGlueMap` is found by computeFlakeGlue() over all 30x30 candidate base-edge pairs (verified
 * unique and consistent for all 30 base edges, and that no non-adjacent vertex pair has any
 * coincidence at all); `subDescr[i]` is `{ scale: r, shift: c*leafPos[i] }` for every `i`, fixed by
 * the same two requirements as dodecahedronFractalDescr() (see its own doc comment). Requirement (2)
 * forces `c/r = phi` here (verified numerically - the unique ratio producing a consistent edge match
 * across all 30 base edges). Requirement (1), with `r + c = 1`, then fixes `r = 1/(1+phi)` exactly -
 * `globalScale = 1/r = 1+phi` (also equal to `phi^2`, the classic 2D "pentaflake" inflation factor) -
 * verified numerically that circumradius(n+1)/circumradius(n) == 1+phi exactly, at both n=1->2 and
 * n=2->3.
 */
export function icosahedronFractalDescr(): FractalDescr {
    if (icosahedronFractalDescrCache) return icosahedronFractalDescrCache;

    const phi = (1 + Math.sqrt(5)) / 2;
    const scale = 0.5; // normalizes edge length (2 at the raw scale above) to exactly 1
    const s = (bit: number) => (bit === 0 ? 1 : -1); // 0/1 sign-bit -> +-1

    const aIdx = (sp: number, sq: number) => sp * 2 + sq;
    const bIdx = (sp: number, sq: number) => 4 + sp * 2 + sq;
    const cIdx = (sp: number, sq: number) => 8 + sp * 2 + sq;

    const leafPos: number[][] = new Array(12);
    for (let sp = 0; sp < 2; sp++)
        for (let sq = 0; sq < 2; sq++) {
            leafPos[aIdx(sp, sq)] = [0, s(sp) * scale, s(sq) * phi * scale];
            leafPos[bIdx(sp, sq)] = [s(sp) * scale, s(sq) * phi * scale, 0];
            leafPos[cIdx(sp, sq)] = [s(sq) * phi * scale, 0, s(sp) * scale];
        }

    const adj = zeroAdj(12);
    const connect = (i: number, j: number) => { adj[i][j] = 1; adj[j][i] = 1; };

    // Same-group edges: flip sp, keep sq.
    for (let sq = 0; sq < 2; sq++) {
        connect(aIdx(0, sq), aIdx(1, sq));
        connect(bIdx(0, sq), bIdx(1, sq));
        connect(cIdx(0, sq), cIdx(1, sq));
    }
    // Cross-group edges: A(sp,sq) ~ B(*,sp); B(sp,sq) ~ C(*,sp); C(sp,sq) ~ A(*,sp).
    for (let sp = 0; sp < 2; sp++)
        for (let sq = 0; sq < 2; sq++)
            for (let free = 0; free < 2; free++) {
                connect(aIdx(sp, sq), bIdx(free, sp));
                connect(bIdx(sp, sq), cIdx(free, sp));
                connect(cIdx(sp, sq), aIdx(free, sp));
            }
    const leafConn: [number, number][] = [];
    for (let i = 0; i < 12; i++)
        for (let j = i + 1; j < 12; j++)
            if (adj[i][j]) leafConn.push([i, j]);

    const r = 1 / (1 + phi);
    const c = phi * r;
    const glue = computeFlakeGlue(leafPos, leafConn, r, c);
    const edgeGlueMap = new Map<string, [number, number, number, number]>(
        glue.map(([ga, gb, gc, gd, ge, gf]) => [`${ga},${gb}`, [gc, gd, ge, gf]]),
    );
    const subDescr: SubDescr[] = leafPos.map(v => ({ scale: r, shift: v.map(x => x * c) }));

    icosahedronFractalDescrCache = {
        leafPos, leafConn, subDescr, edgeGlueMap, nodeGlueMap: new Map(),
        edgeLevelUpMap: growingEdgeLevelUpMap(edgeGlueMap), nodeLevelUpMap: identityNodeLevelUpMap(leafPos),
        globalScale: 1 / r,
    };
    return icosahedronFractalDescrCache;
}

let octahedronFractalDescrCache: FractalDescr | null = null;

/**
 * The static description octahedronFlake() builds on: the same 6 unit-edge regular-octahedron
 * vertices/12 edges as orthoplexBoard(3) (octahedronBoard()'s own construction, reproduced here
 * rather than reused so this function owns its own `leafPos`/`leafConn`) - vertex `2k`/`2k+1` are the
 * `+-scale` points on axis `k`, and are each other's antipode; every non-antipodal pair is an edge.
 *
 * `edgeGlueMap` is found by computeFlakeGlue() over all 12x12 candidate base-edge pairs;
 * `subDescr[i]` is `{ scale: r, shift: c*leafPos[i] }` for every `i`, fixed by the same two
 * requirements as dodecahedronFractalDescr() (see its own doc comment). Requirement (2) forces `c=r`
 * here (unlike dodeca/icosa's `c/r = phi^2`/`phi` - verified numerically: the unique ratio producing
 * a consistent edge match across all 12 base edges), and requirement (1), with `r + c = 1`, then
 * fixes `r = c = 1/2` exactly - a plain midpoint join, `S_i(x) = (x + leafPos[i]) / 2`. Unlike
 * dodeca/icosahedron, octahedron's own non-edges (the 3 antipodal pairs) are deliberately left out of
 * the search entirely - not because they don't coincide (they do: every copy's own
 * antipodal-attachment corner lands on the same shared center point, since `S_i(leafPos[antipode(i)])
 * = r*(-leafPos[i]) + c*leafPos[i] = (c-r)*leafPos[i] = 0` once `c = r`, for every `i`), but because
 * that coincidence needs no glue entry of its own: it already follows transitively from the 12
 * real-edge glue relations, since octahedron's real-edge graph (every vertex adjacent to all but its
 * own antipode) connects any two antipodal copies through a third common neighbor - and
 * nodeEdgeMergeFlakeRec()'s own mergeBoards() call resolves transitive coincidences like that
 * automatically (see mergeBoards()'s own doc comment), unlike the sequential per-step folding this
 * construction originally used and was rewritten away from for exactly this reason. `globalScale =
 * 1/r = 2` is this shape's own levels-grow-by-2 size relation (verified numerically that
 * circumradius(n+1)/circumradius(n) == 2 exactly, at both n=1->2 and n=2->3) - the same role
 * `2+phi`/`1+phi` play for dodeca/icosa.
 */
export function octahedronFractalDescr(): FractalDescr {
    if (octahedronFractalDescrCache) return octahedronFractalDescrCache;

    const edgeScale = 1 / Math.sqrt(2); // matches orthoplexBoard()'s own normalization to unit edge length
    const leafPos: number[][] = [];
    for (let k = 0; k < 3; k++) {
        const plus = [0, 0, 0], minus = [0, 0, 0];
        plus[k] = edgeScale;
        minus[k] = -edgeScale;
        leafPos.push(plus, minus);
    }
    const antipode = (i: number) => i % 2 === 0 ? i + 1 : i - 1;
    const leafConn: [number, number][] = [];
    for (let i = 0; i < 6; i++)
        for (let j = i + 1; j < 6; j++)
            if (j !== antipode(i)) leafConn.push([i, j]);

    const r = 0.5, c = 0.5;
    const glue = computeFlakeGlue(leafPos, leafConn, r, c);
    const edgeGlueMap = new Map<string, [number, number, number, number]>(
        glue.map(([ga, gb, gc, gd, ge, gf]) => [`${ga},${gb}`, [gc, gd, ge, gf]]),
    );
    const subDescr: SubDescr[] = leafPos.map(v => ({ scale: r, shift: v.map(x => x * c) }));

    octahedronFractalDescrCache = {
        leafPos, leafConn, subDescr, edgeGlueMap, nodeGlueMap: new Map(),
        edgeLevelUpMap: growingEdgeLevelUpMap(edgeGlueMap), nodeLevelUpMap: identityNodeLevelUpMap(leafPos),
        globalScale: 1 / r,
    };
    return octahedronFractalDescrCache;
}

/**
 * `r`/`c` for regularPolygonFractalDescr(nSides) - the per-sub-copy `scale`/`shift` it builds are
 * `r`/`c*leafPos[i]`, exactly as for dodeca/icosa/octahedron (see nodeEdgeMergeFlakeRec()'s own doc
 * comment for why every shape here satisfies `r + c = 1`, and regularPolygonFractalDescr()'s own doc
 * comment for exactly which vertex pair(s) requirement (2) below refers to).
 *
 * Fixed by the same two requirements as dodecahedronFractalDescr()/icosahedronFractalDescr()/
 * octahedronFractalDescr() (see dodecahedronFractalDescr()'s own doc comment): (1) leaf-level copies
 * must come out unit edge length, together with `r + c = 1`; (2) the relevant vertex pair(s) (see
 * regularPolygonFractalDescr()'s own doc comment) must actually coincide once transformed, which
 * pins `c/r`. For base edge `(0, 1)`, requirement (2)'s coincidence equation `r*(leafPos[a] -
 * leafPos[b]) = c*(leafPos[1] - leafPos[0])` - solved as complex numbers, `leafPos[j] = R*omega^j` -
 * always reduces to `c/r = sum_{j=-k}^{k} omega^j = 1 + 2*sum_{j=1}^{k} cos(2*pi*j/nSides)` for the
 * appropriate `k`: `k = nSides/4 - 1` when merging by edge (`a,b` = `1+nSides/4, -nSides/4` - see
 * regularPolygonFractalDescr()'s own doc comment), `k = floor(nSides/4)` when merging by node (`a,b`
 * = `1+floor(nSides/4), -floor(nSides/4)`) - verified numerically (both this ratio and the resulting
 * r,c) against an independent mergeClose()-based reference construction, for nSides=3..12 and
 * order=1..3.
 */
function regularPolygonFlakeRC(nSides: number): { r: number; c: number } {
    const isEdgeMerge = nSides % 4 === 0;
    const k = isEdgeMerge ? nSides / 4 - 1 : Math.floor(nSides / 4);
    let cosSum = 0;
    for (let j = 1; j <= k; j++) cosSum += Math.cos((2 * Math.PI * j) / nSides);
    const ratio = 1 + 2 * cosSum; // c/r
    const r = 1 / (1 + ratio), c = ratio / (1 + ratio); // r + c = 1
    return { r, c };
}

// Keyed by `${nSides},${center}` since the two `center` values produce genuinely different descrs.
const regularPolygonFractalDescrCache = new Map<string, FractalDescr>();

/**
 * The static description regularPolygonBoard() and regularPolygonFlake() both build on: nSides
 * unit-edge regular-nSides-gon vertices (`leafPos`, same construction as regularPolygonBoard()
 * itself) and their nSides cycle edges (`leafConn`). Cached per `(nSides, center)` (unlike dodeca/
 * icosa/octahedron's own single-shape caches) since regularPolygonFlake() isn't a fixed shape.
 *
 * For base edge `(i, i+1)`, every copy trivially shares the point `S_i(leafPos[i+1]) =
 * S_{i+1}(leafPos[i])` (true for ANY `r+c=1` construction, on any base graph - see
 * octahedronFractalDescr()'s own doc comment for the same observation) - that alone is a node merge.
 * It upgrades to a genuine edge merge only when a SECOND, independent point also coincides: copy i's
 * own vertex `1 + nSides/4` and copy (i+1)'s own vertex `-nSides/4` (i.e. `nSides - nSides/4`) -
 * which only exist as integer vertex indices when `nSides` is a multiple of 4 (there are exactly 2
 * edges of a regular nSides-gon perpendicular to any given edge exactly when 4 | nSides, and this
 * second point is the far endpoint of one of them - `edgeGlueMap`, found by computeFlakeGlue()). For
 * nSides not a multiple of 4, no such second point exists, so the base edge stays a plain node merge
 * - the nearest analogous point, copy i's own vertex `1 + floor(nSides/4)` and copy (i+1)'s own
 * vertex `-floor(nSides/4)` (`nodeGlueMap`, found by computeNodeGlue()), verified (alongside
 * `edgeGlueMap`'s own search) to be the actual coincidence computeFlakeGlue()/computeNodeGlue() finds,
 * for every nSides tested (3..12) - not merely asserted.
 *
 * If `center` is set and `nSides` is even and greater than 4, one further, auxiliary sub-copy is
 * added (see FractalDescr's own doc comment for what a `subDescr` entry beyond `leafPos.length`
 * means): a copy of the SAME shape sitting at the very center, glued to EVERY one of the nSides
 * regular copies at once, not just its neighbors. Its own vertex `i` is required to coincide with
 * copy `i`'s own vertex `i + nSides/2` (`i`'s antipode - well-defined since `nSides` is even) - by
 * symmetry the central copy can only be centered at the origin (`shift = 0`; a full `nSides`-fold
 * rotational symmetry rules out any other fixed point), so this reduces to one scalar unknown, its
 * own `scale`. Writing `leafPos[j] = R*omega^j` (omega = the nSides-th root of unity) and using
 * `leafPos[i+nSides/2] = -leafPos[i]` (nSides even): the coincidence condition `scale*leafPos[i] =
 * S_i(leafPos[i+nSides/2]) = r*(-leafPos[i]) + c*leafPos[i]` must hold for every `i` simultaneously,
 * which (since `leafPos[i]` sweeps every direction as `i` varies) pins `scale = c - r` exactly - the
 * `nSides > 4` restriction is exactly what keeps this positive: nSides=4 has `c = r` (see
 * regularPolygonFlakeRC()'s own doc comment - it's the edge-merge case with `k=0`, i.e. `c/r=1`),
 * which would degenerate the central copy to a single point. Verified numerically (this `scale`,
 * together with the resulting node/edge/degree counts) against an independent mergeClose()-based
 * reference construction, for nSides=6,8,10,12 and order=1..3.
 */
export function regularPolygonFractalDescr(nSides: number, center: boolean): FractalDescr {
    const cacheKey = `${nSides},${center}`;
    const cached = regularPolygonFractalDescrCache.get(cacheKey);
    if (cached) return cached;

    const R = 1 / (2 * Math.sin(Math.PI / nSides)); // matches regularPolygonBoard()'s own unit-edge scale
    const leafPos: number[][] = [];
    for (let k = 0; k < nSides; k++) {
        const theta = (2 * Math.PI * k) / nSides;
        leafPos.push([R * Math.cos(theta), R * Math.sin(theta)]);
    }
    const leafConn: [number, number][] = [];
    for (let k = 0; k < nSides; k++) {
        const a = k, b = (k + 1) % nSides;
        leafConn.push(a < b ? [a, b] : [b, a]);
    }

    const { r, c } = regularPolygonFlakeRC(nSides);
    const edgeGlueMap = new Map<string, [number, number, number, number]>();
    const nodeGlueMap = new Map<string, [number, number]>();
    if (nSides % 4 === 0) {
        const glue = computeFlakeGlue(leafPos, leafConn, r, c);
        for (const [ga, gb, gc, gd, ge, gf] of glue) edgeGlueMap.set(`${ga},${gb}`, [gc, gd, ge, gf]);
    } else {
        const glue = computeNodeGlue(leafPos, leafConn, r, c);
        for (const [ga, gb, gm, gp] of glue) nodeGlueMap.set(`${ga},${gb}`, [gm, gp]);
    }
    const subDescr: SubDescr[] = leafPos.map(v => ({ scale: r, shift: v.map(x => x * c) }));

    if (center && nSides % 2 === 0 && nSides > 4) {
        const centerIdx = subDescr.length;
        subDescr.push({ scale: c - r, shift: new Array(leafPos[0].length).fill(0) });
        for (let i = 0; i < nSides; i++)
            nodeGlueMap.set(`${i},${centerIdx}`, [(i + nSides / 2) % nSides, i]);
    }

    const descr: FractalDescr = {
        leafPos, leafConn, subDescr, edgeGlueMap, nodeGlueMap,
        edgeLevelUpMap: growingEdgeLevelUpMap(edgeGlueMap), nodeLevelUpMap: identityNodeLevelUpMap(leafPos),
        globalScale: 1 / r,
    };
    regularPolygonFractalDescrCache.set(cacheKey, descr);
    return descr;
}

let centralPentagonFractalDescrCache: FractalDescr | null = null;

/**
 * The static description centralPentagonFlake() builds on: a special case of
 * regularPolygonFractalDescr()'s own `center` feature that pentagon (`nSides=5`) can't use - pentagon
 * is odd, so a same-orientation central copy (that feature's own `scale = c - r`, always positive) has
 * no fixed point to sit at other than a degenerate one. Instead, pentagon's central copy sits at the
 * SAME scale magnitude but OPPOSITE orientation - `scale = -r`, `shift = 0` - a 180-degree rotation of
 * an ordinary sub-copy (found by brute-force distance search over `S_center(leafPos[i])` against every
 * sub-copy/vertex pair, not derived symbolically - pentagon's fifth-root-of-unity algebra doesn't
 * collapse as cleanly as the even-`nSides` case above, since `leafPos[i]` and `leafPos[(i+3)%5]` point
 * in genuinely different directions rather than being antipodal). That search found central copy
 * vertex `i` and vertex `i+1` coinciding with sub-copy `(i+3) % 5`'s own vertex `i` and vertex `i+1`
 * respectively - a whole shared EDGE, unlike regularPolygonFractalDescr()'s own central copy (always a
 * single shared node): `edgeGlueMap` gets one entry per `i`, keyed `((i+3)%5, centerIdx)`.
 *
 * That edge merge needs a growing chain for pentagon's own base edge `(i, i+1)` - normally never
 * computed, since a plain (non-central) pentagon flake merges adjacent copies by a single node (see
 * regularPolygonFractalDescr()'s own doc comment) - so `edgeLevelUpMap` gets an entry per base edge
 * too, same `[[i, i, i+1], [i+1, i, i+1]]` shape as `growingEdgeLevelUpMap()`'s (see FractalDescr's own
 * doc comment), just keyed by the leaf-adjacent pair itself rather than an `edgeGlueMap` key (pentagon
 * has no growing edge of its own between ADJACENT sub-copies - only between a sub-copy and the
 * center). Verified (this `edgeLevelUpMap` design, and the resulting node/edge/degree counts) against
 * an independent mergeClose()-based reference construction, order=1..5.
 */
export function centralPentagonFractalDescr(): FractalDescr {
    if (centralPentagonFractalDescrCache) return centralPentagonFractalDescrCache;

    const nSides = 5;
    const R = 1 / (2 * Math.sin(Math.PI / nSides)); // matches regularPolygonBoard()'s own unit-edge scale
    const leafPos: number[][] = [];
    for (let k = 0; k < nSides; k++) {
        const theta = (2 * Math.PI * k) / nSides;
        leafPos.push([R * Math.cos(theta), R * Math.sin(theta)]);
    }
    const leafConn: [number, number][] = [];
    for (let k = 0; k < nSides; k++) {
        const a = k, b = (k + 1) % nSides;
        leafConn.push(a < b ? [a, b] : [b, a]);
    }

    const { r, c } = regularPolygonFlakeRC(nSides);
    const nodeGlue = computeNodeGlue(leafPos, leafConn, r, c);
    const nodeGlueMap = new Map<string, [number, number]>();
    for (const [ga, gb, gm, gp] of nodeGlue) nodeGlueMap.set(`${ga},${gb}`, [gm, gp]);

    const subDescr: SubDescr[] = leafPos.map(v => ({ scale: r, shift: v.map(x => x * c) }));
    const centerIdx = subDescr.length;
    subDescr.push({ scale: -r, shift: new Array(leafPos[0].length).fill(0) });

    const edgeGlueMap = new Map<string, [number, number, number, number]>();
    const edgeLevelUpMap = new Map<string, [number, number, number][]>();
    for (let i = 0; i < nSides; i++) {
        const a = i, b = (i + 1) % nSides;
        edgeGlueMap.set(`${(i + 3) % nSides},${centerIdx}`, [a, b, b, a]);
        const key = `${Math.min(a, b)},${Math.max(a, b)}`;
        edgeLevelUpMap.set(key, [[a, a, b], [b, a, b]]);
    }

    centralPentagonFractalDescrCache = {
        leafPos, leafConn, subDescr, edgeGlueMap, nodeGlueMap, edgeLevelUpMap,
        nodeLevelUpMap: identityNodeLevelUpMap(leafPos), globalScale: 1 / r,
    };
    return centralPentagonFractalDescrCache;
}
