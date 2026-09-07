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
 * `GlueEntry`'s own doc comment for what an entry means): for each base edge `(i, j)`, the unique
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
 * non-multiple-of-4 case for regularPolygonFractalDescr() - see `GlueEntry`'s own doc comment for
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
 * A "glue object type" - the generalization of "a single node" or "a growing edge" to a shared
 * region with any number of reference points (1 for a node, 2 for an edge; nothing here uses more
 * than 2 yet, but nothing about the mechanism assumes that count is fixed or small). A glue object
 * is entirely COMBINATORIAL: it never touches position, scale, or orientation, only leaf-vertex
 * indices - which is exactly what lets the SAME object correctly handle a reflected or rotated
 * attachment (see `centralPentagonFractalDescr()`'s own central copy, `scale = -r`) with no
 * special-casing at all - see `step()`'s own doc comment below for the full argument.
 *
 * A glue object type is fully characterized by its own `step` function (below) - the generalization
 * of what earlier revisions of this design called `edgeLevelUpMap`/`nodeLevelUpMap`, written by hand,
 * per object, AD HOC: there is deliberately no shared derivation scaffold underneath it. (An earlier
 * attempt here built one around a fixed "reselect one of my own N named reference points" primitive -
 * it correctly handled a single point and a growing edge, since for both of those every sub-piece's
 * own attachment point HAPPENS to coincide with one of the object's own reference points, but that's
 * a coincidence of those two specific shapes, not a general fact: a Menger-sponge-style growing
 * FACE's own sub-squares are mostly edge-midpoint/interior pieces whose attachment point does NOT
 * coincide with any of the face's 4 corners at all, so no fixed reselection primitive can express it
 * - see git history for that attempt and why it doesn't generalize. A second attempt collapsed `step`
 * into a single `addresses(vertices, depth)` function that recursed across ALL depths internally -
 * also not what's specified below: the whole point of splitting `step` out on its own is that it only
 * ever describes ONE recursion level (level `i` to level `i+1`), with the repetition across however
 * many levels the outer shape actually needs left entirely to `glueObjectAddresses()`, generic shared
 * code that has nothing to do with any particular object - see git history for that attempt too.)
 *
 * `step(tags)` performs exactly one recursion step: given the CURRENT leaf-vertex tags realizing this
 * object's own reference points (see `glueObjectAddresses()`'s own doc comment for where these come
 * from and how they compose across levels), returns, for EACH of this object's own sub-pieces (a
 * fixed, ad hoc, per-object list - e.g. `EDGE_GLUE_OBJECT`'s own 2), which outer sub-slot to chase
 * into (`subSlot`) and that sub-piece's own tags one level deeper (`tags`) - each either literally one
 * of the CURRENT tags (an inherited reference point) or a brand-new leaf-vertex value introduced by
 * this object's own hard-coded, per-shape knowledge (a freshly-appearing one, e.g. a growing face's
 * own edge-midpoint - not needed by either object below, but exactly the freedom a future one would
 * use).
 *
 * Must be PURELY combinatorial - never look at position, scale, or orientation, only leaf-vertex
 * indices - which is exactly what lets the SAME function correctly handle a REFLECTED sub-copy
 * (`centralPentagonFractalDescr()`'s own central copy, `scale = -r`) with no orientation-awareness at
 * all: which sub-piece chases into which slot, and with which tags, is a property of the ABSTRACT
 * object, never of how that slot happens to be positioned/oriented in real space (see
 * `EDGE_GLUE_OBJECT`'s own doc comment for a worked explanation of why this specific function has
 * that property). Verified against the pre-generalization, hand-derived mechanism for every existing
 * shape - including the reflected-pentagon case - via an independent scratch reimplementation before
 * this file was updated; see git history.
 */
interface GlueObjectType {
    step(tags: number[]): { subSlot: number; tags: number[] }[];
}

/**
 * The trivial, 1-point glue object: no growth, ever - at every level, the object has exactly one
 * sub-piece, chasing into its own single tag with that SAME tag unchanged. This subsumes the
 * pre-generalization `nodeLevelUpMap`'s self-chase convention (leaf vertex `v` exposed as corner `v`
 * maps to `[v, v]`, i.e. exactly this object's own single step, instantiated with `tags = [v]`) and
 * `nodeGlueMap`'s own single-point merges alike - see `GlueEntry`'s own doc comment for this object's
 * use in full.
 */
const POINT_GLUE_OBJECT: GlueObjectType = {
    step(tags) {
        const [v] = tags;
        return [{ subSlot: v, tags: [v] }];
    },
};

/**
 * The 2-point "growing edge" glue object: each level, the edge splits into 2 sub-pieces - the first
 * tag's own copy of the SAME edge (both tags unchanged), then the second tag's own copy (also
 * unchanged). Subsumes the pre-generalization `growingEdgeLevelUpMap()`'s auto-derived
 * `[[P, P, Q], [Q, P, Q]]` pattern, generalized to work for ANY pair of leaf vertices (not just a
 * glue-map entry's own key) - which is exactly what lets `centralPentagonFractalDescr()`'s reflected
 * central copy reuse this SAME object with no special-casing: this function never distinguishes
 * "which side is canonical" - only "the first tag's own copy" vs "the second tag's own copy", a
 * distinction that survives a 180-degree rotation (a scalar `scale < 0`) untouched, unlike an earlier,
 * single-segment self-referential attempt this session tried before landing on this pattern, which
 * broke under exactly that reflection (see git history).
 */
const EDGE_GLUE_OBJECT: GlueObjectType = {
    step(tags) {
        const [c, d] = tags;
        return [{ subSlot: c, tags: [c, d] }, { subSlot: d, tags: [c, d] }];
    },
};

/**
 * Applies a glue object's own single-step `step()` function repeatedly - GENERIC, shared code, common
 * to every object, that has nothing to do with any particular object's own logic - to enumerate, in a
 * fixed, self-consistent order, every address (see `SubFlakeResult`'s own doc comment for what an
 * address is) realizing `object`'s own structure, instantiated with `tags` (this object's own current
 * reference points, given as leaf-vertex indices of the shape it's being glued within), at recursion
 * depth `depth`. The result is directly valid as a set of keys into an order-`depth` instance's own
 * `labels` map.
 *
 * Base case (`depth === 1`): tag `i` IS leaf vertex `tags[i]`, in order. Recursive case: call
 * `object.step(tags)` ONCE for this level, then concatenate, in that call's own return order, each
 * sub-piece's own full address list (one level shallower, instantiated with its own one-step-deeper
 * tags), each address prefixed by the sub-piece's own chased slot.
 */
function glueObjectAddresses(object: GlueObjectType, tags: number[], depth: number): string[] {
    if (depth === 1) return tags.map(String);
    const addrs: string[] = [];
    for (const { subSlot, tags: childTags } of object.step(tags)) {
        const childAddrs = glueObjectAddresses(object, childTags, depth - 1);
        addrs.push(...childAddrs.map(a => `${subSlot},${a}`));
    }
    return addrs;
}

/**
 * One entry of a `FractalDescr`'s own `glueMap`: sub-copy `P`'s own realization of `object`
 * (instantiated with `selfVertices`) coincides with sub-copy `Q`'s own realization (instantiated with
 * `otherVertices`), point-for-point IN ARRAY ORDER - `selfVertices[i]` glues to `otherVertices[i]`,
 * for every `i`. `otherVertices` is expected to already be arranged in whatever order realizes that
 * correspondence (e.g. reversed relative to `selfVertices`, for an edge glued "backwards" - see
 * `centralPentagonFractalDescr()` for a worked example: `otherVertices = [b, a]` where
 * `selfVertices = [a, b]`).
 *
 * Subsumes the pre-generalization `edgeGlueMap`'s `[C, D, E, F]` (`object = EDGE_GLUE_OBJECT`,
 * `selfVertices = [C, D]`, `otherVertices = [E, F]`) and `nodeGlueMap`'s `[m, p]`
 * (`object = POINT_GLUE_OBJECT`, `selfVertices = [m]`, `otherVertices = [p]`) as two instances of one
 * uniform mechanism - see `FractalDescr`'s own doc comment.
 */
interface GlueEntry {
    object: GlueObjectType;
    selfVertices: number[];
    otherVertices: number[];
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
 * index `i` is "the copy attached near leaf vertex `i`", by convention - every shape here follows it,
 * but nodeEdgeMergeFlakeRec() itself never assumes it; see its own doc comment). `subDescr` MAY have
 * further entries beyond that (e.g. regularPolygonFractalDescr()'s own optional central copy) - these
 * are purely auxiliary internal structure with no leaf vertex of their own, so they're never exposed
 * as one of this shape's own attachment points, only ever referenced as one side of a `glueMap` entry
 * (below).
 *
 * `glueMap` keys every glued pair `(P, Q)` - a pair of `subDescr` indices, not necessarily a
 * `leafConn` edge (an auxiliary entry like a central copy has no `leafConn` edges of its own at all,
 * only a `glueMap` entry) - to a `GlueEntry`: WHICH kind of region is shared (`GlueEntry.object`, e.g.
 * a single point or a whole growing edge) and WHICH of each sub-copy's own leaf vertices realize it
 * (`selfVertices`/`otherVertices`). This single map replaces what used to be four separate maps
 * (`edgeGlueMap`/`nodeGlueMap`/`edgeLevelUpMap`/`nodeLevelUpMap`) - see `GlueEntry`'s own doc comment
 * for how `POINT_GLUE_OBJECT` recovers `nodeGlueMap`'s old behavior and `EDGE_GLUE_OBJECT` recovers
 * `edgeGlueMap`'s, and `GlueObjectType`'s own doc comment for how the growing-chain/self-chase
 * machinery those relied on (`edgeLevelUpMap`/`nodeLevelUpMap`) is now just `GlueEntry.object`'s own
 * `addresses()` function, evaluated fresh whenever needed rather than cached per shape.
 * `selfVertices`/`otherVertices` are found by exhaustive search (`computeFlakeGlue()`/
 * `computeNodeGlue()`) over `leafPos`/`leafConn`, since there is no general closed-form shortcut for
 * either relation.
 *
 * `globalScale`: the overall board at recursion order `n` is built at scale `globalScale ** (n - 1)`
 * (see buildFractal()) - chosen so that leaf-level (deepest, order-1) copies always come out unit
 * edge length, however deep `n` goes.
 */
interface FractalDescr {
    leafPos: number[][];
    leafConn: [number, number][];
    subDescr: SubDescr[];
    glueMap: Map<string, GlueEntry>;
    globalScale: number;
}

/**
 * An ADDRESS identifies one node of an order-`n` `SubFlakeResult` by the exact chase-path used to
 * reach it: a string `"s1,s2,...,s(n-1),v"` (`si` = which `subDescr` slot was chosen at each of the
 * `n-1` recursive steps, `v` = the leaf vertex reached at the bottom; the `n=1` case is just `"v"`).
 * Distinct addresses can - and very often do - land on the SAME merged node (e.g. two leaf vertices
 * that get glued together transitively through several levels), which is exactly why
 * `SubFlakeResult.labels` is a MAP (address -> node index), not the other way around: an address
 * always denotes exactly one node, but a node may answer to many addresses.
 *
 * Mirrors nodeEdgeMergeFlakeRec()'s pre-generalization `corners`/`edgeChains` fields, both now fully
 * recoverable from `labels` alone via `glueObjectAddresses()` (a leaf-level corner `v`'s address, at
 * depth `n`, is `glueObjectAddresses(POINT_GLUE_OBJECT, [v], n)[0]`; a growing edge's own chain is
 * `glueObjectAddresses(EDGE_GLUE_OBJECT, [c, d], n)` in order) - so there is no need to track either
 * separately any more.
 */
interface SubFlakeResult {
    pos: number[][];
    adj: number[][];
    labels: Map<string, number>;
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
 * `descr.glueMap` is computed once by the caller (per shape, cached in the `descr` itself) and
 * threaded through unchanged rather than recomputed on every recursive call - see FractalDescr's own
 * doc comment for exactly what it means. `labels` (returned alongside `pos`/`adj` below) is built
 * directly at the `n = 1` base case (leaf vertex `v` -> address `"v"`) and, at `n > 1`, by prepending
 * each sub-copy's own slot index to ITS OWN `labels` (`subs[s]`'s address `a` becomes this call's own
 * address `${s},${a}`) before combining via mergeBoards() - see that function's own doc comment for
 * how `labels` (like `pos`/`adj`) gets carried through the same merge/remap.
 *
 * The structural merge itself is built as one `merges` list across every `glueMap` entry and resolved
 * by a single mergeBoards() call (see that function's own doc comment for why this - rather than
 * folding subs in one at a time - is what makes octahedronFractalDescr()'s own transitive antipodal-
 * corner coincidence come out correct): for entry `(P, Q) -> {object, selfVertices, otherVertices}`,
 * `glueObjectAddresses(object, selfVertices, n - 1)` and `glueObjectAddresses(object, otherVertices,
 * n - 1)` give two address lists of equal length, already in mutually-corresponding order (see `GlueEntry`'s own doc
 * comment) - looking each one up in `subs[P].labels`/`subs[Q].labels` (already-built order-`(n-1)`
 * instances) and zipping position-by-position gives every merge pair this entry contributes, in one
 * step - a single mechanism covering what used to be a `nodeGlueMap`/`edgeGlueMap` split with
 * separate chain-reversal logic for each.
 */
function nodeEdgeMergeFlakeRec(
    n: number, scale: number, offset: number[], descr: FractalDescr,
): SubFlakeResult {
    const { leafPos, leafConn, subDescr, glueMap } = descr;
    if (n === 1) {
        const pos = leafPos.map(p => p.map((v, d) => scale * v + offset[d]));
        const adj = zeroAdj(leafPos.length);
        for (const [a, b] of leafConn) { adj[a][b] = 1; adj[b][a] = 1; }
        const labels = new Map(leafPos.map((_, v): [string, number] => [String(v), v]));
        return { pos, adj, labels };
    }

    const subs = subDescr.map(sd =>
        nodeEdgeMergeFlakeRec(n - 1, scale * sd.scale, offset.map((o, d) => o + scale * sd.shift[d]), descr));

    const merges: [[number, number], [number, number]][] = [];
    for (const [key, entry] of glueMap) {
        const [P, Q] = key.split(',').map(Number);
        const selfAddrs = glueObjectAddresses(entry.object, entry.selfVertices, n - 1);
        const otherAddrs = glueObjectAddresses(entry.object, entry.otherVertices, n - 1);
        for (let i = 0; i < selfAddrs.length; i++)
            merges.push([[P, subs[P].labels.get(selfAddrs[i])!], [Q, subs[Q].labels.get(otherAddrs[i])!]]);
    }

    // Each sub's own addresses are relative to ITS OWN numbering - prepend its slot index so they
    // become valid addresses at THIS level, before combining (see this function's own doc comment).
    const boardsForMerge = subs.map((s, slot) => ({
        pos: s.pos, adj: s.adj,
        labels: new Map([...s.labels].map(([addr, idx]): [string, number] => [`${slot},${addr}`, idx])),
    }));
    const combined = mergeBoards(boardsForMerge, merges);

    return { pos: combined.pos, adj: combined.adj, labels: combined.labels };
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
 * `glueMap` is found by computeFlakeGlue() over all 30x30 candidate base-edge pairs (verified unique
 * and consistent for all 30 base edges) - every entry uses `EDGE_GLUE_OBJECT`; `subDescr[i]` is
 * `{ scale: r, shift: c*leafPos[i] }` for every `i` - i.e. every sub-copy uses the exact same `r`,
 * `c` here (unlike SubDescr's own doc comment's general case), fixed by two requirements: (1)
 * leaf-level (deepest) copies must come out unit edge length; (2) each glue-map entry's two named
 * edges must actually coincide once transformed. Requirement (2) forces `c/r = phi^2` (verified
 * numerically: the unique ratio producing a consistent edge match across all 30 base edges - a plain
 * per-vertex single-point join, by contrast, only needs c=r, at any scale, which is a fundamentally
 * different and simpler construction that does NOT reproduce a shared edge - see this function's git
 * history for that false start). Requirement (1), combined with `r + c = 1` (see
 * nodeEdgeMergeFlakeRec()'s own doc comment for why every shape here satisfies that), then fixes
 * `r = 1/(2+phi)` exactly - this also reproduces a levels-grow-by-`(2+phi)` size relation for free
 * (`globalScale = 1/r = 2+phi` - verified numerically that circumradius(n+1)/circumradius(n) ==
 * 2+phi exactly, at both n=1->2 and n=2->3), cached since this is all fixed, shape-level data that
 * never changes.
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
    const glueMap = new Map<string, GlueEntry>(
        glue.map(([ga, gb, gc, gd, ge, gf]): [string, GlueEntry] =>
            [`${ga},${gb}`, { object: EDGE_GLUE_OBJECT, selfVertices: [gc, gd], otherVertices: [ge, gf] }]),
    );
    const subDescr: SubDescr[] = leafPos.map(v => ({ scale: r, shift: v.map(x => x * c) }));

    dodecahedronFractalDescrCache = { leafPos, leafConn, subDescr, glueMap, globalScale: 1 / r };
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
 * `glueMap` is found by computeFlakeGlue() over all 30x30 candidate base-edge pairs (verified unique
 * and consistent for all 30 base edges, and that no non-adjacent vertex pair has any coincidence at
 * all) - every entry uses `EDGE_GLUE_OBJECT`; `subDescr[i]` is `{ scale: r, shift: c*leafPos[i] }`
 * for every `i`, fixed by the same two requirements as dodecahedronFractalDescr() (see its own doc
 * comment). Requirement (2) forces `c/r = phi` here (verified numerically - the unique ratio
 * producing a consistent edge match across all 30 base edges). Requirement (1), with `r + c = 1`,
 * then fixes `r = 1/(1+phi)` exactly - `globalScale = 1/r = 1+phi` (also equal to `phi^2`, the
 * classic 2D "pentaflake" inflation factor) - verified numerically that
 * circumradius(n+1)/circumradius(n) == 1+phi exactly, at both n=1->2 and n=2->3.
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
    const glueMap = new Map<string, GlueEntry>(
        glue.map(([ga, gb, gc, gd, ge, gf]): [string, GlueEntry] =>
            [`${ga},${gb}`, { object: EDGE_GLUE_OBJECT, selfVertices: [gc, gd], otherVertices: [ge, gf] }]),
    );
    const subDescr: SubDescr[] = leafPos.map(v => ({ scale: r, shift: v.map(x => x * c) }));

    icosahedronFractalDescrCache = { leafPos, leafConn, subDescr, glueMap, globalScale: 1 / r };
    return icosahedronFractalDescrCache;
}

let octahedronFractalDescrCache: FractalDescr | null = null;

/**
 * The static description octahedronFlake() builds on: the same 6 unit-edge regular-octahedron
 * vertices/12 edges as orthoplexBoard(3) (octahedronBoard()'s own construction, reproduced here
 * rather than reused so this function owns its own `leafPos`/`leafConn`) - vertex `2k`/`2k+1` are the
 * `+-scale` points on axis `k`, and are each other's antipode; every non-antipodal pair is an edge.
 *
 * `glueMap` is found by computeFlakeGlue() over all 12x12 candidate base-edge pairs, every entry
 * using `EDGE_GLUE_OBJECT`; `subDescr[i]` is `{ scale: r, shift: c*leafPos[i] }` for every `i`,
 * fixed by the same two requirements as dodecahedronFractalDescr() (see its own doc comment).
 * Requirement (2) forces `c=r` here (unlike dodeca/icosa's `c/r = phi^2`/`phi` - verified
 * numerically: the unique ratio producing a consistent edge match across all 12 base edges), and
 * requirement (1), with `r + c = 1`, then fixes `r = c = 1/2` exactly - a plain midpoint join,
 * `S_i(x) = (x + leafPos[i]) / 2`. Unlike dodeca/icosahedron, octahedron's own non-edges (the 3
 * antipodal pairs) are deliberately left out of the search entirely - not because they don't
 * coincide (they do: every copy's own antipodal-attachment corner lands on the same shared center
 * point, since `S_i(leafPos[antipode(i)]) = r*(-leafPos[i]) + c*leafPos[i] = (c-r)*leafPos[i] = 0`
 * once `c = r`, for every `i`), but because that coincidence needs no glue entry of its own: it
 * already follows transitively from the 12 real-edge glue relations, since octahedron's real-edge
 * graph (every vertex adjacent to all but its own antipode) connects any two antipodal copies
 * through a third common neighbor - and nodeEdgeMergeFlakeRec()'s own mergeBoards() call resolves
 * transitive coincidences like that automatically (see mergeBoards()'s own doc comment), unlike the
 * sequential per-step folding this construction originally used and was rewritten away from for
 * exactly this reason. `globalScale = 1/r = 2` is this shape's own levels-grow-by-2 size relation
 * (verified numerically that circumradius(n+1)/circumradius(n) == 2 exactly, at both n=1->2 and
 * n=2->3) - the same role `2+phi`/`1+phi` play for dodeca/icosa.
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
    const glueMap = new Map<string, GlueEntry>(
        glue.map(([ga, gb, gc, gd, ge, gf]): [string, GlueEntry] =>
            [`${ga},${gb}`, { object: EDGE_GLUE_OBJECT, selfVertices: [gc, gd], otherVertices: [ge, gf] }]),
    );
    const subDescr: SubDescr[] = leafPos.map(v => ({ scale: r, shift: v.map(x => x * c) }));

    octahedronFractalDescrCache = { leafPos, leafConn, subDescr, glueMap, globalScale: 1 / r };
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
 * octahedronFractalDescr()'s own doc comment for the same observation) - that alone is a node merge
 * (`POINT_GLUE_OBJECT`). It upgrades to a genuine edge merge (`EDGE_GLUE_OBJECT`) only when a SECOND,
 * independent point also coincides: copy i's own vertex `1 + nSides/4` and copy (i+1)'s own vertex
 * `-nSides/4` (i.e. `nSides - nSides/4`) - which only exist as integer vertex indices when `nSides`
 * is a multiple of 4 (there are exactly 2 edges of a regular nSides-gon perpendicular to any given
 * edge exactly when 4 | nSides, and this second point is the far endpoint of one of them - found by
 * computeFlakeGlue()). For nSides not a multiple of 4, no such second point exists, so the base edge
 * stays a plain node merge - the nearest analogous point, copy i's own vertex `1 + floor(nSides/4)`
 * and copy (i+1)'s own vertex `-floor(nSides/4)` (found by computeNodeGlue()), verified (alongside
 * computeFlakeGlue()'s own search) to be the actual coincidence found, for every nSides tested
 * (3..12) - not merely asserted.
 *
 * If `center` is set and `nSides` is even and greater than 4, one further, auxiliary sub-copy is
 * added (see FractalDescr's own doc comment for what a `subDescr` entry beyond `leafPos.length`
 * means): a copy of the SAME shape sitting at the very center, glued to EVERY one of the nSides
 * regular copies at once (each its own `POINT_GLUE_OBJECT` entry), not just its neighbors. Its own
 * vertex `i` is required to coincide with copy `i`'s own vertex `i + nSides/2` (`i`'s antipode -
 * well-defined since `nSides` is even) - by symmetry the central copy can only be centered at the
 * origin (`shift = 0`; a full `nSides`-fold rotational symmetry rules out any other fixed point), so
 * this reduces to one scalar unknown, its own `scale`. Writing `leafPos[j] = R*omega^j` (omega = the
 * nSides-th root of unity) and using `leafPos[i+nSides/2] = -leafPos[i]` (nSides even): the
 * coincidence condition `scale*leafPos[i] = S_i(leafPos[i+nSides/2]) = r*(-leafPos[i]) + c*leafPos[i]`
 * must hold for every `i` simultaneously, which (since `leafPos[i]` sweeps every direction as `i`
 * varies) pins `scale = c - r` exactly - the `nSides > 4` restriction is exactly what keeps this
 * positive: nSides=4 has `c = r` (see regularPolygonFlakeRC()'s own doc comment - it's the
 * edge-merge case with `k=0`, i.e. `c/r=1`), which would degenerate the central copy to a single
 * point. Verified numerically (this `scale`, together with the resulting node/edge/degree counts)
 * against an independent mergeClose()-based reference construction, for nSides=6,8,10,12 and
 * order=1..3.
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
    const glueMap = new Map<string, GlueEntry>();
    if (nSides % 4 === 0) {
        const glue = computeFlakeGlue(leafPos, leafConn, r, c);
        for (const [ga, gb, gc, gd, ge, gf] of glue)
            glueMap.set(`${ga},${gb}`, { object: EDGE_GLUE_OBJECT, selfVertices: [gc, gd], otherVertices: [ge, gf] });
    } else {
        const glue = computeNodeGlue(leafPos, leafConn, r, c);
        for (const [ga, gb, gm, gp] of glue)
            glueMap.set(`${ga},${gb}`, { object: POINT_GLUE_OBJECT, selfVertices: [gm], otherVertices: [gp] });
    }
    const subDescr: SubDescr[] = leafPos.map(v => ({ scale: r, shift: v.map(x => x * c) }));

    if (center && nSides % 2 === 0 && nSides > 4) {
        const centerIdx = subDescr.length;
        subDescr.push({ scale: c - r, shift: new Array(leafPos[0].length).fill(0) });
        for (let i = 0; i < nSides; i++)
            glueMap.set(`${i},${centerIdx}`,
                { object: POINT_GLUE_OBJECT, selfVertices: [(i + nSides / 2) % nSides], otherVertices: [i] });
    }

    const descr: FractalDescr = { leafPos, leafConn, subDescr, glueMap, globalScale: 1 / r };
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
 * single shared node): `glueMap` gets one `EDGE_GLUE_OBJECT` entry per `i`, keyed `((i+3)%5,
 * centerIdx)`, `selfVertices = [i, i+1]`, `otherVertices = [i+1, i]` (the "crossed" correspondence - the
 * center's own vertex `i+1` is what coincides with subflake `(i+3)%5`'s vertex `i`, and vice versa).
 *
 * Unlike the pre-generalization design (see git history), this needs NO further, hand-derived data:
 * `EDGE_GLUE_OBJECT`'s own `addresses()` already describes how a growing edge subdivides, purely
 * combinatorially (see its own doc comment for why this works correctly even though the center is
 * reflected relative to the regular copies, with zero special-casing needed - this was the ORIGINAL
 * motivation for generalizing past the old `edgeGlueMap`/`edgeLevelUpMap` split, which needed a
 * hand-derived orientation fix here that this design makes unnecessary).
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
    const glueMap = new Map<string, GlueEntry>();
    for (const [ga, gb, gm, gp] of nodeGlue)
        glueMap.set(`${ga},${gb}`, { object: POINT_GLUE_OBJECT, selfVertices: [gm], otherVertices: [gp] });

    const subDescr: SubDescr[] = leafPos.map(v => ({ scale: r, shift: v.map(x => x * c) }));
    const centerIdx = subDescr.length;
    subDescr.push({ scale: -r, shift: new Array(leafPos[0].length).fill(0) });

    for (let i = 0; i < nSides; i++) {
        const a = i, b = (i + 1) % nSides;
        glueMap.set(`${(i + 3) % nSides},${centerIdx}`,
            { object: EDGE_GLUE_OBJECT, selfVertices: [a, b], otherVertices: [b, a] });
    }

    centralPentagonFractalDescrCache = { leafPos, leafConn, subDescr, glueMap, globalScale: 1 / r };
    return centralPentagonFractalDescrCache;
}

// Cached per (dim, indicator) - see mengerFractalDescr()'s own doc comment for what these mean and
// how they combine into this cache's own two-level key.
const mengerFractalDescrCache = new Map<number, Map<number, FractalDescr>>();

/**
 * All `base^k` length-`k` tuples with entries in `0..base-1`, in mixed-radix counting order (entry 0
 * most significant - the same order a `k`-deep nested `for` loop, outermost variable first, would
 * produce). Shared by mengerFractalDescr() (leaf-vertex corners, `base=2`; sub-cube grid positions,
 * `base=3`) and mengerFaceGlueObject() (a face's own free-axis grid positions, `base=3`) - the
 * enumeration order only needs to be SOME fixed, deterministic order (see mengerFaceGlueObject()'s own
 * doc comment for why), so this one, shared implementation covers every use.
 */
function radixTuples(k: number, base: number): number[][] {
    let tuples: number[][] = [[]];
    for (let d = 0; d < k; d++) {
        const next: number[][] = [];
        for (const t of tuples) for (let v = 0; v < base; v++) next.push([...t, v]);
        tuples = next;
    }
    return tuples;
}

/**
 * A `dim`-dimensional hypercube leaf vertex `(x_0, ..., x_{dim-1})`, `x_i in {0, 1}`, packed as
 * `encodeMengerVertex(coords) = sum_i coords[i] * 2^(dim-1-i)` (matching dodecahedronFractalDescr()'s
 * own `xIdx` convention, generalized to `dim` axes - `dim=3`'s own `x*4+y*2+z` is this formula's own
 * `dim=3` case) - `decodeMengerVertex` is its inverse, via bit-shifts. mengerFractalDescr()'s own
 * `leafPos` is indexed this way, and `mengerFaceGlueObject()` below decodes/encodes leaf-vertex numbers
 * with these SAME two functions at every recursion level, since the same base hypercube is reused
 * recursively (see `GlueObjectType`'s own doc comment).
 */
function encodeMengerVertex(coords: number[]): number {
    return coords.reduce((acc, c) => acc * 2 + c, 0);
}
function decodeMengerVertex(dim: number, v: number): number[] {
    const coords: number[] = new Array(dim);
    for (let i = 0; i < dim; i++) coords[i] = (v >> (dim - 1 - i)) & 1;
    return coords;
}

/**
 * The `2^(dim-|fixedAxes|)` leaf-vertex tags of a `dim`-dimensional hypercube's own sub-face pinned to
 * `fixedVals[j]` on axis `fixedAxes[j]`, for every `j` (the general form of "a face" - `|fixedAxes|=1`
 * is an ordinary `(dim-1)`-dimensional face; `|fixedAxes|=dim` is a single corner point, R=1). Tag `i`
 * has, on the sub-face's own free axes (every axis not in `fixedAxes`, in ascending order), bit `j` of
 * `i` as free axis `j`'s own coordinate. This "tag `i` has bit `j` of `i` on free axis `j`" convention
 * is exactly what `mengerHyperfaceGlueObject()`'s own `step()` assumes of its own input tags (see that
 * function's own doc comment) - every `glueMap` entry built by `mengerFractalDescr()` uses this SAME
 * function for both `selfVertices` and `otherVertices`, so the convention is self-consistent
 * throughout. `dim=|fixedAxes|` is the degenerate, no-free-axes case: this returns the single tag
 * `[encodeMengerVertex(fixedVals in axis order)]` - a plain point, matching `POINT_GLUE_OBJECT`'s own
 * R=1 (needed whenever two sub-cubes touch only at a shared corner - see mengerFractalDescr()'s own doc
 * comment for when that happens).
 */
function mengerFaceTags(dim: number, fixedAxes: number[], fixedVals: number[]): number[] {
    const freeAxes = [...Array(dim).keys()].filter(a => !fixedAxes.includes(a));
    const tags: number[] = [];
    for (let i = 0; i < (1 << freeAxes.length); i++) {
        const c = new Array(dim).fill(0);
        fixedAxes.forEach((ax, j) => { c[ax] = fixedVals[j]; });
        freeAxes.forEach((ax, j) => { c[ax] = (i >> j) & 1; });
        tags.push(encodeMengerVertex(c));
    }
    return tags;
}

/**
 * Whether the `dim`-length sub-cube grid position `grid` (each entry `0`, `1`, or `2` - `1` meaning
 * "centered" on that axis) survives mengerFractalDescr()'s own removal rule: writing `offCenter` for
 * the count of `grid`'s own non-1 entries (how many axes this position is "off-center" on, `0..dim`),
 * `grid` is kept iff `indicator[offCenter] === 1`. `indicator` has exactly `dim+1` entries, one per
 * possible `offCenter` value `0..dim` - see mengerFractalDescr()'s own doc comment for how this single
 * rule specializes to the Cantor set/Sierpinski carpet/Menger sponge at `dim=1,2,3`.
 */
function isMengerGridKept(grid: number[], indicator: number[]): boolean {
    const offCenter = grid.filter(v => v !== 1).length;
    return indicator[offCenter] === 1;
}

/**
 * The general "hyperface" glue object mengerFractalDescr() glues every touching pair of sub-cubes
 * with, regardless of the dimension of the sub-face they share - see `GlueObjectType`'s own doc
 * comment for why this needs genuine, ad hoc, per-shape logic (a face's own sub-pieces are NOT all
 * reselections of the face's own corners, unlike `POINT_GLUE_OBJECT`/`EDGE_GLUE_OBJECT` above). Closes
 * over `slotOf` (mengerFractalDescr()'s own `(a_0,...,a_{dim-1})` sub-cube grid position -> `subDescr`
 * slot lookup) since which OUTER `subDescr` slot each sub-piece chases into is inherently specific to
 * THIS shape's own sub-cube layout - unlike `POINT_GLUE_OBJECT`/`EDGE_GLUE_OBJECT`, this is
 * deliberately NOT a shape-independent, reusable module-level constant (also closes over
 * `dim`/`indicator`, needed to know which of a sub-face's own grid positions actually survive - see
 * below).
 *
 * Two sub-cubes at grid positions `gA`, `gB` touch (share SOME sub-face, of any dimension down to a
 * single corner point) exactly when every axis has `|gA[axis] - gB[axis]| <= 1` (grid values 0 and 2
 * are NEVER adjacent - there is a real gap between them, spanned by grid value 1, whether or not that
 * middle sub-cube itself survives) and at least one axis actually differs. The set of DIFFERING axes -
 * `fixedAxes` below, from `step()`'s own perspective, since those are exactly the axes the shared
 * sub-face is pinned on - can be ANY non-empty subset of `0..dim-1`, from a single axis (an ordinary
 * `(dim-1)`-face, mengerSpongeFlake()'s own square faces at `dim=3`) up to all `dim` axes at once (a
 * single shared corner point - e.g. `dim=3`, `indicator=[1,0,0]`: the center sub-cube survives
 * alongside the 8 corner sub-cubes, but faces and edges do not, so the center only ever touches a
 * corner at one shared point, needing exactly this R=1 case - see mengerFractalDescr()'s own doc
 * comment).
 *
 * `step(tags)`: decodes all input tags (`mengerFaceTags()`'s own convention) back into their own
 * coordinates and finds the FULL SET of axes every one of them agrees on (`fixedAxes` - inferring this
 * from the tags themselves, rather than being told it, is what lets one `step()` implementation handle
 * every possible sub-face dimension uniformly); the remaining axes are this sub-face's own "free" axes.
 * A hypercube subdivides into a `3^dim` grid of sub-cubes; this sub-face lies at grid position 0 (if
 * `fixedVals[j]=0`) or 2 (if `fixedVals[j]=1`) on each of `fixedAxes`, and spans all
 * `3^(dim-|fixedAxes|)` grid positions of the free axes (`radixTuples(freeAxes.length, 3)`) - each
 * combined with the fixed axes' own positions into a full `dim`-length grid position and checked
 * against `isMengerGridKept()`, exactly the SAME rule `mengerFractalDescr()`'s own `positions` uses
 * (this sub-face's own subdivision is nothing but a slice of the shape's single removal rule, some axes
 * pinned - so which of a sub-face's own sub-pieces survive falls out automatically, rather than needing
 * a separately hard-coded constant per dimension). For each surviving grid position, `subSlot` is that
 * sub-cube's own OUTER slot (via `slotOf`), and `tags` is THAT sub-cube's OWN copy of the SAME sub-face
 * (same fixed axes/values, `mengerFaceTags()`'s own convention again, but now naming the sub-cube's OWN
 * local leaf-vertex numbering - the SAME `0..2^dim-1` range, reused recursively, per `GlueObjectType`'s
 * own doc comment). This uniform per-grid-position formula needs no explicit "inherited vs
 * freshly-introduced" branch at all: for the sub-face's own corner grid positions, one of the local
 * tags it produces numerically coincides with one of the input tags (both computed by the exact same
 * formula on matching free-axis coordinates); for the others, none do - but that fresh-vs-inherited
 * distinction falls out automatically from the shared formula rather than needing to be computed
 * explicitly. (Sub-pieces that share a grid point with EACH OTHER are reconciled by
 * mengerFractalDescr()'s own separate `glueMap` entry for THAT pair of sub-cubes, not by this function:
 * this object's only job is "which nodes on THIS sub-face correspond to which nodes on the other side
 * of THIS SPECIFIC glued pair," exactly as `GlueEntry`'s own doc comment specifies - verified against an
 * independent floating-point reference construction at `dim=1,2,3` (including non-classical `indicator`
 * choices that force the R=1 corner-touching case, not just Cantor/carpet/sponge), orders 1-3; see git
 * history.)
 */
function mengerHyperfaceGlueObject(dim: number, indicator: number[], slotOf: Map<string, number>): GlueObjectType {
    return {
        step(tags) {
            const coords = tags.map(v => decodeMengerVertex(dim, v));
            const fixedAxes: number[] = [];
            for (let axis = 0; axis < dim; axis++)
                if (coords.every(c => c[axis] === coords[0][axis])) fixedAxes.push(axis);
            assert(fixedAxes.length > 0, `mengerHyperfaceGlueObject: tags do not lie on a common sub-face: ${tags}`);
            const freeAxes = [...Array(dim).keys()].filter(a => !fixedAxes.includes(a));
            const fixedVals = fixedAxes.map(ax => coords[0][ax]);
            const fixedGrids = fixedVals.map(v => v === 0 ? 0 : 2);

            const steps: { subSlot: number; tags: number[] }[] = [];
            for (const free of radixTuples(freeAxes.length, 3)) {
                const grid = new Array(dim).fill(1);
                fixedAxes.forEach((ax, j) => { grid[ax] = fixedGrids[j]; });
                freeAxes.forEach((ax, j) => { grid[ax] = free[j]; });
                if (!isMengerGridKept(grid, indicator)) continue;
                const subSlot = slotOf.get(grid.join(','));
                assert(subSlot !== undefined, `mengerHyperfaceGlueObject: no sub-cube at grid ${grid}`);
                steps.push({ subSlot, tags: mengerFaceTags(dim, fixedAxes, fixedVals) });
            }
            return steps;
        },
    };
}

/**
 * The static description mengerSpongeFlake() (and its lower-dimensional analogs) build on: the `dim`-
 * dimensional unit hypercube's own `2^dim` corners (`leafPos`, `(x_0,...,x_{dim-1}) in {0,1}^dim`,
 * indexed via `encodeMengerVertex()`) and its edges (`leafConn` - every corner pair differing in
 * exactly 1 coordinate). Each order n>1 divides into a `3^dim` grid of sub-cubes, keeping only those
 * `isMengerGridKept()` accepts for the given `indicator` (a length-`dim+1` 0/1 list: entry `k` says
 * whether the "exactly `k` axes off-center" class survives, for `k = 0..dim` - including `k = dim`
 * itself, every axis off-center, the hypercube's own `2^dim` corners; a `0` there leaves no
 * sub-hypercube at any of the shape's own extreme positions, an unusual but valid degenerate choice).
 * This single rule specializes to the three classical examples: `dim=1`, `indicator=[0,1]` is the
 * Cantor set (2 of 3 sub-segments kept, the removed-middle-third construction); `dim=2`,
 * `indicator=[0,1,1]` is the Sierpinski carpet (8 of 9 sub-squares kept, only the center removed);
 * `dim=3`, `indicator=[0,0,1,1]` is the classical Menger sponge (20 of 27 sub-cubes kept, the center
 * and 6 face-centers removed) - verified (node/edge counts at orders 1-3, all three, plus further
 * non-classical `indicator` choices at `dim=2,3`) against an independent floating-point
 * mergeClose()-based reference construction; see git history.
 *
 * `positions[i]` is the `i`-th surviving grid position (`dim`-length, entries in `0,1,2`), `slotOf` its
 * inverse - both built once here and closed over by `mengerHyperfaceGlueObject()` above. Sub-cube
 * `positions[i]`'s own `SubDescr` is `{ scale: 1/3, shift: positions[i].map(k => (k-1)/3) }` (`(k-1)/3`
 * is that grid coordinate's own center position within the parent's own `[-0.5, 0.5]` unit-cube range -
 * `leafPos` itself uses `{0,1}^dim` shifted to `x - 0.5` etc., so corners land at +-0.5) - unlike every
 * other shape in this file, `scale`/`shift` here are NOT derived from a single shared `r,c` pair (this
 * construction has no analog of dodeca/icosa/octahedron's uniform "every copy transforms by the same
 * r,c" relation - each grid position gets its own independent shift, only their common `1/3` scale is
 * shared).
 *
 * `glueMap` has one entry per pair of surviving grid positions that TOUCH (every axis differs by at
 * most 1, at least one axis differs - see `mengerHyperfaceGlueObject()`'s own doc comment for why this
 * can be more than one axis at once, e.g. two sub-cubes sharing only a corner) - `object:
 * mengerHyperfaceGlueObject(dim, indicator, slotOf)`, `selfVertices`/`otherVertices` the two cubes' own
 * matching sub-faces (`mengerFaceTags(dim, diffAxes, ...)`, one call per side: on each differing axis,
 * whichever of the two grid positions has the SMALLER coordinate contributes fixedVal 1 there (its own
 * "+axis" side, touching the other), the larger contributes fixedVal 0 - no reflection or rotation ever
 * happens between adjacent sub-cubes here, so the two sub-faces' matching tag `i` line up directly with
 * no reordering needed, unlike centralPentagonFractalDescr()'s own reflected central copy).
 *
 * Cached per `(dim, indicator)`: `indicator`'s own `dim+1` bits, MSB-first, reduced to a single integer
 * key (`indicator.reduce((acc, b) => acc * 2 + b, 0)`), nested under a first-level cache keyed by `dim`
 * itself (`indicator`'s own valid range depends on `dim`, so nesting under `dim` sidesteps needing any
 * arbitrary scheme to pack both into one flat key).
 */
export function mengerFractalDescr(dim: number, indicator: number[]): FractalDescr {
    assert(Number.isInteger(dim) && dim >= 1, `dim must be a positive integer, got ${dim}`);
    assert(indicator.length === dim + 1 && indicator.every(b => b === 0 || b === 1),
        `indicator must be a length-${dim + 1} list of 0/1 entries, got [${indicator}]`);

    const indicatorBits = indicator.reduce<number>((acc, b) => acc * 2 + b, 0);
    let byIndicator = mengerFractalDescrCache.get(dim);
    if (!byIndicator) { byIndicator = new Map<number, FractalDescr>(); mengerFractalDescrCache.set(dim, byIndicator); }
    const cached = byIndicator.get(indicatorBits);
    if (cached) return cached;

    const numCorners = 1 << dim;
    const leafPos: number[][] = new Array(numCorners);
    for (let v = 0; v < numCorners; v++) leafPos[v] = decodeMengerVertex(dim, v).map(x => x - 0.5);
    const leafConn: [number, number][] = [];
    for (let v1 = 0; v1 < numCorners; v1++)
        for (let v2 = v1 + 1; v2 < numCorners; v2++) {
            const c1 = decodeMengerVertex(dim, v1), c2 = decodeMengerVertex(dim, v2);
            const diff = c1.filter((c, i) => c !== c2[i]).length;
            if (diff === 1) leafConn.push([v1, v2]);
        }

    const positions = radixTuples(dim, 3).filter(grid => isMengerGridKept(grid, indicator));
    const slotOf = new Map(positions.map((grid, i): [string, number] => [grid.join(','), i]));
    const subDescr: SubDescr[] = positions.map(grid => ({ scale: 1 / 3, shift: grid.map(k => (k - 1) / 3) }));

    const object = mengerHyperfaceGlueObject(dim, indicator, slotOf);
    const glueMap = new Map<string, GlueEntry>();
    for (let i = 0; i < positions.length; i++)
        for (let j = i + 1; j < positions.length; j++) {
            const gi = positions[i], gj = positions[j];
            const diffAxes: number[] = [];
            let touching = true;
            for (let axis = 0; axis < dim; axis++) {
                const d = gj[axis] - gi[axis];
                if (d === 0) continue;
                if (Math.abs(d) !== 1) { touching = false; break; }
                diffAxes.push(axis);
            }
            if (!touching || diffAxes.length === 0) continue;
            const selfVals = diffAxes.map(ax => gi[ax] < gj[ax] ? 1 : 0);
            const otherVals = selfVals.map(v => 1 - v);
            glueMap.set(`${i},${j}`, {
                object,
                selfVertices: mengerFaceTags(dim, diffAxes, selfVals),
                otherVertices: mengerFaceTags(dim, diffAxes, otherVals),
            });
        }

    const descr: FractalDescr = { leafPos, leafConn, subDescr, glueMap, globalScale: 3 };
    byIndicator.set(indicatorBits, descr);
    return descr;
}
