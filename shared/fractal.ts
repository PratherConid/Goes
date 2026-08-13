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

let mengerSpongeFractalDescrCache: FractalDescr | null = null;

/**
 * A cube leaf vertex `(x, y, z)`, `x,y,z` each `0` or `1`, packed as `encodeMengerVertex(x,y,z) =
 * x*4 + y*2 + z` (matching dodecahedronFractalDescr()'s own `xIdx` convention) - `decodeMengerVertex`
 * is its inverse, via bit-shifts. mengerSpongeFractalDescr()'s own `leafPos` is indexed this way, and
 * `mengerSpongeSquareGlueObject()` below decodes/encodes leaf-vertex numbers with these SAME two
 * functions at every recursion level, since the same base cube is reused recursively (see
 * `GlueObjectType`'s own doc comment).
 */
function encodeMengerVertex(x: number, y: number, z: number): number {
    return x * 4 + y * 2 + z;
}
function decodeMengerVertex(v: number): [number, number, number] {
    return [(v >> 2) & 1, (v >> 1) & 1, v & 1];
}

/**
 * The 4 leaf-vertex tags of a cube's own face at `axis = fixedVal` (`axis` one of 0,1,2 for x,y,z;
 * `fixedVal` 0 or 1): tag `i` has, on the face's own 2 free axes (the 2 axes other than `axis`, in
 * ascending order), coordinates `(i & 1, (i >> 1) & 1)`. This "tag `i` has free-coords `(i&1,
 * (i>>1)&1)`" convention is exactly what `mengerSpongeSquareGlueObject()`'s own `step()` assumes of
 * its own input tags (see that function's own doc comment) - every `glueMap` entry built by
 * `mengerSpongeFractalDescr()` uses this SAME function for both `selfVertices` and `otherVertices`, so
 * the convention is self-consistent throughout.
 */
function mengerFaceTags(axis: number, fixedVal: number): number[] {
    const freeAxes = [0, 1, 2].filter(a => a !== axis);
    const tags: number[] = [];
    for (let i = 0; i < 4; i++) {
        const c = [0, 0, 0];
        c[axis] = fixedVal;
        c[freeAxes[0]] = i & 1;
        c[freeAxes[1]] = (i >> 1) & 1;
        tags.push(encodeMengerVertex(c[0], c[1], c[2]));
    }
    return tags;
}

/**
 * The "square" glue object (R=4) mengerSpongeFractalDescr() glues adjacent sub-cubes' shared faces
 * with - see `GlueObjectType`'s own doc comment for why this needs genuine, ad hoc, per-shape logic (a
 * face's own sub-pieces are NOT all reselections of the face's 4 named corners, unlike
 * `POINT_GLUE_OBJECT`/`EDGE_GLUE_OBJECT` above). Closes over `slotOf` (Menger sponge's own 20-entry
 * `(a,b,c)` sub-cube grid position -> `subDescr` slot lookup - see `mengerSpongeFractalDescr()`'s own
 * doc comment) since which OUTER `subDescr` slot each sub-piece chases into is inherently specific to
 * Menger sponge's own sub-cube layout - unlike `POINT_GLUE_OBJECT`/`EDGE_GLUE_OBJECT`, this is
 * deliberately NOT a shape-independent, reusable module-level constant.
 *
 * `step(tags)`: decodes all 4 input tags (`mengerFaceTags()`'s own convention) back into `(x,y,z)` and
 * finds the FIXED axis (the one all 4 agree on - the face's own normal) and its value; the other 2
 * axes are this face's own "free" axes. A cube subdivides into a 3x3x3 grid of sub-cubes; this face
 * lies at grid position 0 (if `fixedVal=0`) or 2 (if `fixedVal=1`) along the fixed axis, and spans all
 * 3x3=9 grid positions of the two free axes - MINUS the grid center (free-axis grid position `(1,1)`),
 * which is ALWAYS one of the 7 sub-cubes Menger sponge removes (a "face-center" cube: exactly 2 of its
 * 3 grid coordinates equal 1, its fixed-axis one from this face and both free-axis ones from the
 * center) - leaving exactly 8, matching the classic Menger sponge face subdivision this shape is built
 * from. For each of those 8 grid positions, `subSlot` is that sub-cube's own OUTER slot (via
 * `slotOf`), and `tags` is THAT sub-cube's OWN copy of the SAME face (same fixed axis/value,
 * `mengerFaceTags()`'s own convention again, but now naming the sub-cube's OWN local leaf-vertex
 * numbering - the SAME 0..7 range, reused recursively, per `GlueObjectType`'s own doc comment). This
 * uniform per-grid-position formula needs no explicit "inherited vs freshly-introduced" branch at all:
 * for the 4 CORNER grid positions, one of the 4 local tags it produces numerically coincides with one
 * of the input tags (both computed by the exact same formula on matching free-axis coordinates); for
 * the 4 edge-midpoint grid positions, none do - but that fresh-vs-inherited distinction falls out
 * automatically from the shared formula rather than needing to be computed explicitly. (Sub-pieces
 * that share a grid point with EACH OTHER - e.g. two adjacent edge-midpoint cells - are reconciled by
 * `mengerSpongeFractalDescr()`'s own separate `glueMap` entry for THAT pair of sub-cubes, not by this
 * function: this object's only job is "which nodes on THIS face correspond to which nodes on the
 * other side of THIS SPECIFIC glued pair," exactly as `GlueEntry`'s own doc comment specifies -
 * verified against an independent floating-point reference construction, orders 1-3; see git history.)
 */
function mengerSpongeSquareGlueObject(slotOf: Map<string, number>): GlueObjectType {
    return {
        step(tags) {
            const coords = tags.map(decodeMengerVertex);
            let fixedAxis = -1, fixedVal = -1;
            for (let axis = 0; axis < 3; axis++) {
                if (coords.every(c => c[axis] === coords[0][axis])) { fixedAxis = axis; fixedVal = coords[0][axis]; break; }
            }
            assert(fixedAxis !== -1, `mengerSpongeSquareGlueObject: tags do not lie on a common face: ${tags}`);
            const freeAxes = [0, 1, 2].filter(a => a !== fixedAxis);
            const fixedGrid = fixedVal === 0 ? 0 : 2;

            const steps: { subSlot: number; tags: number[] }[] = [];
            for (let g1 = 0; g1 < 3; g1++)
                for (let g2 = 0; g2 < 3; g2++) {
                    if (g1 === 1 && g2 === 1) continue; // this face's own removed center
                    const grid = [0, 0, 0];
                    grid[fixedAxis] = fixedGrid;
                    grid[freeAxes[0]] = g1;
                    grid[freeAxes[1]] = g2;
                    const subSlot = slotOf.get(grid.join(','));
                    assert(subSlot !== undefined, `mengerSpongeSquareGlueObject: no sub-cube at grid ${grid}`);
                    steps.push({ subSlot, tags: mengerFaceTags(fixedAxis, fixedVal) });
                }
            return steps;
        },
    };
}

/**
 * The static description mengerSpongeFlake() builds on: the unit cube's 8 corners (`leafPos`,
 * `(x,y,z) in {0,1}^3`, indexed via `encodeMengerVertex()`) and 12 edges (`leafConn` - every pair
 * differing in exactly 1 coordinate). Each order n>1 divides into the classic Menger sponge's 3x3x3
 * grid of 27 sub-cubes minus the 7 that are removed - the very center `(1,1,1)` and the 6 face-centers
 * (exactly 2 of 3 grid coordinates equal 1) - leaving 20, matching `subDescr`'s own 20 entries
 * (`positions[i]` is grid position `(a,b,c)`, `slotOf` its inverse - both built once here and closed
 * over by `mengerSpongeSquareGlueObject()` above). Sub-cube `(a,b,c)`'s own `SubDescr` is
 * `{ scale: 1/3, shift: [g(a), g(b), g(c)] }` where `g(k) = (k-1)/3` is that grid coordinate's own
 * center position within the parent's own `[-0.5, 0.5]` unit-cube range (`leafPos` itself uses
 * `(x,y,z) in {0,1}^3` shifted to `x - 0.5` etc., so corners land at +-0.5) - unlike every other shape
 * in this file, `scale`/`shift` here are NOT derived from a single shared `r,c` pair (Menger sponge
 * has no analog of dodeca/icosa/octahedron's uniform "every copy transforms by the same r,c" relation
 * - each grid position gets its own independent shift, only their common `1/3` scale is shared).
 *
 * `glueMap` has one entry per pair of the 20 valid sub-cubes that are face-adjacent (differ by 1 in
 * exactly one grid coordinate, both valid) - `object: mengerSpongeSquareGlueObject(slotOf)`,
 * `selfVertices`/`otherVertices` the two cubes' own matching faces (`mengerFaceTags(axis, 1)`/
 * `mengerFaceTags(axis, 0)` respectively - the lower-coordinate cube's own "+axis" face glued to the
 * higher-coordinate cube's own "-axis" face; no reflection or rotation ever happens between adjacent
 * Menger-sponge sub-cubes, so the two faces' matching tag `i` line up directly with no reordering
 * needed, unlike centralPentagonFractalDescr()'s own reflected central copy). Verified (node/edge
 * counts at orders 1-3) against an independent floating-point mergeClose()-based reference
 * construction; see git history.
 */
export function mengerSpongeFractalDescr(): FractalDescr {
    if (mengerSpongeFractalDescrCache) return mengerSpongeFractalDescrCache;

    const leafPos: number[][] = new Array(8);
    for (let x = 0; x < 2; x++)
        for (let y = 0; y < 2; y++)
            for (let z = 0; z < 2; z++)
                leafPos[encodeMengerVertex(x, y, z)] = [x - 0.5, y - 0.5, z - 0.5];
    const leafConn: [number, number][] = [];
    for (let v1 = 0; v1 < 8; v1++)
        for (let v2 = v1 + 1; v2 < 8; v2++) {
            const c1 = decodeMengerVertex(v1), c2 = decodeMengerVertex(v2);
            const diff = c1.filter((c, i) => c !== c2[i]).length;
            if (diff === 1) leafConn.push([v1, v2]);
        }

    const positions: [number, number, number][] = [];
    for (let a = 0; a < 3; a++)
        for (let b = 0; b < 3; b++)
            for (let c = 0; c < 3; c++)
                if ([a, b, c].filter(v => v === 1).length < 2) positions.push([a, b, c]);
    const slotOf = new Map(positions.map(([a, b, c], i): [string, number] => [`${a},${b},${c}`, i]));
    const gridCenter = (k: number) => (k - 1) / 3;
    const subDescr: SubDescr[] = positions.map(([a, b, c]) => ({
        scale: 1 / 3,
        shift: [gridCenter(a), gridCenter(b), gridCenter(c)],
    }));

    const object = mengerSpongeSquareGlueObject(slotOf);
    const glueMap = new Map<string, GlueEntry>();
    for (let i = 0; i < positions.length; i++)
        for (let axis = 0; axis < 3; axis++) {
            const [a, b, c] = positions[i];
            const neighbor: [number, number, number] = [a, b, c];
            neighbor[axis] += 1;
            if (neighbor[axis] > 2) continue;
            const j = slotOf.get(neighbor.join(','));
            if (j === undefined) continue; // neighbor removed
            glueMap.set(`${i},${j}`,
                { object, selfVertices: mengerFaceTags(axis, 1), otherVertices: mengerFaceTags(axis, 0) });
        }

    mengerSpongeFractalDescrCache = { leafPos, leafConn, subDescr, glueMap, globalScale: 3 };
    return mengerSpongeFractalDescrCache;
}
