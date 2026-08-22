// Covers shared/selector.ts's tiny S-expression selector language: parseNodeSelector()/
// parseEdgeSelector()/parseTriangleSelector()/parseSquareSelector() (four mutually recursive
// parsers - see the file's own top comment) and selectNode()/selectEdge()/selectTriangle()/
// selectSquare() (evaluation against a real adjacency matrix).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    parseNodeSelector, parseEdgeSelector, parseTriangleSelector, parseSquareSelector,
    selectNode, selectEdge, selectTriangle, selectSquare, formatSelector,
} from '../shared/selector.ts';

// 0-1-2-3: node 0/3 have degree 1, node 1/2 have degree 2. Edges (0,1), (1,2), (2,3). No
// triangles/squares.
const adj = [
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
];
const pos = [[0], [1], [2], [3]]; // unused by the current grammar, still required by the API

test('parseNodeSelector/parseEdgeSelector reject malformed input (grammar errors)', () => {
    assert.throws(() => parseNodeSelector(''), /empty input/);
    assert.throws(() => parseNodeSelector('(deg eq 5'), /unexpected end of input/);
    assert.throws(() => parseNodeSelector('(deg eq 5) extra'), /unexpected trailing input/);
    assert.throws(() => parseNodeSelector('(foo)'), /unknown node-selector operator/);
    assert.throws(() => parseNodeSelector('(deg xx 5)'), /comparator must be/);
    assert.throws(() => parseNodeSelector('(deg eq -3)'), /nonnegative integer/);
    assert.throws(() => parseNodeSelector('(deg eq 3.5)'), /nonnegative integer/);
    assert.throws(() => parseNodeSelector('(deg eq abc)'), /nonnegative integer/);
});

test('parseNodeSelector/parseEdgeSelector reject an operator of the wrong kind, wherever it appears', () => {
    // fromna only exists in parseEdgeSelExpr/parseTriangleSelExpr/parseSquareSelExpr - unrecognized
    // inside a node-selector context, whether at the top level or nested.
    assert.throws(() => parseNodeSelector('(fromna (deg eq 1))'), /unknown node-selector operator 'fromna'/);
    assert.throws(
        () => parseNodeSelector('(union (deg eq 1) (fromna (deg eq 1)))'),
        /unknown node-selector operator 'fromna'/,
    );
    // deg only exists in parseNodeSelExpr - unrecognized inside an edge-selector context.
    assert.throws(() => parseEdgeSelector('(deg eq 1)'), /unknown edge-selector operator 'deg'/);
    // e2n/n2e no longer exist (replaced by fromna/fromne/tona/tone) - unrecognized anywhere.
    assert.throws(() => parseEdgeSelector('(e2n (deg eq 1))'), /unknown edge-selector operator 'e2n'/);
    assert.throws(() => parseNodeSelector('(n2e (all))'), /unknown node-selector operator 'n2e'/);
});

test('fromna/fromne reject an operand of the wrong kind, and tona/tone reject a bad source token', () => {
    // fromna/fromne's own operand is always parsed via parseNodeSelExpr, which doesn't recognize
    // fromna itself.
    assert.throws(() => parseEdgeSelector('(fromna (fromna (deg eq 1)))'), /unknown node-selector operator 'fromna'/);
    // tona/tone require an 'edge'/'tri'/'sq' token right after the op name.
    assert.throws(() => parseNodeSelector('(tona (all))'), /source kind must be 'edge', 'tri', or 'sq'/);
    assert.throws(() => parseNodeSelector('(tona nope (all))'), /source kind must be 'edge', 'tri', or 'sq'/);
    // and then an operand of THAT kind - (deg ...) is node-only, invalid as tona's edge operand.
    assert.throws(() => parseNodeSelector('(tona edge (deg eq 1))'), /unknown edge-selector operator 'deg'/);
});

test('all/none select every object of whichever kind, resolved by which parser reaches them ' +
    '(no argument)', () => {
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector('(all)')), new Set([0, 1, 2, 3]));
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector('(none)')), new Set());
    assert.deepEqual(
        selectEdge(adj, pos, parseEdgeSelector('(all)')),
        [{ n1: 0, n2: 1 }, { n1: 1, n2: 2 }, { n1: 2, n2: 3 }]);
    assert.deepEqual(selectEdge(adj, pos, parseEdgeSelector('(none)')), []);
    // No triangles/squares in this path graph.
    assert.deepEqual(selectTriangle(adj, pos, parseTriangleSelector('(all)')), []);
    assert.deepEqual(selectSquare(adj, pos, parseSquareSelector('(all)')), []);

    // (all)/(none) take no argument at all - anything extra before the closing paren is a grammar
    // error.
    assert.throws(() => parseNodeSelector('(all edge)'), /expected '\)', got 'edge'/);
    assert.throws(() => parseEdgeSelector('(all node)'), /expected '\)', got 'node'/);
});

test('deg selects nodes by exact/greater/less degree', () => {
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector('(deg eq 2)')), new Set([1, 2]));
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector('(deg gt 1)')), new Set([1, 2]));
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector('(deg lt 2)')), new Set([0, 3]));
});

test('union/inter/diff/compl combine node selectors as plain set operations', () => {
    assert.deepEqual(
        selectNode(adj, pos, parseNodeSelector('(union (deg eq 1) (deg eq 2))')), new Set([0, 1, 2, 3]));
    assert.deepEqual(
        selectNode(adj, pos, parseNodeSelector('(inter (deg eq 1) (deg lt 2))')), new Set([0, 3]));
    assert.deepEqual(
        selectNode(adj, pos, parseNodeSelector('(diff (deg gt 1) (deg eq 2))')), new Set());
    assert.deepEqual(
        selectNode(adj, pos, parseNodeSelector('(compl (deg eq 2))')), new Set([0, 3]));
});

test('fromna selects edges whose nodes are ALL selected, fromne whose nodes have AT LEAST ONE selected', () => {
    // deg eq 2 selects {1, 2} - only edge (1,2) has both endpoints in that set (fromna); edges
    // (0,1) and (1,2) each have at least one endpoint in it (fromne).
    assert.deepEqual(selectEdge(adj, pos, parseEdgeSelector('(fromna (deg eq 2))')), [{ n1: 1, n2: 2 }]);
    assert.deepEqual(
        selectEdge(adj, pos, parseEdgeSelector('(fromne (deg eq 2))')),
        [{ n1: 0, n2: 1 }, { n1: 1, n2: 2 }, { n1: 2, n2: 3 }]);
});

test('tona selects nodes whose every containing edge is selected, tone whose any containing edge is', () => {
    // (fromna (deg eq 2)) = {(1,2)} only.
    const selEdges = '(fromna (deg eq 2))';
    // tone: node 0's only edge is (0,1), not selected -> excluded; node 1's edges are (0,1) [not
    // selected] and (1,2) [selected] -> at least one selected -> included. Node 2 symmetric to 1.
    // Node 3's only edge (2,3) isn't selected -> excluded.
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector(`(tone edge ${selEdges})`)), new Set([1, 2]));
    // tona: node 1 has TWO containing edges, (0,1) and (1,2) - only one is selected, so not ALL are
    // -> excluded. Same for node 2. Nodes 0/3 each have exactly one containing edge, not selected ->
    // excluded too. Nothing in this graph has every containing edge selected.
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector(`(tona edge ${selEdges})`)), new Set());
});

test('tona is vacuously true for a node with no containing objects of the given kind', () => {
    // No triangles at all in this graph - every node "vacuously" satisfies tona (every one of its
    // zero containing triangles is trivially selected), but tone (at least one) is vacuously false.
    assert.deepEqual(
        selectNode(adj, pos, parseNodeSelector('(tona tri (all))')), new Set([0, 1, 2, 3]));
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector('(tone tri (all))')), new Set());
});

test('compl on an edge selector complements within all of the graph\'s edges', () => {
    // All edges: (0,1), (1,2), (2,3). fromna(deg eq 2) = {(1,2)}.
    const edges = selectEdge(adj, pos, parseEdgeSelector('(compl (fromna (deg eq 2)))'));
    assert.deepEqual(edges, [{ n1: 0, n2: 1 }, { n1: 2, n2: 3 }]);
});

test('more expands a node selector to one-edge-away neighbors, keeping the original selection ' +
    'and nothing farther', () => {
    // 5-node path 0-1-2-3-4: endpoints 0/4 have degree 1, nodes 1-3 have degree 2.
    const path5Adj = [
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
    ];
    const path5Pos = [[0], [1], [2], [3], [4]];
    // (deg eq 1) selects the two endpoints {0, 4}; more adds their neighbors {1, 3}, but not node 2
    // (two edges away from both endpoints).
    const nodes = selectNode(path5Adj, path5Pos, parseNodeSelector('(more (deg eq 1))'));
    assert.deepEqual(nodes, new Set([0, 1, 3, 4]));
});

test('more expands an edge selector to edges sharing a node, keeping the original selection and ' +
    'nothing from a disconnected component', () => {
    // A triangle (nodes 0,1,2, each degree 2) plus a disjoint single edge (3,4, each degree 1) -
    // no edges connect the two components.
    const triPlusEdgeAdj = [
        [0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
    ];
    const triPlusEdgePos = [[0], [1], [2], [3], [4]];
    // fromna(deg eq 2) selects the 3 triangle edges; more re-adds every edge incident to nodes
    // 0/1/2 - still just those same 3 edges, since edge (3,4) isn't incident to any of them.
    const edges = selectEdge(triPlusEdgeAdj, triPlusEdgePos, parseEdgeSelector('(more (fromna (deg eq 2)))'));
    assert.deepEqual(edges, [{ n1: 0, n2: 1 }, { n1: 0, n2: 2 }, { n1: 1, n2: 2 }]);
});

test('more is rejected for triangle/square selectors (no adjacency notion is defined for them)', () => {
    assert.throws(() => parseTriangleSelector('(more (all))'), /unknown triangle-selector operator 'more'/);
    assert.throws(() => parseSquareSelector('(more (all))'), /unknown square-selector operator 'more'/);
});

test('more on an already-all selector is a no-op, and formatSelector round-trips it', () => {
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector('(more (all))')), new Set([0, 1, 2, 3]));
    const sel = parseEdgeSelector('(more (fromna (deg eq 2)))');
    assert.equal(formatSelector(sel), '(more (fromna (deg eq 2)))');
});

test('union/inter/diff combine edge selectors as plain (deduplicated) set operations', () => {
    // (fromna (deg gt 0)) is every edge (every node has degree > 0); (fromna (deg eq 2)) is just (1,2).
    const all = '(fromna (deg gt 0))', middle = '(fromna (deg eq 2))';
    assert.deepEqual(
        selectEdge(adj, pos, parseEdgeSelector(`(union ${all} ${middle})`)),
        [{ n1: 0, n2: 1 }, { n1: 1, n2: 2 }, { n1: 2, n2: 3 }]);
    assert.deepEqual(
        selectEdge(adj, pos, parseEdgeSelector(`(inter ${all} ${middle})`)),
        [{ n1: 1, n2: 2 }]);
    assert.deepEqual(
        selectEdge(adj, pos, parseEdgeSelector(`(diff ${all} ${middle})`)),
        [{ n1: 0, n2: 1 }, { n1: 2, n2: 3 }]);
});

// A "bowtie" graph: triangle 0-1-2 and triangle 2-3-4 sharing node 2, plus a pendant node 5 hanging
// off node 0. Degrees: 0:3 (1,2,5), 1:2, 2:4 (0,1,3,4), 3:2, 4:2, 5:1 (belongs to no triangle).
// Triangles (see findTriangles's own u<v<w convention): [0,1,2] and [2,3,4]. No squares.
const bowtieAdj = [
    [0, 1, 1, 0, 0, 1],
    [1, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 1, 0],
    [0, 0, 1, 0, 1, 0],
    [0, 0, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 0],
];
const bowtiePos = bowtieAdj.map((_, i) => [i]);

test('selectTriangle finds every triangle via (all), and none in a triangle-free graph', () => {
    assert.deepEqual(
        selectTriangle(bowtieAdj, bowtiePos, parseTriangleSelector('(all)')),
        [{ n1: 0, n2: 1, n3: 2 }, { n1: 2, n2: 3, n3: 4 }]);
    assert.deepEqual(selectTriangle(adj, pos, parseTriangleSelector('(all)')), []);
});

test('union/inter/diff/compl combine triangle selectors as plain (deduplicated) set operations', () => {
    // Only node 0 has degree 3 - fromne(deg eq 3) selects exactly the triangle containing it, [0,1,2].
    const first = '(fromne (deg eq 3))';
    const second = 'diff (all) ' + first; // the other triangle, [2,3,4]
    assert.deepEqual(
        selectTriangle(bowtieAdj, bowtiePos, parseTriangleSelector(first)), [{ n1: 0, n2: 1, n3: 2 }]);
    assert.deepEqual(
        selectTriangle(bowtieAdj, bowtiePos, parseTriangleSelector(`(${second})`)), [{ n1: 2, n2: 3, n3: 4 }]);
    assert.deepEqual(
        selectTriangle(bowtieAdj, bowtiePos, parseTriangleSelector(`(union ${first} (${second}))`)),
        [{ n1: 0, n2: 1, n3: 2 }, { n1: 2, n2: 3, n3: 4 }]);
    assert.deepEqual(
        selectTriangle(bowtieAdj, bowtiePos, parseTriangleSelector(`(inter ${first} (${second}))`)), []);
    assert.deepEqual(
        selectTriangle(bowtieAdj, bowtiePos, parseTriangleSelector(`(compl ${first})`)), [{ n1: 2, n2: 3, n3: 4 }]);
});

test('fromna/fromne convert a node selector into a triangle selector', () => {
    // (deg eq 3) selects only node 0, which belongs to triangle [0,1,2] alone.
    assert.deepEqual(
        selectTriangle(bowtieAdj, bowtiePos, parseTriangleSelector('(fromne (deg eq 3))')),
        [{ n1: 0, n2: 1, n3: 2 }]);
    // fromna needs ALL 3 nodes of a triangle selected - a single node is never enough.
    assert.deepEqual(selectTriangle(bowtieAdj, bowtiePos, parseTriangleSelector('(fromna (deg eq 3))')), []);
    // All 5 non-pendant nodes selected -> fromna now selects both triangles (all of each one's own
    // 3 nodes are in {0,1,2,3,4}).
    assert.deepEqual(
        selectTriangle(bowtieAdj, bowtiePos, parseTriangleSelector('(fromna (deg gt 1))')),
        [{ n1: 0, n2: 1, n3: 2 }, { n1: 2, n2: 3, n3: 4 }]);
});

test('tona/tone convert a triangle selector back into a node selector, including the vacuous ' +
    'case for the pendant node that belongs to no triangle', () => {
    // Exactly triangle [0,1,2] selected (see fromne(deg eq 3) above).
    const selTri = '(fromne (deg eq 3))';
    // tone: nodes 0/1/2 each have that triangle as a containing one -> selected. Nodes 3/4 only
    // belong to [2,3,4], not selected -> excluded. Node 5 belongs to no triangle -> vacuously false.
    assert.deepEqual(
        selectNode(bowtieAdj, bowtiePos, parseNodeSelector(`(tone tri ${selTri})`)), new Set([0, 1, 2]));
    // tona: nodes 0/1 belong ONLY to [0,1,2] (selected) -> all of their containing triangles are
    // selected -> included. Node 2 belongs to BOTH triangles, and [2,3,4] isn't selected -> excluded.
    // Nodes 3/4 belong only to the non-selected [2,3,4] -> excluded. Node 5 belongs to zero
    // triangles -> vacuously true -> included.
    assert.deepEqual(
        selectNode(bowtieAdj, bowtiePos, parseNodeSelector(`(tona tri ${selTri})`)), new Set([0, 1, 5]));
});

// A single 4-cycle 0-1-2-3-0 (no diagonals - a genuine square), plus a pendant node 4 off node 0.
// Degrees: 0:3 (1,3,4), 1:2, 2:2, 3:2, 4:1 (belongs to no square). Exactly one square: [0,1,2,3].
const squareAdj = [
    [0, 1, 0, 1, 1],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 0],
    [1, 0, 0, 0, 0],
];
const squarePos = squareAdj.map((_, i) => [i]);

test('selectSquare finds the single induced 4-cycle via (all), and none in a square-free graph', () => {
    assert.deepEqual(
        selectSquare(squareAdj, squarePos, parseSquareSelector('(all)')),
        [{ n1: 0, n2: 1, n3: 2, n4: 3 }]);
    assert.deepEqual(selectSquare(bowtieAdj, bowtiePos, parseSquareSelector('(all)')), []);
});

test('fromna/fromne/tona/tone work the same way for squares as for triangles', () => {
    // fromne from the square's own corner nodes selects the square; fromna from just the pendant
    // does not (it isn't part of the square at all).
    assert.deepEqual(
        selectSquare(squareAdj, squarePos, parseSquareSelector('(fromne (deg eq 2))')),
        [{ n1: 0, n2: 1, n3: 2, n4: 3 }]);
    assert.deepEqual(selectSquare(squareAdj, squarePos, parseSquareSelector('(fromna (deg gt 2))')), []);

    // tona/tone from the (only, therefore trivially fully-selected) square: every corner is in it,
    // so both tona and tone select {0,1,2,3}; the pendant node 4 belongs to no square, so tona
    // includes it vacuously while tone excludes it.
    const selSq = '(all)';
    assert.deepEqual(
        selectNode(squareAdj, squarePos, parseNodeSelector(`(tona sq ${selSq})`)), new Set([0, 1, 2, 3, 4]));
    assert.deepEqual(
        selectNode(squareAdj, squarePos, parseNodeSelector(`(tone sq ${selSq})`)), new Set([0, 1, 2, 3]));
});

test('rrmn/rrmp reject a malformed count/portion argument', () => {
    assert.throws(() => parseNodeSelector('(rrmn -1 (all))'), /nonnegative integer/);
    assert.throws(() => parseNodeSelector('(rrmn 1.5 (all))'), /nonnegative integer/);
    assert.throws(() => parseNodeSelector('(rrmn abc (all))'), /nonnegative integer/);
    assert.throws(() => parseNodeSelector('(rrmp -0.5 (all))'), /nonnegative number/);
    assert.throws(() => parseNodeSelector('(rrmp abc (all))'), /nonnegative number/);
});

// rrmn/rrmp are randomized - these tests only check the deterministic invariants (result size, and
// that every kept item really was in the original set), not which specific items got removed.
test('rrmn removes exactly count items (clamped to the set size), chosen from the original set', () => {
    for (const count of [0, 1, 3, 4, 10]) {
        const kept = selectNode(adj, pos, parseNodeSelector(`(rrmn ${count} (all))`));
        assert.equal(kept.size, Math.max(4 - count, 0));
        for (const n of kept) assert.ok([0, 1, 2, 3].includes(n));
    }
});

test('rrmp removes floor(frac * size) items, clamped to the set size', () => {
    const cases: [number, number][] = [[0, 4], [0.25, 3], [0.5, 2], [1, 0], [1.5, 0]];
    for (const [frac, expectedSize] of cases) {
        const kept = selectNode(adj, pos, parseNodeSelector(`(rrmp ${frac} (all))`));
        assert.equal(kept.size, expectedSize, `frac=${frac}`);
        for (const n of kept) assert.ok([0, 1, 2, 3].includes(n));
    }
});

test('rrmn/rrmp also work over edge selectors', () => {
    const kept = selectEdge(adj, pos, parseEdgeSelector('(rrmn 1 (all))'));
    assert.equal(kept.length, 2);
    const all = selectEdge(adj, pos, parseEdgeSelector('(all)'));
    for (const e of kept) assert.ok(all.some(a => a.n1 === e.n1 && a.n2 === e.n2));

    const keptFrac = selectEdge(adj, pos, parseEdgeSelector('(rrmp 0.6666 (all))'));
    // floor(0.6666 * 3) = 1 removed, 2 kept.
    assert.equal(keptFrac.length, 2);
});

test('rrmn/rrmp also work over triangle/square selectors', () => {
    const keptTri = selectTriangle(bowtieAdj, bowtiePos, parseTriangleSelector('(rrmn 1 (all))'));
    assert.equal(keptTri.length, 1);
    const allTri = selectTriangle(bowtieAdj, bowtiePos, parseTriangleSelector('(all)'));
    for (const t of keptTri) assert.ok(allTri.some(a => a.n1 === t.n1 && a.n2 === t.n2 && a.n3 === t.n3));

    const keptSq = selectSquare(squareAdj, squarePos, parseSquareSelector('(rrmp 1 (all))'));
    assert.equal(keptSq.length, 0);
});

test('selectNode/selectEdge/selectTriangle/selectSquare throw when given a selector of the wrong kind', () => {
    const edgeSel = parseEdgeSelector('(fromna (deg eq 2))');
    assert.throws(() => selectNode(adj, pos, edgeSel), /expected a node selector, got an edge selector/);
    const nodeSel = parseNodeSelector('(deg eq 2)');
    assert.throws(() => selectEdge(adj, pos, nodeSel), /expected an edge selector, got a node selector/);
    assert.throws(() => selectTriangle(adj, pos, nodeSel), /expected a triangle selector, got a node selector/);
    assert.throws(() => selectSquare(adj, pos, nodeSel), /expected a square selector, got a node selector/);
});
