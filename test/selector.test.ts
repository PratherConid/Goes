// Covers shared/selector.ts's tiny S-expression selector language: parseNodeSelector()/
// parseEdgeSelector() (two mutually recursive parsers - see the file's own top comment) and
// selectNode()/selectEdge() (evaluation against a real adjacency matrix), on a plain 4-node path
// graph 0-1-2-3.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    parseNodeSelector, parseEdgeSelector, selectNode, selectEdge, formatSelector,
} from '../shared/selector.ts';

// 0-1-2-3: node 0/3 have degree 1, node 1/2 have degree 2. Edges (0,1), (1,2), (2,3).
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
    // e2n only exists in parseEdgeSelExpr - unrecognized inside a node-selector context, whether at
    // the top level or nested.
    assert.throws(() => parseNodeSelector('(e2n (deg eq 1))'), /unknown node-selector operator 'e2n'/);
    assert.throws(
        () => parseNodeSelector('(union (deg eq 1) (e2n (deg eq 1)))'),
        /unknown node-selector operator 'e2n'/,
    );
    // deg only exists in parseNodeSelExpr - unrecognized inside an edge-selector context.
    assert.throws(() => parseEdgeSelector('(deg eq 1)'), /unknown edge-selector operator 'deg'/);
});

test('e2n/n2e reject an operand of the wrong kind', () => {
    // (e2n (deg eq 1)) is edge-only - e2n's own operand is parsed via parseNodeSelExpr, which
    // doesn't recognize e2n.
    assert.throws(() => parseEdgeSelector('(e2n (e2n (deg eq 1)))'), /unknown node-selector operator 'e2n'/);
    // (deg eq 1) is node-only - n2e's own operand is parsed via parseEdgeSelExpr, which doesn't
    // recognize deg.
    assert.throws(() => parseNodeSelector('(n2e (deg eq 1))'), /unknown edge-selector operator 'deg'/);
});

test('all/none select every node/edge or none, resolved by which parser reaches them (no argument)', () => {
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector('(all)')), new Set([0, 1, 2, 3]));
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector('(none)')), new Set());
    assert.deepEqual(
        selectEdge(adj, pos, parseEdgeSelector('(all)')),
        [{ n1: 0, n2: 1 }, { n1: 1, n2: 2 }, { n1: 2, n2: 3 }]);
    assert.deepEqual(selectEdge(adj, pos, parseEdgeSelector('(none)')), []);

    // (all)/(none) take no argument at all now - anything extra before the closing paren is a
    // grammar error.
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

test('e2n selects edges whose both endpoints are selected', () => {
    // deg eq 2 selects {1, 2} - only edge (1,2) has both endpoints in that set.
    const edges = selectEdge(adj, pos, parseEdgeSelector('(e2n (deg eq 2))'));
    assert.deepEqual(edges, [{ n1: 1, n2: 2 }]);
});

test('n2e selects nodes that are an endpoint of some selected edge', () => {
    const nodes = selectNode(adj, pos, parseNodeSelector('(n2e (e2n (deg eq 2)))'));
    assert.deepEqual(nodes, new Set([1, 2]));
});

test('compl on an edge selector complements within all of the graph\'s edges', () => {
    // All edges: (0,1), (1,2), (2,3). e2n(deg eq 2) = {(1,2)}.
    const edges = selectEdge(adj, pos, parseEdgeSelector('(compl (e2n (deg eq 2)))'));
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
    // e2n(deg eq 2) selects the 3 triangle edges; more re-adds every edge incident to nodes 0/1/2 -
    // still just those same 3 edges, since edge (3,4) isn't incident to any of them.
    const edges = selectEdge(triPlusEdgeAdj, triPlusEdgePos, parseEdgeSelector('(more (e2n (deg eq 2)))'));
    assert.deepEqual(edges, [{ n1: 0, n2: 1 }, { n1: 0, n2: 2 }, { n1: 1, n2: 2 }]);
});

test('more on an already-all selector is a no-op, and formatSelector round-trips it', () => {
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector('(more (all))')), new Set([0, 1, 2, 3]));
    const sel = parseEdgeSelector('(more (e2n (deg eq 2)))');
    assert.equal(formatSelector(sel), '(more (e2n (deg eq 2)))');
});

test('union/inter/diff combine edge selectors as plain (deduplicated) set operations', () => {
    // (e2n (deg gt 0)) is every edge (every node has degree > 0); (e2n (deg eq 2)) is just (1,2).
    const all = '(e2n (deg gt 0))', middle = '(e2n (deg eq 2))';
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

test('selectNode/selectEdge throw when given a selector of the wrong kind', () => {
    const edgeSel = parseEdgeSelector('(e2n (deg eq 2))');
    assert.throws(() => selectNode(adj, pos, edgeSel), /expected a node selector, got an edge selector/);
    const nodeSel = parseNodeSelector('(deg eq 2)');
    assert.throws(() => selectEdge(adj, pos, nodeSel), /expected an edge selector, got a node selector/);
});
