// Covers shared/selector.ts's tiny S-expression selector language: parseNodeSelector()/
// parseEdgeSelector() (grammar + selector-kind validation) and selectNode()/selectEdge() (evaluation
// against a real adjacency matrix), on a plain 4-node path graph 0-1-2-3.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseNodeSelector, parseEdgeSelector, selectNode, selectEdge } from '../shared/selector.ts';

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
    assert.throws(() => parseNodeSelector('(foo)'), /unknown operator/);
    assert.throws(() => parseNodeSelector('(deg xx 5)'), /comparator must be/);
    assert.throws(() => parseNodeSelector('(deg eq -3)'), /nonnegative integer/);
    assert.throws(() => parseNodeSelector('(deg eq 3.5)'), /nonnegative integer/);
    assert.throws(() => parseNodeSelector('(deg eq abc)'), /nonnegative integer/);
});

test('parseNodeSelector/parseEdgeSelector reject a syntactically valid selector of the wrong kind', () => {
    // (e2n SEL) always denotes edges - wrong for parseNodeSelector.
    assert.throws(() => parseNodeSelector('(e2n (deg eq 1))'), /expected a node selector, got an edge selector/);
    // (deg ...) always denotes nodes - wrong for parseEdgeSelector.
    assert.throws(() => parseEdgeSelector('(deg eq 1)'), /expected an edge selector, got a node selector/);
});

test('union/inter/diff reject operands of different kinds', () => {
    assert.throws(
        () => parseNodeSelector('(union (deg eq 1) (e2n (deg eq 1)))'),
        /operands must be the same kind/,
    );
});

test('e2n/n2e reject an operand of the wrong kind', () => {
    // (e2n (deg eq 1)) is already edge-typed - e2n requires a node-typed operand.
    assert.throws(() => parseEdgeSelector('(e2n (e2n (deg eq 1)))'), /e2n SEL. requires a node selector/);
    // (deg eq 1) is node-typed - n2e requires an edge-typed operand.
    assert.throws(() => parseNodeSelector('(n2e (deg eq 1))'), /n2e SEL. requires an edge selector/);
});

test('all/none select every node/edge or none, per the explicit kind argument', () => {
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector('(all node)')), new Set([0, 1, 2, 3]));
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector('(none node)')), new Set());
    assert.deepEqual(
        selectEdge(adj, pos, parseEdgeSelector('(all edge)')),
        [{ n1: 0, n2: 1 }, { n1: 1, n2: 2 }, { n1: 2, n2: 3 }]);
    assert.deepEqual(selectEdge(adj, pos, parseEdgeSelector('(none edge)')), []);

    // (all)/(none) with no kind, or an unrecognized kind, are grammar errors.
    assert.throws(() => parseNodeSelector('(all)'), /kind must be .node. or .edge./);
    assert.throws(() => parseNodeSelector('(all edges)'), /kind must be .node. or .edge./);
    // (all node) is node-typed - wrong for parseEdgeSelector.
    assert.throws(() => parseEdgeSelector('(all node)'), /expected an edge selector, got a node selector/);
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

test('selectNode/selectEdge throw when given a selector of the wrong kind', () => {
    const edgeSel = parseEdgeSelector('(e2n (deg eq 2))');
    assert.throws(() => selectNode(adj, pos, edgeSel), /expected a node selector, got an edge selector/);
    const nodeSel = parseNodeSelector('(deg eq 2)');
    assert.throws(() => selectEdge(adj, pos, nodeSel), /expected an edge selector, got a node selector/);
});
