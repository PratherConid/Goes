// Covers shared/selector.ts's tiny S-expression selector language: parseNodeSelector()/
// parseEdgeSelector()/parseTriangleSelector()/parseQuadSelector() (four entry points into one
// context-free, bottom-up-type-inferring parser - see the file's own top comment) and
// selectNode()/selectEdge()/selectTriangle()/selectQuad() (evaluation against a real adjacency
// matrix).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    parseNodeSelector, parseEdgeSelector, parseTriangleSelector, parseQuadSelector, parseSelector,
    selectNode, selectEdge, selectTriangle, selectQuad, selectSimp, formatSelector,
} from '../shared/selector.ts';
import { simpType } from '../shared/types.ts';

// 0-1-2-3: node 0/3 have degree 1, node 1/2 have degree 2. Edges (0,1), (1,2), (2,3). No
// triangles/quads.
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
    assert.throws(() => parseNodeSelector('(foo)'), /unknown selector operator/);
    assert.throws(() => parseNodeSelector('(deg xx 5)'), /comparator must be/);
    assert.throws(() => parseNodeSelector('(deg eq -3)'), /nonnegative integer/);
    assert.throws(() => parseNodeSelector('(deg eq 3.5)'), /nonnegative integer/);
    assert.throws(() => parseNodeSelector('(deg eq abc)'), /nonnegative integer/);
});

test('parse*Selector reject a result of the wrong kind (deg is always node, whatever it\'s parsed as)', () => {
    // deg only ever produces a node-kind selector - parsing it as an edge/triangle/quad selector now
    // fails the post-hoc result-kind check, not because 'deg' itself was unrecognized.
    assert.throws(() => parseEdgeSelector('(deg eq 1)'), /expected an edge selector, got a node selector/);
    assert.throws(() => parseTriangleSelector('(deg eq 1)'), /expected a triangle selector, got a node selector/);
    assert.throws(() => parseQuadSelector('(deg eq 1)'), /expected a quad selector, got a node selector/);
    // e2n/n2e/fromna/fromne/tona/tone no longer exist (replaced by conva/conve) - unrecognized
    // wherever they appear.
    assert.throws(() => parseEdgeSelector('(e2n (deg eq 1))'), /unknown selector operator 'e2n'/);
    assert.throws(() => parseNodeSelector('(n2e (all node))'), /unknown selector operator 'n2e'/);
    assert.throws(() => parseEdgeSelector('(fromna (all node))'), /unknown selector operator 'fromna'/);
    assert.throws(() => parseNodeSelector('(tona edge (all node))'), /unknown selector operator 'tona'/);
});

test('conva/conve require a valid node|edge|simp N|tri|quad result token, and reject simp <-> quad', () => {
    assert.throws(() => parseNodeSelector('(conva (all node))'), /result kind must be 'node', 'edge', 'simp <n>', 'tri', or 'quad'/);
    assert.throws(() => parseNodeSelector('(conva nope (all node))'), /result kind must be 'node', 'edge', 'simp <n>', 'tri', or 'quad'/);
    assert.throws(() => parseTriangleSelector('(conva tri (all quad))'), /no association defined between 'simp' and 'quad'/);
    assert.throws(() => parseQuadSelector('(conve quad (all tri))'), /no association defined between 'simp' and 'quad'/);
    assert.throws(() => parseQuadSelector('(conve quad (all simp 3))'), /no association defined between 'simp' and 'quad'/);
    // conva/conve's own operand is parsed context-free, same as anywhere else - (deg ...) always
    // produces a node selector, so declaring conva's own result kind 'edge' is fine (that's the
    // whole point of conva/conve), but the operand itself is unaffected by that declaration.
    assert.deepEqual(
        parseEdgeSelector('(conva edge (deg eq 1))'),
        { op: 'conva', type: 'edge', from: 'node', a: { op: 'deg', type: 'node', cmp: 'eq', n: 1 } });
});

test('converting a kind to itself is a no-op - conva/conve don\'t even appear in the parsed tree', () => {
    const nodeSel = parseNodeSelector('(deg eq 2)');
    assert.deepEqual(parseNodeSelector('(conva node (deg eq 2))'), nodeSel);
    assert.deepEqual(parseNodeSelector('(conve node (deg eq 2))'), nodeSel);
    const edgeSel = parseEdgeSelector('(all edge)');
    assert.deepEqual(parseEdgeSelector('(conva edge (all edge))'), edgeSel);
});

test('conva/conve reject simp<->quad at evaluation time too (defensive, for a hand-built Selector)', () => {
    const handBuilt = {
        op: 'conva' as const, type: 'quad' as const, from: simpType(2),
        a: { op: 'all' as const, type: simpType(2) },
    };
    assert.throws(
        () => selectQuad(adj, pos, handBuilt), /no association is defined between 'simp' and 'quad'/);
});

test('all/none select every object of the given kind, and reject a missing/invalid kind token', () => {
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector('(all node)')), new Set([0, 1, 2, 3]));
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector('(none node)')), new Set());
    assert.deepEqual(
        selectEdge(adj, pos, parseEdgeSelector('(all edge)')),
        [{ n1: 0, n2: 1 }, { n1: 1, n2: 2 }, { n1: 2, n2: 3 }]);
    assert.deepEqual(selectEdge(adj, pos, parseEdgeSelector('(none edge)')), []);
    // No triangles/quads in this path graph.
    assert.deepEqual(selectTriangle(adj, pos, parseTriangleSelector('(all tri)')), []);
    assert.deepEqual(selectQuad(adj, pos, parseQuadSelector('(all quad)')), []);

    // (all)/(none) now require exactly one kind token - a missing or unrecognized one is a grammar
    // error; a bare (all)/(none) (the old, context-inferred syntax) no longer parses.
    assert.throws(() => parseNodeSelector('(all)'), /kind must be 'node', 'edge', 'simp <n>', 'tri', or 'quad', got '\)'/);
    assert.throws(() => parseNodeSelector('(all nope)'), /kind must be 'node', 'edge', 'simp <n>', 'tri', or 'quad'/);
    assert.throws(() => parseNodeSelector('(all node edge)'), /expected '\)', got 'edge'/);
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

test('union/inter reject zero operands (their kind can\'t be inferred bottom-up) but otherwise ' +
    'take a variadic list: one, and three-or-more', () => {
    // Zero operands: no longer valid syntax - (all <kind>)/(none <kind>) cover the identity cases
    // directly now that they name their own kind explicitly.
    assert.throws(() => parseNodeSelector('(union)'), /needs at least one operand/);
    assert.throws(() => parseNodeSelector('(inter)'), /needs at least one operand/);
    assert.throws(() => parseEdgeSelector('(union)'), /needs at least one operand/);
    assert.throws(() => parseEdgeSelector('(inter)'), /needs at least one operand/);

    // One operand is a plain pass-through for both.
    assert.deepEqual(
        selectNode(adj, pos, parseNodeSelector('(union (deg eq 2))')), new Set([1, 2]));
    assert.deepEqual(
        selectNode(adj, pos, parseNodeSelector('(inter (deg eq 2))')), new Set([1, 2]));

    // Three-plus operands: union of {0}, {1}, {2} is {0,1,2}; a three-way inter that has no common
    // element across all three is empty even though every pair overlaps.
    assert.deepEqual(
        selectNode(adj, pos, parseNodeSelector('(union (deg eq 1) (deg eq 2) (deg eq 1))')),
        new Set([0, 1, 2, 3]));
    assert.deepEqual(
        selectNode(adj, pos, parseNodeSelector(
            '(inter (deg lt 2) (deg gt 0) (deg eq 3))')), // {0,3} inter {1,2} inter {} = {}
        new Set());
});

test('union/inter/diff reject operands of mismatched kinds', () => {
    assert.throws(
        () => parseNodeSelector('(union (deg eq 1) (all edge))'),
        /operands must all be the same kind - operand 1 is node, operand 2 is edge/);
    assert.throws(
        () => parseNodeSelector('(inter (deg eq 1) (all edge) (all node))'),
        /operands must all be the same kind - operand 1 is node, operand 2 is edge/);
    assert.throws(
        () => parseNodeSelector('(diff (deg eq 1) (all edge))'),
        /operands must be the same kind - got node and edge/);
});

test('formatSelector round-trips union/inter at every arity, including one', () => {
    assert.equal(formatSelector(parseNodeSelector('(union (deg eq 1))')), '(union (deg eq 1))');
    assert.equal(
        formatSelector(parseNodeSelector('(union (deg eq 1) (deg eq 2) (deg eq 3))')),
        '(union (deg eq 1) (deg eq 2) (deg eq 3))');
});

test('conva(node) selects edges whose nodes are ALL selected, conve(node) whose nodes have AT ' +
    'LEAST ONE selected', () => {
    // deg eq 2 selects {1, 2} - only edge (1,2) has both endpoints in that set (conva); edges (0,1)
    // and (1,2) each have at least one endpoint in it (conve).
    assert.deepEqual(selectEdge(adj, pos, parseEdgeSelector('(conva edge (deg eq 2))')), [{ n1: 1, n2: 2 }]);
    assert.deepEqual(
        selectEdge(adj, pos, parseEdgeSelector('(conve edge (deg eq 2))')),
        [{ n1: 0, n2: 1 }, { n1: 1, n2: 2 }, { n1: 2, n2: 3 }]);
});

test('conva(edge) selects nodes whose every associated edge is selected, conve(edge) whose any is', () => {
    // (conva edge (deg eq 2)) = {(1,2)} only.
    const selEdges = '(conva edge (deg eq 2))';
    // conve: node 0's only edge is (0,1), not selected -> excluded; node 1's edges are (0,1) [not
    // selected] and (1,2) [selected] -> at least one selected -> included. Node 2 symmetric to 1.
    // Node 3's only edge (2,3) isn't selected -> excluded.
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector(`(conve node ${selEdges})`)), new Set([1, 2]));
    // conva: node 1 has TWO associated edges, (0,1) and (1,2) - only one is selected, so not ALL are
    // -> excluded. Same for node 2. Nodes 0/3 each have exactly one associated edge, not selected ->
    // excluded too. Nothing in this graph has every associated edge selected.
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector(`(conva node ${selEdges})`)), new Set());
});

test('conva is vacuously true for a node with no associated objects of the given kind', () => {
    // No triangles at all in this graph - every node "vacuously" satisfies conva (every one of its
    // zero associated triangles is trivially selected), but conve (at least one) is vacuously false.
    assert.deepEqual(
        selectNode(adj, pos, parseNodeSelector('(conva node (all tri))')), new Set([0, 1, 2, 3]));
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector('(conve node (all tri))')), new Set());
});

test('convlt/conveq/convgt require a valid node|edge|simp N|tri|quad result token, reject simp ' +
    '<-> quad, and require a nonnegative integer count', () => {
    assert.throws(() => parseNodeSelector('(convlt (all node))'), /result kind must be 'node', 'edge', 'simp <n>', 'tri', or 'quad'/);
    assert.throws(() => parseTriangleSelector('(convgt tri 0 (all quad))'), /no association defined between 'simp' and 'quad'/);
    assert.throws(() => parseQuadSelector('(convceq quad 0 (all tri))'), /no association defined between 'simp' and 'quad'/);
    assert.throws(() => parseNodeSelector('(convlt node -1 (all edge))'), /nonnegative integer/);
    assert.throws(() => parseNodeSelector('(convlt node 1.5 (all edge))'), /nonnegative integer/);
    assert.deepEqual(
        parseEdgeSelector('(convgt edge 1 (deg eq 1))'),
        { op: 'convgt', type: 'edge', from: 'node', n: 1, a: { op: 'deg', type: 'node', cmp: 'eq', n: 1 } });
});

test('convlt/conveq/convgt count exactly how many associated selected objects a "to" object has - ' +
    'unlike conva/conve, converting a kind to itself is NOT a no-op', () => {
    // (conva edge (deg eq 2)) = {(1,2)} only (see the conva/conve tests above). Associated edges per
    // node: node 0 -> {(0,1)} (0 selected); node 1 -> {(0,1),(1,2)} (1 selected); node 2 ->
    // {(1,2),(2,3)} (1 selected); node 3 -> {(2,3)} (0 selected).
    const selEdges = '(conva edge (deg eq 2))';
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector(`(convlt node 1 ${selEdges})`)), new Set([0, 3]));
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector(`(conveq node 1 ${selEdges})`)), new Set([1, 2]));
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector(`(convgt node 0 ${selEdges})`)), new Set([1, 2]));
    // Same-kind (node -> node): a node's only associated node is itself, so this reduces to "is n
    // selected" compared against a threshold - convlt(1) is the actual selected set's own complement,
    // NOT the identity the way conva/conve's own same-kind shortcut would suggest.
    const degSel = '(deg eq 2)'; // selects {1, 2}
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector(`(convlt node 1 ${degSel})`)), new Set([0, 3]));
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector(`(convgt node 0 ${degSel})`)), new Set([1, 2]));
});

test('convclt/convceq/convcgt count associated objects that are NOT selected instead', () => {
    // (conve edge (deg eq 1)) = {(0,1), (2,3)} (all edges except (1,2)). NOT-selected-associated
    // counts per node: node 0 -> {(0,1)} all selected -> 0 not-selected; node 1 -> {(0,1),(1,2)} ->
    // (1,2) not selected -> 1; node 2 -> {(1,2),(2,3)} -> (1,2) not selected -> 1; node 3 ->
    // {(2,3)} all selected -> 0.
    const selEdges = '(conve edge (deg eq 1))';
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector(`(convclt node 1 ${selEdges})`)), new Set([0, 3]));
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector(`(convceq node 1 ${selEdges})`)), new Set([1, 2]));
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector(`(convcgt node 0 ${selEdges})`)), new Set([1, 2]));
});

test('(conva K SEL) is exactly (convceq K 0 SEL), and (conve K SEL) is exactly (convgt K 0 SEL)', () => {
    for (const sel of ['(none edge)', '(all edge)', '(conva edge (deg eq 2))']) {
        assert.deepEqual(
            selectNode(adj, pos, parseNodeSelector(`(conva node ${sel})`)),
            selectNode(adj, pos, parseNodeSelector(`(convceq node 0 ${sel})`)));
        assert.deepEqual(
            selectNode(adj, pos, parseNodeSelector(`(conve node ${sel})`)),
            selectNode(adj, pos, parseNodeSelector(`(convgt node 0 ${sel})`)));
    }
});

test('convlt/convclt are vacuously true (and convgt/convcgt vacuously false) for a node with no ' +
    'associated objects of the given kind', () => {
    // No triangles at all in this graph - 0 associated triangles for every node either way.
    assert.deepEqual(
        selectNode(adj, pos, parseNodeSelector('(convlt node 1 (all tri))')), new Set([0, 1, 2, 3]));
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector('(convgt node 0 (all tri))')), new Set());
    assert.deepEqual(
        selectNode(adj, pos, parseNodeSelector('(convclt node 1 (all tri))')), new Set([0, 1, 2, 3]));
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector('(convcgt node 0 (all tri))')), new Set());
});

test('convlt/.../convcgt round-trip through formatSelector', () => {
    const text = '(convlt node 2 (conva edge (deg eq 2)))';
    const sel = parseNodeSelector(text);
    assert.equal(formatSelector(sel), text);
});

test('compl on an edge selector complements within all of the graph\'s edges', () => {
    // All edges: (0,1), (1,2), (2,3). conva(edge, deg eq 2) = {(1,2)}.
    const edges = selectEdge(adj, pos, parseEdgeSelector('(compl (conva edge (deg eq 2)))'));
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
    // conva(edge, deg eq 2) selects the 3 triangle edges; more re-adds every edge incident to nodes
    // 0/1/2 - still just those same 3 edges, since edge (3,4) isn't incident to any of them.
    const edges = selectEdge(triPlusEdgeAdj, triPlusEdgePos, parseEdgeSelector('(more (conva edge (deg eq 2)))'));
    assert.deepEqual(edges, [{ n1: 0, n2: 1 }, { n1: 0, n2: 2 }, { n1: 1, n2: 2 }]);
});

test('more is rejected for triangle/quad selectors (no adjacency notion is defined for them)', () => {
    assert.throws(() => parseTriangleSelector('(more (all tri))'), /requires a node or edge selector, got a triangle selector/);
    assert.throws(() => parseQuadSelector('(more (all quad))'), /requires a node or edge selector, got a quad selector/);
});

test('more on an already-all selector is a no-op, and formatSelector round-trips it', () => {
    assert.deepEqual(selectNode(adj, pos, parseNodeSelector('(more (all node))')), new Set([0, 1, 2, 3]));
    const sel = parseEdgeSelector('(more (conva edge (deg eq 2)))');
    assert.equal(formatSelector(sel), '(more (conva edge (deg eq 2)))');
});

test('more takes an optional leading step count, repeating the one-step expansion that many times', () => {
    // Same 5-node path as above: 0-1-2-3-4.
    const path5Adj = [
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
    ];
    const path5Pos = [[0], [1], [2], [3], [4]];
    // (deg eq 1) selects the endpoints {0, 4}. 1 step (the default, already covered above) reaches
    // {0,1,3,4}; 2 steps also reaches node 2 (two edges from either endpoint) - the whole path.
    assert.deepEqual(
        selectNode(path5Adj, path5Pos, parseNodeSelector('(more 2 (deg eq 1))')), new Set([0, 1, 2, 3, 4]));
    // 0 steps is a no-op - identical to the un-expanded selector.
    assert.deepEqual(
        selectNode(path5Adj, path5Pos, parseNodeSelector('(more 0 (deg eq 1))')), new Set([0, 4]));
    // A step count far beyond the graph's diameter still terminates (the frontier empties out) and
    // simply saturates at every reachable node.
    assert.deepEqual(
        selectNode(path5Adj, path5Pos, parseNodeSelector('(more 100 (deg eq 1))')), new Set([0, 1, 2, 3, 4]));
});

test('more\'s optional step count also works over edge selectors, and rejects a malformed count', () => {
    // A 6-node path 0-1-2-3-4-5: only the two endpoints 0/5 have degree 1.
    const path6Adj = [
        [0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0],
    ];
    const path6Pos = path6Adj.map((_, i) => [i]);
    // conve(node, deg eq 1) selects every edge touching either endpoint: (0,1) and (4,5) only. 1 step
    // adds every edge touching THOSE edges' own nodes - (1,2) and (3,4) - but not the middle edge
    // (2,3), which is 2 hops from either endpoint; a 2nd step reaches it. This is what proves the
    // step count actually chains the expansion rather than saturating after a single hop.
    const oneStep = selectEdge(path6Adj, path6Pos, parseEdgeSelector('(more (conve edge (deg eq 1)))'));
    assert.deepEqual(oneStep, [{ n1: 0, n2: 1 }, { n1: 4, n2: 5 }, { n1: 1, n2: 2 }, { n1: 3, n2: 4 }]);
    const twoSteps = selectEdge(path6Adj, path6Pos, parseEdgeSelector('(more 2 (conve edge (deg eq 1)))'));
    assert.deepEqual(twoSteps, [
        { n1: 0, n2: 1 }, { n1: 4, n2: 5 }, { n1: 1, n2: 2 }, { n1: 3, n2: 4 }, { n1: 2, n2: 3 },
    ]);
    // formatSelector round-trips an explicit step count exactly as written.
    assert.equal(formatSelector(parseEdgeSelector('(more 2 (all edge))')), '(more 2 (all edge))');

    assert.throws(() => parseNodeSelector('(more -1 (all node))'), /nonnegative integer/);
    assert.throws(() => parseNodeSelector('(more abc (all node))'), /nonnegative integer/);
});

test('union/inter/diff combine edge selectors as plain (deduplicated) set operations', () => {
    // (conva edge (deg gt 0)) is every edge (every node has degree > 0); (conva edge (deg eq 2)) is
    // just (1,2).
    const all = '(conva edge (deg gt 0))', middle = '(conva edge (deg eq 2))';
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
// Triangles (see findSimplices's own increasing-order convention): [0,1,2] and [2,3,4]. No quads.
const bowtieAdj = [
    [0, 1, 1, 0, 0, 1],
    [1, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 1, 0],
    [0, 0, 1, 0, 1, 0],
    [0, 0, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 0],
];
const bowtiePos = bowtieAdj.map((_, i) => [i]);

test('selectTriangle finds every triangle via (all tri), and none in a triangle-free graph', () => {
    assert.deepEqual(
        selectTriangle(bowtieAdj, bowtiePos, parseTriangleSelector('(all tri)')),
        [{ nodes: [0, 1, 2] }, { nodes: [2, 3, 4] }]);
    assert.deepEqual(selectTriangle(adj, pos, parseTriangleSelector('(all tri)')), []);
});

test('union/inter/diff/compl combine triangle selectors as plain (deduplicated) set operations', () => {
    // Only node 0 has degree 3 - conve(node, deg eq 3) selects exactly the triangle containing it,
    // [0,1,2].
    const first = '(conve tri (deg eq 3))';
    const second = 'diff (all tri) ' + first; // the other triangle, [2,3,4]
    assert.deepEqual(
        selectTriangle(bowtieAdj, bowtiePos, parseTriangleSelector(first)), [{ nodes: [0, 1, 2] }]);
    assert.deepEqual(
        selectTriangle(bowtieAdj, bowtiePos, parseTriangleSelector(`(${second})`)), [{ nodes: [2, 3, 4] }]);
    assert.deepEqual(
        selectTriangle(bowtieAdj, bowtiePos, parseTriangleSelector(`(union ${first} (${second}))`)),
        [{ nodes: [0, 1, 2] }, { nodes: [2, 3, 4] }]);
    assert.deepEqual(
        selectTriangle(bowtieAdj, bowtiePos, parseTriangleSelector(`(inter ${first} (${second}))`)), []);
    assert.deepEqual(
        selectTriangle(bowtieAdj, bowtiePos, parseTriangleSelector(`(compl ${first})`)), [{ nodes: [2, 3, 4] }]);
    // union/inter are variadic for triangles/quads too, same as node/edge above.
    assert.deepEqual(
        selectTriangle(bowtieAdj, bowtiePos, parseTriangleSelector(`(union ${first} (${second}) ${first})`)),
        [{ nodes: [0, 1, 2] }, { nodes: [2, 3, 4] }]);
});

test('conva/conve convert a node selector into a triangle selector', () => {
    // (deg eq 3) selects only node 0, which belongs to triangle [0,1,2] alone.
    assert.deepEqual(
        selectTriangle(bowtieAdj, bowtiePos, parseTriangleSelector('(conve tri (deg eq 3))')),
        [{ nodes: [0, 1, 2] }]);
    // conva needs ALL 3 nodes of a triangle selected - a single node is never enough.
    assert.deepEqual(selectTriangle(bowtieAdj, bowtiePos, parseTriangleSelector('(conva tri (deg eq 3))')), []);
    // All 5 non-pendant nodes selected -> conva now selects both triangles (all of each one's own
    // 3 nodes are in {0,1,2,3,4}).
    assert.deepEqual(
        selectTriangle(bowtieAdj, bowtiePos, parseTriangleSelector('(conva tri (deg gt 1))')),
        [{ nodes: [0, 1, 2] }, { nodes: [2, 3, 4] }]);
});

test('conva/conve convert an edge selector into a triangle selector (a new cross-type conversion)', () => {
    // Edges touching node 0: (0,1), (0,2), (0,5) - via conve(edge, deg eq 3), since only node 0 has
    // degree 3.
    const someEdges = '(conve edge (deg eq 3))';
    // Triangle [0,1,2]'s own edges are (0,1), (0,2), (1,2) - only 2 of those 3 are in the selected
    // set, so conva (ALL) excludes it; conve (SOME) includes it. Triangle [2,3,4]'s edges share none
    // with the selected set, so neither op includes it.
    assert.deepEqual(selectTriangle(bowtieAdj, bowtiePos, parseTriangleSelector(`(conva tri ${someEdges})`)), []);
    assert.deepEqual(
        selectTriangle(bowtieAdj, bowtiePos, parseTriangleSelector(`(conve tri ${someEdges})`)),
        [{ nodes: [0, 1, 2] }]);
});

test('conva/conve convert a triangle selector back into a node selector, including the vacuous ' +
    'case for the pendant node that belongs to no triangle', () => {
    // Exactly triangle [0,1,2] selected (see conve(tri, deg eq 3) above).
    const selTri = '(conve tri (deg eq 3))';
    // conve: nodes 0/1/2 each have that triangle as an associated one -> selected. Nodes 3/4 only
    // belong to [2,3,4], not selected -> excluded. Node 5 belongs to no triangle -> vacuously false.
    assert.deepEqual(
        selectNode(bowtieAdj, bowtiePos, parseNodeSelector(`(conve node ${selTri})`)), new Set([0, 1, 2]));
    // conva: nodes 0/1 belong ONLY to [0,1,2] (selected) -> all of their associated triangles are
    // selected -> included. Node 2 belongs to BOTH triangles, and [2,3,4] isn't selected -> excluded.
    // Nodes 3/4 belong only to the non-selected [2,3,4] -> excluded. Node 5 belongs to zero
    // triangles -> vacuously true -> included.
    assert.deepEqual(
        selectNode(bowtieAdj, bowtiePos, parseNodeSelector(`(conva node ${selTri})`)), new Set([0, 1, 5]));
});

// A single 4-cycle 0-1-2-3-0 (no diagonals - a genuine quad), plus a pendant node 4 off node 0.
// Degrees: 0:3 (1,3,4), 1:2, 2:2, 3:2, 4:1 (belongs to no quad). Exactly one quad: [0,1,2,3].
const quadAdj = [
    [0, 1, 0, 1, 1],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 0],
    [1, 0, 0, 0, 0],
];
const quadPos = quadAdj.map((_, i) => [i]);

test('selectQuad finds the single induced 4-cycle via (all quad), and none in a quad-free graph', () => {
    assert.deepEqual(
        selectQuad(quadAdj, quadPos, parseQuadSelector('(all quad)')),
        [{ n1: 0, n2: 1, n3: 2, n4: 3 }]);
    assert.deepEqual(selectQuad(bowtieAdj, bowtiePos, parseQuadSelector('(all quad)')), []);
});

test('conva/conve work the same way for quads as for triangles, including edge -> quad', () => {
    // conve from the quad's own corner nodes selects the quad; conva from just the pendant does
    // not (it isn't part of the quad at all).
    assert.deepEqual(
        selectQuad(quadAdj, quadPos, parseQuadSelector('(conve quad (deg eq 2))')),
        [{ n1: 0, n2: 1, n3: 2, n4: 3 }]);
    assert.deepEqual(selectQuad(quadAdj, quadPos, parseQuadSelector('(conva quad (deg gt 2))')), []);

    // conva/conve from the (only, therefore trivially fully-selected) quad: every corner is in it,
    // so both conva and conve select {0,1,2,3}; the pendant node 4 belongs to no quad, so conva
    // includes it vacuously while conve excludes it.
    const selQuad = '(all quad)';
    assert.deepEqual(
        selectNode(quadAdj, quadPos, parseNodeSelector(`(conva node ${selQuad})`)), new Set([0, 1, 2, 3, 4]));
    assert.deepEqual(
        selectNode(quadAdj, quadPos, parseNodeSelector(`(conve node ${selQuad})`)), new Set([0, 1, 2, 3]));

    // Edge -> quad (a new cross-type conversion, mirroring edge -> triangle above): edges touching
    // node 0 are (0,1), (0,3), (0,4) (only node 0 has degree 3) - only 2 of the quad's own 4 edges
    // ((0,1) and (0,3)), so conva (ALL) excludes it; conve (SOME) includes it.
    const someEdges = '(conve edge (deg eq 3))';
    assert.deepEqual(selectQuad(quadAdj, quadPos, parseQuadSelector(`(conva quad ${someEdges})`)), []);
    assert.deepEqual(
        selectQuad(quadAdj, quadPos, parseQuadSelector(`(conve quad ${someEdges})`)),
        [{ n1: 0, n2: 1, n3: 2, n4: 3 }]);
});

test('rrmn/rrmp reject a malformed count/portion argument', () => {
    assert.throws(() => parseNodeSelector('(rrmn -1 (all node))'), /nonnegative integer/);
    assert.throws(() => parseNodeSelector('(rrmn 1.5 (all node))'), /nonnegative integer/);
    assert.throws(() => parseNodeSelector('(rrmn abc (all node))'), /nonnegative integer/);
    assert.throws(() => parseNodeSelector('(rrmp -0.5 (all node))'), /nonnegative number/);
    assert.throws(() => parseNodeSelector('(rrmp abc (all node))'), /nonnegative number/);
});

// rrmn/rrmp are randomized - these tests only check the deterministic invariants (result size, and
// that every kept item really was in the original set), not which specific items got removed.
test('rrmn removes exactly count items (clamped to the set size), chosen from the original set', () => {
    for (const count of [0, 1, 3, 4, 10]) {
        const kept = selectNode(adj, pos, parseNodeSelector(`(rrmn ${count} (all node))`));
        assert.equal(kept.size, Math.max(4 - count, 0));
        for (const n of kept) assert.ok([0, 1, 2, 3].includes(n));
    }
});

test('rrmp removes floor(frac * size) items, clamped to the set size', () => {
    const cases: [number, number][] = [[0, 4], [0.25, 3], [0.5, 2], [1, 0], [1.5, 0]];
    for (const [frac, expectedSize] of cases) {
        const kept = selectNode(adj, pos, parseNodeSelector(`(rrmp ${frac} (all node))`));
        assert.equal(kept.size, expectedSize, `frac=${frac}`);
        for (const n of kept) assert.ok([0, 1, 2, 3].includes(n));
    }
});

test('rrmn/rrmp also work over edge selectors', () => {
    const kept = selectEdge(adj, pos, parseEdgeSelector('(rrmn 1 (all edge))'));
    assert.equal(kept.length, 2);
    const all = selectEdge(adj, pos, parseEdgeSelector('(all edge)'));
    for (const e of kept) assert.ok(all.some(a => a.n1 === e.n1 && a.n2 === e.n2));

    const keptFrac = selectEdge(adj, pos, parseEdgeSelector('(rrmp 0.6666 (all edge))'));
    // floor(0.6666 * 3) = 1 removed, 2 kept.
    assert.equal(keptFrac.length, 2);
});

test('rrmn/rrmp also work over triangle/quad selectors', () => {
    const keptTri = selectTriangle(bowtieAdj, bowtiePos, parseTriangleSelector('(rrmn 1 (all tri))'));
    assert.equal(keptTri.length, 1);
    const allTri = selectTriangle(bowtieAdj, bowtiePos, parseTriangleSelector('(all tri)'));
    for (const t of keptTri) assert.ok(allTri.some(a => a.nodes.join(',') === t.nodes.join(',')));

    const keptQuad = selectQuad(quadAdj, quadPos, parseQuadSelector('(rrmp 1 (all quad))'));
    assert.equal(keptQuad.length, 0);
});

test('rpkn/rpkp reject a malformed count/portion argument', () => {
    assert.throws(() => parseNodeSelector('(rpkn -1 (all node))'), /nonnegative integer/);
    assert.throws(() => parseNodeSelector('(rpkn 1.5 (all node))'), /nonnegative integer/);
    assert.throws(() => parseNodeSelector('(rpkn abc (all node))'), /nonnegative integer/);
    assert.throws(() => parseNodeSelector('(rpkp -0.5 (all node))'), /nonnegative number/);
    assert.throws(() => parseNodeSelector('(rpkp abc (all node))'), /nonnegative number/);
});

// rpkn/rpkp are randomized - these tests only check the deterministic invariants (result size, and
// that every kept item really was in the original set), not which specific items got picked.
test('rpkn picks exactly count items (clamped to the set size), chosen from the original set', () => {
    for (const count of [0, 1, 3, 4, 10]) {
        const kept = selectNode(adj, pos, parseNodeSelector(`(rpkn ${count} (all node))`));
        assert.equal(kept.size, Math.min(count, 4));
        for (const n of kept) assert.ok([0, 1, 2, 3].includes(n));
    }
});

test('rpkp picks floor(frac * size) items, clamped to the set size', () => {
    const cases: [number, number][] = [[0, 0], [0.25, 1], [0.5, 2], [1, 4], [1.5, 4]];
    for (const [frac, expectedSize] of cases) {
        const kept = selectNode(adj, pos, parseNodeSelector(`(rpkp ${frac} (all node))`));
        assert.equal(kept.size, expectedSize, `frac=${frac}`);
        for (const n of kept) assert.ok([0, 1, 2, 3].includes(n));
    }
});

test('rpkn/rpkp also work over edge selectors', () => {
    const kept = selectEdge(adj, pos, parseEdgeSelector('(rpkn 2 (all edge))'));
    assert.equal(kept.length, 2);
    const all = selectEdge(adj, pos, parseEdgeSelector('(all edge)'));
    for (const e of kept) assert.ok(all.some(a => a.n1 === e.n1 && a.n2 === e.n2));

    const keptFrac = selectEdge(adj, pos, parseEdgeSelector('(rpkp 0.6666 (all edge))'));
    // floor(0.6666 * 3) = 1 picked.
    assert.equal(keptFrac.length, 1);
});

test('rpkn/rpkp also work over triangle/quad selectors', () => {
    const keptTri = selectTriangle(bowtieAdj, bowtiePos, parseTriangleSelector('(rpkn 1 (all tri))'));
    assert.equal(keptTri.length, 1);
    const allTri = selectTriangle(bowtieAdj, bowtiePos, parseTriangleSelector('(all tri)'));
    for (const t of keptTri) assert.ok(allTri.some(a => a.nodes.join(',') === t.nodes.join(',')));

    const keptQuad = selectQuad(quadAdj, quadPos, parseQuadSelector('(rpkp 0 (all quad))'));
    assert.equal(keptQuad.length, 0);
});

test('selectNode/selectEdge/selectTriangle/selectQuad throw when given a selector of the wrong kind', () => {
    const edgeSel = parseEdgeSelector('(conva edge (deg eq 2))');
    assert.throws(() => selectNode(adj, pos, edgeSel), /expected a node selector, got an edge selector/);
    const nodeSel = parseNodeSelector('(deg eq 2)');
    assert.throws(() => selectEdge(adj, pos, nodeSel), /expected an edge selector, got a node selector/);
    assert.throws(() => selectTriangle(adj, pos, nodeSel), /expected a triangle selector, got a node selector/);
    assert.throws(() => selectQuad(adj, pos, nodeSel), /expected a quad selector, got a node selector/);
});

// K4 on {0,1,2,3} (every pair adjacent - the graph's only simp-3 object, [0,1,2,3]) plus a pendant
// node 4 attached only to node 0 (degree 1, belongs to no simp-2 or simp-3 object). The K4's own
// 4 simp-2 (triangle) sub-faces are every 3-subset of {0,1,2,3}: [0,1,2], [0,1,3], [0,2,3], [1,2,3].
const k4Adj = [
    [0, 1, 1, 1, 1],
    [1, 0, 1, 1, 0],
    [1, 1, 0, 1, 0],
    [1, 1, 1, 0, 0],
    [1, 0, 0, 0, 0],
];
const k4Pos = k4Adj.map((_, i) => [i]);

test('(all tri) and (all simp 2) parse to the identical Selector and evaluate identically', () => {
    // The whole point of "tri" being sugar for "simp 2" - same AST, not just the same result.
    assert.deepEqual(parseTriangleSelector('(all tri)'), parseTriangleSelector('(all simp 2)'));
    assert.deepEqual(
        selectTriangle(bowtieAdj, bowtiePos, parseTriangleSelector('(all tri)')),
        selectTriangle(bowtieAdj, bowtiePos, parseTriangleSelector('(all simp 2)')));
});

test('selectSimp finds every simp N (N > 2) via (all simp N), generalizing beyond triangles', () => {
    const sel3 = parseSelector('(all simp 3)');
    assert.deepEqual(selectSimp(k4Adj, k4Pos, sel3), [{ nodes: [0, 1, 2, 3] }]);
    // No 4-cliques at all in the triangle-free path graph, or in the bowtie (whose only cliques are
    // its own two triangles).
    assert.deepEqual(selectSimp(adj, pos, sel3), []);
    assert.deepEqual(selectSimp(bowtieAdj, bowtiePos, sel3), []);
    // simp 2 on the K4 finds all 4 triangular sub-faces.
    const sel2 = parseSelector('(all simp 2)');
    assert.deepEqual(
        selectSimp(k4Adj, k4Pos, sel2),
        [{ nodes: [0, 1, 2] }, { nodes: [0, 1, 3] }, { nodes: [0, 2, 3] }, { nodes: [1, 2, 3] }]);
});

test('conva/conve convert simp M <-> simp N (M != N) via the same general containment rule', () => {
    // conve(simp 3, all simp 2): a simp-3 object is selected iff AT LEAST ONE of its simp-2
    // sub-faces is selected - all 4 triangles are selected (all simp 2), so the one simp-3 [0,1,2,3]
    // qualifies.
    const conveSel = parseSelector('(conve simp 3 (all simp 2))');
    assert.deepEqual(selectSimp(k4Adj, k4Pos, conveSel), [{ nodes: [0, 1, 2, 3] }]);
    // conva(simp 2, all simp 3): a simp-2 (triangle) is selected iff ALL of its associated simp-3
    // objects are selected - every triangle here is contained in the one simp-3 [0,1,2,3], which is
    // itself selected (all simp 3), so all 4 triangles qualify.
    const convaSel = parseSelector('(conva simp 2 (all simp 3))');
    assert.deepEqual(
        selectSimp(k4Adj, k4Pos, convaSel),
        [{ nodes: [0, 1, 2] }, { nodes: [0, 1, 3] }, { nodes: [0, 2, 3] }, { nodes: [1, 2, 3] }]);
    // Restricting to just ONE selected triangle ([0,1,2]): conva(simp 3, ...) needs ALL of the
    // simp-3's own sub-faces selected, so the one simp-3 [0,1,2,3] (which has 4 sub-faces, only one
    // selected) is excluded; conve(simp 3, ...) only needs AT LEAST ONE, so it's still included.
    const justOneTriangle = { op: 'raw' as const, type: simpType(2), items: { kind: 'simp' as const, n: 2, value: [{ nodes: [0, 1, 2] }] } };
    assert.deepEqual(selectSimp(k4Adj, k4Pos, { op: 'conva' as const, type: simpType(3), from: simpType(2), a: justOneTriangle }), []);
    assert.deepEqual(
        selectSimp(k4Adj, k4Pos, { op: 'conve' as const, type: simpType(3), from: simpType(2), a: justOneTriangle }),
        [{ nodes: [0, 1, 2, 3] }]);
});

test('conva/conve reject simp <-> quad for any simp arity, not just simp 2', () => {
    assert.throws(
        () => parseQuadSelector('(conva quad (all simp 4))'),
        /no association defined between 'simp' and 'quad'/);
});
