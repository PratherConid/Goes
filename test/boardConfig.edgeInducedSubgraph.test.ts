// Covers shared/boardConfig.ts's edgeInducedSubgraph() and its "eis" BoardModifier wiring
// (parseModifier/applyModifier).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    make, rectangularBoard, edgeInducedSubgraph, parseModifier, applyModifier,
} from '../shared/boardConfig.ts';
import { parseEdgeSelector } from '../shared/selector.ts';

function degrees(adj: number[][]): number[] {
    return adj.map(row => row.reduce((s, v) => s + (v ? 1 : 0), 0));
}

test('drops a node whose only original edges are non-selected, unlike nodeInducedSubgraph', () => {
    // Triangle 0-1-2 (each degree 2 within it) plus a pendant node 3 attached only to node 0 (which
    // is therefore degree 3 overall).
    const bc = make(
        [[0, 0], [1, 0], [0, 1], [2, 0]],
        [
            [0, 1, 1, 1],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
        ],
    );
    // Selects edges where both endpoints have degree > 1: the 3 triangle edges - NOT the pendant
    // edge (0,3), since node 3 has degree 1.
    const sub = edgeInducedSubgraph(bc, parseEdgeSelector('(fromna (deg gt 1))'));
    // Node 3 is adjacent to a surviving node (0) via edge (0,3), but that edge was never selected,
    // so node 3 itself has no selected incident edge and does not survive.
    assert.equal(sub.N, 3);
    assert.deepEqual(degrees(sub.adj), [2, 2, 2]);
});

test('selecting all/no edges reproduces the whole board/an empty board', () => {
    const bc = rectangularBoard(3, 3);
    const all = edgeInducedSubgraph(bc, parseEdgeSelector('(all)'));
    assert.equal(all.N, bc.N);
    assert.deepEqual(degrees(all.adj), degrees(bc.adj));

    const none = edgeInducedSubgraph(bc, parseEdgeSelector('(none)'));
    assert.equal(none.N, 0);
});

test('parseModifier("eis", ...) round-trips through applyModifier the same as calling ' +
    'edgeInducedSubgraph directly', () => {
    const bc = rectangularBoard(3, 3);
    // _parseCommand (src/renderer.ts) splits the whole command line on whitespace before calling
    // parseModifier, so a selector's own internal parens/spaces arrive pre-split like this.
    const modifier = parseModifier('eis', ['(fromna', '(deg', 'eq', '3)', ')']);
    assert.equal(modifier.kind, 'EdgeInducedSubgraph');

    const viaModifier = applyModifier(bc, modifier);
    const direct = edgeInducedSubgraph(bc, parseEdgeSelector('(fromna (deg eq 3))'));
    assert.equal(viaModifier.N, direct.N);
    assert.deepEqual(viaModifier.adj, direct.adj);
});

test('parseModifier("eis", ...) rejects too few arguments or a malformed selector', () => {
    assert.throws(() => parseModifier('eis', []), /eis takes at least 1 argument/);
    assert.throws(() => parseModifier('eis', ['(fromna', '(deg', 'eq']), /unexpected end of input/);
    // (deg ...) is node-only - eis parses its argument via parseEdgeSelector, which doesn't
    // recognize deg.
    assert.throws(
        () => parseModifier('eis', ['(deg', 'eq', '3)']),
        /unknown edge-selector operator 'deg'/,
    );
});
