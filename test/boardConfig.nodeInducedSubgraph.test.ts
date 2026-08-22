// Covers shared/boardConfig.ts's nodeInducedSubgraph() and its "nis" BoardModifier wiring
// (parseModifier/applyModifier), on rectangularBoard(3, 3): a 3x3 grid where the center node has
// degree 4, the 4 edge-midpoints have degree 3, and the 4 corners have degree 2.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    rectangularBoard, nodeInducedSubgraph, parseModifier, applyModifier,
} from '../shared/boardConfig.ts';
import { parseNodeSelector } from '../shared/selector.ts';

function degrees(adj: number[][]): number[] {
    return adj.map(row => row.reduce((s, v) => s + (v ? 1 : 0), 0));
}

test('keeps only the selected nodes, preserving their original positions and mutual adjacency', () => {
    const bc = rectangularBoard(3, 3);
    // Everything except the center (degree 4) - the 8 boundary nodes, which form a cycle around the
    // grid's perimeter (each adjacent to its two boundary neighbors, none adjacent to the center
    // since it's excluded).
    const sub = nodeInducedSubgraph(bc, parseNodeSelector('(deg lt 4)'));

    assert.equal(sub.N, 8);
    assert.deepEqual(degrees(sub.adj), new Array(8).fill(2), 'the boundary ring is an 8-cycle');

    // Positions are carried over unchanged (no averaging/recentering, unlike quotientBoard) - the
    // origin (the center node's own position) must not appear among the survivors.
    assert.ok(!sub.emb.pos.some(p => p[0] === 0 && p[1] === 0), 'the center node must not survive');
    const original = bc.emb.pos.map(p => `${p[0]},${p[1]}`);
    for (const p of sub.emb.pos) assert.ok(original.includes(`${p[0]},${p[1]}`), `${p} not an original position`);
});

test('selecting everything/nothing reproduces the whole board/an empty board', () => {
    const bc = rectangularBoard(3, 3);
    const all = nodeInducedSubgraph(bc, parseNodeSelector('(all)'));
    assert.equal(all.N, 9);
    assert.deepEqual(degrees(all.adj), degrees(bc.adj));

    const none = nodeInducedSubgraph(bc, parseNodeSelector('(none)'));
    assert.equal(none.N, 0);
});

test('parseModifier("nis", ...) round-trips through applyModifier the same as calling ' +
    'nodeInducedSubgraph directly', () => {
    const bc = rectangularBoard(3, 3);
    // _parseCommand (src/renderer.ts) splits the whole command line on whitespace before calling
    // parseModifier, so a selector's own internal parens/spaces arrive pre-split like this.
    const modifier = parseModifier('nis', ['(deg', 'eq', '3)']);
    assert.equal(modifier.kind, 'NodeInducedSubgraph');

    const viaModifier = applyModifier(bc, modifier);
    const direct = nodeInducedSubgraph(bc, parseNodeSelector('(deg eq 3)'));
    assert.equal(viaModifier.N, direct.N);
    assert.deepEqual(viaModifier.adj, direct.adj);
});

test('parseModifier("nis", ...) rejects too few arguments or a malformed selector', () => {
    assert.throws(() => parseModifier('nis', []), /nis takes at least 1 argument/);
    assert.throws(() => parseModifier('nis', ['(deg', 'eq']), /unexpected end of input/);
    // (conva sq ...) is valid at the top level (converting a square selector into this node
    // selector), but its own operand is then parsed as a SQUARE selector - and (deg ...) is
    // node-only, so it's rejected one level down instead.
    assert.throws(
        () => parseModifier('nis', ['(conva', 'sq', '(deg', 'eq', '3))']),
        /unknown square-selector operator 'deg'/,
    );
});
