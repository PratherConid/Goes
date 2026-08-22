// Regression tests for shared/types.ts's makeBoardTriangle/makeBoardQuad - the canonical
// constructors BoardTriangle/BoardQuad values are built through.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeBoardTriangle, makeBoardQuad } from '../shared/types.ts';

test('makeBoardTriangle normalizes any permutation of the same 3 nodes to n1 < n2 < n3', () => {
    const expected = { n1: 2, n2: 5, n3: 9 };
    assert.deepEqual(makeBoardTriangle(2, 5, 9), expected);
    assert.deepEqual(makeBoardTriangle(9, 5, 2), expected);
    assert.deepEqual(makeBoardTriangle(5, 9, 2), expected);
    assert.deepEqual(makeBoardTriangle(9, 2, 5), expected);
});

// Given a cycle a-b-c-d-a, all 8 of its rotation/reflection-equivalent relabelings (4 rotations x 2
// directions) name the exact same quad - makeBoardQuad must canonicalize every one of them to
// the identical BoardQuad.
function allCycleRelabelings(a: number, b: number, c: number, d: number): [number, number, number, number][] {
    const fwd = [a, b, c, d];
    const bwd = [a, d, c, b];
    const out: [number, number, number, number][] = [];
    for (let i = 0; i < 4; i++) {
        out.push([fwd[i], fwd[(i + 1) % 4], fwd[(i + 2) % 4], fwd[(i + 3) % 4]]);
        out.push([bwd[i], bwd[(i + 1) % 4], bwd[(i + 2) % 4], bwd[(i + 3) % 4]]);
    }
    return out;
}

test('makeBoardQuad canonicalizes all 8 rotation/reflection relabelings of the same cycle identically', () => {
    for (const [a, b, c, d] of allCycleRelabelings(2, 8, 5, 9))
        assert.deepEqual(makeBoardQuad(a, b, c, d), { n1: 2, n2: 8, n3: 5, n4: 9 });
});

test('makeBoardQuad preserves cycle structure - unlike a plain sort, it never turns a diagonal into an apparent edge', () => {
    // Cycle 2-8-5-9-2: real edges are (2,8), (8,5), (5,9), (9,2); diagonals are (2,5) and (8,9). A
    // plain ascending sort would give {n1:2,n2:5,n3:8,n4:9}, whose own "edges" (2,5),(5,8),(8,9),(9,2)
    // wrongly include both actual diagonals and drop a real edge.
    const q = makeBoardQuad(2, 8, 5, 9);
    const cycleEdges = new Set([
        `${q.n1},${q.n2}`, `${q.n2},${q.n3}`, `${q.n3},${q.n4}`, `${q.n4},${q.n1}`,
    ]);
    assert.ok(cycleEdges.has('2,8') || cycleEdges.has('8,2'));
    assert.ok(cycleEdges.has('8,5') || cycleEdges.has('5,8'));
    assert.ok(cycleEdges.has('5,9') || cycleEdges.has('9,5'));
    assert.ok(cycleEdges.has('9,2') || cycleEdges.has('2,9'));
    assert.ok(!cycleEdges.has('2,5') && !cycleEdges.has('5,2'), 'diagonal (2,5) must not appear as an edge');
    assert.ok(!cycleEdges.has('8,9') && !cycleEdges.has('9,8'), 'diagonal (8,9) must not appear as an edge');
});
