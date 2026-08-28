// Regression tests for src/camera.ts's quaternion camera math - pure math, no DOM.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    QUAT_IDENTITY, quatMultiply, quatNormalize, quatFromAxisAngle, quatRotateVector, quatToMat3,
    applyOrbitDrag, applyRoll, computeAlpha,
} from '../src/camera.ts';

const EPS = 1e-9;
function assertClose(actual: number, expected: number, msg: string) {
    assert.ok(Math.abs(actual - expected) < EPS, `${msg}: expected ${expected}, got ${actual}`);
}
function assertVecClose(actual: number[], expected: number[], msg: string) {
    for (let i = 0; i < expected.length; i++) assertClose(actual[i], expected[i], `${msg}[${i}]`);
}
function quatMagnitude(q: { w: number; x: number; y: number; z: number }): number {
    return Math.sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
}

test('QUAT_IDENTITY produces the 3x3 identity matrix', () => {
    assert.deepEqual(quatToMat3(QUAT_IDENTITY), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
});

test('QUAT_IDENTITY leaves any vector unchanged', () => {
    assertVecClose(quatRotateVector(QUAT_IDENTITY, [1, 2, 3]), [1, 2, 3], 'identity rotation');
});

test('multiplying by QUAT_IDENTITY on either side is a no-op', () => {
    const q = quatNormalize({ w: 0.5, x: 0.5, y: 0.5, z: 0.5 });
    const left = quatMultiply(QUAT_IDENTITY, q), right = quatMultiply(q, QUAT_IDENTITY);
    assertVecClose([left.w, left.x, left.y, left.z], [q.w, q.x, q.y, q.z], 'identity * q');
    assertVecClose([right.w, right.x, right.y, right.z], [q.w, q.x, q.y, q.z], 'q * identity');
});

test('quatNormalize brings a non-unit quaternion to unit length without changing its direction', () => {
    const q = { w: 2, x: 0, y: 0, z: 0 };
    const n = quatNormalize(q);
    assertClose(quatMagnitude(n), 1, 'normalized magnitude');
    assertVecClose([n.w, n.x, n.y, n.z], [1, 0, 0, 0], 'direction preserved');
});

test('a 90-degree rotation around Z sends the X axis to the Y axis', () => {
    const q = quatFromAxisAngle([0, 0, 1], Math.PI / 2);
    assertClose(quatMagnitude(q), 1, 'axis-angle quaternion is already unit length');
    assertVecClose(quatRotateVector(q, [1, 0, 0]), [0, 1, 0], '90deg Z rotation of X axis');
});

test('a 180-degree rotation around Y sends the X axis to -X', () => {
    const q = quatFromAxisAngle([0, 1, 0], Math.PI);
    assertVecClose(quatRotateVector(q, [1, 0, 0]), [-1, 0, 0], '180deg Y rotation of X axis');
});

test('applyOrbitDrag always returns a unit quaternion, even after many drags', () => {
    let q = QUAT_IDENTITY;
    for (let i = 0; i < 50; i++) q = applyOrbitDrag(q, 17, -23);
    assertClose(quatMagnitude(q), 1, 'orbit-drag result stays unit length');
});

test('a purely horizontal drag (dy=0) is a pure yaw: the world-up axis is left unchanged', () => {
    const q = applyOrbitDrag(QUAT_IDENTITY, 40, 0);
    assertVecClose(quatRotateVector(q, [0, 1, 0]), [0, 1, 0], 'world-up axis invariant under pure yaw');
});

test('applyRoll always returns a unit quaternion, even after many rolls', () => {
    let q = QUAT_IDENTITY;
    for (let i = 0; i < 50; i++) q = applyRoll(q, i % 2 === 0 ? 1 : -1);
    assertClose(quatMagnitude(q), 1, 'roll result stays unit length');
});

test('roll leaves the camera\'s own local forward axis (local Z) unchanged - it only spins around it', () => {
    // Start from a non-trivial orientation (roll composes locally, so this isn't just testing identity).
    const q0 = applyOrbitDrag(QUAT_IDENTITY, 30, 20);
    const forwardBefore = quatRotateVector(q0, [0, 0, 1]);
    const q1 = applyRoll(q0, 1);
    const forwardAfter = quatRotateVector(q1, [0, 0, 1]);
    assertVecClose(forwardAfter, forwardBefore, 'local forward axis is invariant under roll');
});

test('two 1.5-degree rolls in the same direction compose into a 3-degree local-Z rotation', () => {
    const q0 = applyOrbitDrag(QUAT_IDENTITY, 30, 20); // arbitrary starting orientation
    const once = applyRoll(applyRoll(q0, 1), 1);
    // Rolling twice should match a single 3-degree rotation applied in q0's own local frame -
    // the same LOCAL-frame (right-multiply) composition applyRoll itself uses.
    const q3 = quatNormalize(quatMultiply(q0, quatFromAxisAngle([0, 0, 1], Math.PI / 60)));
    assertVecClose(
        [once.w, once.x, once.y, once.z], [q3.w, q3.x, q3.y, q3.z],
        'two 1.5deg rolls == one 3deg local-Z rotation',
    );
});

test('opposite-direction rolls cancel out', () => {
    const q0 = applyOrbitDrag(QUAT_IDENTITY, 10, -15);
    const back = applyRoll(applyRoll(q0, 1), -1);
    assertVecClose([back.w, back.x, back.y, back.z], [q0.w, q0.x, q0.y, q0.z], 'roll then un-roll');
});

test('computeAlpha (clamp): an object nearer than the origin never fades, regardless of rate', () => {
    assertClose(computeAlpha(8, 10, { kind: 'clamp', init: 0, rate: 1 }), 1, 'positive depth (near camera) stays opaque');
    assertClose(computeAlpha(0, 10, { kind: 'clamp', init: 0, rate: 1 }), 1, 'depth exactly at the origin stays opaque');
});

test('computeAlpha (clamp): an object receding behind the origin fades, not one in front of it', () => {
    const alpha = computeAlpha(-8, 10, { kind: 'clamp', init: 0, rate: 1 });
    assertClose(alpha, 0.2, 'depth=-8, dmax=10, init=0, rate=1 -> alpha = 1 - 8*1/10');
});

test('computeAlpha (clamp): fading is clamped to [0, 1]', () => {
    assertClose(computeAlpha(-100, 10, { kind: 'clamp', init: 0, rate: 1 }), 0, 'far past full fade clamps to 0');
    assertClose(computeAlpha(-1, 10, { kind: 'clamp', init: 0, rate: -5 }), 1, 'a negative rate cannot push alpha above 1');
});

test('computeAlpha (clamp): fading only starts once recession exceeds init * dmax', () => {
    const fadecfg = { kind: 'clamp' as const, init: 0.5, rate: 1 };
    assertClose(computeAlpha(-4, 10, fadecfg), 1, 'recession=4 is within the init=0.5*dmax=5 threshold');
    assertClose(computeAlpha(-6, 10, fadecfg), 0.9, 'recession=6 exceeds the threshold by 1, so alpha = 1 - 1/10');
});

test('computeAlpha (clamp): dmax <= 0 never fades, even with a nonzero rate', () => {
    assertClose(computeAlpha(-5, 0, { kind: 'clamp', init: 0, rate: 1 }), 1, 'degenerate single-point board');
});

test('computeAlpha (slice): full opacity within solidThick/2 of z', () => {
    // z=0.2, dmax=10 -> center depth = 2. solidThick=0.4 -> solid half-width = 0.2*dmax = 2, so
    // depths in [0, 4] are fully opaque.
    const fadecfg = { kind: 'slice' as const, z: 0.2, solidThick: 0.4, falloffThick: 0.2 };
    assertClose(computeAlpha(2, 10, fadecfg), 1, 'depth exactly at the slice center');
    assertClose(computeAlpha(0, 10, fadecfg), 1, 'depth at the near edge of the solid region');
    assertClose(computeAlpha(4, 10, fadecfg), 1, 'depth at the far edge of the solid region');
});

test('computeAlpha (slice): falls off linearly from 1 to 0 over falloffThick/2 past the solid region', () => {
    // Same slice as above: solid region is depth in [0, 4] (depth/dmax in [0, 0.4]);
    // falloffThick=0.2 -> falloff half-width = 0.1*dmax = 1, so alpha reaches 0 exactly at depth 5
    // (and depth -1, on the near side).
    const fadecfg = { kind: 'slice' as const, z: 0.2, solidThick: 0.4, falloffThick: 0.2 };
    assertClose(computeAlpha(4.5, 10, fadecfg), 0.5, 'halfway through the far falloff region');
    assertClose(computeAlpha(5, 10, fadecfg), 0, 'exactly at the far edge of visibility');
    assertClose(computeAlpha(6, 10, fadecfg), 0, 'past the far edge stays at 0, not negative');
    assertClose(computeAlpha(-0.5, 10, fadecfg), 0.5, 'halfway through the near falloff region');
    assertClose(computeAlpha(-1, 10, fadecfg), 0, 'exactly at the near edge of visibility');
});

test('computeAlpha (slice): falloffThick <= 0 is a hard cutoff right at the solid region, no division by zero', () => {
    const fadecfg = { kind: 'slice' as const, z: 0, solidThick: 0.2, falloffThick: 0 };
    assertClose(computeAlpha(1, 10, fadecfg), 1, 'depth=1 is within the solid half-width (0.1*10=1)');
    assertClose(computeAlpha(1.01, 10, fadecfg), 0, 'just past the solid region drops straight to 0');
});

test('computeAlpha (slice): dmax <= 0 never fades', () => {
    assertClose(computeAlpha(-5, 0, { kind: 'slice', z: 0, solidThick: 0.1, falloffThick: 0.1 }), 1, 'degenerate single-point board');
});
