// Regression tests for src/camera.ts's quaternion camera math - pure math, no DOM.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    QUAT_IDENTITY, quatMultiply, quatNormalize, quatFromAxisAngle, quatRotateVector, quatToMat3,
    applyOrbitDrag, applyRoll,
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

test('two 15-degree rolls in the same direction compose into a 30-degree local-Z rotation', () => {
    const q0 = applyOrbitDrag(QUAT_IDENTITY, 30, 20); // arbitrary starting orientation
    const once = applyRoll(applyRoll(q0, 1), 1);
    // Rolling twice should match a single 30-degree rotation applied in q0's own local frame -
    // the same LOCAL-frame (right-multiply) composition applyRoll itself uses.
    const q30 = quatNormalize(quatMultiply(q0, quatFromAxisAngle([0, 0, 1], Math.PI / 6)));
    assertVecClose(
        [once.w, once.x, once.y, once.z], [q30.w, q30.x, q30.y, q30.z],
        'two 15deg rolls == one 30deg local-Z rotation',
    );
});

test('opposite-direction rolls cancel out', () => {
    const q0 = applyOrbitDrag(QUAT_IDENTITY, 10, -15);
    const back = applyRoll(applyRoll(q0, 1), -1);
    assertVecClose([back.w, back.x, back.y, back.z], [q0.w, q0.x, q0.y, q0.z], 'roll then un-roll');
});
