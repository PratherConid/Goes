// Orbiting camera state for the 3D board viewport (src/renderer.ts): a unit quaternion
// representing the camera's orientation, always looking at the origin. Pure math, no DOM - the
// game engine/C++ side has no rendering concept and never needs this.

export interface Quaternion { w: number; x: number; y: number; z: number; }

export const QUAT_IDENTITY: Quaternion = { w: 1, x: 0, y: 0, z: 0 };

type Vec3 = [number, number, number];

function cross(a: Vec3, b: Vec3): Vec3 {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/** Hamilton product a*b - composes b's rotation, then a's (applied to a vector as a*(b*v)). */
export function quatMultiply(a: Quaternion, b: Quaternion): Quaternion {
    return {
        w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    };
}

/** Renormalizes q to unit length - needed after every update, since repeated float multiplication drifts. */
export function quatNormalize(q: Quaternion): Quaternion {
    const len = Math.sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z) || 1;
    return { w: q.w / len, x: q.x / len, y: q.y / len, z: q.z / len };
}

/**
 * The inverse of q (q assumed unit-length, so its inverse is just its conjugate: negate the
 * vector part). cameraOrientation represents the camera's own orientation IN WORLD SPACE (e.g.
 * applyOrbitDrag derives the camera's world-space right/up axes via quatRotateVector(q, ...)) -
 * rendering a world point into the camera's view therefore needs this INVERSE rotation, not q
 * itself (see boardLayout(), src/renderer.ts).
 */
export function quatConjugate(q: Quaternion): Quaternion {
    return { w: q.w, x: -q.x, y: -q.y, z: -q.z };
}

/** The unit quaternion rotating by `angle` radians around `axis` (need not be pre-normalized). */
export function quatFromAxisAngle(axis: Vec3, angle: number): Quaternion {
    const len = Math.sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]) || 1;
    const half = angle / 2;
    const s = Math.sin(half) / len;
    return { w: Math.cos(half), x: axis[0] * s, y: axis[1] * s, z: axis[2] * s };
}

/** Rotates v by q (q assumed unit-length) - the standard v + 2w(q_xyz x v) + 2(q_xyz x (q_xyz x v)) form. */
export function quatRotateVector(q: Quaternion, v: Vec3): Vec3 {
    const qv: Vec3 = [q.x, q.y, q.z];
    const t = cross(qv, v).map(c => 2 * c) as Vec3;
    const c2 = cross(qv, t);
    return [v[0] + q.w * t[0] + c2[0], v[1] + q.w * t[1] + c2[1], v[2] + q.w * t[2] + c2[2]];
}

/**
 * The 3x3 rotation matrix for q (q assumed unit-length), in the same row-major number[][] shape
 * shared/boardConfig.ts's projectPoint() expects - so a camera rotation can be applied to an
 * already-projected (x, y, z) point via `projectPoint(quatToMat3(q), point)` directly, no separate
 * matrix-vector-multiply helper needed.
 */
export function quatToMat3(q: Quaternion): number[][] {
    const { w, x, y, z } = q;
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ];
}

const WORLD_UP: Vec3 = [0, 1, 0];
// Tuned so a full main-board-width drag (~600px) is about half a turn (pi radians) of yaw.
const ORBIT_RADIANS_PER_PIXEL = Math.PI / 600;
// 1.5 degrees per arrow-key tap.
const ROLL_STEP_RADIANS = Math.PI / 120;

/**
 * Orbit: dx/dy are the drag delta in pixels since the last tick. Yaw rotates around the fixed
 * world-up axis; pitch rotates around the camera's own *current* local right axis (derived via
 * quatRotateVector) - both axes are expressed in world space, so both compose onto the existing
 * orientation via LEFT-multiplication (yaw * pitch * q), then renormalize to counteract drift.
 */
export function applyOrbitDrag(q: Quaternion, dx: number, dy: number): Quaternion {
    const qYaw = quatFromAxisAngle(WORLD_UP, -dx * ORBIT_RADIANS_PER_PIXEL);
    const right = quatRotateVector(q, [1, 0, 0]);
    const qPitch = quatFromAxisAngle(right, -dy * ORBIT_RADIANS_PER_PIXEL);
    return quatNormalize(quatMultiply(quatMultiply(qYaw, qPitch), q));
}

/**
 * Roll: rotates around the camera's own forward axis (local Z, the line connecting the origin and
 * the camera) by a fixed step, in the given direction - a LOCAL-frame rotation, so it composes via
 * RIGHT-multiplication (q * qRoll): each small rotation transforms the existing quaternion.
 */
export function applyRoll(q: Quaternion, direction: 1 | -1): Quaternion {
    const qRoll = quatFromAxisAngle([0, 0, 1], direction * ROLL_STEP_RADIANS);
    return quatNormalize(quatMultiply(q, qRoll));
}
