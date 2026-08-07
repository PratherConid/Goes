// Orbiting camera state for the 3D board viewport (src/renderer.ts): a unit quaternion
// representing the camera's orientation, always looking at the origin. Pure math, no DOM - the
// game engine/C++ side has no rendering concept and never needs this.

export interface Quaternion { w: number; x: number; y: number; z: number; }

export const QUAT_IDENTITY: Quaternion = { w: 1, x: 0, y: 0, z: 0 };

// init: fraction of dmax (see computeAlpha) beyond which fading starts - 0 means fading starts
// right at the origin's own depth, 1 means it never starts (dmax is the farthest any point can be
// from the origin, so no point's depth can exceed it). rate: how fast alpha falls off per dmax of
// further depth past that point - 0 disables fading entirely (alpha always 1).
export interface FadingConfig { init: number; rate: number; }

// The natural-space point (in units of dmax - see computeAlpha's own doc comment - along each of
// the board's x/y/z render axes, before camera rotation) the camera looks at, instead of the
// origin: boardLayout() (src/renderer.ts) subtracts focus*dmax from every point's raw projected
// coordinates before applying the camera's rotation, so the camera continues orbiting around and
// facing this point exactly as it always did around the origin. [0, 0, 0] (the default) recovers
// the original "always looks at the origin" behavior.
export type Focus = [number, number, number];

export interface Viewport { quat: Quaternion; fadecfg: FadingConfig; focus: Focus; }

// A fresh object per call (not a shared constant) - each ActiveGame needs its own independent
// Viewport, since fi/fr (Renderer._parseCommand) mutate fadecfg's fields in place; sharing one
// instance across games would leak one game's fade settings into every other game.
export function defaultViewport(): Viewport {
    return { quat: QUAT_IDENTITY, fadecfg: { init: 0.0, rate: 0.8 }, focus: [0, 0, 0] };
}

/**
 * The alpha (0-1) an object at `depth` (its z after camera rotation - larger is nearer, see
 * boardLayout()) should render at, given `fadecfg` and `dmax` (the board's own farthest node
 * distance from the origin - rotation-invariant, so the same value regardless of camera
 * orientation). Fading is a depth-cueing effect: objects recede AWAY from the camera (into the
 * screen) as they fade, so what matters is how far *behind* the origin an object sits, i.e.
 * -depth (the origin's own depth is always 0, since rotation is linear and never moves it).
 * Fading starts once -depth exceeds fadecfg.init * dmax, then falls off linearly at fadecfg.rate
 * per dmax of further recession, clamped to [0, 1]. dmax <= 0 (a degenerate single-point board)
 * means no fading, since there's no meaningful distance scale to fade over.
 */
export function computeAlpha(depth: number, dmax: number, fadecfg: FadingConfig): number {
    if (dmax <= 0) return 1;
    const recession = -depth; // how far behind the origin (i.e. away from the camera) this is
    const distInit = fadecfg.init * dmax; // recessionOrigin (always 0) + init * dmax
    if (recession <= distInit) return 1;
    const alpha = 1 - (recession - distInit) * fadecfg.rate / dmax;
    return Math.max(0, Math.min(1, alpha));
}

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

// Tuned so a full main-board-width drag (~600px) is about half a turn (pi radians) of yaw.
const ORBIT_RADIANS_PER_PIXEL = Math.PI / 600;
// 1.5 degrees per arrow-key tap.
const ROLL_STEP_RADIANS = Math.PI / 120;

/**
 * Orbit: dx/dy are the drag delta in pixels since the last tick. A single combined rotation
 * (rather than composing a separate yaw-then-pitch quaternion pair) so a diagonal drag is
 * symmetric in dx/dy instead of depending on an arbitrary composition order. The axis is the
 * linear combination of the camera's own *current* local up axis (yaw, weighted by dx) and local
 * right axis (pitch, weighted by dy) - i.e. [-dy, -dx, 0] expressed directly in the camera's own
 * local frame (x=right, y=up), then rotated into world space via quatRotateVector so it can
 * compose onto the existing orientation via LEFT-multiplication. Using the camera's own local up
 * (rather than the fixed world-up) means yaw always turns around the camera's current vertical,
 * which only coincides with true vertical while the camera is unrolled. Local right/up are always
 * exactly orthonormal, so this axis's own length is exactly hypot(dx, dy) - matching the angle.
 */
export function applyOrbitDrag(q: Quaternion, dx: number, dy: number): Quaternion {
    const axis = quatRotateVector(q, [-dy, -dx, 0]);
    const angle = Math.hypot(dx, dy) * ORBIT_RADIANS_PER_PIXEL;
    return quatNormalize(quatMultiply(quatFromAxisAngle(axis, angle), q));
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
