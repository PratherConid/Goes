/**
 * Dimension-agnostic convex hull utilities, built on a single linear-programming feasibility
 * primitive. Used by rectify() in boardConfig.ts to find, for the mid-edge points around one
 * original vertex, which pairs are connected by an edge on the surface of their convex hull -
 * this works uniformly in any embedding dimension, unlike a facet-enumeration approach (whose
 * facet triangulations would need to be resolved recursively, dimension by dimension, to recover
 * the true 1-skeleton without introducing spurious "diagonal" edges).
 *
 * All computations use floating point with an epsilon tolerance (EPS below); this file (unlike
 * boardConfig.ts) does not assume integer input coordinates.
 */

const EPS = 1e-7;

export interface LinearConstraint {
    coeffs: number[]; // length == LPProblem.numVars
    rhs: number;
}

export interface LPProblem {
    numVars: number;
    nonNegative: boolean[]; // length == numVars; true = variable constrained >= 0, false = free
    equalities: LinearConstraint[];   // coeffs . x == rhs
    inequalities: LinearConstraint[]; // coeffs . x <= rhs
}

/**
 * Standard two-phase-simplex feasibility test: does there exist x satisfying all of `problem`'s
 * constraints? Free variables are split into (x+ - x-) with x+, x- >= 0; each inequality gets a
 * non-negative slack; every row then gets an artificial variable, and phase one minimizes the sum
 * of artificials via primal simplex with Bland's rule (avoids cycling; problem sizes here are tiny
 * so its lack of speed doesn't matter). Feasible iff that minimum is ~0.
 */
export function linearFeasible(problem: LPProblem): boolean {
    const { numVars, nonNegative, equalities, inequalities } = problem;

    const varCols: number[][] = []; // varCols[i] = [col] (nonNegative) or [colPlus, colMinus] (free)
    let col = 0;
    for (let i = 0; i < numVars; i++) {
        if (nonNegative[i]) { varCols.push([col]); col += 1; }
        else { varCols.push([col, col + 1]); col += 2; }
    }
    const numBaseCols = col;

    const rows: { coeffs: number[]; rhs: number; isIneq: boolean }[] = [];
    for (const c of equalities) rows.push({ coeffs: c.coeffs, rhs: c.rhs, isIneq: false });
    for (const c of inequalities) rows.push({ coeffs: c.coeffs, rhs: c.rhs, isIneq: true });

    const numRows = rows.length;
    if (numRows === 0) return true;

    const numSlackCols = inequalities.length;
    const numArtStart = numBaseCols + numSlackCols;
    const numCols = numArtStart + numRows;

    const T: number[][] = Array.from({ length: numRows + 1 }, () => new Array(numCols + 1).fill(0));
    const basis: number[] = new Array(numRows);

    let slackCol = numBaseCols;
    for (let r = 0; r < numRows; r++) {
        const row = rows[r];
        for (let i = 0; i < numVars; i++) {
            const v = row.coeffs[i] ?? 0;
            if (v === 0) continue;
            const cols = varCols[i];
            T[r][cols[0]] += v;
            if (cols.length > 1) T[r][cols[1]] -= v;
        }
        if (row.isIneq) {
            T[r][slackCol] = 1;
            slackCol++;
        }
        let rhs = row.rhs;
        if (rhs < 0) {
            for (let j = 0; j < numArtStart; j++) T[r][j] = -T[r][j];
            rhs = -rhs;
        }
        T[r][numArtStart + r] = 1;
        T[r][numCols] = rhs;
        basis[r] = numArtStart + r;
    }

    // Phase-one objective: minimize sum of artificials. Reduced-cost row = c - c_B^T*T, with
    // c_j = 1 for artificial columns (0 elsewhere) and c_B = all-ones (every basic var is an
    // artificial initially) - so the row is initialized to c and then each constraint row is
    // subtracted off, zeroing out the (basic) artificial columns as required.
    for (let j = numArtStart; j < numCols; j++) T[numRows][j] = 1;
    for (let r = 0; r < numRows; r++)
        for (let j = 0; j <= numCols; j++) T[numRows][j] -= T[r][j];

    while (true) {
        let enter = -1;
        for (let j = 0; j < numCols; j++) {
            if (T[numRows][j] < -EPS) { enter = j; break; }
        }
        if (enter === -1) break; // optimal

        let leave = -1, bestRatio = Infinity;
        for (let i = 0; i < numRows; i++) {
            if (T[i][enter] > EPS) {
                const ratio = T[i][numCols] / T[i][enter];
                if (ratio < bestRatio - EPS || (ratio < bestRatio + EPS && (leave === -1 || basis[i] < basis[leave]))) {
                    bestRatio = ratio;
                    leave = i;
                }
            }
        }
        if (leave === -1) throw new Error('linearFeasible: unexpected unbounded phase-one LP');

        const pivot = T[leave][enter];
        for (let j = 0; j <= numCols; j++) T[leave][j] /= pivot;
        for (let i = 0; i <= numRows; i++) {
            if (i === leave) continue;
            const factor = T[i][enter];
            if (Math.abs(factor) < EPS) continue;
            for (let j = 0; j <= numCols; j++) T[i][j] -= factor * T[leave][j];
        }
        basis[leave] = enter;
    }

    const minArtificialSum = -T[numRows][numCols];
    return Math.abs(minArtificialSum) < EPS;
}

/**
 * Returns the indices (into `points`) of points that are vertices of conv(points), in any
 * embedding dimension. Point i is a hull vertex iff it is NOT expressible as a convex combination
 * of the other points - tested directly via linearFeasible (variables = combination weights,
 * constrained non-negative and summing to 1, with one equality per coordinate).
 *
 * Assumes distinct points: a point exactly coincident with another is reported as non-extreme
 * (trivially reproduced by its duplicate), which is not a meaningful hull vertex distinction.
 */
export function convexHullPoints(points: number[][]): number[] {
    const n = points.length;
    if (n === 0) return [];
    if (n === 1) return [0];
    const d = points[0].length;

    const result: number[] = [];
    for (let i = 0; i < n; i++) {
        const others = points.filter((_, k) => k !== i);
        const m = others.length;
        const equalities: LinearConstraint[] = [];
        for (let dim = 0; dim < d; dim++)
            equalities.push({ coeffs: others.map(p => p[dim]), rhs: points[i][dim] });
        equalities.push({ coeffs: new Array(m).fill(1), rhs: 1 });

        const isConvexCombo = linearFeasible({
            numVars: m,
            nonNegative: new Array(m).fill(true),
            equalities,
            inequalities: [],
        });
        if (!isConvexCombo) result.push(i);
    }
    return result;
}

/**
 * Returns every pair of indices (into `points`, i < j) connected by an edge on the surface of
 * conv(points), in any embedding dimension. {i, j} is a hull edge iff there is a hyperplane
 * through points[i] and points[j] with every other point strictly on one side - tested via
 * linearFeasible for a normal vector c with c.(points[j]-points[i]) = 0 and c.(points[k]-points[i])
 * <= -1 for every other point k (the free scale of c lets any strict "< 0" be normalized to
 * "<= -1").
 *
 * The "every other point" set here is restricted to hull vertices (via convexHullPoints), not all
 * of `points`: a non-vertex point can never itself be an edge endpoint (if points[i] is a nontrivial
 * convex combination of other points, no separating hyperplane through it can exist), but if left
 * in as an obstacle it can wrongly block a real edge between two hull vertices - e.g. a redundant
 * point sitting exactly on the hyperplane through two true corners (on that edge, not at a vertex)
 * would otherwise break the required strict separation. Since conv(points) == conv(hull vertices),
 * dropping non-vertices from the obstacle set is exact, not an approximation.
 *
 * Assumes distinct points (see convexHullPoints).
 *
 * A diagonal of a coplanar convex face (e.g. a square face of a cube) is correctly never returned
 * as an edge - and this is exact, not an artifact of the EPS tolerance. For any convex quadrilateral
 * A, B, C, D in cyclic order, the defining property of the diagonal {A, C} (as opposed to a side) is
 * that B and D sit on strictly opposite sides of it. So for {A, C}, any c satisfying c.(C-A) = 0
 * forces c.(B-A) and c.(D-A) to have opposite sign, so they can never both be <= -1 at once - a hard
 * contradiction with a fixed gap (not one that shrinks to zero and needs EPS to detect).
 */
export function convexHullEdges(points: number[][]): [number, number][] {
    if (points.length < 2) return [];
    const hullIdx = convexHullPoints(points);
    if (hullIdx.length < 2) return [];
    const d = points[0].length;

    const edges: [number, number][] = [];
    for (let a = 0; a < hullIdx.length; a++) {
        for (let b = a + 1; b < hullIdx.length; b++) {
            const i = hullIdx[a], j = hullIdx[b];
            const diffIJ = points[j].map((v, k) => v - points[i][k]);
            const others = hullIdx.filter(k => k !== i && k !== j);

            const equalities: LinearConstraint[] = [{ coeffs: diffIJ, rhs: 0 }];
            const inequalities: LinearConstraint[] = others.map(k => ({
                coeffs: points[k].map((v, dim) => v - points[i][dim]),
                rhs: -1,
            }));

            const isEdge = linearFeasible({
                numVars: d,
                nonNegative: new Array(d).fill(false),
                equalities,
                inequalities,
            });
            if (isEdge) edges.push([i, j]);
        }
    }
    return edges;
}
