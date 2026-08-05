#pragma once
#include <vector>
#include <utility>

// Dimension-agnostic convex hull utilities, built on a single linear-programming feasibility
// primitive. Used by rectify() (board_config.h) to find, for the mid-edge points around one
// original vertex, which pairs are connected by an edge on the surface of their convex hull - this
// works uniformly in any embedding dimension, unlike a facet-enumeration approach (whose facet
// triangulations would need to be resolved recursively, dimension by dimension, to recover the true
// 1-skeleton without introducing spurious "diagonal" edges). Mirrors shared/geometry.ts exactly.
//
// Unlike the rest of this file's board-construction code, this uses double (not integer)
// arithmetic with an epsilon tolerance (EPS in the .cpp): the vectors this gets called on are
// normalized to unit length (see rectify()'s doc comment), and normalization requires sqrt, which
// is irrational for almost every integer input - there is no exact-integer alternative here, unlike
// every other geometric construction in this codebase.

struct LinearConstraint {
    std::vector<double> coeffs; // length == LPProblem.num_vars
    double rhs;
};

struct LPProblem {
    int num_vars;
    std::vector<bool> non_negative; // length == num_vars; true = variable constrained >= 0, false = free
    std::vector<LinearConstraint> equalities;   // coeffs . x == rhs
    std::vector<LinearConstraint> inequalities; // coeffs . x <= rhs
};

// Standard two-phase-simplex feasibility test: does there exist x satisfying every one of
// problem's constraints? Free variables are split into (x+ - x-) with x+, x- >= 0; each inequality
// gets a non-negative slack; every row then gets an artificial variable, and phase one minimizes
// the sum of artificials via primal simplex with Bland's rule (avoids cycling; problem sizes here
// are tiny so its lack of speed doesn't matter). Feasible iff that minimum is ~0.
bool linear_feasible(const LPProblem& problem);

// Returns the indices (into points) of points that are vertices of conv(points), in any embedding
// dimension. Point i is a hull vertex iff it is NOT expressible as a convex combination of the
// other points - tested directly via linear_feasible (variables = combination weights, constrained
// non-negative and summing to 1, with one equality per coordinate).
//
// Assumes distinct points: a point exactly coincident with another is reported as non-extreme
// (trivially reproduced by its duplicate), which is not a meaningful hull vertex distinction.
std::vector<int> convex_hull_points(const std::vector<std::vector<double>>& points);

// Returns every pair of indices (into points, i < j) connected by an edge on the surface of
// conv(points), in any embedding dimension. {i, j} is a hull edge iff there is a hyperplane through
// points[i] and points[j] with every other point strictly on one side - tested via linear_feasible
// for a normal vector c with c.(points[j]-points[i]) = 0 and c.(points[k]-points[i]) <= -1 for
// every other point k (the free scale of c lets any strict "< 0" be normalized to "<= -1").
//
// The "every other point" set here is restricted to hull vertices (via convex_hull_points), not all
// of points: a non-vertex point can never itself be an edge endpoint (if points[i] is a nontrivial
// convex combination of other points, no separating hyperplane through it can exist), but if left
// in as an obstacle it can wrongly block a real edge between two hull vertices - e.g. a redundant
// point sitting exactly on the hyperplane through two true corners (on that edge, not at a vertex)
// would otherwise break the required strict separation. Since conv(points) == conv(hull vertices),
// dropping non-vertices from the obstacle set is exact, not an approximation.
//
// Assumes distinct points (see convex_hull_points).
//
// A diagonal of a coplanar convex face (e.g. a square face of a cube) is correctly never returned
// as an edge - and this is exact, not an artifact of the EPS tolerance. For any convex quadrilateral
// A, B, C, D in cyclic order, the defining property of the diagonal {A, C} (as opposed to a side) is
// that B and D sit on strictly opposite sides of it. So for {A, C}, any c satisfying c.(C-A) = 0
// forces c.(B-A) and c.(D-A) to have opposite sign, so they can never both be <= -1 at once - a hard
// contradiction with a fixed gap (not one that shrinks to zero and needs EPS to detect).
std::vector<std::pair<int,int>> convex_hull_edges(const std::vector<std::vector<double>>& points);
