#pragma once
#include <vector>
#include <utility>

// Mirrors shared/geometry.ts. Unlike the rest of this codebase's exact-integer board-construction
// code, this uses double arithmetic with an epsilon tolerance (EPS in the .cpp): its inputs are
// unit-normalized direction vectors (see rectify()), and normalization requires sqrt.

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

// Mirrors shared/geometry.ts's linearFeasible().
bool linear_feasible(const LPProblem& problem);

// Mirrors shared/geometry.ts's convexHullPoints().
std::vector<int> convex_hull_points(const std::vector<std::vector<double>>& points);

// Mirrors shared/geometry.ts's convexHullEdges().
std::vector<std::pair<int,int>> convex_hull_edges(const std::vector<std::vector<double>>& points);
