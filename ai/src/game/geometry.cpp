#include "game/geometry.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
constexpr double EPS = 1e-7;
}

bool linear_feasible(const LPProblem& problem) {
    int num_vars = problem.num_vars;
    const auto& non_negative = problem.non_negative;
    const auto& equalities = problem.equalities;
    const auto& inequalities = problem.inequalities;

    std::vector<std::vector<int>> var_cols(num_vars); // [col] (non-negative) or [colPlus, colMinus] (free)
    int col = 0;
    for (int i = 0; i < num_vars; i++) {
        if (non_negative[i]) { var_cols[i] = {col}; col += 1; }
        else { var_cols[i] = {col, col + 1}; col += 2; }
    }
    int num_base_cols = col;

    struct Row { std::vector<double> coeffs; double rhs; bool is_ineq; };
    std::vector<Row> rows;
    for (auto& c : equalities) rows.push_back({c.coeffs, c.rhs, false});
    for (auto& c : inequalities) rows.push_back({c.coeffs, c.rhs, true});

    int num_rows = (int)rows.size();
    if (num_rows == 0) return true;

    int num_slack_cols = (int)inequalities.size();
    int num_art_start = num_base_cols + num_slack_cols;
    int num_cols = num_art_start + num_rows;

    std::vector<std::vector<double>> T(num_rows + 1, std::vector<double>(num_cols + 1, 0.0));
    std::vector<int> basis(num_rows);

    int slack_col = num_base_cols;
    for (int r = 0; r < num_rows; r++) {
        auto& row = rows[r];
        for (int i = 0; i < num_vars; i++) {
            double v = (i < (int)row.coeffs.size()) ? row.coeffs[i] : 0.0;
            if (v == 0) continue;
            auto& cols = var_cols[i];
            T[r][cols[0]] += v;
            if (cols.size() > 1) T[r][cols[1]] -= v;
        }
        if (row.is_ineq) {
            T[r][slack_col] = 1;
            slack_col++;
        }
        double rhs = row.rhs;
        if (rhs < 0) {
            for (int j = 0; j < num_art_start; j++) T[r][j] = -T[r][j];
            rhs = -rhs;
        }
        T[r][num_art_start + r] = 1;
        T[r][num_cols] = rhs;
        basis[r] = num_art_start + r;
    }

    // Phase-one objective: minimize sum of artificials. Reduced-cost row = c - c_B^T*T, with
    // c_j = 1 for artificial columns (0 elsewhere) and c_B = all-ones (every basic var is an
    // artificial initially) - so the row is initialized to c and then each constraint row is
    // subtracted off, zeroing out the (basic) artificial columns as required.
    for (int j = num_art_start; j < num_cols; j++) T[num_rows][j] = 1;
    for (int r = 0; r < num_rows; r++)
        for (int j = 0; j <= num_cols; j++) T[num_rows][j] -= T[r][j];

    while (true) {
        int enter = -1;
        for (int j = 0; j < num_cols; j++) {
            if (T[num_rows][j] < -EPS) { enter = j; break; }
        }
        if (enter == -1) break; // optimal

        int leave = -1;
        double best_ratio = std::numeric_limits<double>::infinity();
        for (int i = 0; i < num_rows; i++) {
            if (T[i][enter] > EPS) {
                double ratio = T[i][num_cols] / T[i][enter];
                if (ratio < best_ratio - EPS ||
                    (ratio < best_ratio + EPS && (leave == -1 || basis[i] < basis[leave]))) {
                    best_ratio = ratio;
                    leave = i;
                }
            }
        }
        if (leave == -1) throw std::runtime_error("linear_feasible: unexpected unbounded phase-one LP");

        double pivot = T[leave][enter];
        for (int j = 0; j <= num_cols; j++) T[leave][j] /= pivot;
        for (int i = 0; i <= num_rows; i++) {
            if (i == leave) continue;
            double factor = T[i][enter];
            if (std::abs(factor) < EPS) continue;
            for (int j = 0; j <= num_cols; j++) T[i][j] -= factor * T[leave][j];
        }
        basis[leave] = enter;
    }

    double min_artificial_sum = -T[num_rows][num_cols];
    return std::abs(min_artificial_sum) < EPS;
}

std::vector<int> convex_hull_points(const std::vector<std::vector<double>>& points) {
    int n = (int)points.size();
    if (n == 0) return {};
    if (n == 1) return {0};
    int d = (int)points[0].size();

    std::vector<int> result;
    for (int i = 0; i < n; i++) {
        std::vector<std::vector<double>> others;
        for (int k = 0; k < n; k++) if (k != i) others.push_back(points[k]);
        int m = (int)others.size();

        std::vector<LinearConstraint> equalities;
        for (int dim = 0; dim < d; dim++) {
            std::vector<double> coeffs(m);
            for (int k = 0; k < m; k++) coeffs[k] = others[k][dim];
            equalities.push_back({coeffs, points[i][dim]});
        }
        equalities.push_back({std::vector<double>(m, 1.0), 1.0});

        LPProblem problem{m, std::vector<bool>(m, true), equalities, {}};
        bool is_convex_combo = linear_feasible(problem);
        if (!is_convex_combo) result.push_back(i);
    }
    return result;
}

std::vector<std::pair<int,int>> convex_hull_edges(const std::vector<std::vector<double>>& points) {
    if (points.size() < 2) return {};
    std::vector<int> hull_idx = convex_hull_points(points);
    if (hull_idx.size() < 2) return {};
    int d = (int)points[0].size();

    std::vector<std::pair<int,int>> edges;
    for (size_t a = 0; a < hull_idx.size(); a++) {
        for (size_t b = a + 1; b < hull_idx.size(); b++) {
            int i = hull_idx[a], j = hull_idx[b];
            std::vector<double> diff_ij(d);
            for (int k = 0; k < d; k++) diff_ij[k] = points[j][k] - points[i][k];

            std::vector<int> others;
            for (int k : hull_idx) if (k != i && k != j) others.push_back(k);

            std::vector<LinearConstraint> equalities = {{diff_ij, 0.0}};
            std::vector<LinearConstraint> inequalities;
            for (int k : others) {
                std::vector<double> coeffs(d);
                for (int dim = 0; dim < d; dim++) coeffs[dim] = points[k][dim] - points[i][dim];
                inequalities.push_back({coeffs, -1.0});
            }

            LPProblem problem{d, std::vector<bool>(d, false), equalities, inequalities};
            bool is_edge = linear_feasible(problem);
            if (is_edge) edges.push_back({i, j});
        }
    }
    return edges;
}
