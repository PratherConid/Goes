#pragma once
#include <vector>
#include <utility>

struct BoardConfig {
    int N;
    std::vector<std::vector<int>> adj; // N×N symmetric adjacency matrix (0/1)
};

// Factory functions mirroring board_config.py
BoardConfig rectangular_board(int w, int h);
BoardConfig rectangular_diagonal_board(int w, int h, int m);
BoardConfig cubical_board(int w, int h, int d);
BoardConfig hypercube_board(int w, int h, int d, int t);
BoardConfig triangular_board(int w);
BoardConfig glue_twisted_square_board(int w, int h, int g);
BoardConfig twisted_square_board(int w, int h, int g);
BoardConfig quotient_board(const BoardConfig& bc,
                           const std::vector<std::pair<int,int>>& quot);
