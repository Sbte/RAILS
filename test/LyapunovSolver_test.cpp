#include <limits.h>
#include "gtest/gtest.h"

#include "src/LyapunovSolver.hpp"
#include "src/ScalarWrapper.hpp"

TEST(LyapunovSolverTest, ScalarEigenvalueSolver)
{
    ScalarWrapper A = 2;
    ScalarWrapper B = -4;
    ScalarWrapper V = 1;
    ScalarWrapper T = 4;
    ScalarWrapper H = 0;
    ScalarWrapper eigenvectors = 0;
    ScalarWrapper eigenvalues = 0;

    Lyapunov::Solver<ScalarWrapper, ScalarWrapper, ScalarWrapper> solver(A, B, 0);

    solver.lanczos(A, V, T, H, eigenvectors, eigenvalues, 2);

    EXPECT_EQ(32, H);
    EXPECT_EQ(32, eigenvalues);
}

TEST(LyapunovSolverTest, ScalarDenseSolver)
{
    ScalarWrapper A = 2;
    ScalarWrapper B = -4;
    ScalarWrapper X = -4;

    Lyapunov::Solver<ScalarWrapper, ScalarWrapper, ScalarWrapper> solver(A, B, 0);

    solver.dense_solve(A, B, X);

    EXPECT_EQ(1, X);
}

TEST(LyapunovSolverTest, ScalarSolver)
{
    ScalarWrapper A = 2;
    ScalarWrapper B = -4;
    ScalarWrapper X = -3;
    ScalarWrapper T = 123;

    Lyapunov::Solver<ScalarWrapper, ScalarWrapper, ScalarWrapper> solver(A, B, 0);

    solver.solve(X, T);

    EXPECT_EQ(-4, X * T * X);
}
