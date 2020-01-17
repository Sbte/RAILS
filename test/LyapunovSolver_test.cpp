#include <limits.h>
#include "gtest/gtest.h"

#include "TestHelpers.hpp"

#include "src/LyapunovSolver.hpp"
#include "src/ScalarWrapper.hpp"
#include "src/StlWrapper.hpp"

#include <map>
#include <string>

using namespace RAILS;

TEST(LyapunovSolverTest, ScalarEigenvalueSolver)
{
    ScalarWrapper A = 2;
    ScalarWrapper B = -4;
    ScalarWrapper V = 1;
    ScalarWrapper T = 4;
    ScalarWrapper H = 0;
    ScalarWrapper eigenvectors = 0;
    ScalarWrapper eigenvalues = 0;

    Solver<ScalarWrapper, ScalarWrapper, ScalarWrapper> solver(A, B, 0);

    solver.resid_lanczos(A, V, T, H, eigenvectors, eigenvalues, 2);

    EXPECT_EQ(32, H);
    EXPECT_EQ(32, eigenvalues);
}

TEST(LyapunovSolverTest, ScalarDenseSolver)
{
    ScalarWrapper A = 2;
    ScalarWrapper B = -4;
    ScalarWrapper X = -4;

    Solver<ScalarWrapper, ScalarWrapper, ScalarWrapper> solver(A, B, 0);

    solver.dense_solve(A, B, X);

    EXPECT_EQ(1, X);
}

TEST(LyapunovSolverTest, ScalarSolver)
{
    ScalarWrapper A = 2;
    ScalarWrapper B = -4;
    ScalarWrapper X = -3;
    ScalarWrapper T = 123;

    Solver<ScalarWrapper, ScalarWrapper, ScalarWrapper> solver(A, B, 0);

    int ret = solver.solve(X, T);
    EXPECT_EQ(0, ret);

    EXPECT_EQ(-4, X * T * X);
}

TEST(LyapunovSolverTest, StlDenseSolver)
{
    int n = 20;
    StlWrapper A(n, n);
    A.random();

    StlWrapper B(n, 1);
    B.random();

    StlWrapper C = B.copy();
    C(n-1, 0) = 0.0;
    B -= C;
    B = B * B.transpose();

    StlWrapper X;

    Solver<StlWrapper, StlWrapper, StlWrapper> solver(A, B, B);

    solver.dense_solve(A, B, X);

    // Compute the residual
    StlWrapper R = A * X + X * A.transpose() + B;
    StlWrapper R_exp(n, n);
    R_exp = 0.0;

    EXPECT_VECTOR_NEAR(R_exp, R);
}

TEST(LyapunovSolverTest, StlDenseSolverResize)
{
    int n = 20;
    StlWrapper A(40, 40);
    A.resize(n, n);
    A.random();

    StlWrapper B(n, 1);
    B.random();

    StlWrapper C = B.copy();
    C(n-1, 0) = 0.0;
    B -= C;
    B = B * B.transpose();

    StlWrapper X;

    Solver<StlWrapper, StlWrapper, StlWrapper> solver(A, B, B);

    solver.dense_solve(A, B, X);

    // Compute the residual
    StlWrapper R = A * X + X * A.transpose() + B;
    StlWrapper R_exp(n, n);
    R_exp = 0.0;

    EXPECT_VECTOR_NEAR(R_exp, R);
}

TEST(LyapunovSolverTest, StlSolver)
{
    int n = 20;
    StlWrapper A(n, n);
    A.random();

    StlWrapper B(n, 1);
    B.random();

    StlWrapper C = B.copy();
    C(n-1, 0) = 0.0;
    B -= C;

    StlWrapper X(n,1);
    StlWrapper T;

    StlWrapper R;
    StlWrapper R_exp(n, n);
    R_exp = 0.0;

    Solver<StlWrapper, StlWrapper, StlWrapper> solver(A, B, B);

    int ret = solver.solve(X, T);
    EXPECT_EQ(0, ret);

    // Compute the residual
    R = A * X * T * X.transpose()
        + X * T * X.transpose() * A.transpose() + B * B.transpose();

    EXPECT_VECTOR_NEAR(R_exp, R);

    // Solve twice
    ret = solver.solve(X, T);
    EXPECT_EQ(0, ret);

    // Compute the residual
    R = A * X * T * X.transpose()
        + X * T * X.transpose() * A.transpose() + B * B.transpose();

    EXPECT_VECTOR_NEAR(R_exp, R);
}

class ParameterList
{
    std::map<std::string, double> params_;

public:
    template<typename T>
    T get(std::string const &name, T def)
        {
            auto it = params_.find(name);
            if (it == params_.end())
                return def;
            return it->second;
        }

    template<typename T>
    void set(std::string const &name, T val)
        {
            params_[name] = val;
        }
};

void get_tridiagonal_problem(int n, StlWrapper &A, StlWrapper &B)
{
    A = StlWrapper(n, n);
    A.random();

    // Make A tridiagonal
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (std::abs(i - j) > 1)
                A(i, j) = 0.0;
            else if (i == j)
                A(i, j) *= 3.0;

    B = StlWrapper(n, 1);
    B.random();

    StlWrapper C = B.copy();
    C(n-1, 0) = 0.0;
    B -= C;
}

TEST(LyapunovSolverTest, StlSolverRestart)
{
    int n = 20;
    StlWrapper A, B;
    get_tridiagonal_problem(n, A, B);

    StlWrapper X(n,1);
    StlWrapper T;

    Solver<StlWrapper, StlWrapper, StlWrapper> solver(A, B, B);

    ParameterList params;
    params.set("Restart Size", 19);
    params.set("Reduced Size", 15);
    params.set("Expand Size", 1);
    params.set("Minimize solution space", false);
    solver.set_parameters(params);

    int ret = solver.solve(X, T);
    EXPECT_EQ(0, ret);

    // Compute the residual
    StlWrapper R = A * X * T * X.transpose()
        + X * T * X.transpose() * A.transpose() + B * B.transpose();
    StlWrapper R_exp(n, n);
    R_exp = 0.0;

    EXPECT_GT(n, X.N());

    EXPECT_VECTOR_NEAR(R_exp, R);
}

TEST(LyapunovSolverTest, StlSolverMinimize)
{
    int n = 20;
    StlWrapper A, B;
    get_tridiagonal_problem(n, A, B);

    StlWrapper X(n,1);
    StlWrapper T;

    StlWrapper R;
    StlWrapper R_exp(n, n);
    R_exp = 0.0;

    Solver<StlWrapper, StlWrapper, StlWrapper> solver(A, B, B);

    ParameterList params;
    params.set("Minimize solution space", false);
    params.set("Tolerance", 1e-8);
    solver.set_parameters(params);

    int ret = solver.solve(X, T);
    EXPECT_EQ(0, ret);

    // Compute the residual
    R = A * X * T * X.transpose()
        + X * T * X.transpose() * A.transpose() + B * B.transpose();

    EXPECT_EQ(n, X.N());

    EXPECT_VECTOR_NEAR(R_exp, R);
    params.set("Minimize solution space", true);
    solver.set_parameters(params);

    ret = solver.solve(X, T);
    EXPECT_EQ(0, ret);

    // Compute the residual
    R = A * X * T * X.transpose()
        + X * T * X.transpose() * A.transpose() + B * B.transpose();

    EXPECT_GT(n, X.N());

    EXPECT_VECTOR_NEAR(R_exp, R);
}

TEST(LyapunovSolverTest, StlSolverRestartIterations)
{
    int n = 20;
    StlWrapper A, B;
    get_tridiagonal_problem(n, A, B);

    StlWrapper X(n,1);
    StlWrapper T;

    StlWrapper R;
    StlWrapper R_exp(n, n);
    R_exp = 0.0;

    Solver<StlWrapper, StlWrapper, StlWrapper> solver(A, B, B);

    ParameterList params;
    params.set("Restart iterations", 10);
    params.set("Minimize solution space", false);
    params.set("Expand size", 1);
    solver.set_parameters(params);

    int ret = solver.solve(X, T);
    EXPECT_EQ(0, ret);

    // Compute the residual
    R = A * X * T * X.transpose()
        + X * T * X.transpose() * A.transpose() + B * B.transpose();

    EXPECT_GT(n, X.N());

    EXPECT_VECTOR_NEAR(R_exp, R);
}

TEST(LyapunovSolverTest, StlSolverRestartFromSolution)
{
    int n = 20;
    StlWrapper A, B;
    get_tridiagonal_problem(n, A, B);

    StlWrapper X(n,1);
    StlWrapper T;

    Solver<StlWrapper, StlWrapper, StlWrapper> solver(A, B, B);

    ParameterList params;
    params.set("Minimize solution space", true);
    params.set("Tolerance", 1e-8);
    solver.set_parameters(params);

    int ret = solver.solve(X, T);
    EXPECT_EQ(0, ret);

    EXPECT_GT(n, X.N());

    A(n-1,n-1) = 4.0;

    Solver<StlWrapper, StlWrapper, StlWrapper> solver2(A, B, B);

    params.set("Restart from solution", true);
    solver2.set_parameters(params);

    ret = solver2.solve(X, T);
    EXPECT_EQ(0, ret);

    // Compute the residual
    StlWrapper R = A * X * T * X.transpose()
        + X * T * X.transpose() * A.transpose() + B * B.transpose();
    StlWrapper R_exp(n, n);
    R_exp = 0.0;

    EXPECT_GT(n, X.N());

    EXPECT_VECTOR_NEAR(R_exp, R);
}
