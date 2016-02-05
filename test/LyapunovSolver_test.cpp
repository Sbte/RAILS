#include <limits.h>
#include "gtest/gtest.h"

#include "src/LyapunovSolver.hpp"
#include "src/ScalarWrapper.hpp"
#include "src/StlWrapper.hpp"

#include <map>
#include <string>

#define EXPECT_VECTOR_NEAR(a, b)                                        \
    {                                                                   \
     int m = (a).M();                                                   \
     int n = (a).N();                                                   \
     for (int i = 0; i < m; i++)                                        \
         for (int j = 0; j < n; j++)                                    \
             EXPECT_NEAR((a)(i,j), (b)(i,j), 1e-3);                     \
    }

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

    Lyapunov::Solver<StlWrapper, StlWrapper, StlWrapper> solver(A, B, B);

    solver.dense_solve(A, B, X);

    // Compute the residual
    StlWrapper R = A * X + X * A.transpose() + B;
    StlWrapper R_exp(n, n);
    R_exp.scale(0.0);

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

    Lyapunov::Solver<StlWrapper, StlWrapper, StlWrapper> solver(A, B, B);

    solver.dense_solve(A, B, X);

    // Compute the residual
    StlWrapper R = A * X + X * A.transpose() + B;
    StlWrapper R_exp(n, n);
    R_exp.scale(0.0);

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
    B = B;

    StlWrapper X(n,1);
    StlWrapper T;

    Lyapunov::Solver<StlWrapper, StlWrapper, StlWrapper> solver(A, B, B);

    solver.solve(X, T);

    // Compute the residual
    StlWrapper R = A * X * T * X.transpose()
      + X * T * X.transpose() * A.transpose() + B * B.transpose();
    StlWrapper R_exp(n, n);
    R_exp.scale(0.0);

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

TEST(LyapunovSolverTest, StlSolverRestart)
{
    int n = 20;
    StlWrapper A(n, n);
    A.random();

    StlWrapper B(n, 1);
    B.random();

    StlWrapper C = B.copy();
    C(n-1, 0) = 0.0;
    B -= C;
    B = B;

    StlWrapper X(n,1);
    StlWrapper T;

    Lyapunov::Solver<StlWrapper, StlWrapper, StlWrapper> solver(A, B, B);

    ParameterList params;
    params.set("Restart Size", 19);
    params.set("Reduced Size", 14);
    params.set("Expand Size", 1);
    solver.set_parameters(params);

    solver.solve(X, T);

    // Compute the residual
    StlWrapper R = A * X * T * X.transpose()
      + X * T * X.transpose() * A.transpose() + B * B.transpose();
    StlWrapper R_exp(n, n);
    R_exp.scale(0.0);

    EXPECT_VECTOR_NEAR(R_exp, R);
}
