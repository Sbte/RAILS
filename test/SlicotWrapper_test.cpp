#include "gtest/gtest.h"

#include "src/SlicotWrapper.hpp"

using namespace RAILS;

TEST(SlicotWrapperTest, Scalar)
{
    double A = 2;
    double X = -4;
    int n = 1;

    double scale = 1.0;
    int info = 0;

    sb03md('C', 'X', 'N', 'T', n, &A, n, &X, n, &scale, &info);

    EXPECT_EQ(-1, X);
    EXPECT_EQ(0, info);
}

TEST(SlicotWrapperTest, Small)
{
    // A = [0,1;-5,-5];
    double A[4] = {0, -5, 1, -5};
    double X[4] = {-1, 0, 0, -1};
    int n = 2;

    double scale = 1.0;
    int info = 0;

    sb03md('C', 'X', 'N', 'T', n, A, n, X, n, &scale, &info);

    double X_exp[4] = {0.62, -0.5, -0.5, 0.6};
    for (int i = 0; i < 4; i++)
        EXPECT_NEAR(X_exp[i], X[i], 1e-14);
    EXPECT_EQ(0, info);
}
