#include <limits.h>
#include "gtest/gtest.h"

#include "src/ScalarWrapper.hpp"

TEST(ScalarWrapper, Assignment)
{
    ScalarWrapper a(1);
    ScalarWrapper b = a;
    b += ScalarWrapper(1);

    EXPECT_DOUBLE_EQ(1, a);
    EXPECT_DOUBLE_EQ(2, b);
}

TEST(ScalarWrapper, AdditionAssignment)
{
    ScalarWrapper a(1);
    ScalarWrapper b = a;
    b += a;

    EXPECT_DOUBLE_EQ(1, a);
    EXPECT_DOUBLE_EQ(2, b);
}

TEST(ScalarWrapper, SubtractionAssignment)
{
    ScalarWrapper a(1);
    ScalarWrapper b = a;
    b -= a;

    EXPECT_DOUBLE_EQ(1, a);
    EXPECT_DOUBLE_EQ(0, b);
}

TEST(ScalarWrapper, MultiplicationAssignment)
{
    ScalarWrapper a(3);
    ScalarWrapper b = a;
    b *= ScalarWrapper(2);

    EXPECT_DOUBLE_EQ(3, a);
    EXPECT_DOUBLE_EQ(6, b);
}

TEST(ScalarWrapper, DivisionAssignment)
{
    ScalarWrapper a(8);
    ScalarWrapper b = a;
    b /= ScalarWrapper(4);

    EXPECT_DOUBLE_EQ(8, a);
    EXPECT_DOUBLE_EQ(2, b);
}

TEST(ScalarWrapper, Addition)
{
    ScalarWrapper a(8);
    ScalarWrapper b(3);
    ScalarWrapper c = a + b;

    EXPECT_DOUBLE_EQ(8, a);
    EXPECT_DOUBLE_EQ(3, b);
    EXPECT_DOUBLE_EQ(11, c);
}

TEST(ScalarWrapper, Multiplication)
{
    ScalarWrapper a(8);
    ScalarWrapper b(3);
    ScalarWrapper c = a * b;

    EXPECT_DOUBLE_EQ(8, a);
    EXPECT_DOUBLE_EQ(3, b);
    EXPECT_DOUBLE_EQ(24, c);
}

TEST(ScalarWrapper, Eigs)
{
    ScalarWrapper a(3);
    ScalarWrapper v, d;
    a.eigs(v, d);

    EXPECT_DOUBLE_EQ(3, a);
    EXPECT_DOUBLE_EQ(1, v);
    EXPECT_DOUBLE_EQ(3, d);
}

TEST(ScalarWrapper, Apply)
{
    ScalarWrapper a(8);
    ScalarWrapper b(3);
    ScalarWrapper c = a * b;

    EXPECT_DOUBLE_EQ(8, a);
    EXPECT_DOUBLE_EQ(3, b);
    EXPECT_DOUBLE_EQ(24, c);
}

TEST(ScalarWrapper, Dot)
{
    ScalarWrapper a(8);
    ScalarWrapper b(3);
    ScalarWrapper c = a.dot(b);

    EXPECT_DOUBLE_EQ(8, a);
    EXPECT_DOUBLE_EQ(3, b);
    EXPECT_DOUBLE_EQ(24, c);
}

TEST(ScalarWrapper, View)
{
    ScalarWrapper a(1);
    ScalarWrapper b = a.view(0);
    b += ScalarWrapper(1);

    EXPECT_DOUBLE_EQ(2, a);
    EXPECT_DOUBLE_EQ(2, b);

    ScalarWrapper c(3);
    a.view(0) = c * a.view(0);

    EXPECT_DOUBLE_EQ(6, a);
    EXPECT_DOUBLE_EQ(6, b);
    EXPECT_DOUBLE_EQ(3, c);
}

TEST(ScalarWrapper, View2)
{
    ScalarWrapper a(1);
    a.push_back(3);
    a.view(1) = ScalarWrapper(2);
    EXPECT_DOUBLE_EQ(1, a(0));
    EXPECT_DOUBLE_EQ(2, a(1));
}
