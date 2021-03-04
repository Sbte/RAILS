#include "gtest/gtest.h"

#include "TestHelpers.hpp"

#include <cmath>
#include "src/StlWrapper.hpp"

#ifdef ENABLE_TRILINOS
#include "Epetra_TestableWrappers.hpp"
#endif

using namespace RAILS;

template <class MultiVectorWrapper>
class GenericMultiVectorWrapperTest: public testing::Test
{
protected:
    MultiVectorWrapper a;
    MultiVectorWrapper b;
    MultiVectorWrapper c;
    MultiVectorWrapper d;

    GenericMultiVectorWrapperTest()
        :
        a(10, 10),
        b(10, 10),
        c(10, 10),
        d(10, 10)
        {}

    virtual ~GenericMultiVectorWrapperTest() {}

    virtual void SetUp()
        {
            resize(1);
        }

    virtual void TearDown()
        {
        }

    void resize(int n)
        {
            a.resize(n);
            b.resize(n);
            c.resize(n);
            d.resize(n);
        }
};

#if GTEST_HAS_TYPED_TEST

using testing::Types;

#ifdef ENABLE_TRILINOS
typedef Types<StlWrapper, TestableEpetra_MultiVectorWrapper> Implementations;
#else
typedef Types<StlWrapper> Implementations;
#endif

TYPED_TEST_CASE(GenericMultiVectorWrapperTest, Implementations);

TYPED_TEST(GenericMultiVectorWrapperTest, Resize)
{
    int N;

    this->a.resize(0);
    N = this->a.N();
    EXPECT_EQ(0, N);

    this->a.resize(0);
    N = this->a.N();
    EXPECT_EQ(0, N);

    this->a.resize(2);
    N = this->a.N();
    EXPECT_EQ(2, N);

    this->a.resize(1);
    N = this->a.N();
    EXPECT_EQ(1, N);
}

TYPED_TEST(GenericMultiVectorWrapperTest, PutScalar)
{
    this->a = 2.0;

    for (int i = 0; i < this->a.M(); ++i)
        for (int j = 0; j < this->a.N(); ++j)
           this->b(i, j) = 2.0;

    EXPECT_VECTOR_EQ(this->b, this->a);
}

TYPED_TEST(GenericMultiVectorWrapperTest, Assignment)
{
    this->a.random();

    this->b = this->a;
    this->b = 2.0;

    EXPECT_VECTOR_EQ(this->b, this->a);
}

TYPED_TEST(GenericMultiVectorWrapperTest, MultiplicationAssignment)
{
    this->a.random();

    this->b = this->a.copy();
    this->b *= 2.5;

    for (int i = 0; i < this->a.M(); ++i)
        for (int j = 0; j < this->a.N(); ++j)
           this->c(i, j) = this->a(i, j) * 2.5;

    EXPECT_VECTOR_EQ(this->c, this->b);
}

TYPED_TEST(GenericMultiVectorWrapperTest, AdditionAssignment)
{
    this->a.random();

    this->b = this->a.copy();
    this->b += this->a;

    this->a *= 2.0;

    EXPECT_VECTOR_EQ(this->a, this->b);
}

TYPED_TEST(GenericMultiVectorWrapperTest, SubtractionAssignment)
{
    this->a.random();

    this->b = this->a.copy();
    this->b -= this->a;

    this->a = 0.0;

    EXPECT_VECTOR_EQ(this->a, this->b);
}

TYPED_TEST(GenericMultiVectorWrapperTest, DivisionAssignment)
{
    this->a.random();

    this->b = this->a.copy();
    this->b *= 1.0 / 13.0;

    this->a /= 13;

    EXPECT_VECTOR_EQ(this->a, this->b);
}

TYPED_TEST(GenericMultiVectorWrapperTest, Addition)
{
    this->a.random();

    this->b = this->a.copy();
    this->b *= 2.0;

    this->c = this->a.copy();
    for (int i = 0; i < this->a.M(); ++i)
        for (int j = 0; j < this->a.N(); ++j)
           this->c(i, j) += this->b(i, j);

    this->d = this->a + this->b;

    EXPECT_VECTOR_EQ(this->c, this->d);
}

TYPED_TEST(GenericMultiVectorWrapperTest, Multiplication)
{
    this->a.random();

    this->b = this->a.copy();
    this->b *= 13.0;

    this->c = 13 * this->a;

    EXPECT_VECTOR_EQ(this->b, this->c);
}

TYPED_TEST(GenericMultiVectorWrapperTest, Norm)
{
    this->a.random();

    double nrm = 0.0;
    for (int i = 0; i < this->a.M(); ++i)
        for (int j = 0; j < this->a.N(); ++j)
           nrm += this->a(i, j) * this->a(i, j);

    nrm = sqrt(nrm);

    EXPECT_DOUBLE_EQ(nrm, this->a.norm());

    this->a /= this->a.norm();

    EXPECT_DOUBLE_EQ(1.0, this->a.norm());
}

TYPED_TEST(GenericMultiVectorWrapperTest, NormView)
{
    this->resize(2);
    this->a.random();

    double nrm1 = 0.0;
    for (int i = 0; i < this->a.M(); ++i)
           nrm1 += this->a(i, 0) * this->a(i, 0);
    nrm1 = sqrt(nrm1);

    EXPECT_NE(0.0, nrm1);
    EXPECT_DOUBLE_EQ(nrm1, this->a.view(0).norm());

    double nrm2 = 0.0;
    for (int i = 0; i < this->a.M(); ++i)
           nrm2 += this->a(i, 1) * this->a(i, 1);
    nrm2 = sqrt(nrm2);

    EXPECT_NE(0.0, nrm2);
    EXPECT_DOUBLE_EQ(nrm2, this->a.view(1).norm());

    EXPECT_EQ(2, this->a.N());
    EXPECT_NE(nrm1, nrm2);
    EXPECT_NE(nrm2, this->a.norm());
    EXPECT_NE(nrm1, this->a.norm());
}

TYPED_TEST(GenericMultiVectorWrapperTest, Dot)
{
    this->a.random();
    this->b.random();

    double sum = 0.0;
    for (int i = 0; i < this->a.M(); ++i)
        for (int j = 0; j < this->a.N(); ++j)
           sum += this->a(i, j) * this->b(i, j);

    auto c = this->a.dot(this->b);

    EXPECT_DOUBLE_EQ(sum, c(0, 0));
}

TYPED_TEST(GenericMultiVectorWrapperTest, Dot2)
{
    // Dot with unequal sizes
    this->a.resize(2);
    this->a.random();
    this->b.resize(3);
    this->b.random();

    auto c = this->a.dot(this->b);

    EXPECT_EQ(2, c.M());
    EXPECT_EQ(3, c.N());

    for (int k = 0; k < this->b.N(); ++k)
    {
        for (int j = 0; j < this->a.N(); ++j)
        {
            double sum = 0.0;
            for (int i = 0; i < this->a.M(); ++i)
                sum += this->a(i, j) * this->b(i, k);

            EXPECT_NEAR(sum, c(j, k), 1e-15);
        }
    }
}

TYPED_TEST(GenericMultiVectorWrapperTest, Orthogonalize)
{
    this->resize(2);

    this->a = 0.0;
    this->a(0, 0) = 2.3;
    this->a(0, 1) = 5.3;
    this->a(1, 1) = 2.7;
    this->a.orthogonalize();

    this->b = 0.0;
    this->b(0, 0) = 1.0;
    this->b(1, 1) = 1.0;

    EXPECT_VECTOR_EQ(this->b, this->a);
}

TYPED_TEST(GenericMultiVectorWrapperTest, Orthogonalize2)
{
    this->a = 0.0;
    this->a(0, 0) = 2.3;

    this->b = 0.0;
    this->b(0, 0) = 1.0;

    this->a.orthogonalize();
    EXPECT_VECTOR_EQ(this->b, this->a);

    this->b.resize(2);
    this->b = 0.0;
    this->b(0, 0) = 1.0;
    this->b(1, 1) = 1.0;

    this->c = 0.0;
    this->c(0, 0) = 5.3;
    this->c(1, 0) = 2.7;

    this->a.push_back(this->c);
    this->a.orthogonalize();

    EXPECT_VECTOR_EQ(this->b, this->a);
}

TYPED_TEST(GenericMultiVectorWrapperTest, Orthogonalize3)
{
    // Test with some random vectors. They should still be
    // orthogonal even if we don't know the solution
    this->a.resize(3);
    this->a.random();
    this->a.orthogonalize();

    EXPECT_ORTHOGONAL(this->a);

    this->b.resize(3);
    this->b.random();
    this->a.push_back(this->b);
    this->a.orthogonalize();

    EXPECT_ORTHOGONAL(this->a);

    EXPECT_EQ(6, this->a.N());
}

TYPED_TEST(GenericMultiVectorWrapperTest, Orthogonalize4)
{
    // If we set the vector in some way, we have to reorthogonalize
    this->a.resize(3);
    this->a.random();
    this->a.orthogonalize();
    EXPECT_ORTHOGONAL(this->a);

    this->a.random();
    this->a.orthogonalize();
    EXPECT_ORTHOGONAL(this->a);

    this->a *= 2.0;
    this->a.orthogonalize();
    EXPECT_ORTHOGONAL(this->a);

    this->a *= 3.3;
    this->a.orthogonalize();
    EXPECT_ORTHOGONAL(this->a);

    this->a /= 2.6;
    this->a.orthogonalize();
    EXPECT_ORTHOGONAL(this->a);

    this->a.view(2).random();
    this->a.resize(2);
    this->a.resize(3);
    this->a.orthogonalize();
    EXPECT_ORTHOGONAL(this->a);
}

TYPED_TEST(GenericMultiVectorWrapperTest, Resize2)
{
    // This is to test that if we copy a vector using =, after this
    // the memory that was allocated is still belonging to what is
    // in the vector and not what was there previously

    this->a.resize(20);
    EXPECT_EQ(20, this->a.N());

    this->a.random();
    this->a.resize(0);
    EXPECT_EQ(0, this->a.N());

    this->b.resize(10);
    EXPECT_EQ(10, this->b.N());

    this->b.random();
    this->a = this->b;
    EXPECT_EQ(10, this->a.N());
    EXPECT_VECTOR_EQ(this->b, this->a);

    this->a.resize(10);
    EXPECT_EQ(10, this->a.N());
    EXPECT_VECTOR_EQ(this->b, this->a);
}

TYPED_TEST(GenericMultiVectorWrapperTest, Resize3)
{
    // This is to test that data is still there
    // after a resize with the right capacity

    this->a.resize(1);
    this->a.random();
    this->b = this->a.copy();

    this->a.resize(10);
    this->c = this->a.view(0);

    EXPECT_VECTOR_EQ(this->b, this->c);
}

TYPED_TEST(GenericMultiVectorWrapperTest, View)
{
    this->a.random();

    this->b = this->a.copy();
    this->b.random();

    this->a.view(0) = this->b;

    EXPECT_VECTOR_EQ(this->b, this->a);
}

TYPED_TEST(GenericMultiVectorWrapperTest, View2)
{
    this->a.random();

    this->b = this->a;
    EXPECT_VECTOR_EQ(this->b, this->a);

    this->c.random();
    this->b = this->c;
    EXPECT_VECTOR_EQ(this->b, this->c);
    EXPECT_NE(this->a(0, 0), this->b(0, 0));

    this->b = this->a;
    this->b.view() = this->c;
    EXPECT_VECTOR_EQ(this->a, this->c);
    EXPECT_VECTOR_EQ(this->b, this->c);
}

TYPED_TEST(GenericMultiVectorWrapperTest, Copy)
{
    this->a.random();

    this->b = this->a;
    this->b.random();
    EXPECT_VECTOR_EQ(this->b, this->a);

    this->b = this->a.copy();
    EXPECT_VECTOR_EQ(this->b, this->a);

    this->b.random();
    EXPECT_NE(this->a(0, 0), this->b(0, 0));

    TypeParam other = this->a.copy();
    EXPECT_VECTOR_EQ(this->a, other);

    other.random();
    EXPECT_NE(this->a(0, 0), other(0, 0));
}

TYPED_TEST(GenericMultiVectorWrapperTest, PushBack)
{
    this->a.random();

    this->b = this->a.copy();
    EXPECT_VECTOR_EQ(this->b, this->a);

    this->a.resize(3);
    this->a.random();
    this->b = this->a.view(0).copy();
    this->b.push_back(this->a.view(1));
    this->b.push_back(this->a.view(2));
    EXPECT_VECTOR_EQ(this->b, this->a);

    this->a.resize(3);
    this->a.random();
    this->b = this->a.view(0).copy();
    this->b.push_back(this->a.view(1,2));
    EXPECT_VECTOR_EQ(this->b, this->a);
}

TYPED_TEST(GenericMultiVectorWrapperTest, Transpose)
{
    this->a.resize(10);
    this->a = 0.0;

    this->a(0,0) = 1;
    this->a(0,1) = 2;
    this->a(1,0) = 3;
    this->a(1,1) = 4;

    this->b.resize(1);
    this->b.random();

    this->c = this->a.transpose() * this->b;
    this->d = this->a * this->b;

    EXPECT_NEAR(this->b(0, 0) + 3.0 * this->b(1, 0), this->c(0, 0), 1e-14);
    EXPECT_NEAR(2.0 * this->b(0, 0) + 4.0 * this->b(1, 0), this->c(1, 0), 1e-14);

    EXPECT_NEAR(this->b(0, 0) + 2.0 * this->b(1, 0), this->d(0, 0), 1e-14);
    EXPECT_NEAR(3.0 * this->b(0, 0) + 4.0 * this->b(1, 0), this->d(1, 0), 1e-14);
}

TYPED_TEST(GenericMultiVectorWrapperTest, Transpose2)
{
    this->a.resize(1);
    EXPECT_EQ(10, this->a.M());
    EXPECT_EQ(1,  this->a.N());
    EXPECT_EQ(1,  this->a.transpose().M());
    EXPECT_EQ(10, this->a.transpose().N());
}

#endif
