#include "gtest/gtest.h"

#include "TestHelpers.hpp"

#include "src/StlWrapper.hpp"

#ifdef ENABLE_TRILINOS
#include "Epetra_TestableWrappers.hpp"
#endif

using namespace RAILS;

template <typename A, typename B, typename C>
struct TypeDefinitions
{
    typedef A OperatorWrapper;
    typedef B MultiVectorWrapper;
    typedef C DenseMatrixWrapper;
};

template <typename T>
class GenericOperatorWrapperTest: public testing::Test
{
protected:
    typename T::OperatorWrapper A;
    typename T::OperatorWrapper B;
    typename T::MultiVectorWrapper a;
    typename T::MultiVectorWrapper b;
    typename T::MultiVectorWrapper c;
    typename T::OperatorWrapper E;
    typename T::DenseMatrixWrapper D;
    typename T::MultiVectorWrapper V;
    typename T::DenseMatrixWrapper DV;

    GenericOperatorWrapperTest()
        :
        A(2, 2),
        B(2, 1),
        a(2, 1),
        b(2, 1),
        c(2, 1),
        E(10, 10),
        D(10, 10),
        V(10, 10)
        {}

    virtual ~GenericOperatorWrapperTest() {}

    virtual void SetUp()
        {
        }

    virtual void TearDown()
        {
        }
};

#if GTEST_HAS_TYPED_TEST

using testing::Types;

#ifdef ENABLE_TRILINOS
typedef Types<TypeDefinitions<StlWrapper, StlWrapper, StlWrapper>,
              TypeDefinitions<TestableEpetra_OperatorWrapper,
                              TestableEpetra_MultiVectorWrapper,
                              Epetra_SerialDenseMatrixWrapper> > Implementations;
#else
typedef Types<TypeDefinitions<StlWrapper, StlWrapper, StlWrapper> > Implementations;
#endif

TYPED_TEST_CASE(GenericOperatorWrapperTest, Implementations);

TYPED_TEST(GenericOperatorWrapperTest, Apply)
{
    this->A(0,0) = 1;
    this->A(0,1) = 2;
    this->A(1,0) = 3;
    this->A(1,1) = 4;

    this->a.random();

    this->b = this->A * this->a;

    EXPECT_NEAR(this->a(0, 0) + 2.0 * this->a(1, 0), this->b(0, 0), 1e-14);
    EXPECT_NEAR(3.0 * this->a(0, 0) + 4.0 * this->a(1, 0), this->b(1, 0), 1e-14);
}

TYPED_TEST(GenericOperatorWrapperTest, Transpose)
{
    this->A(0,0) = 1;
    this->A(0,1) = 2;
    this->A(1,0) = 3;
    this->A(1,1) = 4;

    this->a.random();

    this->b = this->A.transpose() * this->a;
    this->c = this->A * this->a;

    EXPECT_NEAR(this->a(0, 0) + 3.0 * this->a(1, 0), this->b(0, 0), 1e-14);
    EXPECT_NEAR(2.0 * this->a(0, 0) + 4.0 * this->a(1, 0), this->b(1, 0), 1e-14);

    EXPECT_NEAR(this->a(0, 0) + 2.0 * this->a(1, 0), this->c(0, 0), 1e-14);
    EXPECT_NEAR(3.0 * this->a(0, 0) + 4.0 * this->a(1, 0), this->c(1, 0), 1e-14);
}

TYPED_TEST(GenericOperatorWrapperTest, Transpose2)
{
    EXPECT_EQ(2, this->B.M());
    EXPECT_EQ(1, this->B.N());
    EXPECT_EQ(1, this->B.transpose().M());
    EXPECT_EQ(2, this->B.transpose().N());
}

TYPED_TEST(GenericOperatorWrapperTest, Eigs)
{
    int n = 10;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            if (i == j)
                this->E(i, i) = i + 1;
            else
                this->E(i, j) = 0.0;
        }


    this->E.eigs(this->V, this->D, 2, 1e-6);

    EXPECT_NEAR(10, this->D(0,0), 1e-6);
    EXPECT_NEAR(9, this->D(1,0), 1e-6);
}

TYPED_TEST(GenericOperatorWrapperTest, Eigs2)
{
    int n = 10;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            if (i == j)
                this->E(i, i) = i + 1;
            else
                this->E(i, j) = 0.0;
        }

    // Check if orthogonaliozing works, which we do in the solver

    this->E.eigs(this->V, this->D, 2, 1e-6);

    EXPECT_NEAR(10, this->D(0,0), 1e-6);
    EXPECT_NEAR(9, this->D(1,0), 1e-6);

    this->V.orthogonalize();
    EXPECT_ORTHOGONAL(this->V);

    this->E.eigs(this->V, this->D, 2, 1e-6);

    EXPECT_NEAR(10, this->D(0,0), 1e-6);
    EXPECT_NEAR(9, this->D(1,0), 1e-6);

    this->V.orthogonalize();
    EXPECT_ORTHOGONAL(this->V);
}

TYPED_TEST(GenericOperatorWrapperTest, Eigs3)
{
    int n = 10;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            if (i == j)
                this->E(i, i) = i + 1;
            else
                this->E(i, j) = 0.0;
        }

    // Tolerance check

    this->E.eigs(this->V, this->D, 2, 9.5);

    EXPECT_NEAR(10, this->D(0,0), 1e-1);

    EXPECT_EQ(1, this->D.M());
}

TYPED_TEST(GenericOperatorWrapperTest, Norm)
{
    int n = 10;
    this->V.view(0).random();

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            if (j == 0)
                this->E(i, j) = this->V(i, 0);
            else
                this->E(i, j) = 0.0;
        }

    EXPECT_NE(0.0, this->E.norm());
    EXPECT_DOUBLE_EQ(this->V.view(0).norm(), this->E.norm());
}

TYPED_TEST(GenericOperatorWrapperTest, Norm2)
{
    int n = 10;
    this->V.random();

    // Symmetrize
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < i; ++j)
            this->V(i, j) = this->V(j, i);

    // Copy to E
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            this->E(i, j) = this->V(i, j);

    // Compute norm using eigs
    this->V.dot(this->V).eigs(this->DV, this->D);

    // Get maximum
    double max = 0.0;
    for (int i = 0; i < n; ++i)
        max = std::max(max, sqrt(std::abs(this->D(i, 0))));

    EXPECT_NE(0.0, max);
    EXPECT_DOUBLE_EQ(max, this->E.norm());
}

#endif
