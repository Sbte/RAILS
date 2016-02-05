#include "gtest/gtest.h"

#include "TestHelpers.hpp"

#include "src/StlWrapper.hpp"

#include "Epetra_TestableWrappers.hpp"

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

typedef Types<TypeDefinitions<StlWrapper, StlWrapper, StlWrapper>,
              TypeDefinitions<TestableEpetra_OperatorWrapper,
                              TestableEpetra_MultiVectorWrapper,
                              Epetra_SerialDenseMatrixWrapper> > Implementations;

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

template<class Matrix, class MultiVector>
class TestOperator
{
public:
    Matrix A;
    MultiVector V;

    TestOperator(Matrix _A, MultiVector _V): A(_A), V(_V) {}
    MultiVector operator *(MultiVector const &other) const
        {
            return A * other;
        }
};

TYPED_TEST(GenericOperatorWrapperTest, FromOperator)
{
    this->A(0,0) = 1;
    this->A(0,1) = 2;
    this->A(1,0) = 3;
    this->A(1,1) = 4;

    this->a.random();

    TestOperator<typename TypeParam::OperatorWrapper,
                 typename TypeParam::MultiVectorWrapper> op(this->A, this->a);
    typename TypeParam::OperatorWrapper mat =
        TypeParam::OperatorWrapper::from_operator(op);
    this->b = mat * this->a;

    EXPECT_NEAR(this->a(0, 0) + 2.0 * this->a(1, 0), this->b(0, 0), 1e-14);
    EXPECT_NEAR(3.0 * this->a(0, 0) + 4.0 * this->a(1, 0), this->b(1, 0), 1e-14);
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

    TestOperator<typename TypeParam::OperatorWrapper,
                 typename TypeParam::MultiVectorWrapper> op(this->E, this->V);
    typename TypeParam::OperatorWrapper mat =
        TypeParam::OperatorWrapper::from_operator(op);

    mat.eigs(this->V, this->D, 2, 1e-6);

    EXPECT_NEAR(10, this->D(0,0), 1e-6);
    EXPECT_NEAR(9, this->D(1,0), 1e-6);
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

    // Check if orthogonaliozing works, which we do in the solver

    TestOperator<typename TypeParam::OperatorWrapper,
                 typename TypeParam::MultiVectorWrapper> op(this->E, this->V);
    typename TypeParam::OperatorWrapper mat =
        TypeParam::OperatorWrapper::from_operator(op);

    mat.eigs(this->V, this->D, 2, 1e-6);

    EXPECT_NEAR(10, this->D(0,0), 1e-6);
    EXPECT_NEAR(9, this->D(1,0), 1e-6);

    this->V.orthogonalize();
    EXPECT_ORTHOGONAL(this->V);

    mat.eigs(this->V, this->D, 2, 1e-6);

    EXPECT_NEAR(10, this->D(0,0), 1e-6);
    EXPECT_NEAR(9, this->D(1,0), 1e-6);

    this->V.orthogonalize();
    EXPECT_ORTHOGONAL(this->V);
}

TYPED_TEST(GenericOperatorWrapperTest, Eigs4)
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

    TestOperator<typename TypeParam::OperatorWrapper,
                 typename TypeParam::MultiVectorWrapper> op(this->E, this->V);
    typename TypeParam::OperatorWrapper mat =
        TypeParam::OperatorWrapper::from_operator(op);

    mat.eigs(this->V, this->D, 2, 9.5);

    EXPECT_NEAR(10, this->D(0,0), 1e-1);

    EXPECT_EQ(1, this->D.M());
}

#endif
