#include "gtest/gtest.h"

#include "src/StlWrapper.hpp"

#include "Epetra_TestableWrappers.hpp"

#define EXPECT_VECTOR_EQ(a, b) {                \
    int m = (a).M();                            \
    int n = (a).N();                            \
    for (int i = 0; i < m; i++)                 \
      for (int j = 0; j < n; j++)               \
        EXPECT_DOUBLE_EQ((a)(i,j), (b)(i,j));   \
    }

template <typename A, typename B>
struct TypeDefinitions
{
    typedef A OperatorWrapper;
    typedef B MultiVectorWrapper;
};

template <typename T>
class GenericOperatorWrapperTest: public testing::Test
{
protected:
    typename T::OperatorWrapper A;
    typename T::MultiVectorWrapper a;
    typename T::MultiVectorWrapper b;
    typename T::MultiVectorWrapper c;

    GenericOperatorWrapperTest()
        :
        A(2, 2),
        a(2, 1),
        b(2, 1),
        c(2, 1)
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

typedef Types<TypeDefinitions<StlWrapper, StlWrapper>,
              TypeDefinitions<TestableEpetra_OperatorWrapper,
                              TestableEpetra_MultiVectorWrapper> > Implementations;

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

#endif
