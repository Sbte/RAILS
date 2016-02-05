#include "gtest/gtest.h"

#include "src/StlWrapper.hpp"

#include "src/Epetra_SerialDenseMatrixWrapper.hpp"
#include "src/Epetra_MultiVectorWrapper.hpp"

#define EXPECT_VECTOR_EQ(a, b) {                \
    int m = (a).M();                            \
    int n = (a).N();                            \
    for (int i = 0; i < m; i++)                 \
      for (int j = 0; j < n; j++)               \
        EXPECT_DOUBLE_EQ((a)(i,j), (b)(i,j));   \
    }


class TestableEpetra_MultiVectorWrapper: public Epetra_MultiVectorWrapper
{
public:
    TestableEpetra_MultiVectorWrapper(Epetra_MultiVectorWrapper const &other)
        :
        Epetra_MultiVectorWrapper(other)
        {}

    TestableEpetra_MultiVectorWrapper(int m, int n)
        :
        Epetra_MultiVectorWrapper(m, n)
        {}

    double &operator ()(int m, int n = 0)
        {
            return Epetra_MultiVectorWrapper::operator()(m, n);
        }

    double const &operator ()(int m, int n = 0) const
        {
            return Epetra_MultiVectorWrapper::operator()(m, n);
        }
};
    
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

typedef Types<StlWrapper, TestableEpetra_MultiVectorWrapper> Implementations;

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

TYPED_TEST(GenericMultiVectorWrapperTest, Scale)
{
    this->a.random();

    this->b = this->a.copy();
    this->b.scale(2.5);

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

    this->a.scale(2.0);

    EXPECT_VECTOR_EQ(this->a, this->b);
}

TYPED_TEST(GenericMultiVectorWrapperTest, SubtractionAssignment)
{
    this->a.random();

    this->b = this->a.copy();
    this->b -= this->a;

    this->a.scale(0.0);

    EXPECT_VECTOR_EQ(this->a, this->b);
}

TYPED_TEST(GenericMultiVectorWrapperTest, MultiplicationAssignment)
{
    this->a.random();

    this->b = this->a.copy();
    this->b.scale(13.0);

    this->a *= 13;

    EXPECT_VECTOR_EQ(this->a, this->b);
}

TYPED_TEST(GenericMultiVectorWrapperTest, DivisionAssignment)
{
    this->a.random();

    this->b = this->a.copy();
    this->b.scale(1.0 / 13.0);

    this->a /= 13;

    EXPECT_VECTOR_EQ(this->a, this->b);
}

TYPED_TEST(GenericMultiVectorWrapperTest, Addition)
{
    this->a.random();

    this->b = this->a.copy();
    this->b.scale(2.0);

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
    this->b.scale(13.0);

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

TYPED_TEST(GenericMultiVectorWrapperTest, Orthogonalize)
{
    this->resize(2);

    this->a.scale(0.0);
    this->a(0, 0) = 2.3;
    this->a(0, 1) = 5.3;
    this->a(1, 1) = 2.7;
    this->a.orthogonalize();

    this->b.scale(0.0);
    this->b(0, 0) = 1.0;
    this->b(1, 1) = 1.0;

    EXPECT_VECTOR_EQ(this->b, this->a);
}

TYPED_TEST(GenericMultiVectorWrapperTest, Orthogonalize2)
{
    this->a.scale(0.0);
    this->a(0, 0) = 2.3;

    this->b.scale(0.0);
    this->b(0, 0) = 1.0;

    this->a.orthogonalize();
    EXPECT_VECTOR_EQ(this->b, this->a);

    this->b.resize(2);
    this->b.scale(0.0);
    this->b(0, 0) = 1.0;
    this->b(1, 1) = 1.0;

    this->c.scale(0.0);
    this->c(0, 0) = 5.3;
    this->c(1, 0) = 2.7;

    this->a.push_back(this->c);
    this->a.orthogonalize();

    EXPECT_VECTOR_EQ(this->b, this->a);
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

TYPED_TEST(GenericMultiVectorWrapperTest, View)
{
    this->a.random();

    this->b = this->a.copy();
    this->b.random();

    this->a.view(0) = this->b;

    EXPECT_VECTOR_EQ(this->b, this->a);
}

#endif
