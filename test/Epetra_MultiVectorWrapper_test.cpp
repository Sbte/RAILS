#include <limits.h>
#include "gtest/gtest.h"

#include "src/Epetra_MultiVectorWrapper.hpp"

#include <Teuchos_RCP.hpp>

#include <Epetra_Map.h>
#include <Epetra_MultiVector.h>
#include <Epetra_SerialComm.h>

#define EXPECT_VECTOR_EQ(a, b) {                                        \
        int m = (b).MyLength();                                         \
        int n = (b).NumVectors();                                       \
        for (int i = 0; i < m; i++)                                     \
            for (int j = 0; j < n; j++)                                 \
                EXPECT_DOUBLE_EQ((a)[j][i], (b)[j][i]);                 \
    }

class Epetra_MultiVectorWrapperTest : public ::testing::Test
{
protected:
    Teuchos::RCP<Epetra_Map> map;
    Teuchos::RCP<Epetra_Comm> comm;

    Epetra_MultiVectorWrapperTest()
        {
        }

    virtual ~Epetra_MultiVectorWrapperTest()
        {
        }

    virtual void SetUp()
        {
            comm = Teuchos::rcp(new Epetra_SerialComm);
            map = Teuchos::rcp(new Epetra_Map(10, 0, *comm));
        }

    virtual void TearDown()
        {
        }
};

TEST_F(Epetra_MultiVectorWrapperTest, VectorAssignment)
{
    Teuchos::RCP<Epetra_MultiVector> a = Teuchos::rcp(new Epetra_MultiVector(*map, 1));
    a->Random();

    Teuchos::RCP<Epetra_MultiVector> b = Teuchos::rcp(new Epetra_MultiVector(*a));
    b->Update(1.0, *a, 1.0);

    Epetra_MultiVectorWrapper aw(a);
    Epetra_MultiVectorWrapper bw(aw);
    bw += aw;

    EXPECT_VECTOR_EQ(*a, *aw);
    EXPECT_VECTOR_EQ(*b, *bw);
}

TEST_F(Epetra_MultiVectorWrapperTest, VectorAdditionAssignment)
{
    Teuchos::RCP<Epetra_MultiVector> a = Teuchos::rcp(new Epetra_MultiVector(*map, 1));
    a->Random();

    Teuchos::RCP<Epetra_MultiVector> b = Teuchos::rcp(new Epetra_MultiVector(*a));
    b->Update(1.0, *a, 1.0);

    Epetra_MultiVectorWrapper aw(a);
    Epetra_MultiVectorWrapper bw(aw);
    bw += aw;

    EXPECT_VECTOR_EQ(*a, *aw);
    EXPECT_VECTOR_EQ(*b, *bw);
}

TEST_F(Epetra_MultiVectorWrapperTest, SubtractionAssignment)
{
    Teuchos::RCP<Epetra_MultiVector> a = Teuchos::rcp(new Epetra_MultiVector(*map, 1));
    a->Random();

    Teuchos::RCP<Epetra_MultiVector> b = Teuchos::rcp(new Epetra_MultiVector(*a));
    b->Update(-1.0, *a, 1.0);

    Epetra_MultiVectorWrapper aw(a);
    Epetra_MultiVectorWrapper bw(aw);
    bw -= aw;

    EXPECT_VECTOR_EQ(*a, *aw);
    EXPECT_VECTOR_EQ(*b, *bw);
}

TEST_F(Epetra_MultiVectorWrapperTest, MultiplicationAssignment)
{
    Teuchos::RCP<Epetra_MultiVector> a = Teuchos::rcp(new Epetra_MultiVector(*map, 1));
    a->Random();

    Teuchos::RCP<Epetra_MultiVector> b = Teuchos::rcp(new Epetra_MultiVector(*a));
    b->Scale(13.0);

    Epetra_MultiVectorWrapper aw(a);
    aw *= 13;

    EXPECT_VECTOR_EQ(*a, *aw);
    EXPECT_VECTOR_EQ(*b, *aw);
}

TEST_F(Epetra_MultiVectorWrapperTest, DivisionAssignment)
{
    Teuchos::RCP<Epetra_MultiVector> a = Teuchos::rcp(new Epetra_MultiVector(*map, 1));
    a->Random();

    Teuchos::RCP<Epetra_MultiVector> b = Teuchos::rcp(new Epetra_MultiVector(*a));
    b->Scale(1.0 / 13.0);

    Epetra_MultiVectorWrapper aw(a);
    aw /= 13;

    EXPECT_VECTOR_EQ(*a, *aw);
    EXPECT_VECTOR_EQ(*b, *aw);
}

TEST_F(Epetra_MultiVectorWrapperTest, Addition)
{
    Teuchos::RCP<Epetra_MultiVector> a = Teuchos::rcp(new Epetra_MultiVector(*map, 1));
    a->Random();

    Teuchos::RCP<Epetra_MultiVector> b = Teuchos::rcp(new Epetra_MultiVector(*a));
    b->Scale(2.0);

    Teuchos::RCP<Epetra_MultiVector> c = Teuchos::rcp(new Epetra_MultiVector(*a));
    c->Update(1.0, *b, 1.0);

    Epetra_MultiVectorWrapper aw(a);
    Epetra_MultiVectorWrapper bw(b);
    Epetra_MultiVectorWrapper cw = aw + bw;

    EXPECT_VECTOR_EQ(*c, *cw);
}

TEST_F(Epetra_MultiVectorWrapperTest, Multiplication)
{
    Teuchos::RCP<Epetra_MultiVector> a = Teuchos::rcp(new Epetra_MultiVector(*map, 1));
    a->Random();

    Teuchos::RCP<Epetra_MultiVector> c = Teuchos::rcp(new Epetra_MultiVector(*a));
    c->Scale(13.0);

    Epetra_MultiVectorWrapper aw(a);
    Epetra_MultiVectorWrapper cw = 13 * aw;

    EXPECT_VECTOR_EQ(*c, *cw);
}

TEST_F(Epetra_MultiVectorWrapperTest, Norm)
{
    Teuchos::RCP<Epetra_MultiVector> a = Teuchos::rcp(new Epetra_MultiVector(*map, 2));
    a->Random();

    Epetra_MultiVectorWrapper aw(a);

    double nrm[2];
    a->Norm2(nrm);

    EXPECT_DOUBLE_EQ(nrm[0], aw.view(0).norm());
    EXPECT_DOUBLE_EQ(nrm[1], aw.view(1).norm());

    aw.view(0) /= aw.view(0).norm();

    EXPECT_DOUBLE_EQ(1.0, aw.view(0).norm());
}

TEST_F(Epetra_MultiVectorWrapperTest, Orthogonalize)
{
    Teuchos::RCP<Epetra_MultiVector> a = Teuchos::rcp(new Epetra_MultiVector(*map, 2));
    a->ReplaceGlobalValue(0, 0, 2.3);
    a->ReplaceGlobalValue(0, 1, 5.3);
    a->ReplaceGlobalValue(1, 1, 2.7);

    Epetra_MultiVectorWrapper aw(a);
    aw.orthogonalize();

    Teuchos::RCP<Epetra_MultiVector> c = Teuchos::rcp(new Epetra_MultiVector(*map, 2));
    c->ReplaceGlobalValue(0, 0, 1.0);
    c->ReplaceGlobalValue(1, 1, 1.0);

    EXPECT_VECTOR_EQ(*c, *aw);
}

TEST_F(Epetra_MultiVectorWrapperTest, Orthogonalize2)
{
    Teuchos::RCP<Epetra_MultiVector> a = Teuchos::rcp(new Epetra_MultiVector(*map, 1));
    a->ReplaceGlobalValue(0, 0, 2.3);

    Teuchos::RCP<Epetra_MultiVector> c = Teuchos::rcp(new Epetra_MultiVector(*map, 2));
    c->ReplaceGlobalValue(0, 0, 1.0);
    c->ReplaceGlobalValue(1, 1, 1.0);
 
    Epetra_MultiVectorWrapper aw(a);
    aw.orthogonalize();

    EXPECT_VECTOR_EQ(*c, *aw);

    Teuchos::RCP<Epetra_MultiVector> b = Teuchos::rcp(new Epetra_MultiVector(*map, 1));
    b->ReplaceGlobalValue(0, 0, 5.3);
    b->ReplaceGlobalValue(1, 0, 2.7);

    aw.push_back(b);
    aw.orthogonalize();

    EXPECT_VECTOR_EQ(*c, *aw);
}

TEST_F(Epetra_MultiVectorWrapperTest, Resize)
{
    Teuchos::RCP<Epetra_MultiVector> a = Teuchos::rcp(new Epetra_MultiVector(*map, 1));
    Epetra_MultiVectorWrapper aw(a);
    int N;

    aw.resize(0);
    N = aw.N();
    EXPECT_EQ(0, N);

    aw.resize(0);
    N = aw.N();
    EXPECT_EQ(0, N);

    aw.resize(2);
    N = aw.N();
    EXPECT_EQ(2, N);
    N = (*aw).NumVectors();
    EXPECT_EQ(2, N);

    aw.resize(1);
    N = aw.N();
    EXPECT_EQ(1, N);
    N = (*aw).NumVectors();
    EXPECT_EQ(1, N);
}

TEST_F(Epetra_MultiVectorWrapperTest, Resize2)
{
    // This is to test that if we copy a vector using =, after this
    // the memory that was allocated is still belonging to what is
    // in the vector and not what was there previously

    Teuchos::RCP<Epetra_MultiVector> a = Teuchos::rcp(new Epetra_MultiVector(*map, 10));
    Epetra_MultiVectorWrapper aw(a);
    aw.random();
    aw.resize(0);

    Teuchos::RCP<Epetra_MultiVector> b = Teuchos::rcp(new Epetra_MultiVector(*map, 5));
    Epetra_MultiVectorWrapper bw(b);
    bw.random();

    aw = bw;
    EXPECT_VECTOR_EQ(*bw, *aw);

    aw.resize(bw.N());
    EXPECT_VECTOR_EQ(*bw, *aw);
}

TEST_F(Epetra_MultiVectorWrapperTest, View)
{
    Teuchos::RCP<Epetra_MultiVector> a = Teuchos::rcp(new Epetra_MultiVector(*map, 1));
    a->Random();

    Epetra_MultiVectorWrapper aw(a);

    Teuchos::RCP<Epetra_MultiVector> b = Teuchos::rcp(new Epetra_MultiVector(*a));
    b->PutScalar(1.2);

    Epetra_MultiVectorWrapper bw(b);

    aw.view(0) = bw;

    EXPECT_VECTOR_EQ(*b, *aw);
}

TEST_F(Epetra_MultiVectorWrapperTest, View2)
{
    Teuchos::RCP<Epetra_MultiVector> a = Teuchos::rcp(new Epetra_MultiVector(*map, 1));
    a->Random();

    Epetra_MultiVectorWrapper aw(a);

    Teuchos::RCP<Epetra_MultiVector> b = Teuchos::rcp(new Epetra_MultiVector(*a));
    b->PutScalar(1.2);

    aw.view(0) = Epetra_MultiVectorWrapper(b);

    EXPECT_VECTOR_EQ(*b, *aw);
}
