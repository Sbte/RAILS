#include <limits.h>
#include "gtest/gtest.h"

#include "src/Epetra_OperatorWrapper.hpp"
#include "src/Epetra_MultiVectorWrapper.hpp"

#include <Teuchos_RCP.hpp>

#include <Epetra_Map.h>
#include <Epetra_MultiVector.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_SerialComm.h>

using namespace RAILS;

class Epetra_OperatorWrapperTest : public ::testing::Test
{
protected:
    Teuchos::RCP<Epetra_Map> map;
    Teuchos::RCP<Epetra_Comm> comm;

    Epetra_OperatorWrapperTest()
        {
        }

    virtual ~Epetra_OperatorWrapperTest()
        {
        }

    virtual void SetUp()
        {
            comm = Teuchos::rcp(new Epetra_SerialComm);
            map = Teuchos::rcp(new Epetra_Map(2, 0, *comm));
        }

    virtual void TearDown()
        {
        }
};

TEST_F(Epetra_OperatorWrapperTest, Apply)
{
    Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *map, 2));

    // A = [1,2;3,4];
    double A_val[4] = {1, 2, 3, 4};
    int A_idx[2] = {0, 1};
    A->InsertGlobalValues(0, 2, A_val, A_idx);
    A->InsertGlobalValues(1, 2, A_val+2, A_idx);
    A->FillComplete();

    Teuchos::RCP<Epetra_MultiVector> a = Teuchos::rcp(new Epetra_MultiVector(*map, 1));
    a->Random();

    Epetra_OperatorWrapper Aw(A);
    Epetra_MultiVectorWrapper aw(a);
    Epetra_MultiVectorWrapper bw = Aw * aw;

    EXPECT_NEAR((*a)[0][0] + 2.0 * (*a)[0][1], (*bw)[0][0], 1e-14);
    EXPECT_NEAR(3.0 * (*a)[0][0] + 4.0 * (*a)[0][1], (*bw)[0][1], 1e-14);
}

TEST_F(Epetra_OperatorWrapperTest, ApplyMaps)
{
    Teuchos::RCP<Epetra_Map> map2 = Teuchos::rcp(new Epetra_Map(1, 0, *comm));
    Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *map2, *map, 2));

    // A = [1,2];
    double A_val[2] = {1, 2};
    int A_idx[2] = {0, 1};
    A->InsertGlobalValues(0, 2, A_val, A_idx);
    A->FillComplete(*map, *map2);

    Teuchos::RCP<Epetra_MultiVector> a = Teuchos::rcp(new Epetra_MultiVector(*map, 1));
    a->Random();

    Epetra_OperatorWrapper Aw(A);
    Epetra_MultiVectorWrapper aw(a);
    Epetra_MultiVectorWrapper bw = Aw * aw;

    EXPECT_EQ(1, bw.M());
    EXPECT_NEAR((*a)[0][0] + 2.0 * (*a)[0][1], (*bw)[0][0], 1e-14);
}

TEST_F(Epetra_OperatorWrapperTest, Transpose)
{
    Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *map, 2));

    // A = [1,2;3,4];
    double A_val[4] = {1, 2, 3, 4};
    int A_idx[2] = {0, 1};
    A->InsertGlobalValues(0, 2, A_val, A_idx);
    A->InsertGlobalValues(1, 2, A_val+2, A_idx);
    A->FillComplete();

    Teuchos::RCP<Epetra_MultiVector> a = Teuchos::rcp(new Epetra_MultiVector(*map, 1));
    a->Random();

    Epetra_OperatorWrapper Aw(A);
    Epetra_MultiVectorWrapper aw(a);
    Epetra_MultiVectorWrapper bw = Aw.transpose() * aw;
    Epetra_MultiVectorWrapper cw = Aw * aw;

    EXPECT_NEAR((*a)[0][0] + 3.0 * (*a)[0][1], (*bw)[0][0], 1e-14);
    EXPECT_NEAR(2.0 * (*a)[0][0] + 4.0 * (*a)[0][1], (*bw)[0][1], 1e-14);

    EXPECT_NEAR((*a)[0][0] + 2.0 * (*a)[0][1], (*cw)[0][0], 1e-14);
    EXPECT_NEAR(3.0 * (*a)[0][0] + 4.0 * (*a)[0][1], (*cw)[0][1], 1e-14);

    EXPECT_EQ(false, A->UseTranspose());
}

TEST_F(Epetra_OperatorWrapperTest, Transpose2)
{
    Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *map, 2));
    A->SetUseTranspose(true);

    // A = [1,2;3,4];
    double A_val[4] = {1, 2, 3, 4};
    int A_idx[2] = {0, 1};
    A->InsertGlobalValues(0, 2, A_val, A_idx);
    A->InsertGlobalValues(1, 2, A_val+2, A_idx);
    A->FillComplete();

    Teuchos::RCP<Epetra_MultiVector> a = Teuchos::rcp(new Epetra_MultiVector(*map, 1));
    a->Random();

    Teuchos::RCP<Epetra_MultiVector> b = Teuchos::rcp(new Epetra_MultiVector(*a));

    Epetra_OperatorWrapper Aw(A);
    Epetra_MultiVectorWrapper aw(a);
    Epetra_MultiVectorWrapper bw = Aw * aw;
    Epetra_MultiVectorWrapper cw = Aw.transpose() * aw;

    EXPECT_NEAR((*a)[0][0] + 3.0 * (*a)[0][1], (*bw)[0][0], 1e-14);
    EXPECT_NEAR(2.0 * (*a)[0][0] + 4.0 * (*a)[0][1], (*bw)[0][1], 1e-14);

    EXPECT_NEAR((*a)[0][0] + 2.0 * (*a)[0][1], (*cw)[0][0], 1e-14);
    EXPECT_NEAR(3.0 * (*a)[0][0] + 4.0 * (*a)[0][1], (*cw)[0][1], 1e-14);

    EXPECT_EQ(true, A->UseTranspose());
}

TEST_F(Epetra_OperatorWrapperTest, ApplyMapsTranspose)
{
    Teuchos::RCP<Epetra_Map> map2 = Teuchos::rcp(new Epetra_Map(1, 0, *comm));
    Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *map2, *map, 2));

    // A = [1,2];
    double A_val[2] = {1, 2};
    int A_idx[2] = {0, 1};
    A->InsertGlobalValues(0, 2, A_val, A_idx);
    A->FillComplete(*map, *map2);

    Teuchos::RCP<Epetra_MultiVector> a = Teuchos::rcp(new Epetra_MultiVector(*map2, 1));
    a->Random();

    Epetra_OperatorWrapper Aw(A);
    Epetra_MultiVectorWrapper aw(a);
    Epetra_MultiVectorWrapper bw = Aw.transpose() * aw;

    EXPECT_EQ(2, bw.M());
    EXPECT_NEAR((*a)[0][0], (*bw)[0][0], 1e-14);
    EXPECT_NEAR(2.0 * (*a)[0][0], (*bw)[0][1], 1e-14);
}
