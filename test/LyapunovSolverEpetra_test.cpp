#include <limits.h>
#include "gtest/gtest.h"

#include "src/LyapunovSolver.hpp"
#include "src/Epetra_OperatorWrapper.hpp"
#include "src/Epetra_MultiVectorWrapper.hpp"
#include "src/Epetra_SerialDenseMatrixWrapper.hpp"

#include <Teuchos_RCP.hpp>

#include <Epetra_LocalMap.h>
#include <Epetra_MultiVector.h>
#include <Epetra_SerialDenseMatrix.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_SerialComm.h>

using namespace RAILS;

TEST(LyapunovSolverEpetraTest, DenseSolver)
{
    Teuchos::RCP<Epetra_Operator> tmp = Teuchos::null;

    // Create the solver object
    Solver<Epetra_OperatorWrapper, Epetra_MultiVectorWrapper,
           Epetra_SerialDenseMatrixWrapper> solver(tmp, tmp, tmp);

    // A = [0,1;-5,-5];
    double A_val[4] = {0, -5, 1, -5};
    Teuchos::RCP<Epetra_SerialDenseMatrix> A = Teuchos::rcp(
        new Epetra_SerialDenseMatrix(Copy, A_val, 2, 2, 2));

    // B = [-1,0;0,-1];
    double B_val[4] = {-1, 0, 0, -1};
    Teuchos::RCP<Epetra_SerialDenseMatrix> B = Teuchos::rcp(
        new Epetra_SerialDenseMatrix(Copy, B_val, 2, 2, 2));

    Epetra_SerialDenseMatrixWrapper AW(A);
    Epetra_SerialDenseMatrixWrapper BW(B);
    Epetra_SerialDenseMatrixWrapper XW;

    int ret = solver.dense_solve(AW, BW, XW);

    EXPECT_EQ(0, ret);
    EXPECT_NEAR(-0.62, (*XW)[0][0], 1e-14);
    EXPECT_NEAR(0.5, (*XW)[1][0], 1e-14);
    EXPECT_NEAR(0.5, (*XW)[0][1], 1e-14);
    EXPECT_NEAR(-0.6, (*XW)[1][1], 1e-14);
}

TEST(LyapunovSolverEpetraTest, Solver)
{
    Teuchos::RCP<Epetra_SerialComm> comm = Teuchos::rcp(new Epetra_SerialComm);
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(2, 0, *comm));

    Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *map, 2));

    // A = [0,1;-5,-5];
    double A_val[4] = {0, 1, -5, -5};
    int A_idx[2] = {0, 1};
    A->InsertGlobalValues(0, 2, A_val, A_idx);
    A->InsertGlobalValues(1, 2, A_val+2, A_idx);
    A->FillComplete();

    Teuchos::RCP<Epetra_CrsMatrix> B = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *map, 1));

    // B = [-1,0;0,-1];
    double B_val = -1;
    int B_idx[2] = {0, 1};
    B->InsertGlobalValues(0, 1, &B_val, B_idx);
    B->InsertGlobalValues(1, 1, &B_val, B_idx+1);
    B->FillComplete();

    Teuchos::RCP<Epetra_Operator> A_operator = A;
    Teuchos::RCP<Epetra_Operator> B_operator = B;
    Teuchos::RCP<Epetra_Operator> M_operator = B;

    // Create the solver object FIXME: Use M here not B
    Solver<Epetra_OperatorWrapper, Epetra_MultiVectorWrapper,
           Epetra_SerialDenseMatrixWrapper > solver(A_operator, B_operator, M_operator);

    Teuchos::ParameterList params;
    params.set("Minimize solution space", false);
    solver.set_parameters(params);

    Teuchos::RCP<Epetra_MultiVector> V = Teuchos::rcp(new Epetra_MultiVector(*map, 2));
    Teuchos::RCP<Epetra_SerialDenseMatrix> T = Teuchos::rcp(new Epetra_SerialDenseMatrix(2, 2));

    Epetra_MultiVectorWrapper VW(V);
    Epetra_SerialDenseMatrixWrapper TW(T);

    int ret = solver.solve(VW, TW);
    EXPECT_EQ(0, ret);

    Teuchos::RCP<Epetra_MultiVector> X = Teuchos::rcp(new Epetra_MultiVector(*V));
    Teuchos::RCP<Epetra_MultiVector> tmp = Teuchos::rcp(new Epetra_MultiVector(*VW));

    Epetra_LocalMap local_map(VW.N(), 0, X->Comm());
    Teuchos::RCP<Epetra_MultiVector> T_mv = Teuchos::rcp(new Epetra_MultiVector(View, local_map, TW, VW.N(), VW.N()));

    tmp->Multiply('N', 'N', 1.0, *VW, *T_mv, 0.0);
    X->Multiply('N', 'T', 1.0, *tmp, *VW, 0.0);

    EXPECT_NEAR(0.62, (*X)[0][0], 1e-14);
    EXPECT_NEAR(-0.5, (*X)[1][0], 1e-14);
    EXPECT_NEAR(-0.5, (*X)[0][1], 1e-14);
    EXPECT_NEAR(0.6, (*X)[1][1], 1e-14);
}

TEST(LyapunovSolverEpetraTest, BVector)
{
    Teuchos::RCP<Epetra_SerialComm> comm = Teuchos::rcp(new Epetra_SerialComm);
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(2, 0, *comm));
    Teuchos::RCP<Epetra_Map> col_map = Teuchos::rcp(new Epetra_Map(1, 0, *comm));

    Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp(
        new Epetra_CrsMatrix(Copy, *map, 2));

    // A = [0,1;-5,-5];
    double A_val[4] = {0, 1, -5, -5};
    int A_idx[2] = {0, 1};
    A->InsertGlobalValues(0, 2, A_val, A_idx);
    A->InsertGlobalValues(1, 2, A_val+2, A_idx);
    A->FillComplete();

    Teuchos::RCP<Epetra_CrsMatrix> B = Teuchos::rcp(
        new Epetra_CrsMatrix(Copy, *map, *col_map, 1));

    // B = [-1;-1];
    double B_val = -1;
    int B_idx[1] = {0};
    B->InsertGlobalValues(0, 1, &B_val, B_idx);
    B->InsertGlobalValues(1, 1, &B_val, B_idx);
    B->FillComplete(*col_map, *map);

    Teuchos::RCP<Epetra_CrsMatrix> M = Teuchos::rcp(
        new Epetra_CrsMatrix(Copy, *map, 1));

    // M = [-1,0;-1,0];
    double M_val = -1;
    int M_idx[1] = {0};
    M->InsertGlobalValues(0, 1, &M_val, M_idx);
    M->InsertGlobalValues(1, 1, &M_val, M_idx);
    M->FillComplete();

    Teuchos::RCP<Epetra_Operator> A_operator = A;
    Teuchos::RCP<Epetra_Operator> B_operator = B;
    Teuchos::RCP<Epetra_Operator> M_operator = M;

    Solver<Epetra_OperatorWrapper, Epetra_MultiVectorWrapper,
           Epetra_SerialDenseMatrixWrapper > solver(A_operator, B_operator, M_operator);

    Teuchos::ParameterList params;
    params.set("Minimize solution space", false);
    solver.set_parameters(params);

    Teuchos::RCP<Epetra_MultiVector> V = Teuchos::rcp(new Epetra_MultiVector(*map, 2));
    Teuchos::RCP<Epetra_SerialDenseMatrix> T = Teuchos::rcp(new Epetra_SerialDenseMatrix(2, 2));

    Epetra_MultiVectorWrapper VW(V);
    Epetra_SerialDenseMatrixWrapper TW(T);

    int ret = solver.solve(VW, TW);
    EXPECT_EQ(0, ret);

    Teuchos::RCP<Epetra_MultiVector> X = Teuchos::rcp(new Epetra_MultiVector(*V));
    Teuchos::RCP<Epetra_MultiVector> tmp = Teuchos::rcp(new Epetra_MultiVector(*VW));

    Epetra_LocalMap local_map(VW.N(), 0, X->Comm());
    Teuchos::RCP<Epetra_MultiVector> T_mv = Teuchos::rcp(new Epetra_MultiVector(View, local_map, TW, VW.N(), VW.N()));

    tmp->Multiply('N', 'N', 1.0, *VW, *T_mv, 0.0);
    X->Multiply('N', 'T', 1.0, *tmp, *VW, 0.0);

    EXPECT_NEAR(0.82, (*X)[0][0], 1e-14);
    EXPECT_NEAR(-0.5, (*X)[1][0], 1e-14);
    EXPECT_NEAR(-0.5, (*X)[0][1], 1e-14);
    EXPECT_NEAR(0.6, (*X)[1][1], 1e-14);
}

TEST(LyapunovSolverEpetraTest, BMultiVector)
{
    Teuchos::RCP<Epetra_SerialComm> comm = Teuchos::rcp(new Epetra_SerialComm);
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(2, 0, *comm));

    Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *map, 2));

    // A = [0,1;-5,-5];
    double A_val[4] = {0, 1, -5, -5};
    int A_idx[2] = {0, 1};
    A->InsertGlobalValues(0, 2, A_val, A_idx);
    A->InsertGlobalValues(1, 2, A_val+2, A_idx);
    A->FillComplete();

    Teuchos::RCP<Epetra_CrsMatrix> M = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *map, 1));

    // M = [-1,0;-1,0];
    double M_val = -1;
    int M_idx[1] = {0};
    M->InsertGlobalValues(0, 1, &M_val, M_idx);
    M->InsertGlobalValues(1, 1, &M_val, M_idx);
    M->FillComplete();

    // B = [-1;-1];
    Teuchos::RCP<Epetra_MultiVector> B = Teuchos::rcp(new Epetra_MultiVector(*map, 1));
    B->ReplaceGlobalValue(0, 0, -1);
    B->ReplaceGlobalValue(1, 0, -1);

    Teuchos::RCP<Epetra_Operator> A_operator = A;
    Teuchos::RCP<Epetra_Operator> M_operator = M;

    Solver<Epetra_OperatorWrapper, Epetra_MultiVectorWrapper,
           Epetra_SerialDenseMatrixWrapper> solver(A_operator, B, M_operator);

    Teuchos::ParameterList params;
    params.set("Minimize solution space", false);
    solver.set_parameters(params);

    Teuchos::RCP<Epetra_MultiVector> V = Teuchos::rcp(new Epetra_MultiVector(*map, 2));
    Teuchos::RCP<Epetra_SerialDenseMatrix> T = Teuchos::rcp(new Epetra_SerialDenseMatrix(2, 2));

    Epetra_MultiVectorWrapper VW(V);
    Epetra_SerialDenseMatrixWrapper TW(T);

    int ret = solver.solve(VW, TW);
    EXPECT_EQ(0, ret);

    Teuchos::RCP<Epetra_MultiVector> X = Teuchos::rcp(new Epetra_MultiVector(*V));
    Teuchos::RCP<Epetra_MultiVector> tmp = Teuchos::rcp(new Epetra_MultiVector(*VW));

    Epetra_LocalMap local_map(VW.N(), 0, X->Comm());
    Teuchos::RCP<Epetra_MultiVector> T_mv = Teuchos::rcp(new Epetra_MultiVector(View, local_map, TW, VW.N(), VW.N()));

    tmp->Multiply('N', 'N', 1.0, *VW, *T_mv, 0.0);
    X->Multiply('N', 'T', 1.0, *tmp, *VW, 0.0);

    EXPECT_NEAR(0.82, (*X)[0][0], 1e-14);
    EXPECT_NEAR(-0.5, (*X)[1][0], 1e-14);
    EXPECT_NEAR(-0.5, (*X)[0][1], 1e-14);
    EXPECT_NEAR(0.6, (*X)[1][1], 1e-14);
}

TEST(LyapunovSolverEpetraTest, Lanczos)
{
    Teuchos::RCP<Epetra_SerialComm> comm = Teuchos::rcp(new Epetra_SerialComm);
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(2, 0, *comm));

    Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *map, 2));

    // A = [0,1;-5,-5];
    double A_val[4] = {0, 1, -5, -5};
    int A_idx[2] = {0, 1};
    A->InsertGlobalValues(0, 2, A_val, A_idx);
    A->InsertGlobalValues(1, 2, A_val+2, A_idx);
    A->FillComplete();

    Teuchos::RCP<Epetra_CrsMatrix> B = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *map, 1));

    // B = [-1,0;0,-1];
    double B_val = -1;
    int B_idx[2] = {0, 1};
    B->InsertGlobalValues(0, 1, &B_val, B_idx);
    B->InsertGlobalValues(1, 1, &B_val, B_idx+1);
    B->FillComplete();

    Teuchos::RCP<Epetra_Operator> A_operator = A;
    Teuchos::RCP<Epetra_Operator> B_operator = B;
    Teuchos::RCP<Epetra_Operator> M_operator = B;

    // Create the solver object FIXME: Use M here not B
    Solver<Epetra_OperatorWrapper, Epetra_MultiVectorWrapper,
           Epetra_SerialDenseMatrixWrapper > solver(A_operator, B_operator, M_operator);

    Teuchos::ParameterList params;
    params.set("Minimize solution space", false);
    solver.set_parameters(params);

    Teuchos::RCP<Epetra_MultiVector> V = Teuchos::rcp(new Epetra_MultiVector(*map, 2));
    Teuchos::RCP<Epetra_SerialDenseMatrix> T = Teuchos::rcp(new Epetra_SerialDenseMatrix(2, 2));

    Epetra_MultiVectorWrapper VW(V);
    Epetra_SerialDenseMatrixWrapper TW(T);
    Epetra_OperatorWrapper AW(A_operator);

    solver.solve(VW, TW);

    Teuchos::RCP<Epetra_MultiVector> X = Teuchos::rcp(new Epetra_MultiVector(*V));
    Teuchos::RCP<Epetra_MultiVector> tmp = Teuchos::rcp(new Epetra_MultiVector(*VW));

    Epetra_MultiVectorWrapper AV = AW * VW;

    int num_eigenvalues = 2;
    Epetra_SerialDenseMatrixWrapper H(num_eigenvalues + 1, num_eigenvalues + 1);
    Epetra_SerialDenseMatrixWrapper eigenvalues(num_eigenvalues + 1, 1);
    Epetra_MultiVectorWrapper eigenvectors;

    solver.resid_lanczos(AV, VW, TW, H, eigenvectors, eigenvalues, num_eigenvalues);

    EXPECT_NEAR(0.0, eigenvalues(0), 1e-14);
}
