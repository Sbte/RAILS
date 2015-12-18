#include <iostream>

#include <Teuchos_RCP.hpp>
#include "Teuchos_ParameterList.hpp"

#include "Epetra_Map.h"
#include "Epetra_Import.h"
#include "Epetra_Vector.h"
#include "Epetra_MultiVector.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_SerialDenseMatrix.h"

#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

#include "EpetraExt_CrsMatrixIn.h"
#include "EpetraExt_OperatorOut.h"
#include "EpetraExt_MultiVectorOut.h"

#include "Epetra_MultiVectorWrapper.hpp"
#include "Epetra_SerialDenseMatrixWrapper.hpp"
#include "Epetra_OperatorWrapper.hpp"
#include "LyapunovSolver.hpp"
#include "LyapunovMacros.hpp"
#include "SchurOperator.hpp"

#define TIMER_ON
#include "Timer.hpp"

// Parallel Projection Lanczos Lyapunov Solver
int main(int argc, char *argv[])
{
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif
#ifdef HAVE_MPI
    Teuchos::RCP<Epetra_MpiComm> Comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
#else
    Teuchos::RCP<Epetra_SerialComm> Comm = Teuchos::rcp(new Epetra_SerialComm());
#endif

    //Get process ID and total number of processes
    int MyPID = Comm->MyPID();
//     int NumProc = Comm->NumProc();

    Epetra_CrsMatrix *A_ptr, *B_ptr, *M_ptr, *SC_ptr;

    std::cout << "Loading matrices" << std::endl;

    CHECK_ZERO(EpetraExt::MatrixMarketFileToCrsMatrix("A.mtx", *Comm, A_ptr));
    CHECK_ZERO(EpetraExt::MatrixMarketFileToCrsMatrix("B.mtx", *Comm, B_ptr));
    CHECK_ZERO(EpetraExt::MatrixMarketFileToCrsMatrix("M.mtx", *Comm, M_ptr));

    Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp(A_ptr);
    Teuchos::RCP<Epetra_CrsMatrix> B = Teuchos::rcp(B_ptr);
    Teuchos::RCP<Epetra_CrsMatrix> M = Teuchos::rcp(M_ptr);

    Epetra_BlockMap const &map = A->Map();

    std::cout << "Computing Schur complement" << std::endl;

    Teuchos::RCP<SchurOperator> Schur = Teuchos::rcp(new SchurOperator(A, M));
    Schur->Compute();

    Epetra_Map const &map2 = Schur->OperatorRangeMap();
    Epetra_Import import(map2, map);

    int MaxNumEntriesPerRow = B->MaxNumEntries();
    Teuchos::RCP<Epetra_CrsMatrix> B22 = Teuchos::rcp(
        new Epetra_CrsMatrix(Copy, map2, map2, MaxNumEntriesPerRow));
    CHECK_ZERO(B22->Import(*B, import, Insert));
    CHECK_ZERO(B22->FillComplete(map2, map2));
    
    std::cout << "Creating solver" << std::endl;

    Teuchos::RCP<Epetra_Operator> Schur_operator = Schur;
    Teuchos::RCP<Epetra_Operator> B22_operator = B22;

    Lyapunov::Solver<Epetra_OperatorWrapper, Epetra_MultiVectorWrapper,
                     Epetra_SerialDenseMatrixWrapper> solver(
                         Schur_operator, B22_operator, B22_operator);

    Teuchos::RCP<Epetra_MultiVector> V = Teuchos::rcp(
        new Epetra_MultiVector(map2, 1000));
    Teuchos::RCP<Epetra_SerialDenseMatrix> T = Teuchos::rcp(
        new Epetra_SerialDenseMatrix(1000, 1000));

    Epetra_MultiVectorWrapper VW(V);
    Epetra_SerialDenseMatrixWrapper TW(T);

    Teuchos::ParameterList params;
    params.set("Maximum iterations", 1000);
    solver.set_parameters(params);

    std::cout << "Performing solve" << std::endl;

    solver.solve(VW, TW);

    if (!MyPID)
        SAVE_PROFILES("");

    EpetraExt::MultiVectorToMatrixMarketFile(
        "V.mtx", *VW);
    EpetraExt::MultiVectorToMatrixMarketFile(
        "T.mtx", *SerialDenseMatrixToMultiVector(View, *TW, V->Comm()));

#ifdef HAVE_MPI
    MPI_Finalize();
#endif
}
