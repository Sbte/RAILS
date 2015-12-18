#include <iostream>

#include <Teuchos_RCP.hpp>
#include "Teuchos_ParameterList.hpp"

#include <AnasaziBasicEigenproblem.hpp>
#include <AnasaziEpetraAdapter.hpp>
#include <AnasaziBlockKrylovSchurSolMgr.hpp>
#include <AnasaziBlockDavidsonSolMgr.hpp>

#include "Epetra_Map.h"
#include "Epetra_LocalMap.h"
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
#include "EpetraExt_MultiVectorIn.h"

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

    // Teuchos::RCP<Epetra_MultiVector> V = Teuchos::rcp(
    //     new Epetra_MultiVector(map2, 1000));
    // Teuchos::RCP<Epetra_SerialDenseMatrix> T = Teuchos::rcp(
    //     new Epetra_SerialDenseMatrix(1000, 1000));

    // Epetra_MultiVectorWrapper VW(V);
    // Epetra_SerialDenseMatrixWrapper TW(T);

    // Teuchos::ParameterList params;
    // params.set("Maximum iterations", 1000);
    // solver.set_parameters(params);

    // std::cout << "Performing solve" << std::endl;

    // solver.solve(VW, TW);
    // EpetraExt::MultiVectorToMatrixMarketFile(
    //     "V.mtx", *VW);
    // EpetraExt::MultiVectorToMatrixMarketFile(
    //     "T.mtx", *SerialDenseMatrixToMultiVector(View, *TW, V->Comm()));

    Epetra_MultiVector *V_ptr, *T_ptr;
    EpetraExt::MatrixMarketFileToMultiVector("V.mtx", map2, V_ptr);
    Epetra_LocalMap local_map(V_ptr->NumVectors(), 0, *Comm);
    EpetraExt::MatrixMarketFileToMultiVector("T.mtx", local_map, T_ptr);

    Teuchos::RCP<Epetra_MultiVector> V = Teuchos::rcp(V_ptr);
    Teuchos::RCP<Epetra_SerialDenseMatrix> T =
      MultiVectorToSerialDenseMatrix(Copy, *T_ptr);
    delete T_ptr;

    Schur->SetSolution(*V, *T);

    START_TIMER("Compute eigenvalues");

    Teuchos::RCP<Epetra_MultiVector> x = Teuchos::rcp(new Epetra_Vector(map));
    Teuchos::RCP<Epetra_MultiVector> out = Teuchos::rcp(new Epetra_Vector(map));
    x->PutScalar(1.0);

    Teuchos::RCP<Anasazi::BasicEigenproblem<
      double, Epetra_MultiVector, Epetra_Operator> > eig_problem =
      Teuchos::rcp(new Anasazi::BasicEigenproblem<
        double, Epetra_MultiVector, Epetra_Operator>(Schur_operator, x));
    eig_problem->setHermitian(true);
    eig_problem->setNEV(100);
    CHECK_ZERO(!eig_problem->setProblem());

    Teuchos::ParameterList eig_params;
    eig_params.set("Verbosity", Anasazi::Errors +
                   Anasazi::IterationDetails +
                   Anasazi::Warnings +
                   Anasazi::FinalSummary);
    eig_params.set("Convergence Tolerance", 1e-6);

    Anasazi::BlockKrylovSchurSolMgr<
        double, Epetra_MultiVector, Epetra_Operator>
        sol_manager(eig_problem, eig_params);
    
    Anasazi::ReturnType ret;
    ret = sol_manager.solve();
    if (ret != Anasazi::Converged)
        std::cout << "Eigensolver did not converge" << std::endl;

    const Anasazi::Eigensolution<double, Epetra_MultiVector> &eig_sol =
        eig_problem->getSolution();

    const std::vector<Anasazi::Value<double> > &evals = eig_sol.Evals;
    int num_eigs = evals.size();

    double sum = 0.0;
    for (int i = 0; i < num_eigs; i++)
        sum += evals[i].realpart;
    for (int i = 0; i < num_eigs; i++)
    {
        std::cout << std::setw(20) << evals[i].realpart
                  << std::setw(20) << evals[i].imagpart
                  << std::setw(20) << evals[i].realpart / sum
                  << std::endl;
    }

    END_TIMER("Compute eigenvalues");

    if (!MyPID)
        SAVE_PROFILES("");

#ifdef HAVE_MPI
    MPI_Finalize();
#endif
}
