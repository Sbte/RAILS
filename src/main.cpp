#include <iostream>

#include <Teuchos_RCP.hpp>
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"

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

#include <AnasaziTypes.hpp>

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

    Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(
        new Teuchos::ParameterList);
    if (argc > 1)
    {
        Teuchos::updateParametersFromXmlFile(argv[1], params.ptr());
    }

    Epetra_CrsMatrix *A_ptr, *B_ptr, *M_ptr;

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
    Schur->set_parameters(params->sublist("Schur Operator"));
    Schur->Compute();

    Epetra_Map const &map2 = Schur->OperatorRangeMap();
    Epetra_Import import(map2, map);

    int MaxNumEntriesPerRow = B->MaxNumEntries();
    Teuchos::RCP<Epetra_CrsMatrix> B22 = Teuchos::rcp(
        new Epetra_CrsMatrix(Copy, map2, B->ColMap(), MaxNumEntriesPerRow));
    CHECK_ZERO(B22->Import(*B, import, Insert));
    CHECK_ZERO(B22->FillComplete(B->DomainMap(), map2));

    std::cout << "Creating solver" << std::endl;

    Teuchos::RCP<Epetra_Operator> Schur_operator = Schur;
    Epetra_OperatorWrapper Schur_wrapper = Schur_operator;
    Teuchos::RCP<Epetra_Operator> B22_operator = B22;

    Lyapunov::Solver<Epetra_OperatorWrapper, Epetra_MultiVectorWrapper,
                     Epetra_SerialDenseMatrixWrapper> solver(
                         Schur_wrapper, B22_operator, B22_operator);

    Epetra_MultiVectorWrapper V;
    Epetra_SerialDenseMatrixWrapper T;

    bool only_eigenvalues = false;
    if (!only_eigenvalues)
    {
        V = Teuchos::rcp(
            new Epetra_MultiVector(map2, 1000));
        T = Teuchos::rcp(
            new Epetra_SerialDenseMatrix(1000, 1000));

        solver.set_parameters(params->sublist("Lyapunov Solver"));

        std::cout << "Performing solve" << std::endl;

        solver.solve(V, T);
        EpetraExt::MultiVectorToMatrixMarketFile(
            "V.mtx", *V);
        EpetraExt::MultiVectorToMatrixMarketFile(
            "T.mtx", *SerialDenseMatrixToMultiVector(View, *T, A->Comm()));
    }
    else
    {
        Epetra_MultiVector *V_ptr, *T_ptr;
        EpetraExt::MatrixMarketFileToMultiVector("V.mtx", map2, V_ptr);
        Epetra_LocalMap local_map(V_ptr->NumVectors(), 0, *Comm);
        EpetraExt::MatrixMarketFileToMultiVector("T.mtx", local_map, T_ptr);

        V = Teuchos::rcp(V_ptr);
        T = MultiVectorToSerialDenseMatrix(Copy, *T_ptr);
        delete T_ptr;
    }

    Schur->SetSolution(*V, *T);
    Schur_wrapper.set_parameters(*params);

    START_TIMER("Compute eigenvalues");

    Epetra_MultiVectorWrapper eigenvectors;
    Epetra_SerialDenseMatrixWrapper eigenvalues(0,0);

    Teuchos::ParameterList &eig_params = params->sublist("Eigenvalue Solver");
    eig_params.set("Verbosity", Anasazi::Errors +
                   // Anasazi::IterationDetails +
                   Anasazi::Warnings +
                   Anasazi::FinalSummary);

    Schur_wrapper.eigs(eigenvectors, eigenvalues,
                       eig_params.get("Number of Eigenvalues", 10));

    END_TIMER("Compute eigenvalues");

    START_TIMER("Compute trace");
    double trace = Schur->Trace();

    int num_eigs = eigenvalues.M();
    for (int i = 0; i < num_eigs; i++)
    {
        std::cout << std::setw(20) << eigenvalues(i)
                  << std::setw(20) << eigenvalues(i) / trace
                  << std::endl;
    }

    END_TIMER("Compute trace");

    if (!MyPID)
        SAVE_PROFILES("");

#ifdef HAVE_MPI
    MPI_Finalize();
#endif
}
