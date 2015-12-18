#include "SchurOperator.hpp"
#include "LyapunovMacros.hpp"

#include <Epetra_LinearProblem.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_MultiVector.h>
#include <Epetra_Map.h>
#include <Epetra_Import.h>

#include <Amesos_Klu.h>

#define TIMER_ON
#include "Timer.hpp"

SchurOperator::SchurOperator(Teuchos::RCP<Epetra_CrsMatrix> const &A,
                             Teuchos::RCP<Epetra_CrsMatrix> const &M)
    :
    A_(A),
    M_(M),
    problem_(Teuchos::rcp(new Epetra_LinearProblem)),
    solver_(new Amesos_Klu(*problem_))
{}

int SchurOperator::Compute()
{
    FUNCTION_TIMER("SchurOperator", "Compute");
    Epetra_BlockMap const &map = A_->Map();

    Epetra_Vector diag_m(map);
    M_->ExtractDiagonalCopy(diag_m);

    int *indices1 = new int[diag_m.MyLength()];
    int *indices2 = new int[diag_m.MyLength()];

    int num_indices1 = 0;
    int num_indices2 = 0;

    // Iterate over M looking for nonzero parts
    for (int i = 0; i < diag_m.MyLength(); i++)
    {
        if (abs(diag_m[i]) < 1e-15)
            indices1[num_indices1++] = map.GID(i);
        else
            indices2[num_indices2++] = map.GID(i);
    }

    Epetra_Map map1(-1, num_indices1, indices1, 0, Comm());
    Epetra_Map map2(-1, num_indices2, indices2, 0, Comm());

    delete[] indices1;
    delete[] indices2;

    Epetra_Map const &colMap = A_->ColMap();

    Epetra_Vector diag_m_col(colMap);
    Epetra_Import colImport(colMap, map);
    diag_m_col.Import(diag_m, colImport, Insert);

    indices1 = new int[colMap.NumMyElements()];
    indices2 = new int[colMap.NumMyElements()];

    num_indices1 = 0;
    num_indices2 = 0;

    // Iterate over M looking for nonzero parts
    for (int i = 0; i < colMap.NumMyElements(); i++)
    {
        if (abs(diag_m_col[i]) < 1e-15)
            indices1[num_indices1++] = colMap.GID(i);
        else
            indices2[num_indices2++] = colMap.GID(i);
    }

    Epetra_Map colMap1(-1, num_indices1, indices1, 0, Comm());
    Epetra_Map colMap2(-1, num_indices2, indices2, 0, Comm());

    delete[] indices1;
    delete[] indices2;

    Epetra_Import import1(map1, map);
    Epetra_Import import2(map2, map);

    int MaxNumEntriesPerRow = A_->MaxNumEntries();
    A11_ = Teuchos::rcp(
        new Epetra_CrsMatrix(Copy, map1, colMap1, MaxNumEntriesPerRow));
    CHECK_ZERO(A11_->Import(*A_, import1, Insert));
    CHECK_ZERO(A11_->FillComplete(map1, map1));
    A12_ = Teuchos::rcp(
        new Epetra_CrsMatrix(Copy, map1, colMap2, MaxNumEntriesPerRow));
    CHECK_ZERO(A12_->Import(*A_, import1, Insert));
    CHECK_ZERO(A12_->FillComplete(map2, map1));
    A21_ = Teuchos::rcp(
        new Epetra_CrsMatrix(Copy, map2, colMap1, MaxNumEntriesPerRow));
    CHECK_ZERO(A21_->Import(*A_, import2, Insert));
    CHECK_ZERO(A21_->FillComplete(map1, map2));
    A22_ = Teuchos::rcp(
        new Epetra_CrsMatrix(Copy, map2, colMap2, MaxNumEntriesPerRow));
    CHECK_ZERO(A22_->Import(*A_, import2, Insert));
    CHECK_ZERO(A22_->FillComplete(map2, map2));

    problem_->SetOperator(A11_.get());
    CHECK_ZERO(solver_->SetUseTranspose(false));
    CHECK_ZERO(solver_->SymbolicFactorization());
    CHECK_ZERO(solver_->NumericFactorization());
}

int SchurOperator::Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
    FUNCTION_TIMER("SchurOperator", "Apply");

    START_TIMER("SchurOperator", "Apply A22");
    CHECK_ZERO(A22_->Apply(X, Y));
    END_TIMER("SchurOperator", "Apply A22");

    Epetra_MultiVector tmp1(A11_->DomainMap(), Y.NumVectors());
    Epetra_MultiVector tmp2(A11_->RangeMap(), Y.NumVectors());
    Epetra_MultiVector tmp3(Y.Map(), Y.NumVectors());

    START_TIMER("SchurOperator", "Apply A12");
    CHECK_ZERO(A12_->Apply(X, tmp1));
    END_TIMER("SchurOperator", "Apply A12");

    START_TIMER("SchurOperator", "Apply A11");
    problem_->SetLHS(&tmp2);
    problem_->SetRHS(&tmp1);
    CHECK_ZERO(solver_->Solve());
    END_TIMER("SchurOperator", "Apply A11");

    START_TIMER("SchurOperator", "Apply A21");
    CHECK_ZERO(A21_->Apply(tmp2, tmp3));
    END_TIMER("SchurOperator", "Apply A21");

    CHECK_ZERO(Y.Update(-1.0, tmp3, 1.0));
}

const Epetra_Comm &SchurOperator:: Comm() const
{
    return A_->Comm();
}

const Epetra_Map & SchurOperator::OperatorDomainMap() const
{
    return A22_->DomainMap();
}

const Epetra_Map & SchurOperator::OperatorRangeMap() const
{
    return A22_->RangeMap();
}

int SchurOperator::ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
    std::cerr << "ApplyInverse not implemented" << std::endl;
    return -1;
}

int SchurOperator::SetUseTranspose(bool UseTranspose)
{
    std::cerr << "SetUseTransose not implemented" << std::endl;
    return -1;
}

double SchurOperator::NormInf() const 
{
    std::cerr << "NormInf not implemented" << std::endl;
    return 0.0;
}

const char * SchurOperator::Label() const
{
    std::cerr << "Label not implemented" << std::endl;
    return "";
}

bool SchurOperator::UseTranspose() const
{
    return false;
}

bool SchurOperator::HasNormInf() const
{
    return false;
}
