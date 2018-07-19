#include "SchurOperator.hpp"
#include "LyapunovMacros.hpp"

#include <Epetra_LinearProblem.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Vector.h>
#include <Epetra_MultiVector.h>
#include <Epetra_Map.h>
#include <Epetra_Import.h>
#include <Epetra_SerialDenseMatrix.h>

#include "Epetra_MultiVectorWrapper.hpp"
#include "Epetra_SerialDenseMatrixWrapper.hpp"

#include "EpetraExt_Transpose_RowMatrix.h"

#include <Amesos_Klu.h>

#define TIMER_ON
#include "Timer.hpp"

SchurOperator::SchurOperator(Teuchos::RCP<Epetra_CrsMatrix> const &A,
                             Teuchos::RCP<Epetra_Vector> const &M,
                             Teuchos::RCP<Epetra_MultiVector> const &border)
:
    A_(A),
    diagM_(M),
    border_(border),
    problem_(Teuchos::rcp(new Epetra_LinearProblem)),
    solver_(new Amesos_Klu(*problem_)),
    hasSolution_(false),
    mvps_(0)
{}

SchurOperator::SchurOperator(Teuchos::RCP<Epetra_CrsMatrix> const &A,
                             Teuchos::RCP<Epetra_CrsMatrix> const &M,
                             Teuchos::RCP<Epetra_MultiVector> const &border)
    :
    A_(A),
    M_(M),
    border_(border),
    problem_(Teuchos::rcp(new Epetra_LinearProblem)),
    solver_(new Amesos_Klu(*problem_)),
    hasSolution_(false),
    mvps_(0)
{}

int SchurOperator::Compute()
{
    FUNCTION_TIMER("SchurOperator", "Compute");
    Epetra_BlockMap const &map = A_->Map();

    if (diagM_ == Teuchos::null)
    {
        diagM_ = Teuchos::rcp(new Epetra_Vector(map));
        M_->ExtractDiagonalCopy(*diagM_);
    }

    int borderSize = 0;
    if (border_ != Teuchos::null)
        borderSize = border_->NumVectors();

    int *indices1 = new int[diagM_->MyLength()+borderSize];
    int *indices2 = new int[diagM_->MyLength()+borderSize];

    int num_indices1 = 0;
    int num_indices2 = 0;

    // Iterate over M looking for nonzero parts
    for (int i = 0; i < diagM_->MyLength(); i++)
    {
        if (abs((*diagM_)[i]) < 1e-15)
            indices1[num_indices1++] = map.GID(i);
        else
            indices2[num_indices2++] = map.GID(i);
    }

    // Add border
    int MyPID = Comm().MyPID();
    int len = diagM_->GlobalLength();
    if (border_ != Teuchos::null)
    {
        if (MyPID == 0)
        {
            for (int i = 0; i < borderSize; i++)
                indices1[num_indices1++] = len + i;
        }
    }

    Epetra_Map map1(-1, num_indices1, indices1, 0, Comm());
    Epetra_Map map2(-1, num_indices2, indices2, 0, Comm());

    delete[] indices1;
    delete[] indices2;

    Epetra_Map const &colMap = A_->ColMap();

    Epetra_Vector diagM_col(colMap);
    Epetra_Import colImport(colMap, map);
    diagM_col.Import(*diagM_, colImport, Insert);

    indices1 = new int[colMap.NumMyElements()+borderSize];
    indices2 = new int[colMap.NumMyElements()+borderSize];

    num_indices1 = 0;
    num_indices2 = 0;

    // Iterate over M looking for nonzero parts
    for (int i = 0; i < colMap.NumMyElements(); i++)
    {
        if (abs(diagM_col[i]) < 1e-15)
            indices1[num_indices1++] = colMap.GID(i);
        else
            indices2[num_indices2++] = colMap.GID(i);
    }

    if (border_ != Teuchos::null)
    {
        for (int i = 0; i < borderSize; i++)
            indices1[num_indices1++] = len + i;
    }

    Epetra_Map colMap1(-1, num_indices1, indices1, 0, Comm());
    Epetra_Map colMap2(-1, num_indices2, indices2, 0, Comm());

    delete[] indices1;
    delete[] indices2;

    Epetra_Import import1(map1, map);
    Epetra_Import import2(map2, map);

    // -1 on the diagonal of M so scale A until we use M
    A_->Scale(-1.0);

    int MaxNumEntriesPerRow = A_->MaxNumEntries() + borderSize;
    A11_ = Teuchos::rcp(
        new Epetra_CrsMatrix(Copy, map1, colMap1, MaxNumEntriesPerRow));
    CHECK_ZERO(A11_->Import(*A_, import1, Insert));
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

    // Add border
    if (border_ != Teuchos::null)
    {
        for (int col = 0; col < borderSize; col++)
        {
            int idx = col + len;
            for (int i = 0; i < A11_->NumMyRows(); i++)
            {
                int gid = A11_->GRID(i);
                double val = (*border_)[col][i];
                if (val != 0)
                {
                    CHECK_ZERO(A11_->InsertGlobalValues(gid, 1, &val, &idx));
                    if (MyPID == 0)
                        CHECK_ZERO(A11_->InsertGlobalValues(idx, 1, &val, &gid));
                }
            }
        }
    }
    CHECK_ZERO(A11_->FillComplete(map1, map1));

    // More efficient in KLU
    A11T_ = Teuchos::RCP<EpetraExt::RowMatrix_Transpose>(
        new EpetraExt::RowMatrix_Transpose());
    problem_->SetOperator(&(*A11T_)(const_cast<Epetra_CrsMatrix&>(*A11_)));
    CHECK_ZERO(solver_->SetUseTranspose(true));

    // problem_->SetOperator(A11_.get());
    // CHECK_ZERO(solver_->SetUseTranspose(false));

    CHECK_ZERO(solver_->SymbolicFactorization());
    CHECK_ZERO(solver_->NumericFactorization());

    return 0;
}

int SchurOperator::SetSolution(const Epetra_MultiVector& V, const Epetra_SerialDenseMatrix& T)
{
    hasSolution_ = true;

    V_ = Teuchos::rcp(&V, false);
    T_ = Teuchos::rcp(&T, false);

    return 0;
}

int SchurOperator::Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
    FUNCTION_TIMER("SchurOperator", "Apply");

    if (!hasSolution_)
    {
        mvps_ += X.NumVectors();
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

        return 0;
    }

    Epetra_MultiVectorWrapper VW(
        Teuchos::rcp_const_cast<Epetra_MultiVector>(V_));
    Epetra_SerialDenseMatrixWrapper TW(
        Teuchos::rcp_const_cast<Epetra_SerialDenseMatrix>(T_));

    // TODO: Fix these maps to work in parallel runs
    Epetra_MultiVector tmp1(A11_->DomainMap(), Y.NumVectors());
    Epetra_MultiVector tmp2(A11_->RangeMap(), Y.NumVectors());

    Epetra_Import import11(A11_->RangeMap(), X.Map());
    Epetra_Import import22(A22_->RangeMap(), X.Map());

    Epetra_MultiVector X1(A11_->DomainMap(), X.NumVectors());
    CHECK_ZERO(X1.Import(X, import11, Insert));

    Epetra_MultiVector X2(A22_->DomainMap(), X.NumVectors());
    CHECK_ZERO(X2.Import(X, import22, Insert));
    Epetra_MultiVectorWrapper X2W(
        Teuchos::rcp_const_cast<Epetra_MultiVector>(Teuchos::rcp(&X2, false)));

    // X22 * X = V * (T * (V' * X));
    Epetra_MultiVectorWrapper X22 = VW * (TW * VW.dot(X2W));

    // X12 * X = - A11 \ (A12 * (X22 * X));
    Epetra_MultiVectorWrapper X12(Teuchos::rcp(&tmp1, false), X.NumVectors());
    CHECK_ZERO(A12_->Apply(*X22, tmp1));
    problem_->SetLHS(&(*X12));
    problem_->SetRHS(&tmp1);
    CHECK_ZERO(solver_->Solve());
    X12 *= -1.0;

    // X21 * X = - X22 * (A12' * (A11' \ X));
    Epetra_MultiVectorWrapper X21(X22, X.NumVectors());
    CHECK_ZERO(solver_->SetUseTranspose(true));
    problem_->SetLHS(&tmp2);
    problem_->SetRHS(&X1);
    CHECK_ZERO(solver_->Solve());
    CHECK_ZERO(solver_->SetUseTranspose(false));
    CHECK_ZERO(A12_->SetUseTranspose(true));
    CHECK_ZERO(A12_->Apply(tmp2, *X21));
    CHECK_ZERO(A12_->SetUseTranspose(false));
    X21 = VW * (TW * VW.dot(X21));
    X21 *= -1.0;

    // X11 * X = - A11 \ (A12 * (X21 * X));
    Epetra_MultiVectorWrapper X11(X12, X.NumVectors());
    CHECK_ZERO(A12_->Apply(*X21, tmp1));
    problem_->SetLHS(&(*X11));
    problem_->SetRHS(&tmp1);
    CHECK_ZERO(solver_->Solve());
    X11 *= -1.0;

    Epetra_Import import1(Y.Map(), A11_->RangeMap());
    Epetra_Import import2(Y.Map(), A22_->RangeMap());

    X11 += X12;
    X22 += X21;
    CHECK_ZERO(Y.Import(*X11, import1, Insert));
    CHECK_ZERO(Y.Import(*X22, import2, Insert));

    return 0;
}

double SchurOperator::Trace() const
{
    if (!hasSolution_)
    {
        std::cerr << "Solution was not set so can't compute the trace" << std::endl;
        return 0.0;
    }

    Epetra_MultiVectorWrapper VW(
        Teuchos::rcp_const_cast<Epetra_MultiVector>(V_));
    Epetra_SerialDenseMatrixWrapper TW(
        Teuchos::rcp_const_cast<Epetra_SerialDenseMatrix>(T_));

    double trace = 0.0;
    for (int i = 0; i < TW.M(); i++)
        trace += TW(i, i);

    // Trace(A11\A12*C22*A12'*A11^{-1}) = Trace(T*V'*A12'*A11\A11'\A12*V)
    Epetra_MultiVector tmp1(A11_->DomainMap(), VW.N());
    Epetra_MultiVector tmp2(A11_->DomainMap(), VW.N());
    Epetra_MultiVector tmp3(A22_->DomainMap(), VW.N());
    A12_->Apply(*VW, tmp1);

    problem_->SetLHS(&tmp2);
    problem_->SetRHS(&tmp1);
    CHECK_ZERO(solver_->Solve());

    CHECK_ZERO(solver_->SetUseTranspose(true));
    problem_->SetLHS(&tmp1);
    problem_->SetRHS(&tmp2);
    CHECK_ZERO(solver_->Solve());
    CHECK_ZERO(solver_->SetUseTranspose(false));

    CHECK_ZERO(A12_->SetUseTranspose(true));
    CHECK_ZERO(A12_->Apply(tmp1, tmp3));
    CHECK_ZERO(A12_->SetUseTranspose(false));

    Epetra_MultiVectorWrapper tmpW(Teuchos::rcp(&tmp3, false));
    Epetra_SerialDenseMatrixWrapper T11 = TW * VW.dot(tmpW);

    for (int i = 0; i < T11.M(); i++)
        trace += T11(i, i);

    return trace;
}

const Epetra_Comm &SchurOperator::Comm() const
{
    return A_->Comm();
}

const Epetra_Map & SchurOperator::OperatorDomainMap() const
{
    return (hasSolution_ ? A_->DomainMap() : A22_->DomainMap());
}

const Epetra_Map & SchurOperator::OperatorRangeMap() const
{
    return (hasSolution_ ? A_->RangeMap() : A22_->RangeMap());
}

int SchurOperator::ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
    std::cerr << "ApplyInverse not implemented" << std::endl;
    return -1;
}

int SchurOperator::SetUseTranspose(bool UseTranspose)
{
    if (UseTranspose)
        std::cerr << "SetUseTranspose not implemented" << std::endl;

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

int SchurOperator::GetMVPs() const
{
    return mvps_;
}
