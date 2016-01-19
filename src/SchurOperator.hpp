#ifndef SCHUROPERATOR_H
#define SCHUROPERATOR_H

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include <Epetra_Operator.h>

class Amesos_BaseSolver;
class Epetra_CrsMatrix;
class Epetra_LinearProblem;
class Epetra_Map;
class Epetra_MultiVector;
class Epetra_SerialDenseMatrix;

class SchurOperator: public Epetra_Operator
{
    Teuchos::RCP<Epetra_CrsMatrix> A_;
    Teuchos::RCP<Epetra_CrsMatrix> M_;

    Teuchos::RCP<Epetra_LinearProblem> problem_;
    Teuchos::RCP<Amesos_BaseSolver> solver_;
    
    Teuchos::RCP<Epetra_CrsMatrix> A11_;
    Teuchos::RCP<Epetra_CrsMatrix> A21_;
    Teuchos::RCP<Epetra_CrsMatrix> A12_;
    Teuchos::RCP<Epetra_CrsMatrix> A22_;

    Teuchos::RCP<const Epetra_MultiVector> V_;
    Teuchos::RCP<const Epetra_SerialDenseMatrix> T_;

    bool hasSolution_;

    int nx_, ny_, nz_;

public:
    SchurOperator(Teuchos::RCP<Epetra_CrsMatrix> const &A,
                  Teuchos::RCP<Epetra_CrsMatrix> const &M);

    virtual ~SchurOperator() {};

    int set_parameters(Teuchos::ParameterList &params);

    int Compute();

    int SetUseTranspose(bool UseTranspose);

    int SetSolution(const Epetra_MultiVector& V, const Epetra_SerialDenseMatrix& T);

    int Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;

    int ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;

    double Trace() const;

    double NormInf() const;

    const char * Label() const;

    bool UseTranspose() const;

    bool HasNormInf() const;

    const Epetra_Comm & Comm() const;

    const Epetra_Map & OperatorDomainMap() const;

    const Epetra_Map & OperatorRangeMap() const;
};

#endif
