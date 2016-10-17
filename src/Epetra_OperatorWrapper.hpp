#ifndef EPETRA_OPERATORWRAPPER_H
#define EPETRA_OPERATORWRAPPER_H

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include "Epetra_MultiVectorWrapper.hpp"

class Epetra_Operator;
class Epetra_BlockMap;
class Epetra_SerialDenseMatrixWrapper;

class Epetra_OperatorWrapper
{
    Teuchos::RCP<Epetra_Operator> ptr_;
    Teuchos::RCP<Teuchos::ParameterList> params_;

    bool transpose_;
public:
    Epetra_OperatorWrapper();
    Epetra_OperatorWrapper(Teuchos::RCP<Epetra_Operator> ptr);
    Epetra_OperatorWrapper(Epetra_OperatorWrapper const &other);

    Epetra_OperatorWrapper transpose() const;

    virtual ~Epetra_OperatorWrapper() {}

    Epetra_Operator &operator *();
    Epetra_Operator const &operator *() const;

    Epetra_MultiVectorWrapper operator *(Epetra_MultiVectorWrapper const &other) const;

    int set_parameters(Teuchos::ParameterList &params);

    int M() const;
    int N() const;

    double norm();
    int eigs(Epetra_MultiVectorWrapper &V,
             Epetra_SerialDenseMatrixWrapper &D,
             int num, double tol = 1e-6) const;

// Test methods that do not have to be exposed
protected:
    Epetra_OperatorWrapper(int m, int n);

    double &operator ()(int m, int n = 0);
    double const &operator ()(int m, int n = 0) const;
};

#endif
