#ifndef EPETRA_OPERATORWRAPPER_H
#define EPETRA_OPERATORWRAPPER_H

#include <Teuchos_RCP.hpp>

class Epetra_Operator;
class Epetra_MultiVectorWrapper;

class Epetra_OperatorWrapper
{
    Teuchos::RCP<Epetra_Operator> ptr_;
public:
    Epetra_OperatorWrapper();
    Epetra_OperatorWrapper(Teuchos::RCP<Epetra_Operator> ptr);
    Epetra_OperatorWrapper(Epetra_OperatorWrapper const &other);

    virtual ~Epetra_OperatorWrapper() {}

    Epetra_Operator &operator *();
    Epetra_Operator const &operator *() const;

    double norm(int n = 0);

    Epetra_MultiVectorWrapper apply(Epetra_MultiVectorWrapper const &other) const;
};

#endif
