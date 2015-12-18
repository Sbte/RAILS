#include "Epetra_OperatorWrapper.hpp"
#include "Epetra_MultiVectorWrapper.hpp"

#include <Epetra_MultiVector.h>
#include <Epetra_Operator.h>
#include <Epetra_CrsMatrix.h>

#define TIMER_ON
#include "Timer.hpp"

Epetra_OperatorWrapper::Epetra_OperatorWrapper()
    :
    ptr_(Teuchos::null)
{}

Epetra_OperatorWrapper::Epetra_OperatorWrapper(Teuchos::RCP<Epetra_Operator> ptr)
    :
    Epetra_OperatorWrapper()
{
    FUNCTION_TIMER("Epetra_OperatorWrapper", "constructor 1");
    ptr_ = ptr;
}

Epetra_OperatorWrapper::Epetra_OperatorWrapper(Epetra_OperatorWrapper const &other)
    :
    Epetra_OperatorWrapper()
{
    FUNCTION_TIMER("Epetra_OperatorWrapper", "constructor 2");
    ptr_ = other.ptr_;
}

Epetra_Operator &Epetra_OperatorWrapper::operator *()
{
    FUNCTION_TIMER("Epetra_OperatorWrapper", "*");
    return *ptr_;
}

Epetra_Operator const &Epetra_OperatorWrapper::operator *() const
{
    FUNCTION_TIMER("Epetra_OperatorWrapper", "* 2");
    return *ptr_;
}
double Epetra_OperatorWrapper::norm(int n)
{
    FUNCTION_TIMER("EpetraWrapper", "norm");
    Teuchos::RCP<Epetra_CrsMatrix> mat =
        Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(ptr_);
    if (!mat.is_null())
        return mat->NormFrobenius();
    return ptr_->NormInf();
}

Epetra_MultiVectorWrapper Epetra_OperatorWrapper::apply(
    Epetra_MultiVectorWrapper const &other) const
{
    FUNCTION_TIMER("EpetraWrapper", "apply 2");
    Epetra_MultiVectorWrapper out(Teuchos::rcp(new Epetra_MultiVector(*other)));
    ptr_->Apply(*other, *out);
    return out;
}
