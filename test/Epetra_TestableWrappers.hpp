#ifndef EPETRA_TESTABLEWRAPPERS_H
#define EPETRA_TESTABLEWRAPPERS_H

#include "src/Epetra_SerialDenseMatrixWrapper.hpp"
#include "src/Epetra_MultiVectorWrapper.hpp"
#include "src/Epetra_OperatorWrapper.hpp"

class TestableEpetra_MultiVectorWrapper: public Epetra_MultiVectorWrapper
{
public:
    TestableEpetra_MultiVectorWrapper(Epetra_MultiVectorWrapper const &other)
        :
        Epetra_MultiVectorWrapper(other)
        {}

    TestableEpetra_MultiVectorWrapper(int m, int n)
        :
        Epetra_MultiVectorWrapper(m, n)
        {}

    double &operator ()(int m, int n = 0)
        {
            return Epetra_MultiVectorWrapper::operator()(m, n);
        }

    double const &operator ()(int m, int n = 0) const
        {
            return Epetra_MultiVectorWrapper::operator()(m, n);
        }
};

class TestableEpetra_OperatorWrapper: public Epetra_OperatorWrapper
{
public:
    TestableEpetra_OperatorWrapper(Epetra_OperatorWrapper const &other)
        :
        Epetra_OperatorWrapper(other)
        {}

    TestableEpetra_OperatorWrapper(int m, int n)
        :
        Epetra_OperatorWrapper(m, n)
        {}

    double &operator ()(int m, int n = 0)
        {
            return Epetra_OperatorWrapper::operator()(m, n);
        }

    double const &operator ()(int m, int n = 0) const
        {
            return Epetra_OperatorWrapper::operator()(m, n);
        }
};

#endif
