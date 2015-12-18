#ifndef EPETRA_SERIALDENSEMATRIXWRAPPER_H
#define EPETRA_SERIALDENSEMATRIXWRAPPER_H

#include <Teuchos_RCP.hpp>

class Epetra_SerialDenseMatrix;

class Epetra_SerialDenseMatrixWrapper
{
    Teuchos::RCP<Epetra_SerialDenseMatrix> ptr_;
public:
    Epetra_SerialDenseMatrixWrapper();
    Epetra_SerialDenseMatrixWrapper(Teuchos::RCP<Epetra_SerialDenseMatrix> ptr);
    Epetra_SerialDenseMatrixWrapper(Epetra_SerialDenseMatrixWrapper const &other);
    Epetra_SerialDenseMatrixWrapper(int m, int n);

    virtual ~Epetra_SerialDenseMatrixWrapper() {}

    Epetra_SerialDenseMatrixWrapper &operator =(Epetra_SerialDenseMatrixWrapper const &other);

    Epetra_SerialDenseMatrix &operator *();
    Epetra_SerialDenseMatrix const &operator *() const;

    operator double*() const;

    double &operator ()(int m, int n = 0);
    double const &operator ()(int m, int n = 0) const;

    double norm_inf() const;

    int scale(double factor);

    void resize(int m, int n);

    Epetra_SerialDenseMatrixWrapper copy(int m = 0, int n = 0) const;

    int M() const;
    int N() const;

    void eigs(Epetra_SerialDenseMatrixWrapper &v, Epetra_SerialDenseMatrixWrapper &d) const;

    Epetra_SerialDenseMatrixWrapper apply(Epetra_SerialDenseMatrixWrapper const &other) const;
};

#endif
