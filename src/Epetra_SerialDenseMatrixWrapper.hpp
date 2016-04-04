#ifndef EPETRA_SERIALDENSEMATRIXWRAPPER_H
#define EPETRA_SERIALDENSEMATRIXWRAPPER_H

#include <Teuchos_RCP.hpp>

class Epetra_SerialDenseMatrix;

class Epetra_SerialDenseMatrixWrapper
{
    Teuchos::RCP<Epetra_SerialDenseMatrix> ptr_;

    bool is_view_;

    bool transpose_;
public:
    Epetra_SerialDenseMatrixWrapper();
    Epetra_SerialDenseMatrixWrapper(Teuchos::RCP<Epetra_SerialDenseMatrix> ptr);
    Epetra_SerialDenseMatrixWrapper(Epetra_SerialDenseMatrixWrapper const &other);
    Epetra_SerialDenseMatrixWrapper(int m, int n);
    Epetra_SerialDenseMatrixWrapper(Epetra_SerialDenseMatrixWrapper &&other);

    virtual ~Epetra_SerialDenseMatrixWrapper() {}

    Epetra_SerialDenseMatrixWrapper transpose() const;

    Epetra_SerialDenseMatrixWrapper &operator =(Epetra_SerialDenseMatrixWrapper &other);
    Epetra_SerialDenseMatrixWrapper &operator =(Epetra_SerialDenseMatrixWrapper const &other);

    Epetra_SerialDenseMatrixWrapper &operator =(double other);

    Epetra_SerialDenseMatrixWrapper &operator *=(double other);

    Epetra_SerialDenseMatrix &operator *();
    Epetra_SerialDenseMatrix const &operator *() const;

    Epetra_SerialDenseMatrixWrapper operator *(Epetra_SerialDenseMatrixWrapper const &other) const;

    operator double*() const;

    double &operator ()(int m, int n = 0);
    double const &operator ()(int m, int n = 0) const;

    double norm_inf() const;

    void resize(int m, int n);

    Epetra_SerialDenseMatrixWrapper view();
    Epetra_SerialDenseMatrixWrapper copy() const;

    int M() const;
    int N() const;
    int LDA() const;

    void eigs(Epetra_SerialDenseMatrixWrapper &v, Epetra_SerialDenseMatrixWrapper &d) const;
};

#endif
