#ifndef EPETRA_MULTIVECTORWRAPPER_H
#define EPETRA_MULTIVECTORWRAPPER_H

#include <Teuchos_RCP.hpp>

#include <Epetra_DataAccess.h>

class Epetra_MultiVector;
class Epetra_SerialDenseMatrix;
class Epetra_Comm;

class Epetra_SerialDenseMatrixWrapper;

class Epetra_MultiVectorWrapper
{
    Teuchos::RCP<Epetra_MultiVector> ptr_;

    Teuchos::RCP<Epetra_MultiVector> ptr_allocated_;

    // Capacity of a multivector
    int capacity_;

    // Size of a multivector (amount of vectors)
    int size_;

    // Amount of vectors that are already orthogonal
    int orthogonalized_;

    // Vector is a view
    bool is_view_;
public:
    Epetra_MultiVectorWrapper();
    Epetra_MultiVectorWrapper(Teuchos::RCP<Epetra_MultiVector> ptr);
    Epetra_MultiVectorWrapper(Epetra_MultiVectorWrapper const &other);
    Epetra_MultiVectorWrapper(Epetra_MultiVectorWrapper const &other, int n);

    virtual ~Epetra_MultiVectorWrapper() {}

    Epetra_MultiVectorWrapper &operator =(Epetra_MultiVectorWrapper &other);
    Epetra_MultiVectorWrapper &operator =(Epetra_MultiVectorWrapper const &other);

    Epetra_MultiVectorWrapper &operator *=(double other);
    Epetra_MultiVectorWrapper &operator /=(double other);

    Epetra_MultiVectorWrapper &operator -=(Epetra_MultiVectorWrapper const &other);
    Epetra_MultiVectorWrapper &operator +=(Epetra_MultiVectorWrapper const &other);

    Epetra_MultiVectorWrapper operator +(Epetra_MultiVectorWrapper const &other) const;

    Epetra_MultiVector &operator *();

    Epetra_MultiVector const &operator *() const;

    int scale(double factor);

    void resize(int m);

    double norm(int n = 0) const;
    double norm_inf(int n = 0) const;

    void orthogonalize();

    Epetra_MultiVectorWrapper view(int m, int n = 0);
    Epetra_MultiVectorWrapper view(int m, int n = 0) const;
    Epetra_MultiVectorWrapper copy(int m = 0, int n = 0) const;

    void push_back(Epetra_MultiVectorWrapper const &other, int m = -1);

    int M() const;
    int N() const;

    Epetra_SerialDenseMatrixWrapper dot(Epetra_MultiVectorWrapper const &other) const;

    void random();

    Epetra_MultiVectorWrapper apply(Epetra_SerialDenseMatrixWrapper const &other) const;
};

// Helper functions
Teuchos::RCP<Epetra_MultiVector> SerialDenseMatrixToMultiVector(
    Epetra_DataAccess CV, Epetra_SerialDenseMatrix const &src,
    Epetra_Comm const &comm);

Epetra_MultiVectorWrapper operator *(double d, Epetra_MultiVectorWrapper const &other);

#endif
