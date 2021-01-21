#ifndef EPETRA_MULTIVECTORWRAPPER_H
#define EPETRA_MULTIVECTORWRAPPER_H

#include <Teuchos_RCP.hpp>

#include <Epetra_DataAccess.h>

class Epetra_MultiVector;
class Epetra_SerialDenseMatrix;
class Epetra_Comm;

namespace RAILS
{

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

    // Vector is used as transpose or not in * methods
    bool transpose_;
public:
    Epetra_MultiVectorWrapper();
    Epetra_MultiVectorWrapper(Teuchos::RCP<Epetra_MultiVector> ptr);
    Epetra_MultiVectorWrapper(Epetra_MultiVectorWrapper const &other);
    Epetra_MultiVectorWrapper(Epetra_MultiVectorWrapper const &other, int n);
    Epetra_MultiVectorWrapper(Epetra_MultiVectorWrapper &&other);

    virtual ~Epetra_MultiVectorWrapper() {}

    Epetra_MultiVectorWrapper &operator =(Epetra_MultiVectorWrapper &other);
    Epetra_MultiVectorWrapper &operator =(Epetra_MultiVectorWrapper const &other);
    Epetra_MultiVectorWrapper &operator =(Epetra_MultiVectorWrapper &&other);

    Epetra_MultiVectorWrapper &operator =(double other);

    Epetra_MultiVectorWrapper &operator *=(double other);
    Epetra_MultiVectorWrapper &operator /=(double other);

    Epetra_MultiVectorWrapper &operator -=(Epetra_MultiVectorWrapper const &other);
    Epetra_MultiVectorWrapper &operator +=(Epetra_MultiVectorWrapper const &other);

    Epetra_MultiVectorWrapper operator +(Epetra_MultiVectorWrapper const &other) const;
    Epetra_MultiVectorWrapper operator *(Epetra_SerialDenseMatrixWrapper const &other) const;
    Epetra_MultiVectorWrapper operator *(Epetra_MultiVectorWrapper const &other) const;

    Epetra_MultiVector &operator *();
    Epetra_MultiVector const &operator *() const;

    void resize(int m);

    double norm() const;
    double norm_inf() const;

    void orthogonalize();

    Epetra_MultiVectorWrapper view(int m = -1, int n = -1);
    Epetra_MultiVectorWrapper view(int m = -1, int n = -1) const;
    Epetra_MultiVectorWrapper copy() const;

    Epetra_MultiVectorWrapper transpose() const;

    void push_back(Epetra_MultiVectorWrapper const &other, int m = -1);

    int M() const;
    int N() const;

    Epetra_SerialDenseMatrixWrapper dot(Epetra_MultiVectorWrapper const &other) const;

    void random();

// Test methods that do not have to be exposed
protected:
    Epetra_MultiVectorWrapper(int m, int n);

    double &operator ()(int m, int n = 0);
    double const &operator ()(int m, int n = 0) const;
};

// Helper functions
Teuchos::RCP<Epetra_MultiVector> SerialDenseMatrixToMultiVector(
    Epetra_DataAccess CV, Epetra_SerialDenseMatrix const &src,
    Epetra_Comm const &comm);
Teuchos::RCP<Epetra_SerialDenseMatrix> MultiVectorToSerialDenseMatrix(
    Epetra_DataAccess CV, Epetra_MultiVector const &src);

Epetra_MultiVectorWrapper operator *(
    double d, Epetra_MultiVectorWrapper const &other);

}

#endif
