#include "EpetraWrapper.hpp"

// Specializations of the apply methods
template<>
template<>
EpetraWrapper<Epetra_MultiVector> EpetraWrapper<Epetra_MultiVector>::apply(EpetraWrapper<Epetra_SerialDenseMatrix> const &other) const
{
    EpetraWrapper<Epetra_MultiVector> out(*this, (*other).N());

    if ((*other).M() != ptr_->NumVectors())
    {
        std::cerr << "Incomplatible matrices of sizes "
                  << ptr_->MyLength() << "x" << ptr_->NumVectors() << " and "
                  << (*other).M() << "x" << (*other).N() << std::endl;
        return out;
    }

    Epetra_LocalMap map((*other).M(), 0, ptr_->Comm());
    Epetra_MultiVector mv(View, map, (*other).A(), (*other).M(), (*other).N());
    (*out).Multiply('N', 'N', 1.0, *ptr_, mv, 0.0);

    return out;
}

template<>
template<>
EpetraWrapper<Epetra_MultiVector> EpetraWrapper<Epetra_CrsMatrix>::apply(EpetraWrapper<Epetra_MultiVector> const &other) const
{
    EpetraWrapper<Epetra_MultiVector> out(Teuchos::rcp(new Epetra_MultiVector(*other)));
    ptr_->Apply(*other, *out);
    return out;
}

template<>
EpetraWrapper<Epetra_SerialDenseMatrix> EpetraWrapper<Epetra_SerialDenseMatrix>::apply(EpetraWrapper<Epetra_SerialDenseMatrix> const &other) const
{
    EpetraWrapper<Epetra_SerialDenseMatrix> out(Teuchos::rcp(new Epetra_SerialDenseMatrix(*other)));
    out.resize(M(), other.N());

    if (N() != other.M())
    {
        std::cerr << "Incomplatible matrices of sizes "
                  << M() << "x" << N() << " and "
                  << other.M() << "x" << other.N() << std::endl;
        return out;
    }

    ptr_->Apply(*other, *out);
    return out;
}

// Specialization of the assignment operator
template<>
EpetraWrapper<Epetra_MultiVector> &EpetraWrapper<Epetra_MultiVector>::operator =(EpetraWrapper<Epetra_MultiVector> const &other)
{
    if (!is_view_)
    {
        ptr_ = other.ptr_;
        capacity_ = other.capacity_;
        size_ = other.size_;
        orthogonalized_ = other.orthogonalized_;
        return *this;
    }

    ptr_->ExtractCopy(other.ptr_->Values(), other.ptr_->MyLength());
    return *this;
}
