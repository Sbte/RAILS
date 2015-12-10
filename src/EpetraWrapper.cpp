#include "EpetraWrapper.hpp"

Teuchos::RCP<Epetra_SerialDenseMatrix> MultiVectorToSerialDenseMatrix(
    Epetra_DataAccess CV, Epetra_MultiVector const &src)
{
    return Teuchos::rcp(new Epetra_SerialDenseMatrix(
                            CV, src.Values(), src.MyLength(),
                            src.MyLength(), src.NumVectors()));;
}

Teuchos::RCP<Epetra_MultiVector> SerialDenseMatrixToMultiVector(
    Epetra_DataAccess CV, Epetra_SerialDenseMatrix const &src,
    Epetra_Comm const &comm)
{
    Epetra_LocalMap map(src.M(), 0, comm);
    return Teuchos::rcp(new Epetra_MultiVector(
                            CV, map, src.A(), src.M(), src.N()));;
}

// Specializations of the apply methods
template<>
template<>
EpetraWrapper<Epetra_MultiVector> EpetraWrapper<Epetra_MultiVector>::apply(
    EpetraWrapper<Epetra_SerialDenseMatrix> const &other) const
{
    FUNCTION_TIMER("EpetraWrapper", "apply 1");
    EpetraWrapper<Epetra_MultiVector> out(*this, (*other).N());

    if ((*other).M() != ptr_->NumVectors())
    {
        std::cerr << "Incomplatible matrices of sizes "
                  << ptr_->MyLength() << "x" << ptr_->NumVectors() << " and "
                  << (*other).M() << "x" << (*other).N() << std::endl;
        return out;
    }

    Teuchos::RCP<Epetra_MultiVector> mv = SerialDenseMatrixToMultiVector(
        View, *other, ptr_->Comm());
    (*out).Multiply('N', 'N', 1.0, *ptr_, *mv, 0.0);

    return out;
}

template<>
template<>
EpetraWrapper<Epetra_MultiVector> EpetraWrapper<Epetra_CrsMatrix>::apply(
    EpetraWrapper<Epetra_MultiVector> const &other) const
{
    FUNCTION_TIMER("EpetraWrapper", "apply 2");
    EpetraWrapper<Epetra_MultiVector> out(Teuchos::rcp(new Epetra_MultiVector(*other)));
    ptr_->Apply(*other, *out);
    return out;
}

template<>
EpetraWrapper<Epetra_SerialDenseMatrix> EpetraWrapper<Epetra_SerialDenseMatrix>::apply(
    EpetraWrapper<Epetra_SerialDenseMatrix> const &other) const
{
    FUNCTION_TIMER("EpetraWrapper", "apply 3");
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

// Specializations of the assignment operators
template<>
EpetraWrapper<Epetra_MultiVector> &EpetraWrapper<Epetra_MultiVector>::operator =(
    EpetraWrapper<Epetra_MultiVector> &other)
{
    FUNCTION_TIMER("EpetraWrapper", "= 1");
    if (!is_view_)
    {
        ptr_ = other.ptr_;
        capacity_ = other.capacity_;
        size_ = other.size_;
        orthogonalized_ = other.orthogonalized_;
        return *this;
    }

    assert(num_vectors() == other.num_vectors());

    other.ptr_->ExtractCopy(ptr_->Values(), ptr_->MyLength());
    return *this;
}

template<>
EpetraWrapper<Epetra_MultiVector> &EpetraWrapper<Epetra_MultiVector>::operator =(
    EpetraWrapper<Epetra_MultiVector> const &other)
{
    FUNCTION_TIMER("EpetraWrapper", "= 2");
    if (!is_view_)
    {
        ptr_ = other.ptr_;
        capacity_ = other.capacity_;
        size_ = other.size_;
        orthogonalized_ = other.orthogonalized_;
        return *this;
    }

    assert(num_vectors() == other.num_vectors());

    other.ptr_->ExtractCopy(ptr_->Values(), ptr_->MyLength());
    return *this;
}

template<>
EpetraWrapper<Epetra_SerialDenseMatrix> &EpetraWrapper<Epetra_SerialDenseMatrix>::operator =(
    EpetraWrapper<Epetra_SerialDenseMatrix> const &other)
{
    FUNCTION_TIMER("EpetraWrapper", "= 3");
    ptr_ = other.ptr_;
    capacity_ = other.capacity_;
    size_ = other.size_;
    orthogonalized_ = other.orthogonalized_;
    return *this;
}
