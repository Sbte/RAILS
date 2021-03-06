#include "Epetra_MultiVectorWrapper.hpp"
#include "Epetra_SerialDenseMatrixWrapper.hpp"

#include <Epetra_SerialDenseMatrix.h>
#include <Epetra_MultiVector.h>
#include <Epetra_SerialComm.h>
#include <Epetra_LocalMap.h>

#define TIMER_ON
#include "Timer.hpp"

namespace RAILS
{

Epetra_MultiVectorWrapper::Epetra_MultiVectorWrapper()
    :
    ptr_(Teuchos::null),
    ptr_allocated_(Teuchos::null),
    capacity_(-1),
    size_(-1),
    orthogonalized_(0),
    is_view_(false),
    transpose_(false)
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "constructor 1");
}

Epetra_MultiVectorWrapper::Epetra_MultiVectorWrapper(Teuchos::RCP<Epetra_MultiVector> ptr)
    :
    Epetra_MultiVectorWrapper()
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "constructor 2");
    ptr_ = ptr;
}

Epetra_MultiVectorWrapper::Epetra_MultiVectorWrapper(Epetra_MultiVectorWrapper const &other)
    :
    Epetra_MultiVectorWrapper()
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "constructor 3");
    if (!other.ptr_.is_null())
        ptr_ = Teuchos::rcp(new Epetra_MultiVector(*other.ptr_));
    capacity_ = other.capacity_;
    size_ = other.size_;
    orthogonalized_ = other.orthogonalized_;
    transpose_ = other.transpose_;
}

Epetra_MultiVectorWrapper::Epetra_MultiVectorWrapper(Epetra_MultiVectorWrapper const &other, int n)
    :
    Epetra_MultiVectorWrapper()
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "constructor 4");
    size_ = n;
    capacity_ = n;
    if (!other.ptr_.is_null())
        ptr_ = Teuchos::rcp(new Epetra_MultiVector(other.ptr_->Map(), n));
}

Epetra_MultiVectorWrapper::Epetra_MultiVectorWrapper(int m, int n)
    :
    Epetra_MultiVectorWrapper()
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "constructor 5");
    size_ = n;
    capacity_ = n;

    // Should only be used for local testing
    Epetra_SerialComm comm;
    Epetra_Map map(m, 0, comm);
    ptr_ = Teuchos::rcp(new Epetra_MultiVector(map, n));
}

Epetra_MultiVectorWrapper::Epetra_MultiVectorWrapper(Epetra_MultiVectorWrapper &&other)
    :
    ptr_(other.ptr_),
    ptr_allocated_(other.ptr_allocated_),
    capacity_(other.capacity_),
    size_(other.size_),
    orthogonalized_(other.orthogonalized_),
    is_view_(other.is_view_),
    transpose_(other.transpose_)
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "constructor 6");
}

Epetra_MultiVectorWrapper &Epetra_MultiVectorWrapper::operator =(
    Epetra_MultiVectorWrapper &other)
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "= 1");
    if (!is_view_)
    {
        ptr_ = other.ptr_;
        size_ = other.size_;
        orthogonalized_ = other.orthogonalized_;
        transpose_ = other.transpose_;
        return *this;
    }

    assert(N() == other.N());

    other.ptr_->ExtractCopy(ptr_->Values(), ptr_->MyLength());
    return *this;
}

Epetra_MultiVectorWrapper &Epetra_MultiVectorWrapper::operator =(
    Epetra_MultiVectorWrapper const &other)
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "= 2");
    if (!is_view_)
    {
        ptr_ = other.ptr_;
        size_ = other.size_;
        orthogonalized_ = other.orthogonalized_;
        transpose_ = other.transpose_;
        return *this;
    }

    assert(N() == other.N());

    other.ptr_->ExtractCopy(ptr_->Values(), ptr_->MyLength());
    return *this;
}

Epetra_MultiVectorWrapper &Epetra_MultiVectorWrapper::operator =(
    Epetra_MultiVectorWrapper &&other)
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "= 3");
    if (!is_view_)
    {
        ptr_ = other.ptr_;
        size_ = other.size_;
        orthogonalized_ = other.orthogonalized_;
        transpose_ = other.transpose_;
        return *this;
    }

    assert(N() == other.N());

    other.ptr_->ExtractCopy(ptr_->Values(), ptr_->MyLength());
    return *this;
}

Epetra_MultiVectorWrapper &Epetra_MultiVectorWrapper::operator =(double other)
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "= 3");
    ptr_->PutScalar(other);
    orthogonalized_ = 0;
    return *this;
}

Epetra_MultiVectorWrapper &Epetra_MultiVectorWrapper::operator *=(double other)
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "*=");
    ptr_->Scale(other);
    orthogonalized_ = 0;
    return *this;
}

Epetra_MultiVectorWrapper &Epetra_MultiVectorWrapper::operator /=(double other)
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "/=");
    ptr_->Scale(1.0 / other);
    orthogonalized_ = 0;
    return *this;
}

Epetra_MultiVectorWrapper &Epetra_MultiVectorWrapper::operator -=(
    Epetra_MultiVectorWrapper const &other)
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "-=");
    ptr_->Update(-1.0, *other, 1.0);
    orthogonalized_ = 0;
    return *this;
}
Epetra_MultiVectorWrapper &Epetra_MultiVectorWrapper::operator +=(
    Epetra_MultiVectorWrapper const &other)
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "+=");
    ptr_->Update(1.0, *other, 1.0);
    orthogonalized_ = 0;
    return *this;
}

Epetra_MultiVectorWrapper Epetra_MultiVectorWrapper::operator +(
    Epetra_MultiVectorWrapper const &other) const
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "+");
    Epetra_MultiVectorWrapper out(*this);
    out += other;
    return out;
}

Epetra_MultiVectorWrapper Epetra_MultiVectorWrapper::operator *(
    Epetra_SerialDenseMatrixWrapper const &other) const
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "* SDM");
    Epetra_MultiVectorWrapper out(*this, other.N());

    if (other.M() != ptr_->NumVectors())
    {
        std::cerr << "Incomplatible matrices of sizes "
                  << ptr_->MyLength() << "x" << ptr_->NumVectors() << " and "
                  << other.M() << "x" << other.N() << std::endl;
        return out;
    }

    if (transpose_)
    {
        std::cerr << "Not correctly implemented for transpose" << std::endl;
        return out;
    }

    Teuchos::RCP<Epetra_MultiVector> mv = SerialDenseMatrixToMultiVector(
        View, *other, ptr_->Comm());
    (*out).Multiply(transpose_ ? 'T' : 'N', 'N', 1.0, *ptr_, *mv, 0.0);

    return out;
}

Epetra_MultiVectorWrapper Epetra_MultiVectorWrapper::operator *(
    Epetra_MultiVectorWrapper const &other) const
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "* MV");
    
    Epetra_LocalMap map(M(), 0, ptr_->Comm());
    Epetra_MultiVectorWrapper out(
        Teuchos::rcp(new Epetra_MultiVector(map, other.N())));

    if (other.M() != N())
    {
        std::cerr << "Incomplatible matrices of sizes "
                  << ptr_->MyLength() << "x" << ptr_->NumVectors() << " and "
                  << other.M() << "x" << other.N() << std::endl;
        return out;
    }

    (*out).Multiply(transpose_ ? 'T' : 'N', 'N', 1.0, *ptr_, *other.ptr_, 0.0);

    return out;
}

Epetra_MultiVector &Epetra_MultiVectorWrapper::operator *()
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "*");
    return *ptr_;
}

Epetra_MultiVector const &Epetra_MultiVectorWrapper::operator *() const
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "* 2");
    return *ptr_;
}

double &Epetra_MultiVectorWrapper::operator ()(int m, int n)
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "()");
    return (*ptr_)[n][m];
}

double const &Epetra_MultiVectorWrapper::operator ()(int m, int n) const
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "() 2");
    return (*ptr_)[n][m];
}

void Epetra_MultiVectorWrapper::resize(int m)
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "resize");
    // Check if ptr_allocated_ is set
    if (ptr_allocated_.is_null())
        ptr_allocated_ = ptr_;

    // Set the capacity if it was not set
    if (capacity_ == -1)
        capacity_ = N();

    // Allocate more memory if needed, and copy the old vector
    if (capacity_ < m)
    {
        Teuchos::RCP<Epetra_MultiVector> new_ptr = Teuchos::rcp(
            new Epetra_MultiVector(ptr_allocated_->Map(), m));
        if (!ptr_.is_null())
            ptr_->ExtractCopy(new_ptr->Values(), new_ptr->MyLength());
        ptr_ = new_ptr;
        ptr_allocated_ = new_ptr;
        capacity_ = m;
    }
    else if (!m)
    {
        ptr_ = Teuchos::null;
    }
    else if (ptr_.is_null() || (ptr_allocated_->Values() == ptr_->Values()))
    {
        // Now only view a part of ptr_allocated_
        ptr_ = Teuchos::rcp(new Epetra_MultiVector(View, *ptr_allocated_, 0, m));
    }
    else
    {
        // Copy to ptr_allocated_ and view only part of that
        ptr_->ExtractCopy(ptr_allocated_->Values(), ptr_allocated_->MyLength());
        ptr_ = Teuchos::rcp(new Epetra_MultiVector(View, *ptr_allocated_, 0, m));
    }
    size_ = m;
    orthogonalized_ = std::min(orthogonalized_, m);
}

double Epetra_MultiVectorWrapper::norm() const
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "norm");
    double *nrm = new double[N()];
    ptr_->Norm2(nrm);

    double out = 0.0;
    for (int i = 0; i < N(); ++i)
      out += nrm[i] * nrm[i];

    delete[] nrm;

    return sqrt(out);
}

double Epetra_MultiVectorWrapper::norm_inf() const
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "norm_inf");
    double out;
    ptr_->NormInf(&out);
    return out;
}

void Epetra_MultiVectorWrapper::orthogonalize()
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "orthogonalize");
    for (int i = orthogonalized_; i < N(); i++)
    {
        Epetra_MultiVectorWrapper v = view(i);
        v /= v.norm();
        if (i)
        {
            Epetra_MultiVectorWrapper V = view(0, i-1);
            for (int k = 0; k < 2; k++)
                v -= V * V.dot(v);
        }
        v /= v.norm();
    }
    orthogonalized_ = N();
}

Epetra_MultiVectorWrapper Epetra_MultiVectorWrapper::view(int m, int n)
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "view");
    Epetra_MultiVectorWrapper out;
    int num = 1;
    if (n > 0 && m >= 0)
        num = n - m + 1;
    else if (m < 0)
    {
        m = 0;
        num = N();
    }
    out.ptr_ = Teuchos::rcp(new Epetra_MultiVector(View, *ptr_, m, num));
    out.is_view_ = true;
    return out;
}

Epetra_MultiVectorWrapper Epetra_MultiVectorWrapper::view(int m, int n) const
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "view");
    Epetra_MultiVectorWrapper out;
    int num = 1;
    if (n > 0 && m >= 0)
        num = n - m + 1;
    else if (m < 0)
    {
        m = 0;
        num = N();
    }
    out.ptr_ = Teuchos::rcp(new Epetra_MultiVector(View, *ptr_, m, num));
    out.is_view_ = true;
    return out;
}

Epetra_MultiVectorWrapper Epetra_MultiVectorWrapper::copy() const
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "copy");
    return *this;
}

Epetra_MultiVectorWrapper Epetra_MultiVectorWrapper::transpose() const
{
    Epetra_MultiVectorWrapper out(*this);
    out.transpose_ = !out.transpose_;
    return out;
}

void Epetra_MultiVectorWrapper::push_back(Epetra_MultiVectorWrapper const &other, int m)
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "push_back");
    int n = N();
    if (m == -1)
        m = other.N();
    resize(m + n);
    memcpy((*ptr_)[n], other.ptr_->Values(), sizeof(double)*m*ptr_->MyLength());
}

int Epetra_MultiVectorWrapper::M() const
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "M");
    return transpose_ ? (size_ ? ptr_->NumVectors() : size_) : ptr_->Map().NumGlobalPoints();
}

int Epetra_MultiVectorWrapper::N() const
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "N");
    return transpose_ ? ptr_->Map().NumGlobalPoints() : (size_ ? ptr_->NumVectors() : size_);
}

Epetra_SerialDenseMatrixWrapper Epetra_MultiVectorWrapper::dot(
    Epetra_MultiVectorWrapper const &other) const
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "dot");

    RAILS_START_TIMER("Epetra_MultiVectorWrapper", "dot copy");
    Teuchos::RCP<Epetra_SerialDenseMatrix> mat = Teuchos::rcp(
        new Epetra_SerialDenseMatrix(N(), other.N()));
    Teuchos::RCP<Epetra_MultiVector> mv = SerialDenseMatrixToMultiVector(
        View, *mat, ptr_->Comm());
    RAILS_END_TIMER("Epetra_MultiVectorWrapper", "dot copy");

    RAILS_START_TIMER("Epetra_MultiVectorWrapper", "dot multiply");
    mv->Multiply('T', 'N', 1.0, *ptr_, *other.ptr_, 0.0);
    RAILS_END_TIMER("Epetra_MultiVectorWrapper", "dot multiply");

    RAILS_START_TIMER("Epetra_MultiVectorWrapper", "dot copy 2");
    Epetra_SerialDenseMatrixWrapper out(mat);
    RAILS_END_TIMER("Epetra_MultiVectorWrapper", "dot copy 2");
    return out;
}

void Epetra_MultiVectorWrapper::random()
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "random");
    ptr_->Random();
    orthogonalized_ = 0;
}

Teuchos::RCP<Epetra_MultiVector> SerialDenseMatrixToMultiVector(
    Epetra_DataAccess CV, Epetra_SerialDenseMatrix const &src,
    Epetra_Comm const &comm)
{
    Epetra_LocalMap map(src.M(), 0, comm);
    return Teuchos::rcp(new Epetra_MultiVector(
                            CV, map, src.A(), src.M(), src.N()));
}

Teuchos::RCP<Epetra_SerialDenseMatrix> MultiVectorToSerialDenseMatrix(
    Epetra_DataAccess CV, Epetra_MultiVector const &src)
{
    return Teuchos::rcp(new Epetra_SerialDenseMatrix(
                            CV, src.Values(), src.MyLength(),
                            src.MyLength(), src.NumVectors()));
}

Epetra_MultiVectorWrapper operator *(
    double d, Epetra_MultiVectorWrapper const &other)
{
    RAILS_FUNCTION_TIMER("Epetra_MultiVectorWrapper", "double *");
    Epetra_MultiVectorWrapper out(other);
    out *= d;
    return out;
}

}
