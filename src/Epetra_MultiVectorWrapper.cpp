#include "Epetra_MultiVectorWrapper.hpp"
#include "Epetra_SerialDenseMatrixWrapper.hpp"

#include <Epetra_SerialDenseMatrix.h>
#include <Epetra_MultiVector.h>
#include <Epetra_LocalMap.h>

#define TIMER_ON
#include "Timer.hpp"

Epetra_MultiVectorWrapper::Epetra_MultiVectorWrapper()
    :
    ptr_(Teuchos::null),
    ptr_allocated_(Teuchos::null),
    capacity_(-1),
    size_(-1),
    orthogonalized_(0),
    is_view_(false)
{}

Epetra_MultiVectorWrapper::Epetra_MultiVectorWrapper(Teuchos::RCP<Epetra_MultiVector> ptr)
    :
    Epetra_MultiVectorWrapper()
{
    FUNCTION_TIMER("Epetra_MultiVectorWrapper", "constructor 1");
    ptr_ = ptr;
}

Epetra_MultiVectorWrapper::Epetra_MultiVectorWrapper(Epetra_MultiVectorWrapper const &other)
    :
    Epetra_MultiVectorWrapper()
{
    FUNCTION_TIMER("Epetra_MultiVectorWrapper", "constructor 2");
    if (!other.ptr_.is_null())
        ptr_ = Teuchos::rcp(new Epetra_MultiVector(*other.ptr_));
    orthogonalized_ = other.orthogonalized_;
}

Epetra_MultiVectorWrapper::Epetra_MultiVectorWrapper(Epetra_MultiVectorWrapper const &other, int n)
    :
    Epetra_MultiVectorWrapper()
{
    FUNCTION_TIMER("Epetra_MultiVectorWrapper", "constructor 3");
    size_ = n;
    capacity_ = n;
    if (!other.ptr_.is_null())
        ptr_ = Teuchos::rcp(new Epetra_MultiVector(other.ptr_->Map(), n));
}

Epetra_MultiVectorWrapper &Epetra_MultiVectorWrapper::operator =(
    Epetra_MultiVectorWrapper &other)
{
    FUNCTION_TIMER("Epetra_MultiVectorWrapper", "= 1");
    if (!is_view_)
    {
        ptr_ = other.ptr_;
        size_ = other.size_;
        orthogonalized_ = other.orthogonalized_;
        return *this;
    }

    assert(N() == other.N());

    other.ptr_->ExtractCopy(ptr_->Values(), ptr_->MyLength());
    return *this;
}

Epetra_MultiVectorWrapper &Epetra_MultiVectorWrapper::operator =(
    Epetra_MultiVectorWrapper const &other)
{
    FUNCTION_TIMER("Epetra_MultiVectorWrapper", "= 2");
    if (!is_view_)
    {
        ptr_ = other.ptr_;
        size_ = other.size_;
        orthogonalized_ = other.orthogonalized_;
        return *this;
    }

    assert(N() == other.N());

    other.ptr_->ExtractCopy(ptr_->Values(), ptr_->MyLength());
    return *this;
}

Epetra_MultiVectorWrapper &Epetra_MultiVectorWrapper::operator *=(double other)
{
    FUNCTION_TIMER("Epetra_MultiVectorWrapper", "*=");
    ptr_->Scale(other);
    return *this;
}

Epetra_MultiVectorWrapper &Epetra_MultiVectorWrapper::operator /=(double other)
{
    FUNCTION_TIMER("Epetra_MultiVectorWrapper", "/=");
    ptr_->Scale(1.0 / other);
    return *this;
}

Epetra_MultiVectorWrapper &Epetra_MultiVectorWrapper::operator -=(Epetra_MultiVectorWrapper const &other)
{
    FUNCTION_TIMER("Epetra_MultiVectorWrapper", "-=");
    ptr_->Update(-1.0, *other, 1.0);
    return *this;
}
Epetra_MultiVectorWrapper &Epetra_MultiVectorWrapper::operator +=(Epetra_MultiVectorWrapper const &other)
{
    FUNCTION_TIMER("Epetra_MultiVectorWrapper", "+=");
    ptr_->Update(1.0, *other, 1.0);
    return *this;
}

Epetra_MultiVectorWrapper Epetra_MultiVectorWrapper::operator +(Epetra_MultiVectorWrapper const &other) const
{
    FUNCTION_TIMER("Epetra_MultiVectorWrapper", "+");
    Epetra_MultiVectorWrapper e(*this);
    e += other;
    return e;
}

Epetra_MultiVector &Epetra_MultiVectorWrapper::operator *()
{
    FUNCTION_TIMER("Epetra_MultiVectorWrapper", "*");
    return *ptr_;
}

Epetra_MultiVector const &Epetra_MultiVectorWrapper::operator *() const
{
    FUNCTION_TIMER("Epetra_MultiVectorWrapper", "* 2");
    return *ptr_;
}

int Epetra_MultiVectorWrapper::scale(double factor)
{
    FUNCTION_TIMER("Epetra_MultiVectorWrapper", "scale");
    return ptr_->Scale(factor);
}

void Epetra_MultiVectorWrapper::resize(int m)
{
    FUNCTION_TIMER("Epetra_MultiVectorWrapper", "resize");
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
        // TODO: Needs a test
        // Copy to ptr_allocated_ and view only part of that
        ptr_->ExtractCopy(ptr_allocated_->Values(), ptr_allocated_->MyLength());
        ptr_ = Teuchos::rcp(new Epetra_MultiVector(View, *ptr_allocated_, 0, m));
    }
    size_ = m;
}

double Epetra_MultiVectorWrapper::norm(int n) const
{
    FUNCTION_TIMER("Epetra_MultiVectorWrapper", "norm");
    if (N() == 1)
    {
        double out;
        ptr_->Norm2(&out);
        return out;
    }
    return view(n).norm(n);
}

double Epetra_MultiVectorWrapper::norm_inf(int n) const
{
    FUNCTION_TIMER("Epetra_MultiVectorWrapper", "norm_inf");
    if (N() == 1)
    {
        double out;
        ptr_->NormInf(&out);
        return out;
    }
    return view(n).norm_inf(n);
}

void Epetra_MultiVectorWrapper::orthogonalize()
{
    FUNCTION_TIMER("Epetra_MultiVectorWrapper", "orthogonalize");
    for (int i = orthogonalized_; i < N(); i++)
    {
        Epetra_MultiVectorWrapper v = view(i);
        v /= v.norm();
        if (i)
        {
            Epetra_MultiVectorWrapper V = view(0, i-1);
            for (int k = 0; k < 2; k++)
                v -= V.apply(V.dot(v));
        }
        v /= v.norm();
    }
    orthogonalized_ = N();
}

Epetra_MultiVectorWrapper Epetra_MultiVectorWrapper::view(int m, int n)
{
    FUNCTION_TIMER("Epetra_MultiVectorWrapper", "view");
    Epetra_MultiVectorWrapper out;
    int num = n ? n-m+1 : 1;
    out.ptr_ = Teuchos::rcp(new Epetra_MultiVector(View, *ptr_, m, num));
    out.is_view_ = true;
    return out;
}

Epetra_MultiVectorWrapper Epetra_MultiVectorWrapper::view(int m, int n) const
{
    FUNCTION_TIMER("Epetra_MultiVectorWrapper", "view");
    Epetra_MultiVectorWrapper out;
    int num = n ? n-m+1 : 1;
    out.ptr_ = Teuchos::rcp(new Epetra_MultiVector(View, *ptr_, m, num));
    out.is_view_ = true;
    return out;
}

Epetra_MultiVectorWrapper Epetra_MultiVectorWrapper::copy(int m, int n) const
{
    FUNCTION_TIMER("Epetra_MultiVectorWrapper", "copy");
    Epetra_MultiVectorWrapper out(*this);
    return out;
}

void Epetra_MultiVectorWrapper::push_back(Epetra_MultiVectorWrapper const &other, int m)
{
    FUNCTION_TIMER("Epetra_MultiVectorWrapper", "push_back");
    int n = N();
    if (m == -1)
        m = other.N();
    resize(m + n);
    memcpy((*ptr_)[n], other.ptr_->Values(), sizeof(double)*m*ptr_->MyLength());
}

int Epetra_MultiVectorWrapper::M() const
{
    FUNCTION_TIMER("Epetra_MultiVectorWrapper", "M");
    return ptr_->Map().NumGlobalPoints();
}

int Epetra_MultiVectorWrapper::N() const
{
    FUNCTION_TIMER("Epetra_MultiVectorWrapper", "N");
    return (size_ ? ptr_->NumVectors() : size_);
}

Epetra_SerialDenseMatrixWrapper Epetra_MultiVectorWrapper::dot(
    Epetra_MultiVectorWrapper const &other) const
{
    FUNCTION_TIMER("Epetra_MultiVectorWrapper", "dot");

    START_TIMER("Epetra_MultiVectorWrapper", "dot copy");
    Teuchos::RCP<Epetra_SerialDenseMatrix> mat = Teuchos::rcp(
        new Epetra_SerialDenseMatrix(N(), other.N()));
    Teuchos::RCP<Epetra_MultiVector> mv = SerialDenseMatrixToMultiVector(
        View, *mat, ptr_->Comm());
    END_TIMER("Epetra_MultiVectorWrapper", "dot copy");

    START_TIMER("Epetra_MultiVectorWrapper", "dot multiply");
    mv->Multiply('T', 'N', 1.0, *ptr_, *other.ptr_, 0.0);
    END_TIMER("Epetra_MultiVectorWrapper", "dot multiply");

    START_TIMER("Epetra_MultiVectorWrapper", "dot copy 2");
    Epetra_SerialDenseMatrixWrapper out(mat);
    END_TIMER("Epetra_MultiVectorWrapper", "dot copy 2");
    return out;
}

void Epetra_MultiVectorWrapper::random()
{
    FUNCTION_TIMER("Epetra_MultiVectorWrapper", "random");
    ptr_->Random();
}

Epetra_MultiVectorWrapper Epetra_MultiVectorWrapper::apply(
    Epetra_SerialDenseMatrixWrapper const &other) const
{
    FUNCTION_TIMER("Epetra_MultiVectorWrapper", "apply 1");
    Epetra_MultiVectorWrapper out(*this, other.N());

    if (other.M() != ptr_->NumVectors())
    {
        std::cerr << "Incomplatible matrices of sizes "
                  << ptr_->MyLength() << "x" << ptr_->NumVectors() << " and "
                  << other.M() << "x" << other.N() << std::endl;
        return out;
    }

    Teuchos::RCP<Epetra_MultiVector> mv = SerialDenseMatrixToMultiVector(
        View, *other, ptr_->Comm());
    (*out).Multiply('N', 'N', 1.0, *ptr_, *mv, 0.0);

    return out;
}

Epetra_MultiVectorWrapper operator *(double d, Epetra_MultiVectorWrapper const &other)
{
    FUNCTION_TIMER("Epetra_MultiVectorWrapper", "friend *");
    Epetra_MultiVectorWrapper e(other);
    e *= d;
    return e;
}

Teuchos::RCP<Epetra_MultiVector> SerialDenseMatrixToMultiVector(
    Epetra_DataAccess CV, Epetra_SerialDenseMatrix const &src,
    Epetra_Comm const &comm)
{
    Epetra_LocalMap map(src.M(), 0, comm);
    return Teuchos::rcp(new Epetra_MultiVector(
                            CV, map, src.A(), src.M(), src.N()));;
}

Teuchos::RCP<Epetra_SerialDenseMatrix> MultiVectorToSerialDenseMatrix(
    Epetra_DataAccess CV, Epetra_MultiVector const &src)
{
    return Teuchos::rcp(new Epetra_SerialDenseMatrix(
                            CV, src.Values(), src.MyLength(),
                            src.MyLength(), src.NumVectors()));;
}
