#include "Epetra_SerialDenseMatrixWrapper.hpp"

#include <Teuchos_LAPACK.hpp>

#include <Epetra_SerialDenseMatrix.h>

#define TIMER_ON
#include "Timer.hpp"

Epetra_SerialDenseMatrixWrapper::Epetra_SerialDenseMatrixWrapper()
    :
    ptr_(Teuchos::null)
{}

Epetra_SerialDenseMatrixWrapper::Epetra_SerialDenseMatrixWrapper(Teuchos::RCP<Epetra_SerialDenseMatrix> ptr)
    :
    Epetra_SerialDenseMatrixWrapper()
{
    FUNCTION_TIMER("Epetra_SerialDenseMatrixWrapper", "constructor 1");
    ptr_ = ptr;
}

Epetra_SerialDenseMatrixWrapper::Epetra_SerialDenseMatrixWrapper(Epetra_SerialDenseMatrixWrapper const &other)
    :
    Epetra_SerialDenseMatrixWrapper()
{
    FUNCTION_TIMER("Epetra_SerialDenseMatrixWrapper", "constructor 2");
    if (!other.ptr_.is_null())
        ptr_ = Teuchos::rcp(new Epetra_SerialDenseMatrix(*other.ptr_));
}

Epetra_SerialDenseMatrixWrapper::Epetra_SerialDenseMatrixWrapper(int m, int n)
    :
    Epetra_SerialDenseMatrixWrapper()
{
    FUNCTION_TIMER("Epetra_SerialDenseMatrixWrapper", "constructor 3");
    ptr_ = Teuchos::rcp(new Epetra_SerialDenseMatrix(m, n));
}

Epetra_SerialDenseMatrixWrapper &Epetra_SerialDenseMatrixWrapper::operator =(
    Epetra_SerialDenseMatrixWrapper const &other)
{
    FUNCTION_TIMER("Epetra_SerialDenseMatrixWrapper", "=");
    ptr_ = other.ptr_;
    return *this;
}

Epetra_SerialDenseMatrixWrapper &Epetra_SerialDenseMatrixWrapper::operator =(
    double other)
{
    FUNCTION_TIMER("Epetra_SerialDenseMatrixWrapper", "=");
    std::fill_n(ptr_->A(), ptr_->LDA() * ptr_->N(), other);
    return *this;
}

Epetra_SerialDenseMatrixWrapper &Epetra_SerialDenseMatrixWrapper::operator *=(double other)
{
    FUNCTION_TIMER("Epetra_SerialDenseMatrixWrapper", "*=");
    ptr_->Scale(other);
    return *this;
}

Epetra_SerialDenseMatrix &Epetra_SerialDenseMatrixWrapper::operator *()
{
    FUNCTION_TIMER("Epetra_SerialDenseMatrixWrapper", "*");
    return *ptr_;
}

Epetra_SerialDenseMatrix const &Epetra_SerialDenseMatrixWrapper::operator *() const
{
    FUNCTION_TIMER("Epetra_SerialDenseMatrixWrapper", "* 2");
    return *ptr_;
}

Epetra_SerialDenseMatrixWrapper Epetra_SerialDenseMatrixWrapper::operator *(
    Epetra_SerialDenseMatrixWrapper const &other) const
{
    FUNCTION_TIMER("Epetra_SerialDenseMatrixWrapper", "* SDM");
    Epetra_SerialDenseMatrixWrapper out(Teuchos::rcp(new Epetra_SerialDenseMatrix(*other)));
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

Epetra_SerialDenseMatrixWrapper::operator double*() const
{
    FUNCTION_TIMER("Epetra_SerialDenseMatrixWrapper", "double*");
    return ptr_->A();
}

double &Epetra_SerialDenseMatrixWrapper::operator ()(int m, int n)
{
    FUNCTION_TIMER("Epetra_SerialDenseMatrixWrapper", "()");
    return (*ptr_)(m, n);
}

double const &Epetra_SerialDenseMatrixWrapper::operator ()(int m, int n) const
{
    FUNCTION_TIMER("Epetra_SerialDenseMatrixWrapper", "() 2");
    return (*ptr_)(m, n);
}

double Epetra_SerialDenseMatrixWrapper::norm_inf() const
{
    FUNCTION_TIMER("Epetra_SerialDenseMatrixWrapper", "norm_inf");
    return ptr_->NormInf();
}

void Epetra_SerialDenseMatrixWrapper::resize(int m, int n)
{
    FUNCTION_TIMER("Epetra_SerialDenseMatrixWrapper", "resize");
    ptr_->Reshape(m, n);
}

Epetra_SerialDenseMatrixWrapper Epetra_SerialDenseMatrixWrapper::copy(int m, int n) const
{
    FUNCTION_TIMER("Epetra_SerialDenseMatrixWrapper", "copy");
    Epetra_SerialDenseMatrixWrapper out(*this);
    return out;
}

int Epetra_SerialDenseMatrixWrapper::M() const
{
    FUNCTION_TIMER("Epetra_SerialDenseMatrixWrapper", "M");
    return ptr_->M();
}

int Epetra_SerialDenseMatrixWrapper::N() const
{
    FUNCTION_TIMER("Epetra_SerialDenseMatrixWrapper", "N");
    return ptr_->N();
}

int Epetra_SerialDenseMatrixWrapper::LDA() const
{
    FUNCTION_TIMER("Epetra_SerialDenseMatrixWrapper", "LDA");
    return ptr_->LDA();
}

void Epetra_SerialDenseMatrixWrapper::eigs(Epetra_SerialDenseMatrixWrapper &v,
                                           Epetra_SerialDenseMatrixWrapper &d) const
{
    FUNCTION_TIMER("Epetra_SerialDenseMatrixWrapper", "eigs");
    int m = M();
    v = copy();

    // Put the diagonal in d
    d.resize(m, 1);
    for (int i = 0; i < m; i++)
        d(i, 0) = v(i, i);

    // Put the offdiagonal in e
    Epetra_SerialDenseMatrix e(m-1, 1);
    for (int i = 0; i < m-1; i++)
        e(i, 0) = v(i+1, i);

    Epetra_SerialDenseMatrix work(std::max(1,2*m-2), 1);

    int info;
    Teuchos::LAPACK<int, double> lapack;
    lapack.STEQR('I', m, (*d).A(),
                 e.A(), (*v).A(),
                 m, work.A(), &info);

    if (info)
        std::cerr << "Eigenvalues info = " << info << std::endl;
}
