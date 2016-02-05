#include "StlWrapper.hpp"

#include "BlasWrapper.hpp"
#include "LapackWrapper.hpp"

#include <cassert>
#include <iostream>
#include <cstring>
#include <cmath>
#include <random>

#define TIMER_ON
#include "Timer.hpp"

StlWrapper::StlWrapper()
    :
    ptr_(nullptr),
    m_(-1),
    n_(-1),
    m_max_(-1),
    n_max_(-1),
    orthogonalized_(0),
    is_view_(false)
{}

StlWrapper::StlWrapper(StlWrapper const &other)
    :
    StlWrapper()
{
    FUNCTION_TIMER("StlWrapper", "constructor 2");
    if (other.ptr_)
        ptr_ = std::make_shared<StlVector>(*other.ptr_);
    m_ = other.m_;
    n_ = other.n_;
    m_max_ = other.m_max_;
    n_max_ = other.n_max_;
    orthogonalized_ = other.orthogonalized_;
}

StlWrapper::StlWrapper(StlWrapper const &other, int n)
    :
    StlWrapper(other.M(), n)
{
    FUNCTION_TIMER("StlWrapper", "constructor 3");
}

StlWrapper::StlWrapper(int m, int n)
    :
    StlWrapper()
{
    FUNCTION_TIMER("StlWrapper", "constructor 4");
    m_ = m;
    n_ = n;
    m_max_ = m;
    n_max_ = n;
    ptr_ = std::make_shared<StlVector>(m, n);
}

StlWrapper &StlWrapper::operator =(StlWrapper &other)
{
    FUNCTION_TIMER("StlWrapper", "= 1");
    if (!is_view_)
    {
        ptr_ = other.ptr_;
        m_ = other.m_;
        n_ = other.n_;
        m_max_ = other.m_max_;
        n_max_ = other.n_max_;
        orthogonalized_ = other.orthogonalized_;
        return *this;
    }

    assert(N() == other.N());

    ptr_->set(*other.ptr_);
    return *this;
}

StlWrapper &StlWrapper::operator =(
    StlWrapper const &other)
{
    FUNCTION_TIMER("StlWrapper", "= 2");
    if (!is_view_)
    {
        ptr_ = other.ptr_;
        m_ = other.m_;
        n_ = other.n_;
        m_max_ = other.m_max_;
        n_max_ = other.n_max_;
        orthogonalized_ = other.orthogonalized_;
        return *this;
    }

    assert(N() == other.N());

    ptr_->set(*other.ptr_);
    return *this;
}

StlWrapper &StlWrapper::operator *=(double other)
{
    FUNCTION_TIMER("StlWrapper", "*=");
    BlasWrapper::DSCAL(m_ * n_, other, ptr_->get());
    return *this;
}

StlWrapper &StlWrapper::operator /=(double other)
{
    FUNCTION_TIMER("StlWrapper", "/=");
    BlasWrapper::DSCAL(m_ * n_, 1.0 / other, ptr_->get());
    return *this;
}

StlWrapper &StlWrapper::operator -=(
    StlWrapper const &other)
{
    FUNCTION_TIMER("StlWrapper", "-=");
    BlasWrapper::DAXPY(m_ * n_, -1.0, other.ptr_->get(), ptr_->get());
    return *this;
}
StlWrapper &StlWrapper::operator +=(
    StlWrapper const &other)
{
    FUNCTION_TIMER("StlWrapper", "+=");
    BlasWrapper::DAXPY(m_ * n_, 1.0, other.ptr_->get(), ptr_->get());
    return *this;
}

StlWrapper StlWrapper::operator +(
    StlWrapper const &other) const
{
    FUNCTION_TIMER("StlWrapper", "+");
    StlWrapper e(*this);
    e += other;
    return e;
}

StlWrapper StlWrapper::operator *(
    StlWrapper const &other) const
{
    FUNCTION_TIMER("StlWrapper", "* S");
    StlWrapper out(*this, other.N());

    if (other.M() != N())
    {
        std::cerr << "Incomplatible matrices of sizes "
                  << M() << "x" << N() << " and "
                  << other.M() << "x" << other.N() << std::endl;
        return out;
    }

    BlasWrapper::DGEMM('N', 'N', M(), other.N(), other.M(), 1.0,
                       ptr_->get(), m_max_, other.ptr_->get(), other.m_max_,
                       0.0, out.ptr_->get(), out.m_max_);

    return out;
}

StlWrapper operator *(double d, StlWrapper const &other)
{
    FUNCTION_TIMER("StlWrapper", "double *");
    StlWrapper e(other);
    e *= d;
    return e;
}

StlVector &StlWrapper::operator *()
{
    FUNCTION_TIMER("StlWrapper", "*");
    return *ptr_;
}

StlVector const &StlWrapper::operator *() const
{
    FUNCTION_TIMER("StlWrapper", "* 2");
    return *ptr_;
}

double &StlWrapper::operator ()(int m, int n)
{
    FUNCTION_TIMER("StlWrapper", "()");
    return (*ptr_)(m, n);
}

double const &StlWrapper::operator ()(int m, int n) const
{
    FUNCTION_TIMER("StlWrapper", "() 2");
    return (*ptr_)(m, n);
}

int StlWrapper::scale(double factor)
{
    FUNCTION_TIMER("StlWrapper", "scale");
    BlasWrapper::DSCAL(m_ * n_, factor, ptr_->get());
    return 0;
}

void StlWrapper::resize(int m)
{
    FUNCTION_TIMER("StlWrapper", "resize");
    resize(m_, m);
}

void StlWrapper::resize(int m, int n)
{
    FUNCTION_TIMER("StlWrapper", "resize 2");
    if (m <= m_max_ && n <= n_max_)
    {
        m_ = m;
        n_ = n;
        return;
    }

    if (m == m_max_)
    {
        std::shared_ptr<StlVector> new_ptr =
            std::make_shared<StlVector>(m, n);
        new_ptr->set(*ptr_);
        ptr_ = new_ptr;
        m_ = m;
        n_ = n;
        n_max_ = n;
    }

    std::cerr << "Warning: data not copied during resize" <<std::endl;

    ptr_ = std::make_shared<StlVector>(m, n);
}

double StlWrapper::norm() const
{
    FUNCTION_TIMER("StlWrapper", "norm");

    // Frobenius norm
    double out = 0.0;
    for (int i = 0; i < m_; ++i)
        for(int j = 0; j < n_; ++j)
        {
            double value = (*ptr_)(i, j);
            out += value * value;
        }
    return sqrt(out);
}

double StlWrapper::norm_inf() const
{
    FUNCTION_TIMER("StlWrapper", "norm_inf");
    double out = 0.0;
    for (int i = 0; i < m_; ++i)
    {
        double row_sum = 0.0;
        for(int j = 0; j < n_; ++j)
            row_sum += std::abs((*ptr_)(i, j));
        out = std::max(out, row_sum);
    }
    return sqrt(out);
}

void StlWrapper::orthogonalize()
{
    FUNCTION_TIMER("StlWrapper", "orthogonalize");
    for (int i = orthogonalized_; i < N(); i++)
    {
        StlWrapper v = view(i);
        v /= v.norm();
        if (i)
        {
            StlWrapper V = view(0, i-1);
            for (int k = 0; k < 2; k++)
                v -= V * V.dot(v);
        }
        v /= v.norm();
    }
    orthogonalized_ = N();
}

StlWrapper StlWrapper::view(int m, int n)
{
    FUNCTION_TIMER("StlWrapper", "view");
    StlWrapper out = *this;
    int num = n ? n-m+1 : 1;
    out.ptr_ = std::make_shared<StlVector>(&(*ptr_)[m_max_ * m], m_max_, num);
    out.n_ = num;
    out.is_view_ = true;
    return out;
}

StlWrapper StlWrapper::view(int m, int n) const
{
    FUNCTION_TIMER("StlWrapper", "view");
    StlWrapper out = *this;
    int num = n ? n-m+1 : 1;
    out.ptr_ = std::make_shared<StlVector>(&(*ptr_)[m_max_ * m], m_max_, num);
    out.n_ = num;
    out.is_view_ = true;
    return out;
}

StlWrapper StlWrapper::copy(int m, int n) const
{
    FUNCTION_TIMER("StlWrapper", "copy");
    StlWrapper out(*this);
    return out;
}

void StlWrapper::push_back(StlWrapper const &other, int m)
{
    FUNCTION_TIMER("StlWrapper", "push_back");
    int n = N();
    if (m == -1)
        m = other.N();
    resize(m + n);
    memcpy(&(*ptr_)(0, n), other.ptr_->get(),
           sizeof(double) * m * m_);
}

int StlWrapper::M() const
{
    FUNCTION_TIMER("StlWrapper", "M");
    return m_;
}

int StlWrapper::N() const
{
    FUNCTION_TIMER("StlWrapper", "N");
    return n_;
}

StlWrapper StlWrapper::dot(StlWrapper const &other) const
{
    FUNCTION_TIMER("StlWrapper", "dot");
    StlWrapper out(N(), other.N());

    if (other.N() != N())
    {
        std::cerr << "Incomplatible matrices of sizes "
                  << M() << "x" << N() << " and "
                  << other.M() << "x" << other.N() << std::endl;
        return out;
    }

    BlasWrapper::DGEMM('T', 'N', N(), other.N(), other.M(), 1.0,
                       ptr_->get(), m_max_, other.ptr_->get(), other.m_max_,
                       0.0, out.ptr_->get(), out.m_max_);

    return out;
}

void StlWrapper::random()
{
    FUNCTION_TIMER("StlWrapper", "random");
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1,1);
    for (int i = 0; i < m_; ++i)
        for(int j = 0; j < n_; ++j)
            (*ptr_)(i, j) = distribution(generator);
}
