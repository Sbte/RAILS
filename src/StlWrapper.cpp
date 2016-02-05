#include "StlWrapper.hpp"

#include "StlTools.hpp"
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
    is_view_(false),
    transpose_(false),
    op_(nullptr)
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
    transpose_ = other.transpose_;
    op_ = other.op_;
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
        transpose_ = other.transpose_;
        op_ = other.op_;
        return *this;
    }

    assert(N() == other.N());

    ptr_->set(*other.ptr_);
    return *this;
}

StlWrapper &StlWrapper::operator =(StlWrapper const &other)
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
        transpose_ = other.transpose_;
        op_ = other.op_;
        return *this;
    }

    assert(N() == other.N());

    ptr_->set(*other.ptr_);
    return *this;
}

StlWrapper &StlWrapper::operator *=(double other)
{
    FUNCTION_TIMER("StlWrapper", "*=");
    scale(other);
    return *this;
}

StlWrapper &StlWrapper::operator /=(double other)
{
    FUNCTION_TIMER("StlWrapper", "/=");
    scale(1.0 / other);
    return *this;
}

StlWrapper &StlWrapper::operator -=(StlWrapper const &other)
{
    FUNCTION_TIMER("StlWrapper", "-=");
    BlasWrapper::DAXPY(m_ * n_, -1.0, other.ptr_->get(), ptr_->get());
    orthogonalized_ = 0;
    return *this;
}
StlWrapper &StlWrapper::operator +=(StlWrapper const &other)
{
    FUNCTION_TIMER("StlWrapper", "+=");
    BlasWrapper::DAXPY(m_ * n_, 1.0, other.ptr_->get(), ptr_->get());
    orthogonalized_ = 0;
    return *this;
}

StlWrapper StlWrapper::operator +(StlWrapper const &other) const
{
    FUNCTION_TIMER("StlWrapper", "+");
    StlWrapper e(*this);
    e += other;
    return e;
}

StlWrapper StlWrapper::operator *(StlWrapper const &other) const
{
    FUNCTION_TIMER("StlWrapper", "* S");

    if (op_)
      return *op_ * other;

    StlWrapper out(*this, other.N());
    if (other.M() != N())
    {
        std::cerr << "Incomplatible matrices of sizes "
                  << M() << "x" << N() << " and "
                  << other.M() << "x" << other.N() << std::endl;
        return out;
    }

    BlasWrapper::DGEMM(transpose_ ? 'T' : 'N', other.transpose_ ? 'T' : 'N',
                       M(), other.N(), other.M(), 1.0,
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

StlWrapper::operator double*() const
{
    FUNCTION_TIMER("StlWrapper", "double*");
    return ptr_->get();
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
    BlasWrapper::DSCAL(m_max_ * n_, factor, ptr_->get());
    orthogonalized_ = 0;
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
        return;
    }

    StlWrapper out(m, n);
    if (m_max_ > 0)
    {
        std::cerr << "Warning: data copied during resize from size "
                  << m_ << "x" << n_ << " to " << m << "x" << n
                  << " with capacity " << m_max_ << "x" << n_
                  << ", which is very inefficient." << std::endl;

        for (int i = 0; i < std::min(n_, n); ++i)
            memcpy(&(*out.ptr_)(0, i), &(*ptr_)(0, i),
                   sizeof(double) * m_);
    }
    *this = out;
}

double StlWrapper::norm() const
{
    FUNCTION_TIMER("StlWrapper", "norm");

    // Frobenius norm
    double out = 0.0;
    for (int i = 0; i < m_; ++i)
        for (int j = 0; j < n_; ++j)
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
        for (int j = 0; j < n_; ++j)
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
    int num = 1;
    if (n > 0 && m >= 0)
        num = n - m + 1;
    else if (m < 0)
    {
        m = 0;
        num = N();
    }
    out.ptr_ = std::make_shared<StlVector>(&(*ptr_)[m_max_ * m], m_max_, num);
    out.n_ = num;
    out.n_max_ = num;
    out.is_view_ = true;
    return out;
}

StlWrapper StlWrapper::view(int m, int n) const
{
    FUNCTION_TIMER("StlWrapper", "view 2");
    StlWrapper out = *this;
    int num = 1;
    if (n > 0 && m >= 0)
        num = n - m + 1;
    else if (m < 0)
    {
        m = 0;
        num = N();
    }
    out.ptr_ = std::make_shared<StlVector>(&(*ptr_)[m_max_ * m], m_max_, num);
    out.n_ = num;
    out.n_max_ = num;
    out.is_view_ = true;
    return out;
}

StlWrapper StlWrapper::copy() const
{
    FUNCTION_TIMER("StlWrapper", "copy");

    if (op_)
    {
        // Copy an operator into a matrix
        int m = op_->M();
        int n = op_->M();

        StlWrapper out(m, n);
        StlWrapper test_vec(m, 1);
        for (int i = 0; i < n; ++i)
        {
            test_vec.scale(0.0);
            test_vec(i, 0) = 1.0;
            out.view(i) = (*op_) * test_vec;
        }
        return out;
    }

    StlWrapper out(*this);
    return out;
}

void StlWrapper::push_back(StlWrapper const &other)
{
    FUNCTION_TIMER("StlWrapper", "push_back");
    int n = N();
    int other_n = other.N();
    resize(other_n + n);
    memcpy(&(*ptr_)(0, n), other.ptr_->get(), sizeof(double) * other_n * m_);
}

int StlWrapper::M() const
{
    FUNCTION_TIMER("StlWrapper", "M");
    return transpose_ ? n_ : m_;
}

int StlWrapper::N() const
{
    FUNCTION_TIMER("StlWrapper", "N");
    return transpose_ ? m_ : n_;
}

int StlWrapper::LDA() const
{
    FUNCTION_TIMER("StlWrapper", "N");
    return transpose_ ? n_max_ : m_max_;
}

StlWrapper StlWrapper::dot(StlWrapper const &other) const
{
    FUNCTION_TIMER("StlWrapper", "dot");
    StlWrapper out(N(), other.N());

    if (other.M() != M())
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
    std::default_random_engine generator(std::rand());
    std::uniform_real_distribution<double> distribution(-1,1);
    for (int i = 0; i < m_; ++i)
        for (int j = 0; j < n_; ++j)
            (*ptr_)(i, j) = distribution(generator);
    orthogonalized_ = 0;
}

StlWrapper StlWrapper::transpose()
{
    FUNCTION_TIMER("StlWrapper", "transpose");
    StlWrapper tmp(*this);
    tmp.transpose_ = !tmp.transpose_;
    return tmp;
}

int StlWrapper::eigs(StlWrapper &v, StlWrapper &d,
                     int num, double tol) const
{
    FUNCTION_TIMER("StlWrapper", "eigs");
    v = copy();

    int m = v.M();

    if (num < 1)
        num = m;

    d.resize(m, 1);

    int info;
    LapackWrapper::DSYEV('V', 'U', m, v.ptr_->get(), v.LDA(),
                          d.ptr_->get(), &info);

    if (num != m || tol > 1e-14)
    {
        std::vector<int> indices;
        find_largest_eigenvalues(d, indices, num);

        StlWrapper tmpv(m, num);
        StlWrapper tmpd(num, 1);
        int idx = 0;
        for (int i = 0; i < num; ++i)
        {
            if (std::abs(d(indices[i], 0)) > tol)
            {
                tmpv.view(idx) = v.view(indices[i]);
                tmpd(idx, 0) = d(indices[i], 0);
                idx++;
            }
        }

        tmpv.resize(idx);
        tmpd.resize(idx, 1);

        v = tmpv;
        d = tmpd;
    }

    if (info)
        std::cerr << "Eigenvalues info = " << info << std::endl;

    return info;
}
