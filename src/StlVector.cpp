#include "StlVector.hpp"

#include <cstring>

StlVector::StlVector(int m, int n)
    :
    ptr_(new double[m * n]),
    is_view_(false),
    m_(m),
    n_(n)
{}

StlVector::StlVector(double *other, int m, int n)
    :
    ptr_(other),
    is_view_(true),
    m_(m),
    n_(n)
{}

StlVector::StlVector(StlVector const &other)
    :
    ptr_(new double[other.m_ * other.n_]),
    is_view_(false),
    m_(other.m_),
    n_(other.n_)
{
    memcpy(ptr_, other.ptr_, sizeof(double) * m_ * n_);
}

StlVector::~StlVector()
{
    if (!is_view_)
        delete[] ptr_;
}

double &StlVector::operator [](int i)
{
    return ptr_[i];
}

double &StlVector::operator ()(int i, int j)
{
    return ptr_[i];
}

int StlVector::set(StlVector const &other)
{
    memcpy(ptr_, other.ptr_, sizeof(double) * other.m_ * other.n_);
    return 0;
}

double *StlVector::get()
{
    return ptr_;
}
