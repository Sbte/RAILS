#ifndef STLWRAPPER_H
#define STLWRAPPER_H

#include "StlVector.hpp"

#include <memory>

namespace RAILS
{

class StlWrapper
{
    std::shared_ptr<StlVector> ptr_;

    // Actual sizes of the matrix
    int m_;
    int n_;

    // Capacity of the matrix
    int m_max_;
    int n_max_;

    // Amount of vectors that are already orthogonal
    int orthogonalized_;

    // Vector is a view
    bool is_view_;

    // Matrix is used as transpose or not in * methods
    bool transpose_;

public:
    StlWrapper();
    StlWrapper(std::shared_ptr<StlVector> ptr);
    StlWrapper(StlWrapper const &other);
    StlWrapper(StlWrapper const &other, int n);
    StlWrapper(int m, int n);

    virtual ~StlWrapper() {}

    StlWrapper &operator =(StlWrapper &other);
    StlWrapper &operator =(StlWrapper const &other);

    StlWrapper &operator =(double other);

    StlWrapper &operator *=(double other);
    StlWrapper &operator /=(double other);

    StlWrapper &operator -=(StlWrapper const &other);
    StlWrapper &operator +=(StlWrapper const &other);

    StlWrapper operator +(StlWrapper const &other) const;
    StlWrapper operator *(StlWrapper const &other) const;

    StlVector &operator *();
    StlVector const &operator *() const;

    operator double*() const;

    double &operator ()(int m, int n = 0);
    double const &operator ()(int m, int n = 0) const;

    void resize(int m);
    void resize(int m, int n);

    double norm() const;
    double norm_inf() const;

    void orthogonalize();

    StlWrapper view(int m = -1, int n = -1);
    StlWrapper view(int m = -1, int n = -1) const;
    StlWrapper copy() const;

    void push_back(StlWrapper const &other);

    int M() const;
    int N() const;
    int LDA() const;

    StlWrapper dot(StlWrapper const &other) const;

    void random();

    StlWrapper transpose() const;

    int eigs(StlWrapper &V, StlWrapper &D,
             int num = -1, double tol = 1e-16) const;

};

StlWrapper operator *(double d, StlWrapper const &other);

}

#endif
