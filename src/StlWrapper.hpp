#ifndef STLWRAPPER_H
#define STLWRAPPER_H

#include "StlVector.hpp"

#include <memory>

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
public:
    StlWrapper();
    StlWrapper(std::shared_ptr<StlVector> ptr);
    StlWrapper(StlWrapper const &other);
    StlWrapper(StlWrapper const &other, int n);
    StlWrapper(int m, int n);

    virtual ~StlWrapper() {}

    StlWrapper &operator =(StlWrapper &other);
    StlWrapper &operator =(StlWrapper const &other);

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

    int scale(double factor);

    void resize(int m);
    void resize(int m, int n);

    double norm() const;
    double norm_inf() const;

    void orthogonalize();

    StlWrapper view(int m, int n = 0);
    StlWrapper view(int m, int n = 0) const;
    StlWrapper copy(int m = 0, int n = 0) const;

    void push_back(StlWrapper const &other, int m = -1);

    int M() const;
    int N() const;

    StlWrapper dot(StlWrapper const &other) const;

    void random();
};

StlWrapper operator *(double d, StlWrapper const &other);

#endif
