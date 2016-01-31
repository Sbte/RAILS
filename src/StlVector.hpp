#ifndef STLVECTOR_H
#define STLVECTOR_H

class StlVector
{
    double *ptr_;

    bool is_view_;

    int m_;
    int n_;

public:
    StlVector(int m, int n);
    StlVector(double *other, int m, int n);
    StlVector(StlVector const &other);

    virtual ~StlVector();

    double &operator [](int i);
    double &operator ()(int i, int j);

    int set(StlVector const &other);
    double *get();
};
#endif
