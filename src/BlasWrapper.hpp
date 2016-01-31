#ifndef BLASWRAPPER_H
#define BLASWRAPPER_H

extern "C"
{
    void dscal_(int *n, double *a, double *x, int *incx);
    void daxpy_(int *n, double *a, double const *x, int *incx,
                double *y, int *incy);
    void dgemm_(char *transa, char* transb, int *m, int *n,
                int *k, double *alpha, double const *a, int *lda,
                double const *b, int *ldb, double *beta,
                double *c, int *ldc);
}

namespace BlasWrapper
{

void DSCAL(int n, double a, double *x)
{
    int inc = 1;
    dscal_(&n, &a, x, &inc);
}

void DAXPY(int n, double a, double const *x, double *y)
{
    int inc = 1;
    daxpy_(&n, &a, x, &inc, y, &inc);
}

void DGEMM(char transa, char transb, int m, int n, int k,
           double alpha, double const *a, int lda,
           double const *b, int ldb, double beta,
           double *c, int ldc)
{
    dgemm_(&transa, &transb, &m, &n, &k, &alpha,
           a, &lda, b, &ldb, &beta, c, &ldc);
}

}

#endif
