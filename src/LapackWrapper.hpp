#ifndef LAPACKWRAPPER_H
#define LAPACKWRAPPER_H

extern "C"
{
    void dsteqr_(char *compz, int *n, double *d, double *e,
                 double *z, int *ldz, double *work, int *info);
    void dsyev_(char *jobz, char *uplo, int *n, double *a, int *lda,
                double *w, double *work, int *ldwork, int *info);
}

namespace RAILS
{

namespace LapackWrapper
{

void DSTEQR(char compz, int n, double *d, double *e,
            double *z, int ldz, double *work, int *info);

void DSYEV(char jobz, char uplo, int n, double *a, int lda,
           double *w, int *info);

}

}

#endif
