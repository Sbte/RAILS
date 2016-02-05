#ifndef LAPACKWRAPPER_H
#define LAPACKWRAPPER_H

extern "C"
{
    void dsteqr_(char *compz, int *n, double *d, double *e,
                 double *z, int *ldz, double *work, int *info);
    void dsyev_(char *jobz, char *uplo, int *n, double *a, int *lda,
                double *w, double *work, int *ldwork, int *info);
}

namespace LapackWrapper
{

void DSTEQR(char compz, int n, double *d, double *e,
            double *z, int ldz, double *work, int *info)
{
    dsteqr_(&compz, &n, d, e,
            z, &ldz, work, info);
}

void DSYEV(char jobz, char uplo, int n, double *a, int lda,
           double *w, int *info)
{
    //Compute optimal space size
    int ldwork = -1;
    double *work = new double[1];
    dsyev_(&jobz, &uplo, &n, a, &lda,
           w, work, &ldwork, info);
    ldwork = work[0];
    delete[] work;

    work = new double[ldwork];
    dsyev_(&jobz, &uplo, &n, a, &lda,
           w, work, &ldwork, info);
    delete[] work;
}

}

#endif
