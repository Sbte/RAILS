#include "LapackWrapper.hpp"

#include <iostream>

namespace RAILS
{

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

    if (*info)
        std::cerr << "DSYEV workspace query info = " << info << std::endl;

    work = new double[ldwork];
    dsyev_(&jobz, &uplo, &n, a, &lda,
           w, work, &ldwork, info);
    delete[] work;

    if (*info)
        std::cerr << "DSYEV info = " << info << std::endl;
}

}

}
