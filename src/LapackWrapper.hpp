#ifndef LAPACKWRAPPER_H
#define LAPACKWRAPPER_H

extern "C"
{
    void dsteqr_(char *compz, int *n, double *d, double *e,
                 double *z, int *ldz, double *work, int *info);
}

namespace LapackWrapper
{

void DSTEQR(char compz, int n, double *d, double *e,
            double *z, int ldz, double *work, int *info)
{
    dsteqr_(&compz, &n, d, e,
            z, &ldz, work, info);
}

}

#endif
