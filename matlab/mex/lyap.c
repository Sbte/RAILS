#include "mex.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

void sb03md_(char *dico, char *job, char *fact, char *trans,
             int *n, double *A, int *lad, double *U, int *ldu,
             double *X, int *ldx, double *scale, double *sep,
             double *ferr, double *alphar, double *alphai, int *iwork,
             double *dwork, int *ldwork, int *info);

void sg03ad_(char *dico, char *job, char *fact, char *trans, char *uplo,
             int *n, double *A, int *lad, double *E, int *lde, double *Q, int *ldq,
             double *Z, int *ldz, double *X, int *ldx, double *scale, double *sep,
             double *ferr, double *alphar, double *alphai, double *beta, int *iwork,
             double *dwork, int *ldwork, int *info);

void sb03md(char dico, char job, char fact, char trans, int n,
            double *A, double *X, double *scale, int *info)
{
/* Maybe the sizes as stated in
   http://www.icm.tu-bs.de/NICONET/doc/SB03MD.html
   are not right or something since valgrind gives error here.
   Might also be a different bug, but for now just allocate more. */
    int n_alloc = n + 2;

    double *U = (double *)malloc(sizeof(double)*n_alloc*n_alloc);

    double sep;
    double ferr;

    double *ar = (double *)malloc(sizeof(double)*n_alloc);
    double *ai = (double *)malloc(sizeof(double)*n_alloc);

    int ldwork = MAX(n_alloc*n_alloc, 3*n_alloc); /* Minimum requirement */
    ldwork *= 2; /* Allocate some extra for better performance */
    double *work = (double *)malloc(sizeof(double)*ldwork);

    sb03md_(&dico, &job, &fact, &trans,
            &n, A, &n, U, &n, X, &n,
            scale, &sep, &ferr, ar, ai, NULL,
            work, &ldwork, info);

    free(U);

    free(ar);
    free(ai);

    free(work);
}

void sg03ad(char dico, char job, char fact, char trans, char uplo, int n,
            double *A, double *E, double *X, double *scale, int *info)
{
    double *Q = (double *)malloc(sizeof(double)*n*n);
    double *Z = (double *)malloc(sizeof(double)*n*n);

    double sep;
    double ferr;

    double *ar = (double *)malloc(sizeof(double)*n);
    double *ai = (double *)malloc(sizeof(double)*n);
    double *b = (double *)malloc(sizeof(double)*n);

    int ldwork = 8*n+16;
    double *work = (double *)malloc(sizeof(double)*ldwork);

    sg03ad_(&dico, &job, &fact, &trans, &uplo,
            &n, A, &n, E, &n, Q,&n, Z, &n, X, &n,
            scale, &sep, &ferr, ar, ai, b, NULL,
            work, &ldwork, info);

    free(Q);
    free(Z);

    free(ar);
    free(ai);
    free(b);

    free(work);
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    if (nrhs > 4)
        mexErrMsgIdAndTxt("SLICOT:lyap:nrhs", "At most 4 input arguments required.");
    if (nrhs == 3)
        mexErrMsgIdAndTxt("SLICOT:lyap:nrhs", "Solving Sylvester equations is not supported.");
    if (nlhs != 1)
        mexErrMsgIdAndTxt("SLICOT:lyap:nlhs", "One output argument required.");

    if (mxIsSparse(prhs[0]) || mxIsSparse(prhs[1]))
        mexErrMsgIdAndTxt("SLICOT:lyap:prhs", "Can't handle sparse matrices.");

    double *A = mxGetPr(prhs[0]);
    double *B = mxGetPr(prhs[1]);
    double *M;
    if (nrhs == 4)
      M = mxGetPr(prhs[3]);

    size_t n = mxGetN(prhs[0]);
    size_t bn = mxGetN(prhs[1]);

    if (bn != n)
        mexErrMsgIdAndTxt("SLICOT:lyap:n", "Incompatible input matrices");

    plhs[0] = mxCreateDoubleMatrix((mwSize)n, (mwSize)n, mxREAL);
    double *X = mxGetPr(plhs[0]);

    double SCALE = 1.0;
    int info = 0;

    /* TODO: actually reusing the Schur factorization is way faster */
    memcpy(X, B, sizeof(double)*n*n);
    for (int i = 0; i < n * n; i++)
      X[i] = -X[i];

    double *Atmp = (double *)malloc(sizeof(double) * n * n);
    memcpy(Atmp, A, sizeof(double) * n * n);

    if (nrhs == 4)
      {
      double *Mtmp = (double *)malloc(sizeof(double) * n * n);
      memcpy(Mtmp, M, sizeof(double)*n*n);

      sg03ad('C', 'X', 'N', 'T', 'L', n, Atmp, Mtmp, X, &SCALE, &info);

      free(Mtmp);
      }
    else
      {
      sb03md('C', 'X', 'N', 'T', n, Atmp, X, &SCALE, &info);
      }

    free(Atmp);

    if (info == n+1)
        mexWarnMsgIdAndTxt("SLICOT:lyap:info", "Info = %d, should be 0", info);
    else if (info != 0)
        mexErrMsgIdAndTxt("SLICOT:lyap:info", "Info = %d, should be 0", info);

    if (abs(SCALE-1.0) > 1.0e-12)
        mexErrMsgIdAndTxt("SLICOT:lyap:scale", "scale = %f, should be 1.0", SCALE);
}
