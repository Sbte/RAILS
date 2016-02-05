#ifndef SLICOTWRAPPER_H
#define SLICOTWRAPPER_H

extern "C" {
    void sb03md_(char *dico, char *job, char *fact, char *trans,
                 int *n, double *A, int *lad, double *U, int *ldu,
                 double *X, int *ldx, double *scale, double *sep,
                 double *ferr, double *alphar, double *alphai, int *iwork,
                 double *dwork, int *ldwork, int *info);
}

namespace Lyapunov {
void sb03md(char dico, char job, char fact, char trans, int n,
            double *A, int lda, double *X, int ldx,
            double *scale, int *info);

}

#endif
