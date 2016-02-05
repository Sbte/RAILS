#include "SlicotWrapper.hpp"
#include <cstring>
#include <algorithm>

namespace Lyapunov {

void sb03md(char dico, char job, char fact, char trans, int n,
            double *A, int lda, double *X, int ldx,
            double *scale, int *info)
{
    double *U = new double[n*n];

    double sep;
    double ferr;

    double *ar = new double[n];
    double *ai = new double[n];

    int ldwork = std::max(n*n, 3*n); // Minimum requirement
    ldwork *= 2; // Allocate some extra for better performance
    double *work = new double[ldwork];

    // Solve A'*X + X*A = scale*C
    // See http://www.icm.tu-bs.de/NICONET/doc/SB03MD.html
    sb03md_(&dico, &job, &fact, &trans,
            &n, A, &lda, U, &n, X, &ldx,
            scale, &sep, &ferr, ar, ai, NULL,
            work, &ldwork, info);

    delete[] U;

    delete[] ar;
    delete[] ai;

    delete[] work;
}

}
