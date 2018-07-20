#include "SlicotWrapper.hpp"
#include <cstring>
#include <iostream>

namespace RAILS
{

void sb03md(char dico, char job, char fact, char trans, int n,
            double *A, int lda, double *X, int ldx,
            double *scale, int *info)
{
    if (n < 1)
    {
        std::cerr << "Slicot error: n < 1 is not supported" << std::endl;
        return;
    }

    // Maybe the sizes as stated in
    // http://www.icm.tu-bs.de/NICONET/doc/SB03MD.html
    // are not right or something since valgrind gives error here.
    // Might also be a different bug, but for now just allocate more.
    int n_alloc = n + 2;

    double *U = new double[n_alloc*n_alloc];

    double sep;
    double ferr;

    double *ar = new double[n_alloc];
    double *ai = new double[n_alloc];

    int ldwork = std::max(n_alloc*n_alloc, 3*n_alloc); // Minimum requirement
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
