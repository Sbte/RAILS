#ifndef LYAPUNOVSOLVERDECL_H
#define LYAPUNOVSOLVERDECL_H

#include "MatrixOrMultiVectorWrapper.hpp"

namespace RAILS
{

template<class Matrix, class MultiVector, class DenseMatrix>
class Solver
{
public:
    template<class MatrixOrMultiVector>
    Solver(Matrix const &A,
           MatrixOrMultiVector const &B,
           Matrix const &M);

    virtual ~Solver() {};
    
    template<class ParameterList>
    int set_parameters(ParameterList &params);

    // Solve A*V*T*V' + V*T*V'*A' + B*B' = 0
    int solve(MultiVector &V, DenseMatrix &T);

    // Solve A*X + X*A' + B = 0
    int dense_solve(DenseMatrix const &A, DenseMatrix const &B, DenseMatrix &X);

    // Get the eigenvectors of the residual R = AV*T*V' + V*T*AV' + B*B'
    int resid_lanczos(MultiVector const &AV, MultiVector const &V, DenseMatrix const &T,
                      DenseMatrix &H, MultiVector &eigenvectors, DenseMatrix &eigenvalues,
                      int max_iter);

    // Get the restart vectors based on the eigenvalues of T
    int compute_restart_vectors(DenseMatrix &X, DenseMatrix const &T, int num, double tol);
protected:
    Matrix A_;
    MatrixOrMultiVectorWrapper<Matrix, MultiVector> B_;
    Matrix M_;

    int max_iter_;
    double tol_;
    int expand_size_;
    int lanczos_iterations_;
    int restart_size_;
    int reduced_size_;
    int restart_iterations_;
    double restart_tolerance_;
    bool minimize_solution_space_;
    bool restart_from_solution_;
};

}

#endif
