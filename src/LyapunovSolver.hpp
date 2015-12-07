#ifndef LYAPUNOVSOLVER_H
#define LYAPUNOVSOLVER_H

#include <utility>
#include <vector>
#include <algorithm>

#include "LyapunovSolverDecl.hpp"
#include "SlicotWrapper.hpp"

namespace Lyapunov {

template<class Matrix, class MultiVector, class DenseMatrix>
Solver<Matrix, MultiVector, DenseMatrix>::Solver(Matrix const &A,
                                                 Matrix const &B,
                                                 Matrix const &M)
    :
    A_(A),
    B_(B),
    M_(M),
    max_iter_(200),
    tol_(1e-5)
{
}

template<class Matrix, class MultiVector, class DenseMatrix>
template<class ParameterList>
int Solver<Matrix, MultiVector, DenseMatrix>::set_parameters(ParameterList params)
{
    max_iter_ = params.get("Maximum iterations", max_iter_);
    tol_ = params.get("Tolerance", tol_);

    return 0;
}

template<class Matrix, class MultiVector, class DenseMatrix>
int Solver<Matrix, MultiVector, DenseMatrix>::solve(MultiVector &V, DenseMatrix &T)
{
    int n = V.GlobalLength();
    MultiVector AV(V, max_iter_);
    AV.resize(0);

    V.resize(1);
    V.random();
    V.orthogonalize();

    MultiVector W(V);

    DenseMatrix VAV(max_iter_, max_iter_);
    double r0 = B_.norm_frobenius();
    for (int iter = 0; iter < max_iter_; iter++)
    {
        // Perform VAV = V'*A*V by doing VAV = [[VAV; W'*AV], V'*AW]

        // First compute AW
        MultiVector AW = A_.apply(W);

        // Now resize VAV. Resizing should not remove what was in VAV
        // previously
        int num_vectors_V = V.num_vectors();
        int num_vectors_AV = AV.num_vectors();
        int s = num_vectors_AV + W.num_vectors();
        VAV.resize(s, s);

        // Now compute W'*AV and put it in VAV
        // AV only exists after the first iteration
        if (iter > 0)
        {
            DenseMatrix WAV = W.dot(AV);
            int WAV_m = WAV.M();
            int WAV_n = WAV.N();
            for (int i = 0; i < WAV_m; i++)
                for (int j = 0; j < WAV_n; j++)
                    VAV(i + num_vectors_AV, j) = WAV(i, j);
        }

        // Now compute V'*AW and put it in VAV
        DenseMatrix VAW = V.dot(AW);
        int VAW_m = VAW.M();
        int VAW_n = VAW.N();
        for (int i = 0; i < VAW_m; i++)
            for (int j = 0; j < VAW_n; j++)
                VAV(i, j + num_vectors_AV) = VAW(i, j);

        // Now expand AV
        AV.push_back(AW);

        // Compute VBV = V'*B*B'*V. TODO: This can also be done more
        // efficiently I think
        MultiVector BV = B_.apply(V);
        DenseMatrix VBV = BV.dot(BV);

        dense_solve(VAV, VBV, T);

        int num_eigenvalues = 10;
        DenseMatrix H(num_eigenvalues + 1, num_eigenvalues + 1);
        DenseMatrix eigenvalues(num_eigenvalues + 1, 1);
        MultiVector eigenvectors;
        lanczos(AV, V, T, H, eigenvectors, eigenvalues, num_eigenvalues);

        double res = eigenvalues.norm_inf();

        std::cout << "Iteration " << iter
                  << ". Estimate Lanczos, absolute: " << res
                  << ", relative: " << abs(res) / r0 / r0 << std::endl;

        if (abs(res) / r0 / r0 < tol_ || iter >= max_iter_ || V.num_vectors() >= n)
            break;

        int expand_vectors = std::min(std::min(3, eigenvalues.M()), n - V.num_vectors());

        // Find the vectors belonging to the largest eigenvalues
        std::vector<int> indices;
        find_largest_eigenvalues(eigenvalues, indices, expand_vectors);

        for (int i = 0; i < expand_vectors; i++)
            V.push_back(eigenvectors.view(indices[i]));
        V.orthogonalize();

        W = V.view(num_vectors_V, num_vectors_V+expand_vectors-1);
    }
    return 0;
}

template<class Matrix, class MultiVector, class DenseMatrix>
int Solver<Matrix, MultiVector, DenseMatrix>::dense_solve(DenseMatrix const &A, DenseMatrix const &B, DenseMatrix &X)
{
    X = B.copy();
    DenseMatrix A_copy = A.copy();
    double scale = 1.0;
    int info = 0;
    int n = A.M();
    sb03md('C', 'X', 'N', 'T', n, A_copy, X, &scale, &info);

    return info;
}

template<class Matrix, class MultiVector, class DenseMatrix>
int Solver<Matrix, MultiVector, DenseMatrix>::lanczos(MultiVector const &AV, MultiVector const &V, DenseMatrix const &T, DenseMatrix &H, MultiVector &eigenvectors, DenseMatrix &eigenvalues, int max_iter)
{
    MultiVector Q(V, max_iter+1);
    Q.resize(1);
    Q.random();
    Q.view(0) /= Q.norm();

    double alpha = 0.0;
    double beta = 0.0;

    int iter = 0;
    for (; iter < max_iter; iter++)
    {
        Q.resize(iter + 2);
        Q.view(iter+1) = B_.apply(Q.view(iter));
        Q.view(iter+1) = B_.apply(Q.view(iter+1));

        DenseMatrix Z = V.dot(Q.view(iter));
        Z = T.apply(Z);
        Q.view(iter+1) += AV.apply(Z);

        Z = AV.dot(Q.view(iter));
        Z = T.apply(Z);
        Q.view(iter+1) += V.apply(Z);
        
        alpha = Q.view(iter+1).dot(Q.view(iter))(0, 0);
        H(iter, iter) = alpha;

        Q.view(iter+1) -= alpha * Q.view(iter);
        if (iter > 0)
            Q.view(iter+1) -= beta * Q.view(iter-1);

        beta = Q.view(iter+1).norm();
        if (beta < 1e-14)
            break;

        H(iter+1, iter) = beta;
        H(iter, iter+1) = beta;

        Q.view(iter+1) /= beta;
    }

    H.resize(iter+1, iter+1);
    Q.resize(iter+1);

    DenseMatrix v;
    H.eigs(v, eigenvalues);

    eigenvectors = Q.apply(v);

    return 0;
}

static bool eigenvalue_sorter(std::pair<int, double> const &a, std::pair<int, double> const &b)
{
    return std::abs(a.second) > std::abs(b.second);
}

template<class Matrix, class MultiVector, class DenseMatrix>
int Solver<Matrix, MultiVector, DenseMatrix>::find_largest_eigenvalues(DenseMatrix const &eigenvalues, std::vector<int> &indices, int num_vectors)
{
    std::vector<std::pair<int, double> > index_to_value;
    for (int i = 0; i < eigenvalues.M(); i++)
        index_to_value.push_back(std::pair<int, double>(i, eigenvalues(i)));
    
    std::sort(index_to_value.begin(), index_to_value.end(), eigenvalue_sorter);

    for (int i = 0; i < num_vectors; i++)
        indices.push_back(index_to_value[i].first);

    return 0;
}

}

#endif
