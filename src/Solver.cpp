
namespace Lyapunov {

template<class Matrix, class MultiVector>
Solver::Solver(Matrix A,
               Matrix B,
               Matrix M,
               Teuchos::ParameterList params)
    :
    A_(A),
    B_(B),
    M_(M)
{
    SetParameterList(params);
}

Solver::solve(MultiVector V, DenseMatrix T)
{
    MultiVector AV(V, max_iter_);
    AV.resize(0);

    DenseMatrix VAV(max_iter_, max_iter_);
    double ro = B.norm();
    for (int iter = 0; iter < max_iter_; iter++)
    {
        // Perform VAV = V'*A*V by doing VAV = [[VAV; W'*AV], V'*AW]

        // First compute AW
        MultiVector AW = A.apply(W);

        // Now resize VAV. Resizing should not remove what was in VAV
        // previously
        int num_vectors_V = V.num_vectors();
        int s = num_vectors_V + W.num_vectors();
        VAV.resize(s, s);

        // Now compute W'*AV and put it in VAV
        DenseMatrix WAV = W.dot(AV);
        int WAV_m = WAV.m()
        int WAV_n = WAV.n();
        for (int i = 0; i < WAV_m; i++)
            for (int j = 0; j < WAV_n; j++)
                VAV(i + num_vectors_V, j) = WAV(i, j);

        // Now compute V'*AW and put it in VAV
        DenseMatrix VAW = V.dot(AW);
        int VAW_m = VAW.m()
        int VAW_n = VAW.n();
        for (int i = 0; i < VAW_m; i++)
            for (int j = 0; j < VAW_n; j++)
                VAV(i, j + num_vectors_V) = VAW(i, j);

        // Compute VBV = V'*B*B'*V. TODO: This can also be done more
        // efficiently I think
        MultiVector BV = B.apply(V);
        DenseMatrix VBV = BV.dot(BV);

        dense_solve(VAV, VBV, T);

        lanczos(AV, V, T, H, eigV, 10);

        double res = H.norm();

        std::cout << "Estimate Lanczos, absolute: " << res << ", relative: "
                  << abs(res) / r0 / r0 << std::endl;

        if (abs(res) / r0 / r0 < tol || iter >= max_iter_ || V.num_vectors() >= B.m())
            break;

        V.push_back(eigV, 3);
        V.orthogonalize();

        W = V.view(num_vectors_V, num_vectors_V+3);
    }
}

Solver::dense_solve(DenseMatrix A, DenseMatrix B, DenseMatrix X)
{
    X = B.copy();
    double scale = 1.0;
    int info = 0;
    int n = A.n();
    sb03md('C', 'X', 'N', 'T', n, A, X, &SCALE, &info);
}

Solver::lanczos(MultiVector AV, MultiVector V, DenseMatrix T, DenseMatrix H, MultiVector eigV, int max_iter)
{
    MultiVector Q(V, max_iter+1);
    Q.resize(1);
    Q.random();

    int iter = 0
    for (; iter < max_iter; iter++)
    {
        Q.view(iter+1) = B.apply(Q.view(iter));
        Q.view(iter+1) = B.Apply(Q.view(iter+1));

        DenseMatrix Z = V.dot(Q.view(iter));
        Z = T.apply(Z);
        Q.view(iter+1) += AV.dot(Z);

        Z = AV.dot(Q.view(iter));
        Z = T.apply(Z);
        Q.view(iter+1) += V.dot(Z);

        double alpha = Q.view(iter+1).dot(Q.view(iter));
        H(iter, iter) = alplha;

        Q.view(iter+1) -= alpha * Q.view(iter);
        if (iter > 0)
            Q.view(iter+1) -= beta * Q.view(iter-1);

        double beta = Q.view(iter+1).norm();
        if (beta < 1e-14)
            break;

        H(iter+1, iter) = beta;
        H(iter, iter+1) = beta;

        Q.view(iter+1) /= beta;
    }

    H.resize(iter, iter);
    Q.resize(iter);

    DenseMatrix v;
    DenseMatrix d;
    H.eigs(v, d);

    eigV = Q.apply(v);
}
