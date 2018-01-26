function test_suite = test_Laplace
    try
        test_functions = localfunctions();
        test_suite = functiontests(test_functions);
    catch
    end

    try
        initTestSuite;
    catch
    end
end

function A = laplacian2(n)
    m = sqrt(n);
    I = speye(m);
    e = ones(m,1);
    T = spdiags([e -4*e e], -1:1, m, m);
    S = spdiags([e e], [-1 1], m, m);
    A = (kron(I,T) + kron(S,I));
end

function seed()
    if ~exist('rng')
        rand('state', 4634);
    else
        rng(4634)
    end
end

function test_Laplace_64(t)
    seed;
    n = 64;
    A = laplacian2(n);
    M = spdiags(rand(n,1), 0, n, n);
    B = rand(n,1);

    [V, S, res, iter] = RAILSsolver(A,M,B);

    t.assertLessThan(iter, n-10);
    t.assertLessThan(res * norm(B'*B), 1E-2);
    t.assertLessThan(res, 1E-4);
    t.assertLessThan(norm(A*V*S*V'*M'+M*V*S*V'*A'+B*B') / norm(B'*B), 1E-4);
end

function test_Laplace_256(t)
    seed;
    n = 256;
    A = laplacian2(n);
    M = spdiags(rand(n,1), 0, n, n);
    B = rand(n,1);

    [V, S, res, iter] = RAILSsolver(A,M,B);

    t.assertLessThan(iter, n-10);
    t.assertLessThan(res * norm(B'*B), 1E-2);
    t.assertLessThan(res, 1E-4);
    t.assertLessThan(norm(A*V*S*V'*M'+M*V*S*V'*A'+B*B') / norm(B'*B), 1E-4);
end

function test_Laplace_maxit(t)
    seed;
    n = 64;
    A = laplacian2(n);
    M = spdiags(rand(n,1), 0, n, n);
    B = rand(n,1);

    t.assertWarning(@(x) RAILSsolver(A, M, B, 10), 'RAILSsolver:ProjectionMethod');
end

function test_Laplace_singular(t)
    seed;
    n = 64;
    A = laplacian2(n);
    M = spdiags(rand(n,1), 0, n, n);
    M(n, n) = 0;
    B = rand(n,1);

    t.assertWarning(@(x) RAILSsolver(A, M, B, 10), 'RAILSsolver:SingularMassMatrix');
end

function test_Laplace_equivalence(t)
% Here we show that the Laplace problem is also a Lyapunov problem
    seed;

    n = 1024;
    m = sqrt(n);

    A_lapl = laplacian2(n);
 
    e = ones(n, 1);
    A = spdiags([e, -2*e, e], -1:1, m, m);
    I = speye(m);
    A_kron = kron(A, I) + kron(I, A);
    t.assertEqual(A_lapl, A_kron);

    B = rand(m, 1);
    b = -reshape(B*B', n, 1);

    x_lapl = A_lapl \ b;

    opts.restart_upon_convergence = false;
    [V, S, res, iter] = RAILSsolver(A, [], B, opts);

    t.assertLessThan(res * norm(B'*B), 1E-2);
    t.assertLessThan(res, 1E-4);
    t.assertLessThan(norm(A*V*S*V'+V*S*V'*A'+B*B') / norm(B'*B), 1E-4);

    x_lyap = reshape(V*S*V', n, 1);
    t.assertLessThan(norm(x_lapl-x_lyap), 1E-4);
end