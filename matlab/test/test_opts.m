function test_suite = test_opts
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

function test_tol(t)
    seed;
    n = 256;
    A = laplacian2(n);
    M = spdiags(rand(n,1), 0, n, n);
    B = rand(n,1);

    tol = 5e-5;
    [V, S, res, iter] = RAILSsolver(A,M,B,tol);

    t.assertLessThan(iter, n-10);
    t.assertLessThan(res, tol);
    t.assertLessThan(norm(A*V*S*V'*M'+M*V*S*V'*A'+B*B') / norm(B'*B), tol);
    t.assertGreaterThan(norm(A*V*S*V'*M'+M*V*S*V'*A'+B*B') / norm(B'*B), tol / 10);
end

function test_restart(t)
    seed;
    n = 256;
    A = laplacian2(n);
    M = spdiags(rand(n,1), 0, n, n);
    B = rand(n,1);

    clear opts;
    opts.restart_size = 50;
    opts.reduced_size = 10;
    [V, S, res, iter] = RAILSsolver(A,M,B,opts);

    t.assertEqual(size(V,2), 10);
    t.assertLessThan(iter, 100);
    t.assertEqual(size(V, 2), size(S, 2));
    t.assertLessThan(res * norm(B'*B), 1E-2);
    t.assertLessThan(res, 1E-4);
    t.assertLessThan(norm(A*V*S*V'*M'+M*V*S*V'*A'+B*B') / norm(B'*B), 1E-4);
end

function test_restart2(t)
    seed;
    n = 256;
    A = laplacian2(n);
    M = spdiags(rand(n,1), 0, n, n);
    B = rand(n,1);

    maxit = 110;
    clear opts;
    opts.reduced_size = 15;
    opts.restart_iterations = 40;
    [V, S, res, iter] = RAILSsolver(A,M,B,maxit,opts);

    t.assertLessThan(iter, maxit);
    t.assertEqual(size(V, 2), size(S, 2));
    t.assertLessThan(res * norm(B'*B), 1E-2);
    t.assertLessThan(res, 1E-4);
    t.assertLessThan(norm(A*V*S*V'*M'+M*V*S*V'*A'+B*B') / norm(B'*B), 1E-4);
end

function test_restart3(t)
    seed;
    n = 256;
    A = laplacian2(n);
    M = spdiags(rand(n,1), 0, n, n);
    B = rand(n,1);

    maxit = 150;
    clear opts;
    opts.restart_size = 50;
    opts.reduced_size = 10;
    opts.restart_iterations = 20;
    opts.restart_tolerance = 1e-2;
    [V, S, res, iter] = RAILSsolver(A,M,B,maxit,opts);

    t.assertLessThan(iter, maxit);
    t.assertEqual(size(V, 2), size(S, 2));
    t.assertLessThan(res * norm(B'*B), 1E-2);
    t.assertLessThan(res, 1E-4);
    t.assertLessThan(norm(A*V*S*V'*M'+M*V*S*V'*A'+B*B') / norm(B'*B), 1E-4);
end

function test_wrong_restart(t)
    seed;
    n = 256;
    A = laplacian2(n);
    M = spdiags(rand(n,1), 0, n, n);
    B = rand(n,1);

    clear opts;
    opts.restart_size = 10;
    opts.reduced_size = 50;
    t.assertError(@() RAILSsolver(A,M,B,opts), 'RAILSsolver:InvalidOption');
end

function test_wrong_expand(t)
    seed;
    n = 256;
    A = laplacian2(n);
    M = spdiags(rand(n,1), 0, n, n);
    B = rand(n,1);

    clear opts;
    opts.expand = 3;
    t.assertError(@() RAILSsolver(A,M,B,opts), 'RAILSsolver:InvalidOption');
end

function test_wrong_space(t)
    seed;
    n = 256;
    A = laplacian2(n);
    M = spdiags(rand(n,1), 0, n, n);
    B = rand(n,1);

    clear opts;
    opts.space = rand(n-1,1);
    t.assertError(@() RAILSsolver(A,M,B,opts), 'RAILSsolver:InvalidOption');
end

function test_space(t)
    seed;
    n = 256;
    A = laplacian2(n);
    M = spdiags(rand(n,1), 0, n, n);
    B = rand(n,1);

    maxit = 150;
    clear opts;
    opts.restart_size = 50;
    opts.reduced_size = 10;
    [V, S, res, iter] = RAILSsolver(A,M,B,maxit,opts);
    
    opts.space = V(:,1:9);
    [V, S, res, iter2] = RAILSsolver(A,M,B,maxit,opts);

    t.assertLessThan(iter2, iter);
    t.assertEqual(size(V, 2), size(S, 2));
    t.assertLessThan(res * norm(B'*B), 1E-2);
    t.assertLessThan(res, 1E-4);
    t.assertLessThan(norm(A*V*S*V'*M'+M*V*S*V'*A'+B*B') / norm(B'*B), 1E-4);
end

function test_morth(t)
    seed;
    n = 256;
    A = laplacian2(n);
    M = spdiags(rand(n,1), 0, n, n);
    B = rand(n,1);

    opts.ortho = 'M';
    [V, S, res, iter] = RAILSsolver(A,M,B,opts);

    t.assertLessThan(iter, n-10);
    t.assertLessThan(res * norm(B'*B), 1E-2);
    t.assertLessThan(res, 1E-4);
    t.assertLessThan(norm(A*V*S*V'*M'+M*V*S*V'*A'+B*B') / norm(B'*B), 1E-4);
end

function test_nullspace(t)
    seed;
    n = 256;
    A = laplacian2(n);
    M = spdiags(rand(n,1), 0, n, n);
    B = rand(n,1);

    Q = rand(n,1);
    Q = Q / norm(Q);
    P = speye(n) - Q * Q';
    A = P*A*P;
    B = P*B;
    M = P*M*P;
    opts.nullspace = Q;
    warning('off', 'RAILSsolver:SingularMassMatrix');
    [V, S, res, iter] = RAILSsolver(A,M,B,opts);
    warning('on', 'RAILSsolver:SingularMassMatrix');

    t.assertLessThan(norm(Q'*V), 1E-10);
    t.assertLessThan(res * norm(B'*B), 1E-2);
    t.assertLessThan(res, 1E-4);
    t.assertLessThan(norm(A*V*S*V'*M'+M*V*S*V'*A'+B*B') / norm(B'*B), 1E-4);
end