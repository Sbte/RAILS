function tests = test_random
    tests = functiontests(localfunctions);
end

function test_random_ev(t)
    rng(4634);
    n = 64;
    A = sprand(n,n,10/n);
    M = speye(n);
    [B,~] = eigs(A,1);

    [V, S, res, iter] = RAILSsolver(A,M,B,64);

    t.assertLessThan(iter, 10);
    t.assertLessThan(res * norm(B'*B), 1E-2);
    t.assertLessThan(res, 1E-4);
    t.assertLessThan(norm(A*V*S*V'*M+M*V*S*V'*A'+B*B') / norm(B'*B), 1E-4);
end

function test_random_64(t)
    rng(4634);
    n = 64;
    A = sprand(n,n,10/n);
    B = rand(n,1);
    M = spdiags(rand(n,1), 0, n, n);

    [V, S, res] = RAILSsolver(A,M,B,65);

    t.assertLessThan(res * norm(B'*B), 1E-2);
    t.assertLessThan(res, 1E-4);
    t.assertLessThan(norm(A*V*S*V'*M+M*V*S*V'*A'+B*B') / norm(B'*B), 1E-4);
end