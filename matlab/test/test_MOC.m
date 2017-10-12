function tests = test_MOC
    tests = functiontests(localfunctions);
end

function test_MOC_Erik(t)
    addpath([pwd, '/../DataErik']);
    [A,M,B] = thcm2matlab(1);

    % Put it back into the right form, Erik is awesome
    A = group2block(A);
    M = group2block(M);
    B = group2block(B);

    n = size(A, 1);
    t.assertEqual(n, 8*8*4*6);

    % Need positive M for the orthogonalisation
    A = -A;
    M = -M;

    res = 0;

    warning('off', 'MATLAB:nearlySingularMatrix');
    [V,D] = eigs(A, 2, 'sm');
    A2 = sparse(n+2, n+2);
    A2(1:n, 1:n) = A;
    A2(n+1:n+2, 1:n) = V';
    A2(1:n, n+1:n+2) = V;
    warning('on', 'MATLAB:nearlySingularMatrix');

    idx1 = [];
    idx2 = [];
    for i = 1:n+2
        m = mod(i - 1, 6);
        if m == 0 || m == 1 || m == 2 || m == 3
            idx1 = [idx1 i];
            if (i <= n && M(i,i)) M(i, i) = 0; end
        else
            idx2 = [idx2 i];
        end
    end

    A11 = A2(idx1, idx1);
    A12 = A2(idx1, idx2);
    A21 = A2(idx2, idx1);
    A22 = A2(idx2, idx2);

    fac = A11 \ A12;
    SC = A22 - A21 * fac;
    BS = B(idx2, idx2);
    [SX, S]  = RAILSsolver(SC, [], BS, 1000, 1e-5);

    fac = A11 \ [A12(1:end-2,:); zeros(2,size(A12,2))];
    X22 = SX * S * SX';
    X12 = -fac * X22;
    X11 = -fac * X12';

    res = norm(SC*X22 + X22*SC' + BS*BS', 'fro');
    t.assertLessThan(SC*X22 + X22*SC' + BS*BS', 1E-4);

    idx1 = idx1(1:end-2);

    X = sparse(n, n);
    X(idx1, idx1) = X11(1:end-2, 1:end-2);
    X(idx1, idx2) = X12(1:end-2, :);
    X(idx2, idx1) = X12(1:end-2, :)';
    X(idx2, idx2) = X22;

    res = norm(A*X*M' + M*X*A' + B*B', 'fro');
    t.assertLessThan(res, 1E-4);

    rmpath([pwd, '/../DataErik']);
end