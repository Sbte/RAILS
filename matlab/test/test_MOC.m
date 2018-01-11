function tests = test_MOC
    tests = functiontests(localfunctions);
end

function test_MOC_Erik(t)
    [A,M,B] = get_MOC_data();

    n = size(A, 1);
    t.assertEqual(n, 8*8*4*6);

    res = 0;

    % Add the nullspace to the matrix as border
    warning('off', 'MATLAB:nearlySingularMatrix');
    [V,D] = eigs(A, 2, 'sm');
    A2 = sparse(n+2, n+2);
    A2(1:n, 1:n) = A;
    A2(n+1:n+2, 1:n) = V';
    A2(1:n, n+1:n+2) = V;
    warning('on', 'MATLAB:nearlySingularMatrix');

    M2 = sparse(n+2, n+2);
    M2(1:n, 1:n) = M;

    B2 = sparse(n+2, size(B,2));
    B2(1:n, size(B,2)) = B;

    % Solve the Schur complement system
    [S, MS, BS, Sinv, Vtrans] = RAILSschur(A2, M2, B2);
    [V, T] = RAILSsolver(S, MS, BS, 1000, 1e-3);

    res = norm(S(V)*T*(V'*MS') + (MS*V)*(S(V)*T)' + BS*BS', 'fro');
    t.assertLessThan(res, 1E-3);

    V = Vtrans(V);
    V = V(1:n,:);

    res = norm(A*V*T*V'*M' + M*V*T*V'*A' + B*B', 'fro');
    t.assertLessThan(res, 1E-3);
end

function [A,M,B] = get_MOC_data()
    Abeg  = load([pwd, '/../DataErik/Ap1.beg']);
    Aco   = load([pwd, '/../DataErik/Ap1.co']);
    Ainfo = load([pwd, '/../DataErik/Ap1.info']);
    Ajco  = load([pwd, '/../DataErik/Ap1.jco']);
    Arl   = load([pwd, '/../DataErik/Ap1.rl']);
    Mco   = load([pwd, '/../DataErik/Bp1.co']);
    F     = load([pwd, '/../DataErik/Frcp1.co']);

    n = Ainfo(1);
    nnz = Ainfo(2);

    ivals = zeros(nnz,1);
    jvals = Ajco;
    vals  = Aco;

    row = 1;
    idx = 1;
    while row <= n
        for k = Abeg(row):Abeg(row+1)-1
            ivals(idx) = row;
            idx = idx + 1;
        end
        row = row + 1;
    end
    A = sparse(ivals, jvals, vals, n, n);
    M = sparse(1:n, 1:n, Mco, n, n);

    % Set everything but temperature and salinity to zero
    idx = 1:6:n;
    idx = [idx, idx+1, idx+2, idx+3];
    M(idx,idx) = 0;

    % Set everything but salinity to zero
    idx = 1:6:n;
    idx = [idx, idx+1, idx+2, idx+3, idx+4];
    F(idx) = 0;

    % Spatially correlated noise
    B = 0.1 * F;
end
