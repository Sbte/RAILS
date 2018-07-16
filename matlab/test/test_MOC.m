function test_suite = test_MOC
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

function test_MOC_Erik(t)
    [A,M,B] = get_MOC_data();

    n = size(A, 1);
    t.assertEqual(n, 8*8*4*6);

    res = 0;

    % Add the nullspace to the matrix as border
    A2 = sparse(n+2, n+2);
    A2(1:n, 1:n) = A;
    for j = 0:n-1
        if (mod(j, 6) == 3)
            if mod((mod(floor(j / 6), 4) + mod(floor(floor(j / 6) / 4), 16)), 2) == 0
                A2(n+1, j+1) = 1;
                A2(j+1, n+1) = 1;
            else
                A2(n+2, j+1) = 1;
                A2(j+1, n+2) = 1;
            end
        end
    end

    M2 = sparse(n+2, n+2);
    M2(1:n, 1:n) = M;

    B2 = sparse(n+2, size(B,2));
    B2(1:n, 1:size(B,2)) = B;

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
    dir = fileparts(mfilename('fullpath'));
    Abeg  = load([dir, '/../DataErik/Ap1.beg']);
    Aco   = load([dir, '/../DataErik/Ap1.co']);
    Ainfo = load([dir, '/../DataErik/Ap1.info']);
    Ajco  = load([dir, '/../DataErik/Ap1.jco']);
    Arl   = load([dir, '/../DataErik/Ap1.rl']);
    Mco   = load([dir, '/../DataErik/Bp1.co']);
    F     = load([dir, '/../DataErik/Frcp1.co']);

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
