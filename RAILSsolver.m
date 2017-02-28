function [V,S,res,iter,resvec,timevec] = RAILSsolver(A, M, B, varargin)
% Solver for A*V*S*V'*M+M*V*S*V'*A'+B*B'=0
% [V,S,res,iter] = RAILSsolver(A, M, B, maxit, tol, opts);
%
% Input should be clear. M can be left empty for standard Lyapunov.
%
% opts.projection_method:
%   Projected equations that are solver are of the form
%   V'*A*V*S*V'*M'*V+V'*M*V*S*V'*A'*V+V'*B*B'*V=0
%   where the options are
%   1: V = r (default)
%   1.1: V = A^{-1}r (start with A^{-1}V_0)
%   1.2: V = A^{-1}r (start with A^{-1}B)
%
% opts.invA or opts.Ainv:
%   Function that performs A^{-1}x. This can be approximate. It is
%   only referenced when projection_method > 1
%
% opts.space:
%   Initial space. The default is a random vector, but it can be
%   set to for instance B. It will be orthonormalized within this
%   function.
%
% opts.restart:
%   Maximum amount of vectors that V can contain (default not
%   used).
%
% opts.expand:
%   Amount of vectors used to expand the space. (default 3)
%
% opts.nullspace:
%   Space we want to be projected out of V in each step. (default
%   none)
%
% opts.ortho:
%   Orthogonalization method. Default is standard
%   orthogonalization, but one can also choose to use
%   M-orthogonalization by setting this to 'M'
%
% opts.restart_iterations:
%   Amount of iterations after which the method is restarted.
%   Should be used in combination with a restart tolerance, not
%   with the restart option mentioned above
%
% opts.restart_tolerance:
%   Tolerance used for retaining eigenvectors. If this is > 0 the
%   amount of vectors in V is dependent on this tolerance. (default
%   tol * 1e-3)
%
% opts.lanczos_iterations:
%   Maximum amount of iterations that is used by the lanczos
%   method. Note that one could also just replace the lanczos
%   method by a call to eigs if one wants.
%
% opts.reduced:
%   Amount of vectors that is used after a restart. This can be set
%   in combination with the restart option. Note that by setting this
%   it is possible to throw away too much information at every
%   restart

    if nargin < 3
        error('Not enough input arguments');
    end
    args = 3;

    if nargin < args+1 || ~isnumeric(varargin{args-2}) || varargin{args-2} - ceil(varargin{args-2}) ~=0
        if  nargin >= args+1 && isempty(varargin{args-2})
            args = args + 1;
        end
        maxit = 100;
    else
        maxit = varargin{args-2};
            args = args + 1;
    end

    hasM = ~isempty(M);
    invA = [];
    V = [];

    if nargin < args+1 || ~isnumeric(varargin{args-2})
        if  nargin >= args+1 && isempty(varargin{args-2})
            args = args + 1;
        end
        tol = 1e-4;
    else
        tol = varargin{args-2};
            args = args + 1;
    end

    restart_size = -1;
    restart_iterations = -1;
    lanczos_iterations = 10;
    reduced_size = -1;
    expand = min(3, size(B, 2));
    mortho = false;
    ortho = true;
    nullspace = [];
    projection_method = 1;
    restart_tolerance = tol * 1e-3;
    restart_upon_convergence = true;

    % Options
    verbosity = 0;
    if nargin > args
        opts = varargin{args-2};
        if ~isa(opts, 'struct')
            error('RAILSsolver:OptionsNotStructure',...
                  'the last argument is not a structure');
        end
        if isfield(opts, 'verbosity')
            if strcmp(opts.verbosity, 'Verbose')
                verbosity = 1;
            else
                verbosity = opts.verbosity;
            end
        end
        if isfield(opts, 'invA')
            invA = opts.invA;
        end
        if isfield(opts, 'Ainv')
            invA = opts.Ainv;
        end
        if isfield(opts, 'space')
            V = opts.space;
            if ~isempty(V) && size(V, 1) ~= size(B, 1)
                error('RAILSsolver:InvalidOption',...
                      'opts.space should have the same row dimension as A');
            end
        end
        if isfield(opts, 'space_is_orthogonalized')
            if isempty(V)
                error('RAILSsolver:InvalidOption',...
                      'opts.space has not been defined');
            end
            ortho = ~opts.space_is_orthogonalized;
        end
        if isfield(opts, 'restart')
            restart_size = opts.restart;
        end
        if isfield(opts, 'restart_upon_convergence')
            restart_upon_convergence = opts.restart_upon_convergence;
        end
        if isfield(opts, 'expand')
            expand = opts.expand;
            if expand > size(B, 2)
                error('RAILSsolver:InvalidOption',...
                      'opts.expand is larger than the column dimension of B');
            end
        end
        if isfield(opts, 'nullspace')
            nullspace = Morth(opts.nullspace, []);
        end
        if isfield(opts, 'projection_method')
            projection_method = opts.projection_method;
        end
        if isfield(opts, 'ortho') && opts.ortho == 'M'
            mortho = true;
        end
        if isfield(opts, 'restart_iterations')
            restart_iterations = opts.restart_iterations;
        end
        if isfield(opts, 'restart_tolerance')
            restart_tolerance = opts.restart_tolerance;
        end
        if isfield(opts, 'lanczos_iterations')
            lanczos_iterations = opts.lanczos_iterations;
        end
        if isfield(opts, 'reduced')
            reduced_size = opts.reduced;
            if reduced_size > 0 && restart_size > 0 && reduced_size >= restart_size
                error('RAILSsolver:InvalidOption',...
                      'opts.reduced should be smaller than opts.restart');
            end
        elseif restart_size > 0
            reduced_size = restart_size / 2;
        end
    end

    tstart = tic;

    % Allow for matrices and functions as input
    [atype, afun, afcnstr] = iterchk(A);
    if strcmp(atype, 'matrix')
        % Check matrix and right hand side vector inputs have appropriate sizes
        [m,n] = size(A);
        if (m ~= n)
            error('RAILSsolver:SquareMatrix', 'A should be a square matrix');
        end
    else
        m = size(B,1);
        n = m;
    end

    if isempty(V)
        % We use a random vector
        V = (rand(n,1) - .5) * 2;
    end

    if ~isempty(invA) && abs(projection_method - floor(projection_method) - 0.1) < sqrt(eps)
        V = invA(V);
    end
    if ~isempty(invA) && abs(projection_method - floor(projection_method) - 0.2) < sqrt(eps)
        V = invA(full(B));
    end

    if mortho
        V = Morth(V, M, nullspace);
    elseif ortho
        V = Morth(V, [], nullspace);
    end

    AV = [];
    VAV = [];

    MV = [];
    VMV = [];

    BV = [];
    VBV = [];
    H = [];

    S = [];
    resvec = [];
    timevec = [];
    iter = 0;
    iter_since_restart = 0;
    converged = false;
    reduced = false;

    r0 = norm(full(B'*B), 2);

    % Main loop
    for i=1:maxit
        iter = i;
        iter_since_restart = iter_since_restart + 1;
        if verbosity > 0
            fprintf('Iter %d\n', i);
        end

        new_indices = size(VAV,2)+1:size(V,2);
        % VAV = V'*A*V;
        AVnew = iterapp('mtimes', afun, atype, afcnstr, V(:, new_indices));
        if isempty(VAV)
            VAV = V'*AVnew;
        else
            VAV = [[VAV; V(:,new_indices)'*AV], V'*AVnew];
        end
        AV = [AV, AVnew];

        % VBV = V'*B*B'*V;
        BVnew = B' * V(:,new_indices);
        if isempty(VBV)
            VBV = BVnew'*BVnew;
        else
            t = BVnew'*BV;
            VBV = [[VBV; t], [t'; BVnew'*BVnew]];
        end
        BV = [BV, BVnew];

        % This is not needed if we use a normal
        % Lyapunov solver instead of a general one
        % VMV = V'*M*V;
        if hasM && ~mortho
            MVnew = M*V(:, new_indices);
            if isempty(VMV)
                VMV = V'*MVnew;
            else
                VMV = [[VMV; V(:, new_indices)'*MV], V'*MVnew];
            end
            MV = [MV, MVnew];

            S = lyap(VAV, VBV, [], VMV);
        else
            S = lyap(VAV, VBV);
        end

        [~,H,V2,D2] = rlanczos(AV, B, M, V, S, max(lanczos_iterations, expand * 1 + 1), expand, verbosity);

        % Sort eigenvectors by size of the eigenvalue
        [~,I2] = sort(abs(diag(D2)), 1, 'descend');
        V2 = V2(:, I2);
        D2 = D2(I2, I2);

        res = norm(D2, inf);
        if verbosity > 0
            fprintf('Estimate Lanczos, absolute: %e, relative: %e, iterations: %d\n', res, res / r0, size(V2, 2));
        end
        res = res / r0;

        if iter_since_restart > 1 || i == 1
            resvec = [resvec; res];
            timevec = [timevec; toc(tstart)];
        end

        if abs(res) < tol
            if converged
                if verbosity > 0
                    fprintf('Converged with space size %d\n', size(V, 2));
                end
                break;
            end
            converged = true;
            if ~restart_upon_convergence
                reduced = true;
            end
        end

        if i >= maxit || size(V,2) >= m
            break;
        end

        if (restart_iterations > 0 && iter_since_restart == restart_iterations) ...
                || (isempty(H) && reduced_size > 0 && size(S,1) > reduced_size) ...
                || (restart_size > 0 && size(S, 1) > restart_size) ...
                || (converged && ~reduced)

            if reduced_size > 0 && reduced_size < size(V, 2)
                if verbosity > 0
                    fprintf(['Decreasing search space size. Trying %d ' ...
                             'vectors.\n'], reduced_size);
                end
                [V3, D3] = eigs(S, reduced_size);
            else
                if verbosity > 0
                    fprintf(['Decreasing search space size. Trying %d ' ...
                             'vectors.\n'], size(V, 2));
                end
                [V3, D3] = eig(S);
            end

            d = abs(diag(D3));
            I3 = find(d > restart_tolerance);

            % Restart using only the part of the largest eigenvectors
            [~,s] = sort(d(I3),'descend');
            V3 = V3(:, I3(s));
            V = V * V3;

            if verbosity > 0
                fprintf('Restarted with %d vectors.\n', size(V, 2));
            end
            reduced = converged;

            reortho = 0;
            if reortho
                VAV = [];
                AV = [];
                VMV = [];
                MV = [];
                VBV = [];
                BV = [];
            else
                VAV = V3' * VAV * V3;
                AV = AV * V3;
                if hasM && ~mortho
                    VMV = V3' * VMV * V3;
                    MV = MV * V3;
                end
                VBV = V3' * VBV * V3;
                % Make this symmetric for safety reasons
                VBV = (VBV + VBV') / 2;
                BV = BV * V3;
            end

            iter_since_restart = 0;
            continue
        end

        % Max number of expansion vectors we want
        num = min(min(expand, size(H,2)), m - size(V,2));

        W = V2(:, 1:num);

        if ~isempty(invA) && projection_method < 2 && projection_method > 1
            W = [W, invA(W)];
            % W = invA(W);
        end

        if mortho
            V = Morth(W, M, nullspace, V);
        else
            V = Morth(W, [], nullspace, V);
        end
    end

    if verbosity > 3 && verbosity < 5
        semilogy(resvec);
    end
end

function [Q, H, V, d] = rlanczos(AW, B, M, W, S, iters, nev, verbosity)
    n = size(B,1);
    m = 5;
    if nargin > 5
        m = iters;
    end
    
    if nargin < 7
        nev = 0;
    end

    q1 = (rand(n,1) - .5) * 2;
    Q = zeros(n, m+2);
    Q(:,2) = q1/ norm(q1);
    H = zeros(m+1, m+1);
    beta = 0;

    V = [];

    iter = 0;
    for k=1:m
        if isempty(M)
            z = AW*(S*(W'*Q(:,k+1))) + W*(S*(AW'*Q(:,k+1))) + B*(B'*Q(:,k+1));
        else
            z = AW*(S*(W'*(M'*Q(:,k+1)))) + M*(W*(S*(AW'*Q(:,k+1)))) + B*(B'*Q(:,k+1));
        end

        alpha = z'*Q(:,k+1);
        H(k, k) = alpha;

        z = z - alpha * Q(:,k+1) - beta * Q(:,k);
        z = Morth2(z, [], V);
        beta = norm(z);
        if beta < 1e-14
            iter = iter + 1;
            break;
        end

        if nev > 0
            [v,d] = eig(H(1:k,1:k));
            [~,idx] = sort(abs(diag(d)), 1, 'descend');
            v = v(:, idx);

            % Determine residuals and check for convergence
            conv = [];
            for i=1:k
                r = abs(beta * v(k, i)) / abs(d(idx(i), idx(i)));
                if r < 1e-12
                    conv = [conv i];
                end
            end

            % If eigenvectors converged put them in V
            if ~isempty(conv)
                V = Morth(Q(:,2:k+1) * v(:, conv), []);
            end

            if isequal(conv(1:min(nev, length(conv))), 1:nev)
                iter = iter + 1;
                break;
            end
        end

        H(k+1,k) = beta;
        H(k,k+1) = beta;
        Q(:,k+2) = z / beta;

        iter = iter + 1;
    end

    H = H(1:iter,1:iter);
    Q = Q(:,2:iter+1);

    [v,d] = eig(H);

    % Compute Ritz vectors
    if nargout > 2
        V = Q*v;
    end
end

function V = Morth(W, M, nullspace, V)
% GS M-orthogonalisation
    if nargin < 4
        V = [];
    end
    if nargin < 3
        nullspace = [];
    end
    if isa(nullspace, 'function_handle')
        nullspace_op = nullspace;
    else
        nullspace_op = @(x)  x - nullspace*(nullspace'*x);
    end

    for i = 1:size(W,2)
        v1 = W(:,i) / norm(W(:,i));
        for k = 1:2
            if ~isempty(nullspace)
                v1 = nullspace_op(v1);
            end
            if ~isempty(V)
                if isempty(M)
                    v1 = v1 - V*(V'*v1);
                else
                    v1 = v1 - V*(V'*M*v1);
                end
            end
        end
        nrm = norm(v1);
        if isempty(M)
            V(:,size(V,2)+1) = v1 / nrm;
        else
            V(:,size(V,2)+1) = v1 / sqrt(v1'*M*v1);
        end
    end
end

function V = Morth2(W, M, nullspace, V)
% GS M-orthogonalisation
    if nargin < 4
        V = [];
    end
    if nargin < 3
        nullspace = [];
    end

    for i = 1:size(W,2)
        v1 = W(:,i);
        for k = 1:3
            if ~isempty(nullspace)
                v1 = v1 - nullspace*(nullspace'*v1);
            end
            if ~isempty(V)
                if isempty(M)
                    v1 = v1 - V*(V'*v1);
                else
                    v1 = v1 - V*(V'*M*v1);
                end
            end
        end
        V(:,size(V,2)+1) = v1;
    end
end
