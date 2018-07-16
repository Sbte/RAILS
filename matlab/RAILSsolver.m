function [V,T,res,iter,resvec,timevec,restart_data] = RAILSsolver(A, M, B, varargin)
% Solver for A*V*T*V'*M+M*V*T*V'*A'+B*B'=0
% [V,T,res,iter,resvec] = RAILSsolver(A, M, B, maxit, tol, opts);
%
% Input should be clear. M can be left empty for standard Lyapunov.
%
% opts.projection_method:
%   Projected equations that are solver are of the form
%   V'*A*V*T*V'*M'*V+V'*M*V*T*V'*A'*V+V'*B*B'*V=0
%   where the options are
%   1: V = r (default, start with V_0)
%   1.1: V = A^{-1}r (start with A^{-1}V_0)
%   1.2: V = A^{-1}r (start with A^{-1}B)
%   1.3: V = A^{-1}r (start with V_0)
%   2.1: V = [r, A^{-1}r] (start with [V_0, A^{-1}V_0])
%   2.2: V = [r, A^{-1}r] (start with [B, A^{-1}B])
%   2.3: V = [r, A^{-1}r] (start with V_0)
%
% opts.invA or opts.Ainv:
%   Function that performs A^{-1}x. This can be approximate. It is
%   only referenced when projection_method > 1. The default uses \
%   but of course it is better to factorize this beforehand.
%   (default: @(x) A\x).
%
% opts.space:
%   Initial space V_0. The default is a random vector, but it can be
%   set to for instance B. It will be orthonormalized within this
%   function. (default: random vector).
%
% opts.expand:
%   Amount of vectors used to expand the space. (default: 3).
%
% opts.nullspace:
%   Space we want to be projected out of V in each step. (default: []).
%
% opts.ortho:
%   Orthogonalization method. Default is standard orthogonalization,
%   but one can also choose to use M-orthogonalization by setting this
%   to 'M'. (default: []).
%
% opts.restart_iterations:
%   Amount of iterations after which the method is restarted. The
%   method will not be restarted after a set amount of iterations if
%   this is set to -1. This option should be used in combination with
%   a restart tolerance, not with the restart size option mentioned
%   below. (default: -1)
%
% opts.restart_tolerance:
%   Tolerance used for retaining eigenvectors. If this is > 0 the
%   amount of vectors in V is dependent on this tolerance. (default
%   tol).
%
% opts.restart_upon_convergence: Perform a restart upon convergence to
%   reduce the size of the space. This is the same as doing a rank
%   reduction as a post-processing step except that it will also
%   reiterate to make sure the convergence tolerance is still
%   reached. (default: true).
%
% opts.restart_size:
%   Maximum amount of vectors that V can contain. The amount of
%   vectors will not be limited in case this is set to -1. Setting
%   this may cause the method to not converge, since a minimum rank
%   of the solution is required to reach a certain tolerance. Using
%   a combination of restart_iterations and restart_tolerance is
%   more robust. (default: -1).
%
% opts.reduced_size:
%   Amount of vectors that is used after a restart. This can be set in
%   combination with the restart size option. Note that by setting
%   this it is possible to throw away too much information at every
%   restart. The reduced size is not limited in case this is set to
%   -1. (default: -1).
%
% opts.lanczos_vectors:
%   Amount of eigenvectors that you want to compute using Lanczos.
%   For this we use eigs at the moment. Usually it is a good idea to
%   use more vectors than the amount of expand vectors since it can
%   happen that eigenvectors of the residual are already in the
%   space. (default: 2 * expand).
%
% opts.lanczos_tolerance
%   Tolerance for computing the eigenvectors/values.
%
% opts.fast_orthogonalization
%   Uses vector operations + ortho instead of doing modified GS.
%   This is less stable but about 10x as fast. (default: true).

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
    AV = [];
    VAV = [];

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
    restart_tolerance = tol;
    restart_upon_convergence = true;
    eigs_tol = [];
    nev = [];
    fast_orthogonalization = true;

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
        if isfield(opts, 'V')
            V = opts.V;
            if ~isempty(V) && size(V, 1) ~= size(B, 1)
                error('RAILSsolver:InvalidOption',...
                      'opts.V should have the same row dimension as A');
            end
        end
        if isfield(opts, 'restart_data')
            if ~isempty(opts.restart_data)
                if ~isfield(opts.restart_data, 'V') || ...
                        ~isfield(opts.restart_data, 'AV') || ~isfield(opts.restart_data, 'VAV')
                    error('RAILSsolver:InvalidOption',...
                          'opts.restart_data does not contain valid restart data');
                end
                V = opts.restart_data.V;
                if ~isempty(V) && size(V, 1) ~= size(B, 1)
                    error('RAILSsolver:InvalidOption',...
                          'opts.restart_data.V should have the same row dimension as A');
                end
                AV = opts.restart_data.AV;
                if ~isempty(AV) && size(V, 1) ~= size(AV, 1)
                    error('RAILSsolver:InvalidOption',...
                          'opts.restart_data.AV should have the same row dimension as A');
                end
                VAV = opts.restart_data.VAV;
                if ~isempty(VAV) && size(V, 2) ~= size(VAV, 1)
                    error('RAILSsolver:InvalidOption',...
                          'opts.restart_data.VAV should have the same column dimension as V');
                end
            end
        end
        if isfield(opts, 'space_is_orthogonalized')
            if isempty(V)
                error('RAILSsolver:InvalidOption',...
                      'opts.space has not been defined');
            end
            ortho = ~opts.space_is_orthogonalized;
        end
        if isfield(opts, 'restart_size')
            restart_size = opts.restart_size;
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
        if isfield(opts, 'lanczos_tolerance')
            eigs_tol = opts.lanczos_tolerance;
        end
        if isfield(opts, 'lanczos_vectors')
            nev = opts.lanczos_vectors;
        end
        if isfield(opts, 'fast_orthogonalization')
            fast_orthogonalization = opts.fast_orthogonalization;
        end
        if isfield(opts, 'reduced_size')
            reduced_size = opts.reduced_size;
            if reduced_size > 0 && restart_size > 0 && reduced_size >= restart_size
                error('RAILSsolver:InvalidOption',...
                      'opts.reduced_size should be smaller than opts.restart_size');
            end
        elseif restart_size > 0
            reduced_size = restart_size / 2;
        end
    end

    tstart = tic;

    % Allow for matrices and functions as input
    Afun = A;
    if isa(A, 'double')
        % Check matrix and right hand side vector inputs have appropriate sizes
        [m,n] = size(A);
        if (m ~= n)
            error('RAILSsolver:SquareMatrix', 'A should be a square matrix');
        end
        Afun = @(x) A * x;
    else
        m = size(B,1);
        n = m;
    end

    % Check for a singular mass matrix
    if ~isempty(M) && 1 / condest(M) < 1e-12
        warning('RAILSsolver:SingularMassMatrix', ...
                ['Your M matrix appears to be singular. ' ...
                 'It is advised to use the provided RAILSschur method.']);
    end

    % Check projection method usage
    if ~isempty(invA) && projection_method == 1
        warning('RAILSsolver:InverseNotUsed', ...
                ['An inverse application method is provided, but the current ', ...
                 'projection method does not make use of this']);
    elseif isempty(invA)
        invA = @(x) A \ x;
    end

    if isempty(V)
        % We use a random vector
        V = (rand(n,1) - .5) * 2;
    end
    W = V;

    if ~isempty(invA) && abs(projection_method - floor(projection_method) - 0.1) < sqrt(eps)
        W = invA(V);
    end
    if ~isempty(invA) && abs(projection_method - floor(projection_method) - 0.2) < sqrt(eps)
        V = full(B);
        W = invA(V);
    end

    if ~isempty(invA) && floor(projection_method) == 2 && ...
            abs(projection_method - floor(projection_method) - 0.3) > sqrt(eps)
        V = [V, W];
    elseif ~isempty(invA) && floor(projection_method) == 1 && ...
            abs(projection_method - floor(projection_method) - 0.3) > sqrt(eps)
        V = W;
    end

    if mortho
        V = Morth(V, M, nullspace, [], fast_orthogonalization);
    elseif ortho
        V = Morth(V, [], nullspace, [], fast_orthogonalization);
    end

    if isempty(nev)
        nev = expand * 2;
    end

    MV = [];
    VMV = [];

    BV = [];
    VBV = [];
    H = [];

    T = [];
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

        % VAV = V'*A*V;
        new_indices = size(VAV,2)+1:size(V,2);
        AVnew = Afun(V(:, new_indices));
        if isempty(VAV)
            VAV = V'*AVnew;
        else
            VAV = [[VAV; V(:,new_indices)'*AV], V'*AVnew];
        end
        AV = [AV, AVnew];

        % VBV = V'*B*B'*V;
        new_indices = size(VBV,2)+1:size(V,2);
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
        if hasM
            old_indices = 1:size(MV,2);
            new_indices = size(MV,2)+1:size(V,2);
            MVnew = M*V(:, new_indices);
            MV = [MV, MVnew];
        end

        if hasM && ~mortho
            % VMV = V'*M*V;
            if isempty(VMV)
                VMV = V'*MVnew;
            else
                VMV = [[VMV; V(:, new_indices)'*MV(:,old_indices)], V'*MVnew];
            end
            T = lyap(VAV, VBV, [], VMV);
        else
            T = lyap(VAV, VBV);
        end

        % Compute eigenvalues and vectors of the residual, but make
        % sure we do not compute too many
        eopts.issym = true;
        eopts.tol = eigs_tol;
        if hasM
            [V2,D2] = eigs(@(x) AV*(T*(MV'*x)) + MV*(T*(AV'*x)) + B*(B'*x), n, nev, 'lm');
        else
            [V2,D2] = eigs(@(x) AV*(T*(V'*x))  + V*(T*(AV'*x))  + B*(B'*x), n, nev, 'lm');
        end
        H = D2;

        % Sort eigenvectors by size of the eigenvalue
        [~,I2] = sort(abs(diag(D2)), 1, 'descend');
        V2 = V2(:, I2);
        D2 = D2(I2, I2);

        % Orthogonalize so we do not use too eigenvectors that are
        % already in the space
        if mortho
            V2 = Morth(V2, M, V, [], fast_orthogonalization);
        else
            V2 = Morth(V2, [], V, [], fast_orthogonalization);
        end

        res = norm(D2, inf);
        if verbosity > 0
            fprintf('Estimate Lanczos, absolute: %e, relative: %e, num eigenvectors: %d\n',...
                    res, res / r0, size(V2, 2));
        end
        res = res / r0;

        if iter_since_restart > 1 || i == 1
            resvec = [resvec; res];
            timevec = [timevec; toc(tstart)];
        end

        if abs(res) < tol
            if converged || ~restart_upon_convergence
                if verbosity > 0
                    fprintf('Converged with space size %d\n', size(V, 2));
                end
                if nargout > 6
                    restart_data.V = V;
                    restart_data.AV = AV;
                    restart_data.VAV = VAV;
                end
                break;
            end
            converged = true;
        end

        if i >= maxit || size(V,2) >= m
            if nargout > 6
                restart_data.V = V;
                restart_data.AV = AV;
                restart_data.VAV = VAV;
            end
            if projection_method == 1
                warning('RAILSsolver:ProjectionMethod', ...
                        ['Convergence has not been achieved with ' ...
                         'opts.projection_method = 1. ' ...
                         'It is advised to set opts.projection_method ' ...
                         'to a different value. For instance ' ...
                         'opts.projection_method = 1.2.']);
            end
            break;
        end

        if (restart_iterations > 0 && iter_since_restart == restart_iterations) ...
                || (isempty(H) && reduced_size > 0 && size(T,1) > reduced_size) ...
                || (restart_size > 0 && size(T, 1) > restart_size) ...
                || (converged && ~reduced)

            if reduced_size > 0 && reduced_size < size(V, 2)
                if verbosity > 0
                    fprintf(['Decreasing search space size. Trying %d ' ...
                             'vectors.\n'], reduced_size);
                end
                [V3, D3] = eigs(T, reduced_size);
            else
                if verbosity > 0
                    fprintf(['Decreasing search space size. Trying %d ' ...
                             'vectors.\n'], size(V, 2));
                end
                [V3, D3] = eig(T);
            end

            d = abs(diag(D3));
            I3 = find(d / max(d) > restart_tolerance);

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
                if ~isempty(MV)
                    MV = MV * V3;
                end
                if ~isempty(VMV)
                    VMV = V3' * VMV * V3;
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
        num = min(min(expand, size(V2,2)), m - size(V,2));

        W = V2(:, 1:num);

        if ~isempty(invA) && projection_method < 2 && projection_method > 1
            W = invA(W);
        elseif ~isempty(invA) && projection_method < 3 && projection_method > 2
            W = [W, invA(W)];
        end

        if mortho
            V = Morth(W, M, nullspace, V, fast_orthogonalization);
        else
            V = Morth(W, [], nullspace, V, fast_orthogonalization);
        end
    end

    if verbosity > 3 && verbosity < 5
        semilogy(resvec);
    end
end

function V = Morth(W, M, nullspace, V, fast_orthogonalization)
% MGS M-orthogonalisation
    if nargin < 5
        fast_orthogonalization = false;
    end

    if nargin < 4
        V = [];
    end
    if nargin < 3
        nullspace = [];
    end

    num_iter = 1;
    tol = 1e-8;

    if isempty(M)
        if fast_orthogonalization
            % Really fast orthogonalization, which is not so stable
            if ~isempty(nullspace)
                W = W - nullspace*(nullspace'*W);
            end
            if ~isempty(V)
                W = W - V*(V'*W);
            end
            V = [V, orth(W)];
        else
            % Actual modified GS, which is supposed to be about as fast
            % but is actually 10x as slow...
            for i = 1:size(W,2)
                v1 = W(:,i) / norm(W(:,i));
                for k = 1:num_iter
                    v1 = nullspace_op(v1);
                    for j=1:size(V,2)
                        v1 = v1 - V(:,j)*(V(:,j)'*(v1));
                    end
                end
                nrm = norm(v1);
                if nrm < tol
                    continue;
                end
                V(:,size(V,2)+1) = v1 / nrm;
            end
        end
    else
        % Modified GS with M-inner product
        for i = 1:size(W,2)
            v1 = W(:,i) / norm(W(:,i));
            for k = 1:num_iter
                v1 = nullspace_op(v1);
                for j=1:size(V,2)
                    v1 = v1 - V(:,j)*(V(:,j)'*(M*v1));
                end
            end
            nrm = sqrt(v1'*M*v1);
            if nrm < tol
                continue;
            end
            V(:,size(V,2)+1) = v1 / nrm;
        end
    end

    function y = nullspace_op(x)
        y = x;
        if ~isempty(nullspace)
            if isa(nullspace, 'function_handle')
                y = nullspace(x);
            else
                if isempty(M)
                    for ii=1:size(nullspace, 2)
                        y = y - nullspace(:,ii) * (nullspace(:,ii)' * y);
                    end
                else
                    for ii=1:size(nullspace, 2)
                        y = y - nullspace(:,ii) * (nullspace(:,ii)' * (M * y));
                    end
                end
            end
        end
    end
end