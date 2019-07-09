function [S, MS, BS, Sinv, Vtrans] = RAILSschur(A, M, B)
% [S, MS, BS, Sinv, Vtrans] = RAILSschur(A, M, B)
%
% Used for problems with a singular M. After this, RAILSsolver can
% be used as
%
% [...] = RAILSsolver(S, MS, BS, ...)
%
% and for the inverse (opts.projection_method > 1), one can use
%
% opts.Ainv = Sinv;
%
% However, it is better to adjust this method in such a way that it
% uses a factorization/iterative method that is suitable for your problem.
%
% The solution to the original problem, instead of the solution to
% the problem reduced to the nonsingular part of M, can be found
% using Vorig = Vtrans(V), after which the low-rank approximation to the
% solution is given by Vorig * T * Vorig'.

    idx1 = find(abs(diag(M)) < 1e-12);
    idx2 = find(abs(diag(M)) > 1e-12);

    A11 = A(idx1, idx1);
    A12 = A(idx1, idx2);
    A21 = A(idx2, idx1);
    A22 = A(idx2, idx2);

    AS = @(x) A22 * x - A21 * (A11 \ (A12 * x));
    ASt = @(x) A22' * x - A12' * (A11' \ (A21' * x));
    S = @(y, varargin) transfun(y, AS, ASt, varargin{:});

    MS = [];
    if ~isempty(M)
        MS = M(idx2, idx2);
    end

    BS = B(idx2,:);

    if norm(B(idx1,:), inf) > sqrt(eps)
        BS = restrict(B);
        warning('B is not zero in the singular part');
    end

    idxr = [idx1; idx2];
    idxl = zeros(size(A,1),1);
    for i=1:size(A,1)
        idxl(i) = find(idxr==i);
    end

    Xreorder = @(x) x(idxl,:);
    X2 = @(x) x(idx2,:);

    Sinv = @(x) X2(A \ (Xreorder([zeros(size(A,1)-size(x,1), size(x,2)); x])));

    Vtrans = @(V) transform(V);

    function y = restrict(x)
        y = x(idx2,:)- A21 * (A11 \ x(idx1,:));
    end

    function y = prolongate(x)
        y = Xreorder([-A11 \ (A12 * x); x]);
    end

    function y = transform(x)
        if size(x, 1) == size(A, 1)
            y = restrict(x);
        elseif size(x, 1) == size(A22, 1)
            y = prolongate(x);
        else
            error(['size of x =', num2str(size(x, 1))]);
        end
    end
end 

function y = transfun(x, f, tf, trans)
    if nargin < 4 || ~strcmp(trans, 'transp')
        y = f(x);
    else
        y = tf(x);
    end
end