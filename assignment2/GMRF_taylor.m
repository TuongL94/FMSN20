function [logp, D_logp, D2_logp]= GMRF_taylor(x_0, y, A, Q, E, pars)
% GMRF_TAYLOR_SKELETON  Taylor expansion of the conditional for non-Gaussian observations
%
% [logp, D_logp, D2_logp]= GMRF_taylor_Po(x_0, y, A, Q, E)
%
% x_0 = value at which to compute taylor expansion, as a column with N elements
% y = the data vector, as a column with n elements
% A = the observation matrix, sparse n-by-N
% Q = the precision matrix, sparse N-by-N
% E = the population count in each region, as a column with n elements
% pars = possibly additional parameters needed by the observation likelihood
%
% Function should return taylor expansion, gradient and Hessian.
%
% $Id: gmrf_taylor_skeleton.m 5107 2017-11-12 13:35:17Z johanl $

%ensure that E=1 if E not given (i.e. same/no population weight in all regions)
if nargin<5, E=1; end
if nargin<6, par=[]; end

%compute observed part of the field
z = A*x_0;

%compute log(p(y|z,theta))
b = zeros(length(y),1);
for i = 1:length(y)
    b(i)  = sum(log(1:y(i)));
end
f = sum(y.*(log(E) + z) - E.*exp(z) - b);

%compute -log p(x|y,theta)
logp = 0.5*x_0'*Q*x_0 - f;

if nargout>1
  %compute derivatives (if needed, i.e. nargout>1)
  df = y - E.*exp(z);
  D_logp = Q*x_0 - A'*df;
end

if nargout>2
  %compute hessian (if needed, i.e. nargout>2)
  d2f = -E.*exp(z); % diagonal elements of the hessian of f
  n = size(A,1);
  D2_logp = Q - A'*spdiags(d2f,0,n,n)*A;
end

