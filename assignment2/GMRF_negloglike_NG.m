function negloglike = GMRF_negloglike_NG(theta, y, A, B, G, E)
% GMRF_NEGLOGLIKE_NG_SKELETON Calculate the GMRF likelihood for non-Gaussian observations
%
% negloglike = GMRF_negloglike_NG(theta, y, A, B, G, E)
%
% theta - log of parameters
% y - data vector, as a column with n elements
% A - Observation matrix, sparse n-by-N
% B - Matrix of covariates, matrix of size n-by-Nbeta 
% C,G,G2 = matrices used to build a Matern-like CAR/SAR precision,
%          see matern_prec_matrices, sparse N-by-N
% qbeta = Precision for the regression parameters, scalar.
% E = The population count in each region, used for Poisson observations
%
% $Id: gmrf_negloglike_NG_skeleton.m 5109 2017-11-12 20:08:26Z johanl $

%ensure that E=1 if E not given (i.e. same/no population weight in all regions)
if nargin<6
    E=1;
end

%extract parameters (and transform to natural parameters)
par = exp(theta);

%compute Q matrices  (for a CAR(1) or SAR(1) process)
Q_x = par(1)*G; % CAR(1) process
%compute Q for beta-parameters
Qbeta = 1e-3*speye(size(B,2));
Q_eps = 1/par(2)*speye(size(A,2));

%combine all components of Q using blkdiag
Qall = blkdiag(Q_x,Q_eps,Qbeta);
%also compute the observation matric by combining A and B matrices
Aall = [A A B];

%declare x_mode as global so that we start subsequent optimisations from
%the previous mode (speeds up nested optimisation).
global x_mode;
if isempty(x_mode)
  %no existing mode, use zeros as start
  x_mode = zeros(size(Qall,1),1);
end

%compute reorder (if needed)
p = amd(Qall + Aall'*Aall); %sparsity and reorder of Q_xy
Qall = Qall(p,p); %reorder
Aall = Aall(:,p);
x_mode = x_mode(p);

%find mode - using Newton-Raphson optimisation
x_mode = fminNR(@(x) GMRF_taylor(x, y, Aall, Qall, E), x_mode);

%find the Laplace approximation of the denominator computed at the mode
[logp,~,Q_xy] = GMRF_taylor(x_mode, y, Aall, Qall, E);

%Compute choleskey factor of Q_xy
[R_xy, p_xy] = chol(Q_xy);
if p_xy~=0
  %choleskey factor fail -> (almost) semidefinite matrix -> 
  %-> det(Q) ~ 0 -> log(det(Q)) ~ -inf -> negloglike ~ inf
  %Set negloglike to a REALLY big value
  negloglike = realmax;
  return;
end

%also compute determinant of Qall (might be simplified)
log_det_Qall = log(par(1))*size(A,2) - size(A,2)*log(par(2)) - 3*size(B,2)*log(10);
log_det_Q_xy = sum(log(diag(R_xy)));
%note that logp = -log_obs + x_mode'*Q*x_mode/2.
negloglike = logp - 0.5*log_det_Qall + log_det_Q_xy;

%inverse reorder before returning
x_mode(p) = x_mode;

%print diagnostic information (progress)
% fprintf(1, 'fval: %11.4e\n', negloglike);
