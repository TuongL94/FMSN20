%load data
load HA2_Brazil.mat
%extract observations, E and covariates
Y = Insurance(:,2);
E = Insurance(:,1);
B = Insurance(:,3:end);
%observation matrix for all locations
A = speye(length(Y));
%find missing (nan) observations
I = ~isnan(Y);
%we need a global variable for x_mode to reuse it
%between optimisation calls
global x_mode;
%attempt to estimate parameters (optim in optim...)
%subset to only observed points here
theta0 = [0 0];

options = optimset('OutputFcn', @outfun);
par = fminsearch(@(theta)GMRF_negloglike_NG(theta, Y(I), A(I,:), B(I,:),G,E(I)),theta0,options);

%conditional mean is now given be the mode
E_xy = x_mode;

%%

%extract parameters (and transform to natural parameters)
par = exp(par);

%compute Q matrices  (for a CAR(1) or SAR(1) process)
Q_x = par(1)*G; % CAR(1) process
%compute Q for beta-parameters
Qbeta = 1e-3*speye(size(B(I,:),2));
Q_eps = 1/par(2)*speye(size(A(I,:),2));

%combine all components of Q using blkdiag
Q_tilde = blkdiag(Q_x,Q_eps,Qbeta);

%use the taylor expansion to compute posterior precision
[~, ~, Q_xy] = GMRF_taylor(E_xy,Y(I),A(I,:),Q_tilde,E(I));

e = [zeros(size(Q_xy,1)-size(B,2), size(B,2)) eye(size(B,2))];
var_beta = e'*(Q_xy\e);