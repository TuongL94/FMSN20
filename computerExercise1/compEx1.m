% Covariance matrices and simulation of Gaussian fields

% parameters for matern covariance function
sigma2 = 0.1*ones(1,3);
kappa = [0.1 1 10];
nu = [1 1 1];

h = linspace(0,80,100);

% plots the matern covariance function for various parameters
for i = 1:1
    figure
    plot(h,matern_covariance(h,sigma2(i),kappa(i),nu(i)));
end

% simulation of a sample from a Gaussian field
for i = 1:1
    dims = [50 60];
    N = dims(1)*dims(2);
    [x,y] = ndgrid(1:dims(1),1:dims(2));
    A = [x(:),y(:)];
    D = distance_matrix(A);
    Sigma = matern_covariance(D,sigma2(i),kappa(i),nu(i));
    mu = 2*ones(N,1);
    R = chol(Sigma + eye(size(Sigma))*1e-5);
    eta = mu + R'*randn(N,1);
    eta_image = reshape(eta,dims);
    figure
    imagesc(eta_image)
end

%% Non-parametric estimation of covariances

% estimate covariance function using product of errors
sigma_epsilon = 0.1;
y = eta + randn(N,1)*sigma_epsilon;
z = y - mu;
figure
plot(D,z*z','.k');

% estimate covariance function using binned covariance estimation
Kmax = 500;
Dmax = 30;
[rhat,s2hat,m,n,d]=covest_nonparametric(D,z,Kmax,Dmax);
figure
plot(d,rhat,'o',0,s2hat,'o')

%% Parametric estimation of covariances

% estimate parameters of covariance function using least squares
par=covest_ls(rhat, s2hat, m, n, d, 'matern',[0 0 1 0]);
figure
plot(h,matern_covariance(h,par(1),par(2),par(3)));

%% Kriging

I_obs = (rand(dims) <= 0.01);

Sigma = matern_covariance(D,par(1),par(2),par(3));
Sigma_yy = Sigma + sigma_epsilon^2*eye(size(Sigma));
Sigma_uu = Sigma_yy(~I_obs,~I_obs);
Sigma_uo = Sigma_yy(~I_obs,I_obs);
Sigma_oo = Sigma_yy(I_obs,I_obs);

y_o = y(I_obs);
y_u = y(~I_obs);
mu_u = mu(~I_obs);
mu_o = mu(I_obs);
y_rec = nan(dims);
y_rec(I_obs) = y_o;

y_rec(~I_obs) = mu_u + Sigma_uo*(Sigma_oo\(y_o - mu_o));
rec_image = reshape(y_rec,dims);
figure
imagesc(rec_image)

real_image = reshape(y,dims);
figure
imagesc(real_image)
