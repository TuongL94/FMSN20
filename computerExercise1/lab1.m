sigma2 = 0.1;
kappa = 0.1;
nu = 1;
figure
plot(0:80, matern_covariance(0:80, sigma2, kappa, nu));

for i=1:length(nu)
    dims = [50 60];
    N = dims(1)*dims(2);
    [x,y]= ndgrid(1:dims(1), 1:dims(2));
    D = distance_matrix([x(:), y(:)]);
    Sigma = matern_covariance(D, sigma2, kappa(i), nu(i));
    mu = 2*ones(N,1);
    R = chol(Sigma + eye(size(Sigma))*1e-5); % Calculate the Cholesky factorisation
    eta = mu+R'*randn(N,1); % Simulate a sample
    eta_image = reshape(eta,dims); %reshape the column to an image
    figure
    imagesc(eta_image)
end

sigma_epsilon = 0.1;
y = eta + randn(N,1)*sigma_epsilon;
z = y-mu;
figure
plot(D, z*z', '.k');

nu = [0.01 0.1 1 10];
Kmax = 500;
Dmax = 30;%max(D(:));
[rhat,s2hat,m,n,d]=covest_nonparametric(D,z,Kmax, Dmax);
figure
plot(d,rhat,'o',0,s2hat,'or')

figure
for i = 1:4
    par = covest_ls(rhat,s2hat,m,n,d, [0 0 nu(i) 0]);
    hold on
    plot(0:80, matern_covariance(0:80, par(1), par(2), par(3)))
end
hold off

%% part 2: kriging from incomplete data
p = 1;
I_obs = (rand(dims)>=p);
Sigma_yy = Sigma + sigma_epsilon^2*eye(size(Sigma));
Sigma_uu = Sigma_yy(~I_obs, ~I_obs);
Sigma_uo = Sigma_yy(~I_obs, I_obs);
Sigma_oo = Sigma_yy(I_obs, I_obs);
y_o = y(I_obs);
y_u = y(~I_obs);

X = ones(N,1);
X_u = X(~I_obs);
X_o = X(I_obs);

y_rec = nan(dims);
y_rec(I_obs) = y_o;
mu = 2;
y_rec(~I_obs) = mu + Sigma_uo*(Sigma_oo\(y_o - mu*ones(size(y_o))));
%%
close all
figure
imagesc(eta_image)
figure
imagesc(y_rec)
figure 
imagesc(I_obs)
