load HA1_Parana_Jan
sz = [43 67];
nbr_of_obs = length(ParanaObs(:,5)); % number of observations
nbr_of_valid_data = 60; % number of observation use for validation
valid_data_ind = randperm(nbr_of_obs,nbr_of_valid_data); % indices of observations that will be used for validation
train_indices = ~ismember(1:nbr_of_obs,valid_data_ind); % indices of observations used for training model
nbr_grid = length(ParanaGrid(:,1)); % number of points in grid
nbr_train = nbr_of_obs - nbr_of_valid_data;

% observed data
long_obs = ParanaObs(:,1);
lat_obs = ParanaObs(:,2);
dist_to_coast_obs = ParanaObs(:,4);
precip_obs = ParanaObs(:,5);

% training data
long_train = long_obs(train_indices);
lat_train = lat_obs(train_indices);
dist_to_coast_train = dist_to_coast_obs(train_indices);
precip_train = precip_obs(train_indices);

% validation data
long_valid = long_obs(valid_data_ind);
lat_valid = lat_obs(valid_data_ind);
dist_to_coast_valid = dist_to_coast_obs(valid_data_ind);
precip_valid = precip_obs(valid_data_ind);

% grid data
long_grid = ParanaGrid(:,1);
lat_grid = ParanaGrid(:,2);
dist_to_coast_grid = ParanaGrid(:,4);
in_parana_grid = ParanaGrid(:,5);

%% ordinary least squares

% regression to estimate unknown parameters of our model (the model is that
% the precipitation depends linearly on longitude, latitude and distance to
% coast)
X_k = [ones(nbr_train,1) long_train lat_train dist_to_coast_train];
[beta, beta_int] = regress(precip_train,X_k);

% interpolation of precipitation on the grid
X_u = [ones(nbr_grid,1) long_grid lat_grid dist_to_coast_grid];
y_pred_ols = X_u*beta;

err = precip_train - X_k*beta; % residuals

sigma_eps_2_hat = norm(err)^2/(nbr_train-3);
v_beta = (X_k'*X_k)\(eye(4)*sigma_eps_2_hat);
v_mu_ols = sum((X_u*v_beta).*X_u,2); 
v_pred_ols = sigma_eps_2_hat + v_mu_ols; % variance of predicted points

nbr_itr = 100; % number of permutation for bootstrap
D_train = distance_matrix([long_train,lat_train]);
Dmax = max(D_train(:))/2;
Kmax = 100;
R = zeros(nbr_itr,Kmax+1);
[rhat,s2hat,m,n,d]=covest_nonparametric(D_train,err,Kmax,Dmax); % binned ls

% bootstrap
for i = 1:nbr_itr
    random_permutation = randperm(length(long_train));
    err_temp = err(random_permutation);
   
    % estimate the covariance function using binned least squares
    [rhat_b,s2hat_b,m_b,n_b,d_b]=covest_nonparametric(D_train,err_temp,Kmax,Dmax);
    R(i,:) = rhat_b;
end
Q = quantile(R,0.95,1);

%% universal kriging
z = precip_train - X_k*beta;

par = covest_ml(D_train, z);
sigma2 = par(1);
kappa = par(2);
nu = par(3);
sigma2_epsilon = par(4); 

D_all = distance_matrix([long_train lat_train; long_grid lat_grid]);
Sigma = matern_covariance(D_all, sigma2, kappa, nu);

Sigma_yy = Sigma + sigma2_epsilon*eye(size(Sigma));
Sigma_kk = Sigma_yy(1:nbr_train, 1:nbr_train);
Sigma_uk = Sigma_yy(nbr_train+1:end, 1:nbr_train);
Sigma_ku = Sigma_uk';
Sigma_uu = Sigma_yy(nbr_train+1:end, nbr_train+1:end);

y_k = precip_train;
y_rec = nan(nbr_train+nbr_grid,1);
y_rec(1:nbr_train) = y_k;
mu_k = X_k*beta;
mu_u = X_u*beta;
y_rec(nbr_train+1:end) = mu_u + Sigma_uk*(Sigma_kk\(y_k - mu_k));

v_pred = diag(Sigma_uu - (Sigma_uk*(Sigma_kk\Sigma_ku)));

%% plots

% reconstructed percipitation using ols
figure
imagesc('XData',long_grid,'YData',lat_grid,'CData',reshape(y_pred_ols,sz))
plotBorder(Border);
colorbar

% variance of the predicted points for ols
figure
imagesc('XData',long_grid,'YData',lat_grid,'CData',reshape(v_pred_ols,sz))
plotBorder(Border);
colorbar

% reconstructed percipitation using universal kriging
figure
imagesc('XData',long_grid,'YData',lat_grid,'CData',reshape(y_rec(nbr_train+1:end),sz));
plotBorder(Border);
colorbar

% variance of the predicted points for universal kriging
figure
imagesc('XData',long_grid,'YData',lat_grid,'CData',reshape(sqrt(v_pred),sz));
plotBorder(Border);
colorbar

% nonparametric covariance estimation
figure
plot(d,rhat,'o',0,s2hat,'ro')

hold on
x = 0:0.01:4;
plot(x,matern_covariance(x, sigma2, kappa, nu), 'r'); % parametric covariance estimation (maximum likelihood)
plot(d,Q,'g') % 95% quantiles from bootstrap
hold off

% validation data
figure
scatter(long_valid,lat_valid, 20, precip_valid,'filled')
colorbar
