load HA1_Parana_Jan
sz = [43 67];
nbr_of_obs = length(ParanaObs(:,5)); % number of observation
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

% regression to estimate unknown parameters of our model (the model is that
% the precipitation depends linearly on longitude, latitude and distance to
% coast)
X_k = [ones(nbr_train,1) long_train lat_train dist_to_coast_train];
beta = regress(precip_train,X_k);

% interpolation of precipitation on the grid 
X_u = [ones(nbr_grid,1) ParanaGrid(:,1) ParanaGrid(:,2) ParanaGrid(:,4)];
Y_pred_ols = X_u*beta;

err = precip_train - X_k*beta; % residuals
sigma_2_hat = norm(err)^2/(nbr_train-3);
v_beta = (X_k'*X_k)\(eye(4)*sigma_2_hat);

v_mu_ols = sum((X_u*v_beta).*X_u,2); 
v_pred_ols = sigma_2_hat + v_mu_ols; % variance of predicted points


nbr_itr = 100; % number of permutation for bootstrap
D = distance_matrix([long_train,lat_train]);
Dmax = max(D(:))/2;
Kmax = 100;
R = zeros(nbr_itr,Kmax+1);
[rhat,s2hat,m,n,d]=covest_nonparametric(D,err,Kmax,Dmax); % binned ls

% bootstrap
for i = 1:nbr_itr
    random_permutation = randperm(length(long_train));
    err_temp = err(random_permutation);
   
    % estimate the covariance function using binned least squares
    [rhat_b,s2hat_b,m_b,n_b,d_b]=covest_nonparametric(D,err_temp,Kmax,Dmax);
    R(i,:) = rhat_b;
end
Q = quantile(R,0.95,1);

%% Plots

% reconstructed percipitation using ols
imagesc(reshape(Y_pred_ols,sz))

% variance of the predicted points for ols
figure
imagesc(reshape(v_pred_ols,sz))

% covariance function using binned ls
figure
plot(d,rhat,'o',0,s2hat,'o')
hold on
plot(d,Q)


