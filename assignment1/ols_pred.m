load HA1_Parana_Jan
sz = [43 67];
nbr_of_obs = length(ParanaObs(:,5)); % number of observation
nbr_of_valid_data = 60; % number of observation use for validation
valid_data_ind = randperm(nbr_of_obs,nbr_of_valid_data); % indices of observations that will be used for validation
train_indices = ~ismember(1:nbr_of_obs,valid_data_ind); % indices of observations used for training model

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
X = [ones(nbr_of_obs-nbr_of_valid_data,1) long_train lat_train dist_to_coast_train];
beta = regress(precip_train,X);

% interpolation of precipitation on the grid 
X0 = [ones(length(ParanaGrid(:,1)),1) ParanaGrid(:,1) ParanaGrid(:,2) ParanaGrid(:,4)];
Y_pred = X0*beta;
figure
imagesc(reshape(Y_pred,sz))

err = precip_train - X*beta; % residuals
sigma_2_hat = norm(err)^2/(length(err)-3);
v_beta = (X'*X)\(eye(4)*sigma_2_hat);

v_mu_1 = sum((X0*v_beta).*X0,2); 
v_mu_2 = sum((X*v_beta).*X,2);
v_pred_1 = sigma_2_hat + v_mu_1; % variance of predicted points
v_pred_2 = sigma_2_hat + v_mu_2; % variance of observed training points
figure
imagesc(reshape(v_pred_1,sz))
% scatter(ParanaGrid(:,1), ParanaGrid(:,2),10, sqrt(v_pred_1), 'filled')
% hold on
% scatter(long_train, lat_train,10, sqrt(v_pred_2), 'filled')


% nbr_itr = 10; % number of permutation for bootstrap
% Dmax = 0;
% 
% for i = 1:nbr_itr
%     random_permutation = randi(length(long_train),length(long_train),1);
%     
%     % permute known data randomly and compute percipitation at the
%     % locations where precipitation have been measured
%     Y_pred = [ones(nbr_of_obs-nbr_of_valid_data,1) long_train(random_permutation) ...
%         lat_train(random_permutation) dist_to_coast_train(random_permutation)]*beta; 
%     % figure
%     % scatter(long_obs(random_permutation), lat_obs(random_permutation),20, Y_pred)
% 
%     err = precip_train(random_permutation)-Y_pred; % residuals
% 
%     D = distance_matrix([long_train(random_permutation),lat_train(random_permutation)]);
% 
%     % estimate the covariance function using binned least squares
%     Kmax = 500;
%     if Dmax == 0
%         Dmax = max(D(:))/2;
%     end
%     [rhat,s2hat,m,n,d]=covest_nonparametric(D,err,Kmax,Dmax);
%     figure
%     plot(d,rhat,'o',0,s2hat,'ro')
% end