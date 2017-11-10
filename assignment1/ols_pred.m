load HA1_Parana_Jan
sz = [43 67];
nbr_of_obs = length(ParanaObs(:,5)); % number of observation

% observed data
long_obs = ParanaObs(:,1);
lat_obs = ParanaObs(:,2);
dist_to_coast_obs = ParanaObs(:,4);
precip_obs = ParanaObs(:,5);

% regression to estimate unknown parameters of our model (the model is that
% the precipitation depends linearly on longitude, latitude and distance to
% coast)
X = [ones(nbr_of_obs,1) long_obs lat_obs dist_to_coast_obs];
beta = regress(precip_obs,X);

% interpolation of precipitation on the grid 
% Y_pred = [ones(length(ParanaGrid(:,1)),1) ParanaGrid(:,1) ParanaGrid(:,2) ParanaGrid(:,4)]*beta;
% figure
% imagesc(reshape(Y_pred,sz))

nbr_itr = 10; % number of permutation for bootstrap
Dmax = 0;

for i = 1:nbr_itr
    random_permutation = randperm(nbr_of_obs);
    
    % permute known data randomly and compute percipitation at the
    % locations where precipitation have been measured
    Y_pred = [ones(nbr_of_obs,1) long_obs(random_permutation) ...
        lat_obs(random_permutation) dist_to_coast_obs(random_permutation)]*beta; 
    % figure
    % scatter(long_obs(random_permutation), lat_obs(random_permutation),20, Y_pred)

    err = precip_obs(random_permutation)-Y_pred; % residuals

    D = distance_matrix([long_obs(random_permutation),lat_obs(random_permutation)]);

    % estimate the covariance function using binned least squares
    Kmax = 500;
    if Dmax == 0
        Dmax = max(D(:))/2;
    end
    [rhat,s2hat,m,n,d]=covest_nonparametric(D,err,Kmax,Dmax);
    figure
    plot(d,rhat,'o',0,s2hat,'ro')
end