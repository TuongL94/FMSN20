%% 
close all
clear all
load HA1_Parana_Jan.mat

%extract covariates and reshape to images
sz = [43 67];
long = reshape(ParanaGrid(:,1), sz);
lat = reshape(ParanaGrid(:,2), sz);
elev = reshape(ParanaGrid(:,3), sz);
dist = reshape(ParanaGrid(:,4), sz);
Ind = reshape(ParanaGrid(:,5), sz);

%%

X = [ones(598,1) ParanaObs(:,1) ParanaObs(:,2) ParanaObs(:,4)];
Y = ParanaObs(:,5);
beta = regress(Y,X);
mu1 = [ones(2881,1) long(:) lat(:) dist(:)]*beta;
max(mu1)
%mu2 = [ones(598,1) ParanaObs(:,3) 1./ParanaObs(:,4)]*beta;

imagesc(long(:), lat(:), reshape(mu1, sz));

r = Y - X*beta;
mean(r)

D = distance_matrix([ParanaObs(:,2), ParanaObs(:,1)]);
figure
plot(D, r*r', '.k');

nu = 0.1;
Kmax = 500;
Dmax = max(D(:))/2;
obs = ParanaObs;
figure
hold on
for i=1:1
    %sl?ng om punkter
    %r?kna ut r och D
    idx = randperm(598);
    obs = obs(idx,:);
    X = X(idx,:);
    Y = Y(idx,:);
    r = Y - X*beta;
    D = distance_matrix(obs(:,2),obs(:,1));
    [rhat,s2hat,m,n,d]=covest_nonparametric(D,r,Kmax, Dmax);
    %r?kna ut n?n kvantil
    plot(d,rhat)
end

