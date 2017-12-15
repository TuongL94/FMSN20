% Load data
load fmri.mat

sz = size(img); %size of data

% parameters for classification algorithms
K = 5; % number of classes
maxiter = 100; % maximum iteration for K-means and GMM
Nsim = [200 100];
plotflag = 0;

%% Option 1: regress onto indicator functions
beta = X\colstack(img)';
beta = icolstack(beta',sz(1:2));
%and treat the beta:s as the image
[y_beta, ~, P_beta] = pca(colstack(beta));
y_beta_pca = y_beta(:,[1,3,4]); % components with most information

%%
y_beta_invcol = icolstack(y_beta, sz(1:2));
figure
subplot(3,4,1)
semilogy(P_beta/sum(P_beta))
axis tight
for i=1:size(y_beta,3)
  subplot(3,4,i+1)
  imagesc(y_beta_invcol(:,:,i))
  title(i)
end

%% K-means classification for option 1
[cl1_km,theta1_km] = kmeans(y_beta_pca,K,maxiter,0);

% plot classification of pixels
figure
imagesc(reshape(cl1_km,[sz(1:end-1) 1]))

%% This section is only for visualization of the reconstructed image
rec1_km = zeros(length(cl1_km),size(y_beta_pca,2)); % reconstruction
% reconstruct image
for i = 1:K
    rec1_km = rec1_km + (cl1_km==i) * theta1_km{i}.mu;
end
rec1_km = reshape(rec1_km,[sz(1:end-1),size(y_beta_pca,2)]);
% plot components of the reconstructed image
for i = 1:size(y_beta_pca,2)
    subplot(1,size(y_beta_pca,2),i)
    imagesc(rec1_km(:,:,i))
    axis off;
end

%% GMM classification for option 1
[theta1_gmm, prior1_gmm,p1_gmm,samples1_gmm] = normmix_gibbs(y_beta_pca,K,Nsim,plotflag);
[~,ind1] = max(p1_gmm,[],2);
ind1 = reshape(ind1,[sz(1:end-1) 1]);

% plot classification of pixels
figure
imagesc(ind1)

%% DMRF classification for option 1
N = [0 1 0;1 0 1;0 1 0]; % 4-neighbourhood pattern
alpha0 = zeros(K-1,1);
alpha_post = 0;
iter = 1000; % number of iterations for Gibbs sampler
negLogPl = zeros(iter,1); % negative log psuedo likelihood
beta0 = zeros(iter+1,1); % vector of betas (plus one for initial beta), assuming same beta for all classes
theta1 = theta1_gmm; % initial class parameters is set to the values obtained from GMM
im = icolstack(y_beta_pca,sz(1:end-1));

z = zeros(sz(1),sz(2),K); % initial indicator image based on GMM classification
for i = 1:sz(1)
    for j = 1:sz(2)
        z(i,j,ind1(i,j)) = 1; 
    end
end

% inference for the DMRF model
for i = 1:iter
    alpha_post = mrf_gaussian_post([0; alpha0],theta1,im);
    z = mrf_sim(z,N,alpha_post,beta0(i),1);
    [~,~,Mf,~] = mrf_sim(z,N,alpha_post,beta0(i),1);
    
    colstacked_im = colstack(im);
    for j = 1:K
        ind = z(:,:,j);
        [theta1{j}.mu, theta1{j}.Sigma] = gibbs_mu_sigma(colstacked_im(ind(:) == 1,:));
    end
    
    [negLogPl(i),~,~] = mrf_negLogPL(alpha0,beta0(i),z,Mf,1e-1);
    [alpha0,beta0(i+1),acc] = gibbs_alpha_beta(alpha0,beta0(i),z,N,1e-1,1e-4);
    
    figure(20)
    image(rgbimage(z))
    drawnow
end

% plots the trajectory of the negative log psuedo likelihood
figure
plot(1:iter,negLogPl)

%% Option 2: Compute SVD directly on the data
[y1,V,P] = pca(colstack(img));
y1_pca = y1(:,[1 3 4]); % components with most information

%%
y = icolstack(y1, sz(1:2));

%study the temporal components to find those with 20s periodicity
figure
subplot(3,4,1)
semilogy(P/sum(P))
axis tight
for i=1:11
  subplot(3,4,i+1)
  plot(V(:,i))
  axis tight
  title(i)
end

figure
subplot(3,4,1)
semilogy(P/sum(P))
axis tight
for i=1:11
  subplot(3,4,i+1)
  imagesc(y(:,:,i))
  title(i)
end

%% K-means classification for option 2
[cl2_km,theta2_km] = kmeans(y1_pca,K,maxiter,0);

% plot classification of pixels
figure
imagesc(reshape(cl2_km,[87 102 1]))

%% This section is only for visualization of the reconstructed image
rec2_km = zeros(length(cl2_km),size(y1_pca,2)); % reconstruction
% reconstruct image
for i = 1:K
    rec2_km = rec2_km + (cl2_km==i) * theta2_km{i}.mu;
end

rec2_km = reshape(rec2_km,[sz(1:end-1),size(y1_pca,2)]);
%plot components of the reconstructed image
for i = 1:size(y1_pca,2)
    subplot(1,size(y1_pca,2),i)
    imagesc(rec2_km(:,:,i))
    axis off;
end

%% GMM classification for option 2
[theta2_gmm, prior2_gmm,p2_gmm,samples2_gmm] = normmix_gibbs(y1_pca,K,Nsim,plotflag);
[~,ind2] = max(p2_gmm,[],2);
ind2 = reshape(ind2,[sz(1:end-1) 1]);

% plot classification of pixels
figure
imagesc(ind2)

%% DMRF classification for option 2
N = [0 1 0;1 0 1;0 1 0]; % 4-neighbourhood pattern
alpha0 = zeros(K-1,1);
alpha_post = 0;
iter = 1000; % number of iterations for Gibbs sampler
negLogPl = zeros(iter,1); % negative log psuedo likelihood
beta0 = zeros(iter+1,1); % vector of betas (plus one for initial beta), assuming same beta for all classes
theta = theta2_gmm; % initial class parameters is set to the values obtained from GMM
im = icolstack(y1_pca,sz(1:end-1));

z = zeros(sz(1),sz(2),K); % initial indicator image based on GMM classification
for i = 1:sz(1)
    for j = 1:sz(2)
        z(i,j,ind2(i,j)) = 1; 
    end
end

% inference for the DMRF model
for i = 1:iter
    alpha_post = mrf_gaussian_post([0; alpha0],theta,im);
    z = mrf_sim(z,N,alpha_post,beta0(i),1);
    [~,~,Mf,~] = mrf_sim(z,N,alpha_post,beta0(i),1);
    
    colstacked_im = colstack(im);
    for j = 1:K
        ind = z(:,:,j);
        [theta{j}.mu, theta{j}.Sigma] = gibbs_mu_sigma(colstacked_im(ind(:) == 1,:));
    end
    
    [negLogPl(i),~,~] = mrf_negLogPL(alpha0,beta0(i),z,Mf,1e-1);
    [alpha0,beta0(i+1),acc] = gibbs_alpha_beta(alpha0,beta0(i),z,N,1e-1,1e-4);
    
    figure(30)
    image(rgbimage(z))
    drawnow
end

% plots the trajectory of the negative log psuedo likelihood
figure
plot(1:iter,negLogPl)
