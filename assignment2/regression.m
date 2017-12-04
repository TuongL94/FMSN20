%% regression
B = Insurance(:,3:end);
[beta beta_ci] = regress(Y,B)
B = B(:,[2 4 5 6 9 10]); %choosing significant covariates