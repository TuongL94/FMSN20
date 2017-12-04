%load data
load HA2_Brazil

%available covariates
disp(InsuranceNames)
%For now we ignore the average cost of collisions (avg.cost.col)

%For the number of events we want to model
%  no.collisions ~ Po( no.vehicles * exp(log_risk) )
%thus 
%  log_risk ~ log(no.collisions/no.vehicles)
%compute the risk of collisions
log_risk = log(Insurance(:,2)./Insurance(:,1));

%plot all data
figure
subplot(3,3,1)
plotMap(BrazilMap, log_risk, 'none') %third element is ege colour, no edge
colorbar
axis tight
title('log risk')
for i=1:8
  subplot(3,3,i+1)
  plotMap(BrazilMap, Insurance(:,i+3), 'none')
  colorbar
  axis tight
  title(InsuranceNames{i+3})
end

%and compare to covariates
figure
for i=4:length(InsuranceNames)
	subplot(3,3,i-3)
	plot(Insurance(:,i),log_risk, '.')
	axis tight
	title( InsuranceNames{i} )
end

%illustrate sparsity (matrices already have near-optimal order)
figure
subplot(121)
spy(G)
subplot(122)
spy(G*G)