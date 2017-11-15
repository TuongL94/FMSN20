function [ ] = plotBorder(Border)
%PLOTBORDER Summary of this function goes here
%   Detailed explanation goes here
axis xy tight; hold on
plot(Border(:,1),Border(:,2),'k',...
  Border(1034:1078,1),Border(1034:1078,2),'r')
hold off;
end

