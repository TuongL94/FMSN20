function stop = outfun(theta,optimValues, state)
%OUTFUN Summary of this function goes here
%   Detailed explanation goes here
stop = false;
fprintf('minimum point (log(theta)): %f %f \n ',theta(1),theta(2))
fprintf('fval: %11.4e \n ',optimValues.fval)
end

