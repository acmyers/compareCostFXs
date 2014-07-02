function [final_wealth] = costFunctionDRW(theta, X, y)
%COSTFUNCTIONDRW computes the mutual information between a student's marginal
%   probability of acceptance/rejection and our predicted probabilities. The
%   function returns the final wealth on initial pot of $1, therefore the function 
%   should be referred to as a "benefit function" or "objective function".

% Initialize useful values
m = length(y); 
% Marginal probability of acceptance
marg_pA = sum(y)/m;
% Marginal probability of rejection
marg_pR = 1 - marg_pA;

% Return the following variables 
initial_wealth = 1; % initial wealth = $1
final_wealth_individual = zeros(m);
final_wealth = 0;

% =============================================================

pred = sigmoid(X*theta);

final_wealth_individual = (pred/marg_pA).*y + ((1-pred)/marg_pR).*(1-y);
final_wealth = prod(final_wealth_individual);


% =============================================================

end
