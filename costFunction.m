function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize useful values
m = length(y); 

% Return the following variables
J = 0;
grad = zeros(size(theta));

% ============================================================

pred = sigmoid(X*theta); 

J = -(1/m)*(log(pred)'*y + log(1 - pred)'*(1 - y)); 

grad = (1/m)*X'*(pred - y); 
                         

% =============================================================

end
