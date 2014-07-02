function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% Return the following variables 
g = zeros(size(z));

% =============================================================

g = 1./(1+exp(-z));


% =============================================================

end
