function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

H = sigmoid(X * theta) % Hypothesis

%Theta(1) is always 0 since it won't be regularized
%Start regularized from Theta(2) onwards
temptheta = zeros(size(theta))
temptheta(2:size(theta,1),:) = theta(2:size(theta,1),:)

% Regularized cost function for Logistic Regression
J = 1/m * sum ( -y.*log(H) - (1-y).*log(1-H) ) + lambda/(2*m) * sum(temptheta.^2)

% Regularized gradient 
grad = 1/m * sum ( repmat(H - y,1,size(X,2)) .* X ) + (lambda/m * temptheta)'



% =============================================================

end
