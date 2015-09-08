function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    %hypothesis=theta(1)+theta(2).*X(:,2)
    
    %temp0=theta(1) - alpha * (1/m) * sum(hypothesis-y)
    %temp1=theta(2) - alpha * (1/m) * sum((hypothesis-y) .* X(:,2))   

    %theta(1) = temp0
    %theta(2) = temp1
    
    n = size(X,2) % number of features (include the all ones in first column)
    
    %setup hypothesis
    hypothesis = zeros(size(X,1),1)
    for i = 1:n
        hypothesis = hypothesis + theta(i).*X(:,i)
    end

    theta = theta - ( alpha/m * sum(repmat((hypothesis-y),1,3) .* X) )' 









    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
