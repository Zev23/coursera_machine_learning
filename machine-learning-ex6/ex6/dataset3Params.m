function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

Steps = [ 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

results = [];

for sC = Steps
    for ssigma = Steps
        model = svmTrain(X, y, sC, @(x1, x2) gaussianKernel(x1, x2, ssigma));
        predictions = svmPredict(model, Xval);
        predicterror = mean(double(predictions ~= yval));
        results = [results; predicterror sC ssigma];
    end
end

min_error = min(results(:,1));
row_index = find(results(:,1) == min_error);
C = results(row_index, 2);
sigma = results(row_index, 3);

% =========================================================================

end
