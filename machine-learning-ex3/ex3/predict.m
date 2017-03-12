function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% Add intercept term to x and X_test
X = [ones(m, 1) X];
% Theta1 = [ones(1, size(Theta1, 2)) Theta1'];
% Theta1 = [ones(size(Theta1, 1), 1) Theta];
% Theta2 = [ones(size(Theta2, 1), 1) Theta2];

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

Z_of_2 = Theta1 * X';

a_of_2 = sigmoid(Z_of_2)';

a_of_2 = [ones(size(a_of_2), 1) a_of_2]';

z_of_3 = Theta2 * a_of_2; 

a_of_3 = sigmoid(z_of_3);


[max_value, p]  = max(a_of_3', [], 2);
% p = max(sigmoid(X * num_labels),[],2) >= 0.5;






% =========================================================================


end
