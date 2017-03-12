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
num_iters = length(theta);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



Z = X *  theta;

logOfH = log(sigmoid(Z));
logOfOneMinusH = log(1-sigmoid(Z));

lhs = ((-1).* y') * logOfH; 
rhs = ((1-y)'*logOfOneMinusH);
costFunction = (1/m)*(lhs - rhs);

regularizationPartForJ = (lambda / (2 * m )) * sum(theta(2: size(theta,1)))
J = costFunction + regularizationPartForJ;

for iter = 1:num_iters
   if iter== 1
      grad(iter) = ((sum((sigmoid(Z) - y)'* X(:, iter)) / m)  ) ;
   else
      grad(iter) = ((sum((sigmoid(Z) - y)'* X(:, iter)) / m) + (lambda / m) * theta(iter) ) ;
   end
   
end 


% =============================================================

end
