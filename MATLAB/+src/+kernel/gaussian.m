function K = gaussian(X, Y, l, f, s)
%% function K = gaussian(X, Y, l, f, s)
% Compute the Gaussian kernel matrix between X and Y
%
% Inputs:
%   X: m x d matrix of m points in d-dimensional space
%   Y: n x d matrix of n points in d-dimensional space
%   l: length scale parameter
%   f: signal variance parameter
%   s: noise variance parameter
%
% Outputs:
%   K: m x n matrix of Gaussian kernel values between X and Y 
%      K(i,j) = f^2 * exp(-||X(i,:) - Y(j,:)||^2 / (2 * l^2)) + s^2 * δ(i,j)
%
% Example:
%   X = rand(10, 3);
%   Y = rand(5, 3);
%   K = src.kernel.gaussian(X, Y, 1.0, 1.0, 0.1);

   XY = X * Y';
   XX = sum(X.^2, 2);
   YY = sum(Y.^2, 2);

   D2 = bsxfun(@plus, bsxfun(@plus, -2 * XY, XX), YY');
   K = f^2*exp(-D2 / (2*l^2)) + s^2*eye(size(X,1), size(Y,1));

end