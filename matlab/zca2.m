function [Z] = zca2(x)
epsilon = 1e-4;
% You should be able to use the code from your PCA/ZCA exercise
% Retain all of the components from the ZCA transform (i.e. do not do
% dimensionality reduction)
%%% YOUR CODE HERE %%%

% x is the input patch data of size
% z is the ZCA transformed data. The dimenison of z = x.

%%================================================================
%% Step 0b: Zero-mean the data (by row)
%  You can make use of the mean and repmat/bsxfun functions.
av = mean(x,2);
x=bsxfun(@minus,x,av);

%%================================================================
%% Step 1a: Implement PCA to obtain xRot
%  Implement PCA to obtain xRot, the matrix in which the data is expressed
%  with respect to the eigenbasis of sigma, which is the matrix U.
sigma = x*x';
[U,S,V] = svd(sigma);
xRot=U'*x;


%%================================================================
%% Step 4a: Implement PCA with whitening and regularisation
%  Implement PCA with whitening and regularisation to produce the matrix
%  xPCAWhite. 
eigs=sum(S);
num = xRot;
den= sqrt(eigs+epsilon);
xPCAWhite = bsxfun(@rdivide,num,den');

%%================================================================
%% Step 5: Implement ZCA whitening
%  Now implement ZCA whitening to produce the matrix xZCAWhite. 
%  Visualise the data and compare it to the raw data. You should observe
%  that whitening results in, among other things, enhanced edges.

%%% YOUR CODE HERE %%%

Z=U*xPCAWhite;

