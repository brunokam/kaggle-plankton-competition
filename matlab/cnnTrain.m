clear all;
%% Convolution Neural Network Exercise
parpool(3);
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started in building a single.
%  layer convolutional nerual network. In this exercise, you will only
%  need to modify cnnCost.m and cnnminFuncSGD.m. You will not need to 
%  modify this file.

%%======================================================================
%% STEP 0: Initialize Parameters and Load Data
%  Here we initialize some parameters used for the exercise.

% Configuration
imageDim=25
numClasses = 121;  % Number of classes (MNIST images fall into 10 classes)
filterDim =6;% Filter size for conv layer
numFilters = 20;   % Number of filters for conv layer
poolDim = 2;      % Pooling dimension, (should divide imageDim-filterDim+1)

% Load MNIST Train
load('imageLabels.mat')
load('planktonTrainingImages25.mat')
load('planktonImageRatio')


% 
imageData=planktonTrainingImages25;
labelData=imageLabels;




%pre processing
imageData = whiten(imageData);
%sd=zca2(sd);

images = imageData(1:27000,:)'; %loadMNISTImages('../common/train-images-idx3-ubyte');
images = reshape(images,imageDim,imageDim,[]);
imageRatio=planktonImageRatio;

labels = labelData(1:27000,1); %loadMNISTLabels('../common/train-labels-idx1-ubyte');
%labels(labels==0) = 2; % Remap 0 to 10

% %data augmentation flip 90 180 270
% flup=flipud(images(:,:,:));
% fllr=fliplr(images(:,:,:));
% images = cat(3, images, flup,fllr,flipud(fllr(:,:,:)));
% imageRatio=[planktonImageRatio;planktonImageRatio;planktonImageRatio;planktonImageRatio];
% 
% %labels for augemnted data
% labels=[labels;labels;labels;labels];


% Initialize Parameters
theta = cnnInitParams(imageDim,filterDim,numFilters,poolDim,numClasses);
save('theta','theta');
%%======================================================================
%% STEP 1: Implement convNet Objective
%  Implement the function cnnCost.m.

%%======================================================================
%% STEP 2: Gradient Check
%  Use the file computeNumericalGradient.m to check the gradient
%  calculation for your cnnCost.m function.  You may need to add the
%  appropriate path or copy the file to this directory.

% DEBUG=false;  % set this to true to check gradient
% if DEBUG
%     % To speed up gradient checking, we will use a reduced network and
%     % a debugging data set
%     db_numFilters = 2;
%     db_filterDim = 9;
%     db_poolDim = 5;
%     db_images = images(:,:,1:10);
%     db_labels = labels(1:10);
%     db_theta = cnnInitParams(imageDim,db_filterDim,db_numFilters,...
%                 db_poolDim,numClasses);
%     
%     [cost grad] = cnnCost(db_theta,db_images,db_labels,numClasses,...
%                                 db_filterDim,db_numFilters,db_poolDim);
%     
% 
%     % Check gradients
%     numGrad = computeNumericalGradient( @(x) cnnCost(x,db_images,...
%                                 db_labels,numClasses,db_filterDim,...
%                                 db_numFilters,db_poolDim), db_theta);
%  
%     % Use this to visually compare the gradients side by side
%     disp([numGrad grad]);
%     
%     diff = norm(numGrad-grad)/norm(numGrad+grad);
%     % Should be small. In our implementation, these values are usually 
%     % less than 1e-9.
%     disp(diff); 
%  
%     assert(diff < 1e-9,...
%         'Difference too large. Check your gradient computation again');
%     
% end;

%%======================================================================
%% STEP 3: Learn Parameters
%  Implement minFuncSGD.m, then train the model.

options.epochs = 8;
options.minibatch = 1080;
options.alpha = 0.1;
options.momentum = .95;

opttheta = minFuncSGD(@(x,y,z,zz) cnnCost(x,y,z,zz,numClasses,filterDim,...
                      numFilters,poolDim),theta,images,labels,options,imageRatio);

%%======================================================================
%% STEP 4: Test
%  Test the performance of the trained model using the MNIST test set. Your
%  accuracy should be above 97% after 3 epochs of training

testImages = imageData(27001:end,:)';
testImages = reshape(testImages,imageDim,imageDim,[]);
testLabels = labelData(27001:end,1);
testRatio = planktonImageRatio(27001:end);
%testLabels(testLabels==0) = 2; % Remap 0 to 10


[~,cost,preds]=cnnCost(opttheta,testImages,testLabels,testRatio,numClasses,...
                filterDim,numFilters,poolDim,true);

acc = sum(preds==testLabels)/length(preds);

%print accuracy
fprintf('Accuracy is %f\n',acc);
save('acc','acc');

