function [net, info] = cnn_plankton(varargin)

run(fullfile(fileparts(mfilename('fullpath')),...
  '..', 'matlab', 'vl_setupnn.m')) ;

opts.expDir = fullfile('data','plankton-baseline') ;
opts.train.batchSize = 100 ;
opts.train.numEpochs = 80 ;
opts.train.continue = true ;
opts.train.useGpu = false ;
%opts.train.learningRate = [0.01 * ones(1, 40) 0.001 * ones(1, 20) 0.0001 * ones(1, 20)] ;
opts.train.learningRate = 0.001 ;
opts.train.expDir = opts.expDir ;
opts = vl_argparse(opts, varargin) ;


% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

imdb = getImages(opts) ;
mkdir(opts.expDir) ;

% Define a network similar to LeNet
f = 1/100 ;
net.layers = {} ;
% 1 layer
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(3,3,1,34, 'single'), ...
                           'biases', zeros(1, 34, 'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;                      
% 2 layer                      
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(5,5,34,50, 'single'),...
                           'biases', zeros(1,50,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;                       
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
% 3 layer                  
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(4,4,50,600, 'single'),...
                           'biases', zeros(1,600,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'dropout', ...
                           'rate', 0.5) ;
% 4 layer                    
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(1,1,600,121, 'single'),...
                           'biases', zeros(1,121,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;                       
net.layers{end+1} = struct('type', 'softmaxloss') ;


% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

% Take the mean out and make GPU if needed
imdb.images.data = bsxfun(@minus, imdb.images.data, mean(imdb.images.data,4));

if opts.train.useGpu
  imdb.images.data = gpuArray(imdb.images.data) ;
end
[net, info] = cnn_train(net, imdb, @getBatch, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;


% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
rrr=randi(4);
zzz = imdb.images.data(:,:,:,batch);

if rrr==1
    im=zzz;
end
if rrr==2
    im=flipdim(zzz,1);
end
if rrr==3
    im=flipdim(zzz,2);
end
if rrr==4
    imm=flipdim(zzz,1);
    im=flipdim(imm,2);
end

labels = imdb.images.labels(1,batch);


% --------------------------------------------------------------------
function imdb = getImages(opts)
% --------------------------------------------------------------------
size = 40;
load('trainImages40.mat');
load('labels.mat');
images = trainImages40;
trainImages = images(1:28000,:)';
trainImages = reshape(trainImages, size, size, []);

testImages = images(28001:end, :)';
testImages = reshape(testImages, size, size, []);

x1 = trainImages;
x2 = testImages;

y1 = labels(1:28000, 1)';
y1 = [y1];
y1 = cell2mat(y1);

y2 = labels(28001:end, 1)';
y2 = cell2mat(y2);

imdb.images.data = single(reshape(cat(3, x1, x2), size, size, 1, [])) ;

imdb.images.labels = cat(2, y1, y2) ;
imdb.images.set = [ones(1,numel(y1)) 3*ones(1,numel(y2))] ;
imdb.meta.sets = {'train', 'val', 'test'} ;

classes = cell(1,121);
for i = 1:121
    classes{1,i} = i;
end
imdb.meta.classes = classes;
