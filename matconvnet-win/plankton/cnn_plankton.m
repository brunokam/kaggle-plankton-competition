function [net, info] = cnn_mnist(varargin)
% CNN_MNIST  Demonstrated MatConNet on MNIST

run(fullfile(fileparts(mfilename('fullpath')),...
  '..', 'matlab', 'vl_setupnn.m')) ;

opts.dataDir = fullfile('data','mnist') ;
opts.expDir = fullfile('data','mnist-baseline') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train.batchSize = 100 ;
opts.train.numEpochs = 80 ;
opts.train.continue = true ;
opts.train.useGpu = false ;
opts.train.learningRate = 0.001 ;
opts.train.expDir = opts.expDir ;
opts = vl_argparse(opts, varargin) ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getMnistImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% Define a network similar to LeNet
f=1/100 ;
net.layers = {} ;
%1
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
                       
%2                       
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
                       
 %3                      
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(4,4,50,600, 'single'),...
                           'biases', zeros(1,600,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'dropout', ...
                           'rate', 0.5) ;
 %4                      
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
function imdb = getMnistImdb(opts)
% --------------------------------------------------------------------
files = {'train-images-idx3-ubyte', ...
         'train-labels-idx1-ubyte', ...
         't10k-images-idx3-ubyte', ...
         't10k-labels-idx1-ubyte'} ;

mkdir(opts.dataDir) ;
% for i=1:4
%   if ~exist(fullfile(opts.dataDir, files{i}), 'file')
%     url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{i}) ;
%     fprintf('downloading %s\n', url) ;
%     gunzip(url, opts.dataDir) ;
%   end
% end


load('planktonTrainingImages40.mat');
load('labs');

imageData=planktonTrainingImages40;
images = imageData(1:28000,:)'; %loadMNISTImages('../common/train-images-idx3-ubyte');
images = reshape(images,40,40,[]);

%dataaug
% flup=flipud(images(:,:,:));
% fllr=fliplr(images(:,:,:));
% fluplr=fliplr(flup(:,:,:));

% images45=images;
% images45(images45==0)=1;
% images45=imrotate(images45,45);
% images45=images45(7:34,7:34,:);
% images45(images45==0)=255;
% flup45=flipud(images45(:,:,:));
% fllr45=fliplr(images45(:,:,:));
% fluplr45=fliplr(flup45(:,:,:));


%concat augmented images
% images = cat(3, images, flup);
% clear flup;
% images = cat(3,images,fllr);
% clear fllr;
% images = cat(3,images,fluplr);
% clear fluplr;
%images = cat(3,images,images45,flup45,fllr45,fluplr45);


% clear fluplr;
% clear flup45;
% clear fllr45;
% clear fluplr45;


testImages = imageData(28001:end,:)';
testImages = reshape(testImages,40,40,[]);

x1=images;
x2=testImages;

y1=labs(1:28000,1)';
y1=[y1];
y1=cell2mat(y1);


y2=labs(28001:end,1)';
y2 = cell2mat(y2);


imdb.images.data = single(reshape(cat(3, x1, x2),40,40,1,[])) ;

imdb.images.labels = cat(2, y1, y2) ;
imdb.images.set = [ones(1,numel(y1)) 3*ones(1,numel(y2))] ;
imdb.meta.sets = {'train', 'val', 'test'} ;
aa=cell(1,121);
for i = 1:121
    aa{1,i}=i;
end
imdb.meta.classes = aa;%arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;
