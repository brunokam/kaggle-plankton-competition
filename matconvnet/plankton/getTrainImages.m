load('labelledImages.mat')  % Load labels
parpool(4);  % Set workers for paraller computations

% Set parameters
size = 40;  % Size of images desired
imagesNumber = 30336;  % Total number of images to read

% Read images
images = zeros(imagesNumber, size * size);  % Initialize images matrix
parfor i = 1 : imagesNumber
    img = imresize(imread(labelledImages{i,1}), [size size]);
    images(i,:) = img(:);
    if mod(i, 100) == 0
        fprintf('%i / %i \n', i, imagesNumber);
    end
end

% Save images
trainImages40 = images;
mkdir('../../imageData')
save(strcat('../../imageData/trainImages', int2str(size)), 'trainImages40', '-v7.3'); 
clear images;
