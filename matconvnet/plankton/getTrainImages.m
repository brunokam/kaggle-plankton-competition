load('labelledImages.mat')  % Load labels
parpool(4);  % Set workers for paraller computations

% Set parameters
sizeImg = 28;  % Size of images desired
imagesNumber = 30336;  % Total number of images to read

% Read images
images = zeros(imagesNumber, sizeImg * sizeImg);  % Initialize images matrix
for i = 1 : imagesNumber
    img = imresize(imread(labelledImages{i, 1}), [sizeImg sizeImg]);
    images(i, :) = img(:);
    if mod(i, 100) == 0
        fprintf('%i / %i \n', i, imagesNumber);
    end
end

% Save images
trainImages28 = images;
mkdir('../../imageData')
save(strcat('../../imageData/trainImages', int2str(sizeImg)), 'trainImages28', '-v7.3');
clear images;
delete(gcp);
