load('testNames.mat')  % Load file names
parpool(4);  % Set workers for paraller computations

% Set parameters
size = 40;  % Size of images desired
imagesNumber = 60000;  % Total number of images to read

% Read images
images = zeros(imagesNumber, size * size);  % Initialize images matrix
parfor i = 1 : imagesNumber
    img = imresize(imread(testNames{i,1}), [size size]);
    images(i,:) = img(:);
    if mod(i, 100) == 0
        fprintf('%i / %i \n', i, imagesNumber);
    end
end

% Save images
testImages40one = images;
mkdir('../../imageData')
save(strcat('../../imageData/testImages', int2str(size), 'one'), 'testImages40one', '-v7.3');
clear images;


% Set parameters for second set
imagesNumberTwo = 70400;  % Total number of images to read

% Read second set
images = zeros(imagesNumberTwo, size * size);  % Initialize images matrix
parfor i = (imagesNumber + 1) : imagesNumberTwo
    img = imresize(imread(testNames{i,1}), [size size]);
    images(i - imagesNumber,:) = img(:);
    if mod(i, 100) == 0
        fprintf('%i / %i \n', i, imagesNumberTwo);
    end
end

% Save second set
testImages40two = images;
mkdir('../../imageData')
save(strcat('../../imageData/testImages', int2str(size), 'two'), 'testImages40two', '-v7.3');
clear images;
