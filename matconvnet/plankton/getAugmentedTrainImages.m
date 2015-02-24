load('../../imageData/trainImages28.mat'); % Loads train images
load('weakClasses.mat'); % Loads weak classes of images
load('labels.mat'); % Loads labels of images
images = trainImages28;

% Sets parameters
[imagesNum, ~] = size(images);  % Total number of images to read
[weakClassesNum, ~] = size(weakClasses); % Total number of weak classes
imgSize = 28;  % Desired size of images
rotationsPerImgNum = 3; % Number of rotations applied to single image

% Calculates number of rotations
fprintf('Calculating number of rotations... ');
rotatedImagesNum = 0;
numByClasses = zeros(weakClassesNum, 1);
for i = 1 : weakClassesNum
    ids = find(labels == weakClasses(i));
    [num, ~] = size(ids);

    numByClasses(i) = num;
    rotatedImagesNum = rotatedImagesNum + (num * rotationsPerImgNum);
end
fprintf('Done.\n');

% Rotates images
fprintf('Rotating images... ');
rotatedImages = zeros(rotatedImagesNum, imgSize * imgSize);
randomIds = zeros(rotatedImagesNum, 1);
id = 1;
for i = 1 : weakClassesNum
    for j = 1 : numByClasses(i)
        img = images(i, :);

        for k = 1 : rotationsPerImgNum % Do specified numer of rotations
            rotatedImg = imrotate(img, k * (360 / (rotationsPerImgNum + 1)));

            randomIds(id) = randi(imagesNum + rotatedImagesNum - 1);
            rotatedImages(id, :) = rotatedImg(:);
            id = id + 1;
        end
    end

    % fprintf('%i / %i classes\n', i, weakClassesNum);
end
fprintf('Done.\n');

% Merges images
fprintf('Merging images... ');
finalImages = zeros(imagesNum + rotatedImagesNum, imgSize * imgSize);
for i = 1 : rotatedImagesNum
    finalImages(randomIds(i), :) = rotatedImages(i);
end

j = 1;
for i = 1 : imagesNum
    if any(finalImages(i, :)) == 0 % Writes original images between rotated ones
        finalImages(i, :) = images(j, :);
        j = j + 1;
    end
end
fprintf('Done.\n');

% Saves images
augmentedTrainImages28 = finalImages;
save(strcat('../../imageData/augmentedTrainImages', int2str(imgSize)), 'augmentedTrainImages28', '-v7.3');
