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
    classImageIds = find(labels == weakClasses(i));
    [num, ~] = size(classImageIds);

    numByClasses(i) = num;
    rotatedImagesNum = rotatedImagesNum + (num * rotationsPerImgNum);
end

ids = zeros(sum(numByClasses), 2);
k = 1;
for i = 1 : weakClassesNum
    classId = weakClasses(i);
    classImageIds = find(labels == classId);

    for j = 1 : numByClasses(i)
        ids(k, :) = [classImageIds(j), classId];
        k = k + 1;
    end
end
fprintf('Done.\n');

% Rotates images
fprintf('Rotating images... ');
rotatedImages = zeros(rotatedImagesNum, imgSize * imgSize);
rotatedImageClassIds = zeros(rotatedImagesNum, 1);
randomIds = zeros(rotatedImagesNum, 1);
k = 1;
m = 1;
for i = 1 : weakClassesNum
    for j = 1 : numByClasses(i)
        img = images(ids(k, 1), :);

        for l = 1 : rotationsPerImgNum % Do specified numer of rotations
            rotatedImg = imrotate(img, l * (360 / (rotationsPerImgNum + 1)));

            rnd = randi(imagesNum + rotatedImagesNum - 1);
            while any(randomIds == rnd) == 1
                rnd = randi(imagesNum + rotatedImagesNum - 1);
            end

            randomIds(m) = rnd;
            rotatedImages(m, :) = rotatedImg(:);
            rotatedImageClassIds(m) = ids(k, 2);
            m = m + 1;
        end

        k = k + 1;
    end

    % fprintf('%i / %i classes\n', i, weakClassesNum);
end
fprintf('Done.\n');

% Merges images
fprintf('Merging images... ');
finalImages = zeros(imagesNum + rotatedImagesNum, imgSize * imgSize);
finalLabels = zeros(imagesNum + rotatedImagesNum, 1);
for i = 1 : rotatedImagesNum
    finalImages(randomIds(i), :) = rotatedImages(i, :);
    finalLabels(randomIds(i)) = rotatedImageClassIds(i);
end

j = 1;
for i = 1 : (imagesNum + rotatedImagesNum)
    if any(randomIds == i) == 0 % Writes original images between rotated ones
        finalImages(i, :) = images(j, :);
        finalLabels(i) = labels(j);
        j = j + 1;
    end
end
fprintf('Done.\n');

% Saves images
fprintf('Saving images... ');
augmentedTrainImages28 = finalImages;
augmentedLabels = finalLabels;
save(strcat('../../imageData/augmentedTrainImages', int2str(imgSize)), 'augmentedTrainImages28', '-v7.3');
save('augmentedLabels', 'augmentedLabels', '-v7.3');
fprintf('Done.\n');
