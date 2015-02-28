function getAugmentedTrainImages()
    load('../../imageData/trainImages28.mat'); % Loads train images
    load('weakClasses.mat'); % Loads weak classes of images
    load('labels.mat'); % Loads labels of images
    originalImages = trainImages28;
    originalLabels = labels;

    % Sets parameters
    [originalImageNum, ~] = size(originalImages);  % Total number of images to read
    [weakClassNum, ~] = size(weakClasses); % Total number of weak classes
    imgSize = 28;  % Desired size of images

    params.originalImageNum = originalImageNum;
    params.imgSize = imgSize;
    params.originalImages = originalImages;
    params.originalLabels = originalLabels;
    params.weakClassNum = weakClassNum;
    params.weakClasses = weakClasses;

    % Rotates images
    [rotatedImageNum, rotatedImageClassIds, rotatedRandomIds, rotatedImages] = rotate(params);

    % Translates images
    [translatedImageNum, translatedImageClassIds, translatedRandomIds, translatedImages] = translate(params);


    % Merges rotated images
    params.newImageNum = rotatedImageNum;
    params.newImageClassIds = rotatedImageClassIds;
    params.randomIds = rotatedRandomIds;
    params.newImages = rotatedImages;

    [mergedImages, mergedLabels] = merge(params);


    % Merges translated images
    params.originalImageNum = originalImageNum + rotatedImageNum;
    params.originalImages = mergedImages;
    params.originalLabels = mergedLabels;

    params.newImageNum = translatedImageNum;
    params.newImageClassIds = translatedImageClassIds;
    params.randomIds = translatedRandomIds;
    params.newImages = translatedImages;

    [mergedImages, mergedLabels] = merge(params);

    % Saves images
    fprintf('Saving images... ');
    augmentedTrainImages28 = mergedImages;
    augmentedLabels = mergedLabels;
    save(strcat('../../imageData/augmentedTrainImages', int2str(imgSize)), 'augmentedTrainImages28', '-v7.3');
    save('augmentedLabels', 'augmentedLabels', '-v7.3');
    fprintf('Done.\n');
end



function [mergedImages, mergedLabels] = merge(params)
    % Merges images
    fprintf('Merging images... ');
    mergedImages = zeros(params.originalImageNum + params.newImageNum, params.imgSize * params.imgSize);
    mergedLabels = zeros(params.originalImageNum + params.newImageNum, 1);
    for i = 1 : params.newImageNum
        mergedImages(params.randomIds(i), :) = params.newImages(i, :);
        mergedLabels(params.randomIds(i)) = params.newImageClassIds(i);
    end

    j = 1;
    for i = 1 : (params.originalImageNum + params.newImageNum)
        if any(params.randomIds == i) == 0 % Writes original images between rotated ones
            mergedImages(i, :) = params.originalImages(j, :);
            mergedLabels(i) = params.originalLabels(j);
            j = j + 1;
        end
    end
    fprintf('Done.\n');
end



function [newImageNum, newImageClassIds, randomIds, newImages] = rotate(params)
    augmentationFactor = 3; % Number of different images retrieved from a single image

    % Calculates number of rotations
    fprintf('Calculating number of images to rotate... ');
    newImageNum = 0;
    numByWeakClass = zeros(params.weakClassNum, 1);
    for i = 1 : params.weakClassNum
        classImageIds = find(params.originalLabels == params.weakClasses(i));
        [num, ~] = size(classImageIds);

        numByWeakClass(i) = num;
        newImageNum = newImageNum + (num * augmentationFactor);
    end

    weakImageIds = zeros(sum(numByWeakClass), 2);
    k = 1;
    for i = 1 : params.weakClassNum
        classId = params.weakClasses(i);
        classImageIds = find(params.originalLabels == classId);

        for j = 1 : numByWeakClass(i)
            weakImageIds(k, :) = [classImageIds(j), classId];
            k = k + 1;
        end
    end
    fprintf('Done.\n');

    fprintf('Rotating images... ');
    newImages = zeros(newImageNum, params.imgSize * params.imgSize);
    newImageClassIds = zeros(newImageNum, 1);
    randomIds = zeros(newImageNum, 1);
    k = 1;
    m = 1;
    for i = 1 : params.weakClassNum
        for j = 1 : numByWeakClass(i)
            img = reshape(params.originalImages(weakImageIds(k, 1), :), [params.imgSize, params.imgSize]);

            for l = 1 : augmentationFactor % Do specified numer of rotations
                rotatedImg = imrotate(img, l * (360 / (augmentationFactor + 1)));

                rnd = randi(params.originalImageNum + newImageNum - 1);
                while any(randomIds == rnd) == 1
                    rnd = randi(params.originalImageNum + newImageNum - 1);
                end

                randomIds(m) = rnd;
                newImages(m, :) = reshape(rotatedImg, [1, params.imgSize * params.imgSize]);
                newImageClassIds(m) = weakImageIds(k, 2);
                m = m + 1;
            end

            k = k + 1;
        end
    end
    fprintf('Done.\n');
end




function [newImageNum, newImageClassIds, randomIds, newImages] = translate(params)
    augmentationFactor = 4; % Maximal number of different images retrieved from a single image
    threshold = floor(params.imgSize / 5); % Minimal number of pixels to do translation

    % Calculates number of rotations
    fprintf('Calculating number images to translate... ');
    numByWeakClass = zeros(params.weakClassNum, 1);
    for i = 1 : params.weakClassNum
        classImageIds = find(params.originalLabels == params.weakClasses(i));
        [num, ~] = size(classImageIds);

        numByWeakClass(i) = num;
    end

    weakImageIds = zeros(sum(numByWeakClass), 2);
    k = 1;
    for i = 1 : params.weakClassNum
        classId = params.weakClasses(i);
        classImageIds = find(params.originalLabels == classId);

        for j = 1 : numByWeakClass(i)
            weakImageIds(k, :) = [classImageIds(j), classId];
            k = k + 1;
        end
    end

    newImageNum = 0;
    borders = zeros(sum(numByWeakClass), 4);
    k = 1;
    for i = 1 : params.weakClassNum
        for j = 1 : numByWeakClass(i)
            id = weakImageIds(k, 1);
            img = reshape(params.originalImages(id, :), [params.imgSize, params.imgSize]);
            brd = scanImage(img, params.imgSize); % Retrieves borders of a single image

            num = 0;
            for l = 1 : augmentationFactor
                switch l
                    case 1
                        num = num + any([brd(4), brd(1)] >= threshold);
                    case 2
                        num = num + any([brd(2), brd(1)] >= threshold);
                    case 3
                        num = num + any([brd(2), brd(3)] >= threshold);
                    otherwise
                        num = num + any([brd(4), brd(3)] >= threshold);
                end
            end

            newImageNum = newImageNum + num; % Adds 4 to the sum and subtracts the non-existing borders
            borders(k, :) = brd(:);
            k = k + 1;
        end
    end
    fprintf('Done.\n');

    fprintf('Translating images... ');
    newImages = zeros(newImageNum, params.imgSize * params.imgSize);
    newImageClassIds = zeros(newImageNum, 1);
    randomIds = zeros(newImageNum, 1);
    k = 1;
    m = 1;
    for i = 1 : params.weakClassNum
        for j = 1 : numByWeakClass(i)
            img = params.originalImages(weakImageIds(k, 1), :);
            img = reshape(img, [params.imgSize, params.imgSize]);

            for l = 1 : augmentationFactor
                border = [0, 0];
                shift = [0, 0];
                switch l
                    case 1
                        border = [borders(k, 4), borders(k, 1)];
                        shift = [-border(1), -border(2)];
                    case 2
                        border = [borders(k, 2), borders(k, 1)];
                        shift = [border(1), -border(2)];
                    case 3
                        border = [borders(k, 2), borders(k, 3)];
                        shift = [border(1), border(2)];
                    otherwise
                        border = [borders(k, 4), borders(k, 3)];
                        shift = [-border(1), border(2)];
                end

                if any(border >= threshold)
                    translatedImg = imtranslate(img, shift);

                    switch l
                        case 1
                            translatedImg((params.imgSize - border(2) + 1):params.imgSize, :) = 255;
                            translatedImg(:, (params.imgSize - border(1) + 1):params.imgSize) = 255;
                        case 2
                            translatedImg((params.imgSize - border(2) + 1):params.imgSize, :) = 255;
                            translatedImg(:, 1:border(1)) = 255;
                        case 3
                            translatedImg(1:border(2), :) = 255;
                            translatedImg(:, 1:border(1)) = 255;
                        otherwise
                            translatedImg(1:border(2), :) = 255;
                            translatedImg(:, (params.imgSize - border(1) + 1):params.imgSize) = 255;
                    end

                    %subplot(1, 2, 1), imshow(uint8(img));
                    %subplot(1, 2, 2), imshow(uint8(reshape(translatedImg, [28, 28])));
                    %pause(2);
                
                    rnd = randi(params.originalImageNum + newImageNum - 1);
                    while any(randomIds == rnd) == 1 % Chooses random position for the new image
                        rnd = randi(params.originalImageNum + newImageNum - 1);
                    end
                    
                    translatedImg = reshape(translatedImg, [1, params.imgSize * params.imgSize]);

                    randomIds(m) = rnd;
                    newImages(m, :) = translatedImg(:);
                    newImageClassIds(m) = weakImageIds(k, 2);
                    m = m + 1;
                end
            end

            k = k + 1;
        end
    end
    fprintf('Done.\n');
end

function result = scanImage(originalImg, imgSize)
    threshold = 240;
    filteredImg = originalImg > threshold;

    result = zeros(1, 4);

    % Scanning top
    i = 1;
    while any(filteredImg(i, :) == 0) == 0
        i = i + 1;
        result(1) = result(1) + 1;
    end

    % Scanning right
    i = imgSize;
    while any(filteredImg(:, i) == 0) == 0
        i = i - 1;
        result(2) = result(2) + 1;
    end

    % Scanning bottom
    i = imgSize;
    while any(filteredImg(i, :) == 0) == 0
        i = i - 1;
        result(3) = result(3) + 1;
    end

    % Scanning left
    i = 1;
    while any(filteredImg(:, i) == 0) == 0
        i = i + 1;
        result(4) = result(4) + 1;
    end
end
