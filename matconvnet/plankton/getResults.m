vl_setupnn;

size = 40;
net.layers{end}.type = 'softmax';
submission = zeros(130400, 121);

% Read first set of test images
load('testImages40one');
planks = testImages40one;
clear testImages40one;
planks = planks';
planks = reshape(planks, size, size, 1, []);
planks = bsxfun(@minus, planks, mean(planks,4));
planks = single(planks);

% Test first data set
for i = 1 : 60000
    res = vl_simplenn(net, planks(:,:,1,i));
    for j = 1:121
        submission(i,j) = mean(mean(res(13).x(:,:,j)));
    end
    if mod(i, 100) == 0
        fprintf('%i / %i \n', i, 130400);
    end
end
clear planks;

% Load second of test images
load('testImages40two');
planks = testImages40two;
clear testImages40two;
planks = planks';
planks = reshape(planks, size, size, 1,[]);
planks = bsxfun(@minus, planks, mean(planks,4));
planks = single(planks);

% Test second data set
for i = 1 : 70400
    res = vl_simplenn(net, planks(:,:,1,i));
    i_current = i + 60000;
    for j = 1:121
        submission(i_current, j) = mean(mean(res(13).x(:,:,j)));
    end
    if mod(i, 100) == 0
        fprintf('%i / %i \n', i_current, 130400);
    end
end

% Save submission
save('submission', 'submission', '-v7.3');
