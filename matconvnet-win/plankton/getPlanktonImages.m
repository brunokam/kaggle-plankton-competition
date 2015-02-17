




siz=28; %size of images desired
imgs=zeros(44515,784);
parfor i = 1:44515
    a=imresize(imread(newNames{i,1}), [siz siz]);
    imgs(i,:)=a(:);
end
planktonTestImages28New=imgs;
save('planktonTestImages28New','planktonTestImages28New');  