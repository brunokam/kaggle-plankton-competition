



parpool(2);
siz=40; %size of images desired
imgs=zeros(60000,1600);
parfor i = 1:60000
    a=imresize(imread(testNames{i,1}), [siz siz]);
    imgs(i,:)=a(:);
    i
end
planktonTestImages40one=imgs;
save('planktonTestImages40one','planktonTestImages40one','-v7.3'); 
clear planktonTestImages40one;

siz=40; %size of images desired
imgs=zeros(70400,1600);
parfor i = 60001:130400
    a=imresize(imread(testNames{i,1}), [siz siz]);
    imgs(i-60000,:)=a(:);
    i
end
planktonTestImages40two=imgs;
save('planktonTestImages40two','planktonTestImages40two','-v7.3');  

