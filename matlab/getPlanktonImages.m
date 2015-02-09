
siz=40; %size of images desired
load('labelledImages')
imgs=zeros(30336,1600);
for i = 1:30336
    a=imresize(imread(labelledImages{i,1}), [siz siz]);
    a=a(:)';
imgs(i,:)=a;   
i
end
planktonTrainingImages40=imgs;
save('planktonTrainingImages40','-v7.3');  