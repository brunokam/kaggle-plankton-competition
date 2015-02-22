vl_setupnn;
load('planktonTrainingImages40');
planks=planktonTrainingImages40;
planks=planks';
planks=reshape(planks,40,40,1,[]);
planks = bsxfun(@minus, planks, mean(planks,4));
planks=single(planks);
net.layers{end}.type = 'softmax';
net.layers{1,9}.rate = 0;

predictions=zeros(30336,121);
for i = 1:30336
    res = vl_simplenn(net, planks(:,:,1,i));
    for j = 1:121
        a1(i,j)=mean(mean(res(13).x(:,:,j)));
    end
    
end
save('predictions','predictions');


load('labels');
[maxPred guess] = max(predictions,[],2);
right = zeros(121,1);
wrong =zeros(121,1);
accuracy = cell(121,1);
prevalence = cell(121,1);
for i = 1:121
    prevalence{i,1}=0;
end

for i = 1:30336;   
    correct=labs{i,1};
    prevalence{correct,1}=prevalence{correct,1}+1;
    if guess(i,1)== correct
        right(correct,1)=right(correct,1) + 1;
    else
        wrong(correct,1)=wrong(correct,1) + 1;
    end
end
accuracy=right./(wrong+right);
prevalence_accuracy = {labels accuracy prevalence};





        