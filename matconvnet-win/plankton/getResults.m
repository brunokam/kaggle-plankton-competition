vl_setupnn;
load('planktonTestImages40one');
planks=planktonTestImages40one;
clear planktonTestImages40one;
planks=planks';
planks=reshape(planks,40,40,1,[]);
planks = bsxfun(@minus, planks, mean(planks,4));
planks=single(planks);
%net.layers{end}.type = 'softmax';


a1=zeros(130400,121);
for i = 1:60000
    res = vl_simplenn(net, planks(:,:,1,i));
    for j = 1:121
        a1(i,j)=mean(mean(res(13).x(:,:,j)));
    end
    i
end
clear planks;


load('planktonTestImages40two');
planks=planktonTestImages40two;
clear planktonTestImages40two;
planks=planks';
planks=reshape(planks,40,40,1,[]);
planks = bsxfun(@minus, planks, mean(planks,4));
planks=single(planks);

for i = 1:70400
    res = vl_simplenn(net, planks(:,:,1,i));
    for j = 1:121
        a1(i+60000,j)=mean(mean(res(13).x(:,:,j)));
    end
    i
end

save('a1','a1');