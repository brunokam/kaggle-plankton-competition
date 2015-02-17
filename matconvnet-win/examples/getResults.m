
load('planktonTestImages28');
planks=planktonTestImages28;
planks = bsxfun(@minus, planks, mean(planks,4));
planks=single(planks);
net.layers{end}.type = 'softmax';
net.layers{1,7}.rate=0;

aa=zeros(130400,121);
for i = 1: 130400
res = vl_simplenn(net, planks(:,:,1,i));
aa(i,:)=res(10).x(:,:,:);
i
end
save('aa','aa');

