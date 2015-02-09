

load('labelledImages')
planktonImageRatio=zeros(30336,1);
for i = 1:30336
    a=imread(labelledImages{i,1});
    [x,y]=size(a);
    planktonImageRatio(i)=min(x,y)/max(x,y);
end
save('planktonImageRatio','planktonImageRatio');
