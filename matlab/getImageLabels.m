labelledImages=cell(160733,2);   %get all the training image names
a=1;
for i = 1:160733
    b=num2str(i)
    name=[b '.jpg'];
    if exist(name)
        labelledImages{a,1}=name;
        a=a+1;
    end
end

for i = 1:30336      %assign labels to training images
    i
    aa=which(labelledImages{i,1});
    s=length(labelledImages{i,1})+1;
    aa=aa(45:end-s);
    for j = 1:121
        if strcmp(aa,labels{j,2})==1
            labelledImages{i,2}=labels{j,1};
        end
    end
end

        

        
        
