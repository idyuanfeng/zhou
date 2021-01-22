
% 本程序是基于对比散度的快速学习RMN的程序
%
% maxepoch  -- 最大迭代次数
% numhid    -- 隐层节点数
% batchdata -- 分批数量
% restart   -- set to 1 if learning starts from beginning 

epsilonw      = 0.01 ;         % 权重的学习率 
epsilonvb     = 1;         % 可见层偏置 
epsilonhb     = 1;         % 隐藏单元
weightcost  = 0.0002;        %权重衰减系数
initialmomentum  = 0.5;      %动量系数
finalmomentum    = 0.9;

[numcases, numdims, numbatches]=size(batchdata);

% if restart ==1
%   restart=0;
  epoch=1;

% Initializing symmetric weights and biases. 
  vishid     = 0.1*randn(numdims, numhid);%  numdims*numhid   （numhid=256）  w  的初值
  hidbiases  = zeros(1,numhid);  %隐藏层的偏置
  visbiases  = zeros(1,numdims); %可见层的偏置
% 
  poshidprobs = zeros(numcases,numhid);% 100*256   
  neghidprobs = zeros(numcases,numhid);% 100*256
  posprods    = zeros(numdims,numhid);%  24*256
  negprods    = zeros(numdims,numhid);%  24*256
  vishidinc  = zeros(numdims,numhid);%   24*256  最终的  权重的 偏置
  hidbiasinc = zeros(1,numhid);% 最终隐藏层偏置
  visbiasinc = zeros(1,numdims);%最终可见层偏置
  batchposhidprobs=zeros(numcases,numhid,numbatches);
% end
a=zeros(1,maxepoch);
for epoch = epoch:maxepoch
    fprintf(1,'epoch %d\r',epoch);
    errsum=0;
    for batch = 1:numbatches  %分了几批数据
        fprintf(1,'epoch %d batch %d\r',epoch,batch);%打印第几次迭代的第几批
        
        %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        data = batchdata(:,:,batch);% v1
        poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)));% P(h1. =1 |v1)
                                                   %repmat(hidbiases,numcases,1)产生numcases个hidbiases矩阵，整体作为一列
        batchposhidprobs(:,:,batch)=poshidprobs;
        posprods    = data' * poshidprobs;% v1' * P(h1.=1|v1)
        poshidact   = sum(poshidprobs);
        posvisact = sum(data);% 
        
        %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        poshidstates = poshidprobs > rand(numcases,numhid);  %  二值化  转化为 0  1
        
        %%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));  %隐藏层到可见层 P(h2.=1|v2)
        neghidprobs = 1./(1 + exp(-negdata*vishid - repmat(hidbiases,numcases,1)));    %可见层到隐藏层 
        negprods  = negdata'*neghidprobs;%   v2' * P(h2.=1|v2)
        neghidact = sum(neghidprobs);
        negvisact = sum(negdata);  % v2
        
        %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        err= sum(sum( (data-negdata).^2 ));
        errsum = err + errsum;
        a(epoch)=errsum;
        
        if epoch>5
            momentum=finalmomentum;%0.9
        else
            momentum=initialmomentum;%0.1
        end
        
        %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        vishidinc = momentum*vishidinc + ...
            epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);  %  w    weightcost权重衰减系数
                % posprods: v1' * P(h1.=1|v1)    (posprods-negprods)h1乘以v1的转置 - h2*v2的转置
        visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);% v1 - v2
        hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);% h1 - h2
        
        vishid = vishid + vishidinc;
        visbiases = visbiases + visbiasinc;
        hidbiases = hidbiases + hidbiasinc;
        
        %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    end
    fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum);

end
%figure
% plot(a);