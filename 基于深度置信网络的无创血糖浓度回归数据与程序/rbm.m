
% �������ǻ��ڶԱ�ɢ�ȵĿ���ѧϰRMN�ĳ���
%
% maxepoch  -- ����������
% numhid    -- ����ڵ���
% batchdata -- ��������
% restart   -- set to 1 if learning starts from beginning 

epsilonw      = 0.01 ;         % Ȩ�ص�ѧϰ�� 
epsilonvb     = 1;         % �ɼ���ƫ�� 
epsilonhb     = 1;         % ���ص�Ԫ
weightcost  = 0.0002;        %Ȩ��˥��ϵ��
initialmomentum  = 0.5;      %����ϵ��
finalmomentum    = 0.9;

[numcases, numdims, numbatches]=size(batchdata);

% if restart ==1
%   restart=0;
  epoch=1;

% Initializing symmetric weights and biases. 
  vishid     = 0.1*randn(numdims, numhid);%  numdims*numhid   ��numhid=256��  w  �ĳ�ֵ
  hidbiases  = zeros(1,numhid);  %���ز��ƫ��
  visbiases  = zeros(1,numdims); %�ɼ����ƫ��
% 
  poshidprobs = zeros(numcases,numhid);% 100*256   
  neghidprobs = zeros(numcases,numhid);% 100*256
  posprods    = zeros(numdims,numhid);%  24*256
  negprods    = zeros(numdims,numhid);%  24*256
  vishidinc  = zeros(numdims,numhid);%   24*256  ���յ�  Ȩ�ص� ƫ��
  hidbiasinc = zeros(1,numhid);% �������ز�ƫ��
  visbiasinc = zeros(1,numdims);%���տɼ���ƫ��
  batchposhidprobs=zeros(numcases,numhid,numbatches);
% end
a=zeros(1,maxepoch);
for epoch = epoch:maxepoch
    fprintf(1,'epoch %d\r',epoch);
    errsum=0;
    for batch = 1:numbatches  %���˼�������
        fprintf(1,'epoch %d batch %d\r',epoch,batch);%��ӡ�ڼ��ε����ĵڼ���
        
        %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        data = batchdata(:,:,batch);% v1
        poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)));% P(h1. =1 |v1)
                                                   %repmat(hidbiases,numcases,1)����numcases��hidbiases����������Ϊһ��
        batchposhidprobs(:,:,batch)=poshidprobs;
        posprods    = data' * poshidprobs;% v1' * P(h1.=1|v1)
        poshidact   = sum(poshidprobs);
        posvisact = sum(data);% 
        
        %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        poshidstates = poshidprobs > rand(numcases,numhid);  %  ��ֵ��  ת��Ϊ 0  1
        
        %%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));  %���ز㵽�ɼ��� P(h2.=1|v2)
        neghidprobs = 1./(1 + exp(-negdata*vishid - repmat(hidbiases,numcases,1)));    %�ɼ��㵽���ز� 
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
            epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);  %  w    weightcostȨ��˥��ϵ��
                % posprods: v1' * P(h1.=1|v1)    (posprods-negprods)h1����v1��ת�� - h2*v2��ת��
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