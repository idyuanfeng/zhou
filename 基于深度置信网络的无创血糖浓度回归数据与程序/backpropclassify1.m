maxepoch=1000;%΢��  ��������
fprintf(1,'\nTraining discriminative model on MNIST by minimizing cross entropy error. \n');
fprintf(1,' batches of 1000 cases each. \n');

load mnistvhclassify  %��һ�����
load mnisthpclassify
load mnisthp2classify
% load mnisthp3classify
makebatches;
[numcases numdims numbatches]=size(batchdata);
N=numcases; 

%%%% PREINITIALIZE WEIGHTS OF THE DISCRIMINATIVE MODEL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w1=[vishid; hidrecbiases];%��һ�� �� Ȩ�غ�ƫ��  151*128
w2=[hidpen; penrecbiases];
w3=[hidpen2; penrecbiases2];
% w4=[hidpen3; penrecbiases3];
w_class = 0.1*randn(size(w3,2)+1,1);%ȫ���Ӳ�Ȩ�س�ʼ�� 129*1
 

%%%%%%%%%% END OF PREINITIALIZATIO OF WEIGHTS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

l1=size(w1,1)-1;%150
l2=size(w2,1)-1;%128
l3=size(w3,1)-1;%128
l4=size(w_class,1)-1;%128
l5=1; 
% test_err=[];
% train_err=[];

a=zeros(1,maxepoch);
for epoch = 1:maxepoch

%%%%%%%%%%%%%%%%%%%% COMPUTE TRAINING MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%����ѵ����
 err=0; 
 err_cr=0;
% counter=0;
[numcases numdims numbatches]=size(batchdata);
N=numcases;
 for batch = 1:numbatches  % ����  numbatches ������
  data = [batchdata(:,:,batch)];
  target = [batchtargets(:,:,batch)];%��ǩ
  data = [data ones(N,1)];% �� data �������һ��  1  ���һ�к�ƫ�����
  w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)];  % [100*151 ] * [151*128]  
  w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];% [100*129] * [129*128]
  w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];% [100*129] * [129*128]
%   targetout = exp(w3probs*w_class);% [100*129] * [129*1]
% targetout =w3probs*w_class;% [100*129] * [129*1]
targetout =1./(1 + exp(-w3probs*w_class));
%   targetout = targetout./repmat(sum(targetout,2),1,39);%����10������ֵ
% targetout = targetout./sum(targetout);
%   [I, J]=max(targetout,[],2);%����Ԥ��ı�ǩ    ������һ��  I�Ǹ���ֵ  J�� ����ǩ
%   [I1, J1]=max(target,[],2); %ԭʼ��ǩ
%   counter=counter+length(find(J==J1));%ͳ�Ʒ�����ȷ�ĸ���
%   err_cr = err_cr- sum( target(:,1:end).*log(targetout)) ;%[100*1] .* [100*1]
% err_cr = sum( target(:,1:end).*log(targetout)) ;
%  err_cr = err_cr+sqrt((1/N)*sum((1/2)*(target-targetout).^2));
 err_cr = err_cr+sqrt((1/N)*sum((target-targetout).^2));
%  err_cr = sum((1/N)*(target-targetout).^2);

 end    %target .*log(targetout)  target��0-1���󣬱����� �б�ǩ��Ӧ�ĸ���ֵ��ȥ���������ĸ���ֵ�������
%  train_err(epoch)=(numcases*numbatches-counter);%  ͳ�Ʒ������ĸ��� ������-��ȷ����
%  train_crerr(epoch)=err_cr/numbatches;
a(epoch)=err_cr;
%%%%%%%%%%%%%% END OF COMPUTING TRAINING MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%






% %%%%%%%%%%%%%%%%%%%% COMPUTE TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %������֤��
%  err=0;
% err_cr=0;
% % counter=0;
% [testnumcases, testnumdims, testnumbatches]=size(testbatchdata);
% N=testnumcases;
% for batch = 1:testnumbatches
%   data = [testbatchdata(:,:,batch)];
%   target = [testbatchtargets(:,:,batch)];
%   data = [data ones(N,1)];
%   w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)];
%   w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
%   w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
%   targetout = exp(w3probs*w_class);
% % targetout = w3probs*w_class;% [100*129] * [129*1]
% %   targetout = targetout./repmat(sum(targetout,2),1,10);
%  targetout = targetout./sum(targetout);
% % err=sqrt((1/N)*sum((target-targetout).*(target-targetout)));
% % err_cr = err_cr+sqrt((1/N)*sum((target-targetout).*(target-targetout)));
% 
%   err_cr = err_cr- sum( target(:,1:end).*log(targetout)) ;
% end
% %  test_err(epoch)=(testnumcases*testnumbatches-counter);
% %  test_crerr(epoch)=err_cr/testnumbatches;
% %  fprintf(1,'Before epoch %d Train # misclassified: %d (from %d). Test # misclassified: %d (from %d) \t \t \n',...
% %            epoch,train_err(epoch),numcases*numbatches,test_err(epoch),testnumcases*testnumbatches);

%%%%%%%%%%%%% END OF COMPUTING TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 tt=0;
 for batch = 1:numbatches/1
 fprintf(1,'epoch %d batch %d\r',epoch,batch);

%%%%%%%%%%% COMBINE 10 MINIBATCHES INTO 1 LARGER MINIBATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 tt=tt+1; 
 data=[];
 targets=[]; 
 for kk=1:1
  data=[data 
        batchdata(:,:,(tt-1)*1+kk)]; 
  targets=[targets
        batchtargets(:,:,(tt-1)*1+kk)];
 end 

%%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ͨ��3�� �������� ʵ�ֹ����ݶ�
max_iter=3;

  if epoch<6  % �����������С��6�Σ���ֱ�ӵ���������Ȩ��. 
    N = size(data,1);
    XX = [data ones(N,1)];
    w1probs = 1./(1 + exp(-XX*w1)); w1probs = [w1probs  ones(N,1)];
    w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
    w3probs = 1./(1 + exp(-w2probs*w3)); %w3probs = [w3probs  ones(N,1)];

    VV = [w_class(:)']'; %��Ϊ1������   2570*1
    Dim = [l4; l5];% 256   10
    [X, fX] = minimize(VV,'CG_CLASSIFY_INIT1',max_iter,Dim,w3probs,targets);%targets ʵ�ʱ�ǩ
    %  X �Ż����������������һ��������
    w_class = reshape(X,l4+1,l5);% X��140554*1

  else
    VV = [w1(:)' w2(:)' w3(:)' w_class(:)']';
    Dim = [l1; l2; l3; l4; l5];%24 256 256 256 1
    [X, fX] = minimize(VV,'CG_CLASSIFY1',max_iter,Dim,data,targets);

    w1 = reshape(X(1:(l1+1)*l2),l1+1,l2);
    xxx = (l1+1)*l2;
    w2 = reshape(X(xxx+1:xxx+(l2+1)*l3),l2+1,l3);
    xxx = xxx+(l2+1)*l3;
    w3 = reshape(X(xxx+1:xxx+(l3+1)*l4),l3+1,l4);
    xxx = xxx+(l3+1)*l4;
    w_class = reshape(X(xxx+1:xxx+(l4+1)*l5),l4+1,l5);

  end
%%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 end

 save mnistclassify_weights w1 w2 w3 w_class
%  save mnistclassify_error test_err test_crerr train_err train_crerr;
end
figure
plot(a);

