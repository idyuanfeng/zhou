%% DBN������ȡ
clc
clear
tic;
%��������
load A4
% load dataall
x_train=dataA{1,5}(:,[52,73,83,104,136,138,146]);%52,73,84,104,136,
y_train=dataA{1,5}(:,end);
x_test=dataA{2,5}(:,[52,73,83,104,136,138,146]);%52,73,84,104,136,
y_test=dataA{2,5}(:,end);
% x_train=traindata(:,1:150);
% y_train=traindata(:,end);
% x_test=testdata(:,1:150);
% y_test=testdata(:,end);
%% ����ѵ��������Χ�ڵĲ��Լ�
%���ݹ�һ��
[xn_train,inputps] = mapminmax(x_train',0,1);
x_train = xn_train';
xn_test = mapminmax('apply',x_test',inputps);
x_test = xn_test';
[yn_train,outputps] = mapminmax(y_train',0,1);
y_train = yn_train';
yn_test = mapminmax('apply',y_test',outputps);
y_test = yn_test';
% save ps inputps
% DBNѵ��
maxepoch=100; %�������� 
numhid1=256;numhid2=256; numhid3=256; 
num=39*32;
fprintf(1,'  This uses %3i epochs.\n', maxepoch);
makebatches;%�����ݷ���
[numcases numdims numbatches]=size(batchdata);
%%  ��һ��RBMѵ��
fprintf(1,'Pretraining Layer 1 with RBM: %d-%d \n',numdims,numhid1);
numhid =numhid1;
% restart=1;
rbm;
hidrecbiases=hidbiases; 
save mnistvhclassify vishid hidrecbiases visbiases;
%% �ڶ���RBMѵ��
fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d \n',numhid1,numhid2);
batchdata=batchposhidprobs;
numhid=numhid2;
% restart=1;
rbm;
hidpen=vishid; penrecbiases=hidbiases; hidgenbiases=visbiases;
save mnisthpclassify hidpen penrecbiases hidgenbiases;
%% ������RBMѵ��
fprintf(1,'\nPretraining Layer 3 with RBM: %d-%d \n',numhid2,numhid3);
batchdata=batchposhidprobs;
numhid=numhid3;
% restart=1;
rbm;
hidpen2=vishid; penrecbiases2=hidbiases; hidgenbiases2=visbiases;
save mnisthp2classify hidpen2 penrecbiases2 hidgenbiases2;
% a12=a;
% figure
% plot(a);
%%
backpropclassify1; %���򴫲�
% a4=a;
% save DBNerr_64_32_16 a4
%%
mean_dbn;
%% ������rbm�������Ϊ����
y_train=dataA{1,5}(:,end);
y_test=dataA{2,5}(:,end);
% y_train=traindata(:,end);
% y_test=testdata(:,end);
dbn_datatrain=[data3_train,y_train];
data1_test=[];
data2_test=[];
data3_test=[];
p=length(x_test(:,1));
data=[x_test,ones(p,1)];
  w1probs = 1./(1 + exp(-data*w1)); data1_test=[data1_test;w1probs];w1probs = [w1probs  ones(p,1)];
  w2probs = 1./(1 + exp(-w1probs*w2));data2_test=[data2_test;w2probs]; w2probs = [w2probs ones(p,1)]; 
  w3probs = 1./(1 + exp(-w2probs*w3));data3_test=[data3_test;w3probs]; w3probs = [w3probs  ones(p,1)]; 
  targetouttest =1./(1 + exp(-w3probs*w_class));
  dbn_datatest=[data3_test,y_test];
%  predict_1 = mapminmax('reverse',targetout,outputps);
%  predict_2 = mapminmax('reverse',targetouttest,outputps);
%  ty=y_train*18;
% typ=predict_1*18;
% tty=y_test*18;
% ttyp=predict_2*18;
% [total1, percentage1] = clarke1(ty,typ)
% [total2, percentage2] = clarke1(tty,ttyp)
save dbnoutputdata_5_40_256_256_256 dbn_datatrain dbn_datatest 
% DBNPLS1;
DBNSVR1;
toc;
