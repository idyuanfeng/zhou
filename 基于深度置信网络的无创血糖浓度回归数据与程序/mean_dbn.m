
load('mnistclassify_weights.mat')
% makebatches;
%—µ¡∑ºØ
[numcases numdims numbatches]=size(batchdata);
N=numcases;
data1_train=[];
data2_train=[];
data3_train=[];
% data4_train=[];
% targetout=[];
% Cm1=1;Cm2=0;

for batch = 1:numbatches
  data = [batchdata(:,:,batch)];
  %target = [batchtargets(:,:,batch)];
  data = [data ones(N,1)];
  w1probs = 1./(1 + exp(-data*w1)); data1_train=[data1_train;w1probs];w1probs = [w1probs  ones(N,1)];
  w2probs = 1./(1 + exp(-w1probs*w2));data2_train=[data2_train;w2probs]; w2probs = [w2probs ones(N,1)];
  w3probs = 1./(1 + exp(-w2probs*w3));data3_train=[data3_train;w3probs]; w3probs = [w3probs  ones(N,1)]; 
%   w4probs = 1./(1 + exp(-w3probs*w_class));data4_train=[data4_train;w4probs]; w4probs = [w4probs  ones(N,1)]; 
%   targetout=w3probs*w_class;
targetout =1./(1 + exp(-w3probs*w_class));
end

% ≤‚ ‘ºØ
% data1_test=[];
% data2_test=[];
% data3_test=[];
% p=length(x_test(:,1));
% data=[x_test,ones(p,1)];
%   w1probs = 1./(1 + exp(-data*w1)); data1_test=[data1_test;w1probs];w1probs = [w1probs  ones(p,1)];
%   w2probs = 1./(1 + exp(-w1probs*w2));data2_test=[data2_test;w2probs]; w2probs = [w2probs ones(p,1)]; 
%   w3probs = 1./(1 + exp(-w2probs*w3));data3_test=[data3_test;w3probs]; 
%  [numcases numdims numbatches]=size(testbatchdata);
% N=numcases;
% data1_test=[];
% data2_test=[];
% data3_test=[];
% % Cm1=1;Cm2=0;
% 
% for batch = 1:numbatches
%   data = [testbatchdata(:,:,batch)];
%   %target = [batchtargets(:,:,batch)];
%   data = [data ones(N,1)];
%   w1probs = 1./(1 + exp(-data*w1)); data1_test=[data1_test;w1probs];w1probs = [w1probs  ones(N,1)];
%   w2probs = 1./(1 + exp(-w1probs*w2));data2_test=[data2_test;w2probs]; w2probs = [w2probs ones(N,1)]; 
%   w3probs = 1./(1 + exp(-w2probs*w3));data3_test=[data3_test;w3probs]; 
% %   w3probs = [w3probs  ones(N,1)];
% %   w4probs = 1./(1 + exp(-w3probs*w_class));data4=[data4;w4probs]; 
% end

 save finaldata data3_train 
