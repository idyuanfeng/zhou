
%该函数实现的功能是计算网络代价函数值f，以及f对网络中各个参数值的偏导数df，
%权值和偏置值是同时处理。其中参数VV为网络中所有参数构成的列向量，
%参数Dim为每层网络的节点数构成的向量，XX为训练样本集合。
%f和df分别表示网络的代价函数和偏导函数值。 

function [f, df] = CG_CLASSIFY_INIT(VV,Dim,w3probs,target)
l1 = Dim(1);
l2 = Dim(2);
N = size(w3probs,1);
% Do decomversion.
  w_class = reshape(VV,l1+1,l2);
  w3probs = [w3probs  ones(N,1)];  

%   targetout = exp(w3probs*w_class);
%   targetout = targetout./repmat(sum(targetout,2),1,10);

% targetout =w3probs*w_class;
targetout =1./(1 + exp(-w3probs*w_class));
% f = sum((1/2)*(target-targetout).^2) %误差
%   f = sqrt((1/N)*sum((1/2)*(target-targetout).^2)) ;%均方根误差
f = sqrt((1/N)*sum((target-targetout).^2));
IO = (targetout-target);
Ix_class=IO; 
dw_class =  w3probs'*Ix_class; 

df = [dw_class(:)']'; 

