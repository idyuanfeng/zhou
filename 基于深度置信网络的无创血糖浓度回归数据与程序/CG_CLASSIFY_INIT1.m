
%�ú���ʵ�ֵĹ����Ǽ���������ۺ���ֵf���Լ�f�������и�������ֵ��ƫ����df��
%Ȩֵ��ƫ��ֵ��ͬʱ�������в���VVΪ���������в������ɵ���������
%����DimΪÿ������Ľڵ������ɵ�������XXΪѵ���������ϡ�
%f��df�ֱ��ʾ����Ĵ��ۺ�����ƫ������ֵ�� 

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
% f = sum((1/2)*(target-targetout).^2) %���
%   f = sqrt((1/N)*sum((1/2)*(target-targetout).^2)) ;%���������
f = sqrt((1/N)*sum((target-targetout).^2));
IO = (targetout-target);
Ix_class=IO; 
dw_class =  w3probs'*Ix_class; 

df = [dw_class(:)']'; 

