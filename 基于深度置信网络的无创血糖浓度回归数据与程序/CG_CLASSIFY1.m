function [f, df] = CG_CLASSIFY1(VV,Dim,XX,target)
l1 = Dim(1);
l2 = Dim(2);
l3= Dim(3);
l4= Dim(4);
l5= Dim(5);
N = size(XX,1);

% Do decomversion.
 w1 = reshape(VV(1:(l1+1)*l2),l1+1,l2);
 xxx = (l1+1)*l2;
 w2 = reshape(VV(xxx+1:xxx+(l2+1)*l3),l2+1,l3);
 xxx = xxx+(l2+1)*l3;
 w3 = reshape(VV(xxx+1:xxx+(l3+1)*l4),l3+1,l4);
 xxx = xxx+(l3+1)*l4;
 w_class = reshape(VV(xxx+1:xxx+(l4+1)*l5),l4+1,l5);


  XX = [XX ones(N,1)];
  w1probs = 1./(1 + exp(-XX*w1)); w1probs = [w1probs  ones(N,1)];
  w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
  w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
% targetout =w3probs*w_class;
targetout =1./(1 + exp(-w3probs*w_class));
%   targetout = exp(w3probs*w_class);
%   targetout = targetout./repmat(sum(targetout,2),1,39);
%   f = -sum(sum( target(:,1:end).*log(targetout))) ;
% f = sum((1/2)*(target-targetout).^2) %Îó²î
% f = sqrt((1/N)*sum((1/2)*(target-targetout).^2));
f = sqrt((1/N)*sum((target-targetout).^2));
IO = (targetout-target);
Ix_class=IO; 
dw_class =  w3probs'*Ix_class; 


Ix3 = (Ix_class*w_class').*w3probs.*(1-w3probs);
Ix3 = Ix3(:,1:end-1);
dw3 =  w2probs'*Ix3;

Ix2 = (Ix3*w3').*w2probs.*(1-w2probs); 
Ix2 = Ix2(:,1:end-1);
dw2 =  w1probs'*Ix2;

Ix1 = (Ix2*w2').*w1probs.*(1-w1probs); 
Ix1 = Ix1(:,1:end-1);
dw1 =  XX'*Ix1;

df = [dw1(:)' dw2(:)' dw3(:)' dw_class(:)']'; 
