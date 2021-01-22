clc
clear
%% 数据导入
% load A1
% dbn_datatrain=dataA{1,5}(:,[52,73,83,104,136,138,146,151]);
% dbn_datatest=dataA{2,5}(:,[52,73,83,104,136,138,146,151]);
load dbnoutputdata_5_20_256_256_256
% load dbnoutputdataB_4_20_256_256_256
z=dbn_datatrain;
% load pz1.txt %原始数据存放在纯文本文件pz1.txt 中
mu=mean(z);sig=std(z); %求均值和标准差
rr=corrcoef(z); %求相关系数矩阵
data=zscore(z); %数据标准化,变量记做X*和Y*
n=256;m=1; %n 是自变量的个数,m 是因变量的个数
x0=z(:,1:n);y0=z(:,end); %原始的自变量和因变量数据
e0=data(:,1:n);f0=data(:,end); %标准化后的自变量和因变量数据
num=size(e0,1);%求样本点的个数
chg=eye(n); %w 到w*变换矩阵的初始化
for i=1:n
%以下计算w，w*和t 的得分向量，
matrix=e0'*f0*f0'*e0;
[vec,val]=eig(matrix); %求特征值和特征向量
val=diag(val); %提出对角线元素，即提出特征值
[val,ind]=sort(val,'descend');
w(:,i)=vec(:,ind(1)); %提出最大特征值对应的特征向量
w_star(:,i)=chg*w(:,i); %计算w*的取值
t(:,i)=e0*w(:,i); %计算成分ti 的得分
alpha=e0'*t(:,i)/(t(:,i)'*t(:,i)); %计算alpha_i
chg=chg*(eye(n)-w(:,i)*alpha'); %计算w 到w*的变换矩阵
e=e0-t(:,i)*alpha'; %计算残差矩阵
e0=e;
%以下计算ss(i)的值
beta=t\f0; %求回归方程的系数，数据标准化，没有常数项
cancha=f0-t*beta; %求残差矩阵
ss(i)=sum(sum(cancha.^2)); %求误差平方和
%以下计算press(i)
for j=1:num
t1=t(:,1:i);f1=f0;
she_t=t1(j,:);she_f=f1(j,:); %把舍去的第j 个样本点保存起来
t1(j,:)=[];f1(j,:)=[]; %删除第j 个观测值
beta1=[t1,ones(num-1,1)]\f1; %求回归分析的系数,这里带有常数项
cancha=she_f-she_t*beta1(1:end-1,:)-beta1(end,:); %求残差向量
press_i(j)=sum(cancha.^2); %求误差平方和
end
press(i)=sum(press_i);
Q_h2(1)=1;
if i>1, Q_h2(i)=1-press(i)/ss(i-1); end
if Q_h2(i)<0.0975
fprintf('提出的成分个数r=%d',i); break
end
end
beta_z=t\f0; %求Y*关于t 的回归系数
xishu=w_star*beta_z; %求Y*关于X*的回归系数，每一列是一个回归方程
mu_x=mu(1:n);mu_y=mu(n+1:end); %提出自变量和因变量的均值
sig_x=sig(1:n);sig_y=sig(n+1:end); %提出自变量和因变量的标准差
ch0=mu_y-(mu_x./sig_x*xishu).*sig_y; %计算原始数据回归方程的常数项
for i=1:m
xish(:,i)=xishu(:,i)./sig_x'*sig_y(i); %计算原始数据回归方程的系数
end
sol=[ch0;xish] %显示回归方程的系数，每一列是一个方程，每一列的第一个数是常数项
% save DBNPLSmydata1 x0 y0 num xishu ch0 xish
% %回归系数的直方图
% figure(1)
% bar(xishu')
%体能训练预测图
% load DBNPLSmydata1
ch1=repmat(ch0,num,1);
yhat1=ch1+x0*xish; %计算Y 的预测值
y1max=max(yhat1); %求预测值的最大值
y2max=max(y0); %求观测值的最大值
ymax=max([y1max;y2max]) %求预测值和观测值的最大值
cancha1=yhat1-y0; %计算残差
mse1=sqrt((sum(cancha1.^2))/num);
r1=min(corrcoef(y0,yhat1));
%画直线y=x,并画预测图
% figure(2)
% plot(3:10,3:10,y0(:,1),yhat1(:,1),'*')


%% 测试集
x_t=dbn_datatest(:,1:n);y_t=dbn_datatest(:,end);  %原始的自变量和因变量数据
num1=size(x_t,1);
ch2=repmat(ch0,num1,1);
yhat=ch2+x_t*xish; %计算Y 的预测值
y1max=max(yhat); %求预测值的最大值
y2max=max(y_t); %求观测值的最大值
ymax=max([y1max;y2max]) %求预测值和观测值的最大值
cancha2=yhat-y_t; %计算残差
mse2=sqrt((sum(cancha2.^2))/num1);
r2=min(corrcoef(y_t,yhat));
%画直线y=x,并画预测图
% figure(3)
% plot(3:10,3:10,y_t(:,1),yhat(:,1),'*')

ty_1=y0*18;
typ_1=yhat1*18;
[total1, percentage1] = clarke1(ty_1,typ_1);

ty=y_t*18;
typ=yhat*18;
[total2, percentage2] = clarke1(ty,typ);


