clc
clear
%% ���ݵ���
% load A1
% dbn_datatrain=dataA{1,5}(:,[52,73,83,104,136,138,146,151]);
% dbn_datatest=dataA{2,5}(:,[52,73,83,104,136,138,146,151]);
load dbnoutputdata_5_20_256_256_256
% load dbnoutputdataB_4_20_256_256_256
z=dbn_datatrain;
% load pz1.txt %ԭʼ���ݴ���ڴ��ı��ļ�pz1.txt ��
mu=mean(z);sig=std(z); %���ֵ�ͱ�׼��
rr=corrcoef(z); %�����ϵ������
data=zscore(z); %���ݱ�׼��,��������X*��Y*
n=256;m=1; %n ���Ա����ĸ���,m ��������ĸ���
x0=z(:,1:n);y0=z(:,end); %ԭʼ���Ա��������������
e0=data(:,1:n);f0=data(:,end); %��׼������Ա��������������
num=size(e0,1);%��������ĸ���
chg=eye(n); %w ��w*�任����ĳ�ʼ��
for i=1:n
%���¼���w��w*��t �ĵ÷�������
matrix=e0'*f0*f0'*e0;
[vec,val]=eig(matrix); %������ֵ����������
val=diag(val); %����Խ���Ԫ�أ����������ֵ
[val,ind]=sort(val,'descend');
w(:,i)=vec(:,ind(1)); %����������ֵ��Ӧ����������
w_star(:,i)=chg*w(:,i); %����w*��ȡֵ
t(:,i)=e0*w(:,i); %����ɷ�ti �ĵ÷�
alpha=e0'*t(:,i)/(t(:,i)'*t(:,i)); %����alpha_i
chg=chg*(eye(n)-w(:,i)*alpha'); %����w ��w*�ı任����
e=e0-t(:,i)*alpha'; %����в����
e0=e;
%���¼���ss(i)��ֵ
beta=t\f0; %��ع鷽�̵�ϵ�������ݱ�׼����û�г�����
cancha=f0-t*beta; %��в����
ss(i)=sum(sum(cancha.^2)); %�����ƽ����
%���¼���press(i)
for j=1:num
t1=t(:,1:i);f1=f0;
she_t=t1(j,:);she_f=f1(j,:); %����ȥ�ĵ�j �������㱣������
t1(j,:)=[];f1(j,:)=[]; %ɾ����j ���۲�ֵ
beta1=[t1,ones(num-1,1)]\f1; %��ع������ϵ��,������г�����
cancha=she_f-she_t*beta1(1:end-1,:)-beta1(end,:); %��в�����
press_i(j)=sum(cancha.^2); %�����ƽ����
end
press(i)=sum(press_i);
Q_h2(1)=1;
if i>1, Q_h2(i)=1-press(i)/ss(i-1); end
if Q_h2(i)<0.0975
fprintf('����ĳɷָ���r=%d',i); break
end
end
beta_z=t\f0; %��Y*����t �Ļع�ϵ��
xishu=w_star*beta_z; %��Y*����X*�Ļع�ϵ����ÿһ����һ���ع鷽��
mu_x=mu(1:n);mu_y=mu(n+1:end); %����Ա�����������ľ�ֵ
sig_x=sig(1:n);sig_y=sig(n+1:end); %����Ա�����������ı�׼��
ch0=mu_y-(mu_x./sig_x*xishu).*sig_y; %����ԭʼ���ݻع鷽�̵ĳ�����
for i=1:m
xish(:,i)=xishu(:,i)./sig_x'*sig_y(i); %����ԭʼ���ݻع鷽�̵�ϵ��
end
sol=[ch0;xish] %��ʾ�ع鷽�̵�ϵ����ÿһ����һ�����̣�ÿһ�еĵ�һ�����ǳ�����
% save DBNPLSmydata1 x0 y0 num xishu ch0 xish
% %�ع�ϵ����ֱ��ͼ
% figure(1)
% bar(xishu')
%����ѵ��Ԥ��ͼ
% load DBNPLSmydata1
ch1=repmat(ch0,num,1);
yhat1=ch1+x0*xish; %����Y ��Ԥ��ֵ
y1max=max(yhat1); %��Ԥ��ֵ�����ֵ
y2max=max(y0); %��۲�ֵ�����ֵ
ymax=max([y1max;y2max]) %��Ԥ��ֵ�͹۲�ֵ�����ֵ
cancha1=yhat1-y0; %����в�
mse1=sqrt((sum(cancha1.^2))/num);
r1=min(corrcoef(y0,yhat1));
%��ֱ��y=x,����Ԥ��ͼ
% figure(2)
% plot(3:10,3:10,y0(:,1),yhat1(:,1),'*')


%% ���Լ�
x_t=dbn_datatest(:,1:n);y_t=dbn_datatest(:,end);  %ԭʼ���Ա��������������
num1=size(x_t,1);
ch2=repmat(ch0,num1,1);
yhat=ch2+x_t*xish; %����Y ��Ԥ��ֵ
y1max=max(yhat); %��Ԥ��ֵ�����ֵ
y2max=max(y_t); %��۲�ֵ�����ֵ
ymax=max([y1max;y2max]) %��Ԥ��ֵ�͹۲�ֵ�����ֵ
cancha2=yhat-y_t; %����в�
mse2=sqrt((sum(cancha2.^2))/num1);
r2=min(corrcoef(y_t,yhat));
%��ֱ��y=x,����Ԥ��ͼ
% figure(3)
% plot(3:10,3:10,y_t(:,1),yhat(:,1),'*')

ty_1=y0*18;
typ_1=yhat1*18;
[total1, percentage1] = clarke1(ty_1,typ_1);

ty=y_t*18;
typ=yhat*18;
[total2, percentage2] = clarke1(ty,typ);


