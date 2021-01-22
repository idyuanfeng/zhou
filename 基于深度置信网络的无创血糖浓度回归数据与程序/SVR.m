%% svr
clc
clear
%下载数据
load B1
% x_train=traindata(:,1:150);
% y_train=traindata(:,end);
% x_test=testdata(:,1:150);
% y_test=testdata(:,end);
x_train=dataA{1,4}(:,[52,73,83,104,136,138,146]);
y_train=dataA{1,4}(:,end);
x_test=dataA{2,4}(:,[52,73,83,104,136,138,146]);
y_test=dataA{2,4}(:,end);
[xn_train,inputps] = mapminmax(x_train',0,1);
xn_train = xn_train';
xn_test = mapminmax('apply',x_test',inputps);
xn_test = xn_test';
[yn_train,outputps] = mapminmax(y_train',0,1);
yn_train = yn_train';
yn_test = mapminmax('apply',y_test',outputps);
yn_test = yn_test';
%1. 寻找最佳c参数/g参数  %网格寻优
[c,g] = meshgrid(-2:0.5:2,-2:0.5:2);
[m,n] = size(c);
cg = zeros(m,n);
eps = 10^(-4);
v = 5;
bestc = 0;
bestg = 0;
error = Inf;
for i = 1:m
    for j = 1:n
        cmd = ['-v ',num2str(v),' -t 2',' -c ',num2str(2^c(i,j)),' -g ',num2str(2^g(i,j) ),' -s 3 -p 0.01'];
        cg(i,j) = svmtrain(yn_train,xn_train,cmd);
        if cg(i,j) < error
            error = cg(i,j);
            bestc = 2^c(i,j);
            bestg = 2^g(i,j);
        end
        if abs(cg(i,j) - error) <= eps && bestc > 2^c(i,j)
            error = cg(i,j);
            bestc = 2^c(i,j);
            bestg = 2^g(i,j);
        end
    end
end

 
%%
% 2. 创建/训练SVM  
cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p 0.01'];
model= svmtrain(yn_train,xn_train,cmd);
% save model
%% V. SVM仿真预测
[Predict_1,acc_1,pro] = svmpredict(yn_train,xn_train,model);
[Predict_2,acc_2,pro] = svmpredict(yn_test,xn_test,model);
%%
% 1. 反归一化
predict_1 = mapminmax('reverse',Predict_1,outputps);
predict_2 = mapminmax('reverse',Predict_2,outputps);

 
%%
% 2. 结果对比
result_1 = [ y_train predict_1];
result_2 = [y_test predict_2];
num1=length(predict_1);
num2=length(predict_2);
RMSE1=sqrt(sum((y_train-predict_1).^2)/num1);
RMSE2=sqrt(sum((y_test-predict_2).^2)/num2);
% SVRERR_1=y_test-predict_2;
% save SVRERR_1 SVRERR_1
SVRERR_2=y_test-predict_2;
save SVRERR_2 SVRERR_2
%% VI. 绘图
% figure(1)
% plot(1:length(y_train),y_train,'r*',1:length(y_train),predict_1,'bo')
% grid on
% legend('真实值','预测值')
% xlabel('样本编号')
% ylabel('血糖浓度')
% string_1 = {'训练集预测结果对比';
%            ['mse = ' num2str(acc_1(2)) ' R^2 = ' num2str(acc_1(3))]};
% title(string_1)
% figure(2)
% plot(1:length(y_test),y_test,'r*',1:length(y_test),predict_2,'bo')
% grid on
% legend('真实值','预测值')
% xlabel('样本编号')
% ylabel('血糖浓度')
% string_2 = {'测试集预测结果对比';
%            ['mse = ' num2str(acc_2(2)) ' R^2 = ' num2str(acc_2(3))]};
% title(string_2)
% figure(3)
% plot(y_train,predict_1,'*b');
% hold on
% x=3:10;
% y=x;
% plot(x,y,'g')
% grid on
% xlabel('真实浓度')
% ylabel('预测浓度')
% string_3={'训练集效果图'};
% title(string_3)
% 
% figure(4)
% plot(y_test,predict_2,'*b');
% hold on
% x=3:10;
% y=x;
% plot(x,y,'g')
% grid on 
% xlabel('真实浓度')
% ylabel('预测浓度')
% string_4={'测试集效果图'};
% title(string_4)
% 
% %% 误差分布
% figure(5)
% hist(y_train-predict_1,10);
% grid on
% xlabel('误差分布区间')
% ylabel('分布个数')
% string_5={'svm训练集误差分布图'};
% title(string_5)
% figure(6)
% hist(y_test-predict_2,10);
% grid on
% xlabel('误差分布区间')
% ylabel('分布个数')
% string_6={'svm测试集误差分布图'};
% title(string_6)
% %%
% figure(7)
% x=3:10;
% y=x;
% plot(x,y,'g')
% hold on
% k1=unique(y_train);
% k=k1(k1~=0);
% target_col=1;%要搜索的目标列
% for u=1:length(k)
%     target_val=k(u);%要搜索的目标值
%     [row,col]=find(result_1(:,target_col)==target_val);
%     result_row=row;
%     result=result_1(result_row,:);
%     max_result=max(result(:,2));
%     min_result=min(result(:,2));
%     last_result=[target_val max_result;
%                  target_val min_result];
%     plot(last_result(:,1),last_result(:,2),'.-b')
%     hold on
% end
% grid on
% xlabel('真实浓度')
% ylabel('预测浓度')
% string_7={'训练示意图'};
% title(string_7)
% 
% figure(8)
% x=3:10;
% y=x;
% plot(x,y,'g')
% hold on
% k1=unique(y_test);
% k=k1(k1~=0);
% target_col=1;%要搜索的目标列
% for u=1:length(k)
%     target_val=k(u);%要搜索的目标值
%     [row,col]=find(result_2(:,target_col)==target_val);
%     result_row=row;
%     result=result_2(result_row,:);
%     max_result=max(result(:,2));
%     min_result=min(result(:,2));
%     last_result=[target_val max_result;
%                  target_val min_result];
%     plot(last_result(:,1),last_result(:,2),'.-b')
%     hold on
% end
% grid on
% xlabel('真实浓度')
% ylabel('预测浓度')
% string_8={'测试示意图'};
% title(string_8)
 ty=y_train*18;
typ=predict_1*18;
tty=y_test*18;
ttyp=predict_2*18;
[total, percentage] = clarke1(ty,typ)
[total, percentage] = clarke1(tty,ttyp)
