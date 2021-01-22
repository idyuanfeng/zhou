% DBNtrain;
% %% dbnsvr
% clc
% clear
% %数据下载
% load dbnoutputdata
% load dbnoutputdata_5_20_256_256_256
% load dbnoutputdataB_4_20_256_256_256
x_train=dbn_datatrain(:,1:256);
y_train=dbn_datatrain(:,end);
x_test=dbn_datatest(:,1:256);
y_test=dbn_datatest(:,end);
[xn_train,inputps] = mapminmax(x_train',0,1);
xn_train = xn_train';
xn_test = mapminmax('apply',x_test',inputps);
xn_test = xn_test';
[yn_train,outputps] = mapminmax(y_train');
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
model1 = svmtrain(yn_train,xn_train,cmd);
% save model1
 save model1
%% V. SVM仿真预测
[Predict_1,acc_1,pro1] = svmpredict(yn_train,xn_train,model1);
[Predict_2,acc_2,pro2] = svmpredict(yn_test,xn_test,model1);

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
%% VI. 绘图
% DBNSVRERR_1=y_test-predict_2;
% save DBNSVRERR_1 DBNSVRERR_1
DBNSVRERR_2=y_test-predict_2;
save DBNSVRERR_2 DBNSVRERR_2
%%
 ty=y_train*18;
typ=predict_1*18;
tty=y_test*18;
ttyp=predict_2*18;
[total1, percentage1] = clarke1(ty,typ)
[total2, percentage2] = clarke1(tty,ttyp)
