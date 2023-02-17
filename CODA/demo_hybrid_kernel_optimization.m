%{
    A demo for hybrid-kernel RVM model with Parameter Optimization
%}


clc
clear
close all
addpath(genpath(pwd))
% 
% k = randperm(size(AAA1,1));
% AAA(:,2)=AAA(:,2)*(-1);
% AAA(:,3:4)=AAA(:,3:4)*10;

% AAA_mean=mean(AAA(k(1:1903),1:5));
% for i=1:5
%     AAA(:,i)=AAA(:,i)./AAA_mean(i);
% end
% AAA(:,1:5)=zscore(AAA(:,1:5));
% AAA(:,1)=AAA(:,1)*b(1);
% AAA(:,2)=AAA(:,2)*b(2);
% AAA(:,3)=AAA(:,3)*b(3);
% AAA(:,4)=AAA(:,4)*b(4);
% AAA(:,5)=AAA(:,5)*b(5);
%  normalize each row to unit
% AAA(:,[1 2 3 5]) = AAA(:,[1 2 3 5])./repmat(sqrt(sum(AAA(:,[1 2 3 5]).^2,2)),1,size(AAA(:,[1 2 3 5]),2));
% %  normalize each column to unit
% AAA(:,[1 2 3 5]) = AAA(:,[1 2 3 5])./repmat(sqrt(sum(AAA(:,[1 2 3 5]).^2,1)),size(AAA(:,[1 2 3 5]),1),1);

k = randperm(size(AAA,1));
AAA_P1=AAA(k(1:1903),[1 3 4 5]);AAA_T1=AAA(k(1:1903),6);
 AAA_P1=sigmoid(AAA_P1);
AAA_P2=AAA(k(1904:end),[1 3 4 5]);AAA_T2=AAA(k(1904:end),6);
 AAA_P2=sigmoid(AAA_P2);
% [COEFF,AAA_P1,latent,tsquared,explained,mu]=pca(AAA_P1);
% [COEFF1,AAA_P2,latent1,tsquared1,explained1,mu1]=pca(AAA_P2);


trainData = AAA_P1;
trainLabel = AAA_T1;
testData = AAA_P2;
testLabel = AAA_T2;

% kernel function
kernel_1 = Kernel('type', 'gaussian', 'gamma',52.7193068869578);
kernel_2 = Kernel('type', 'polynomial', 'degree',7.58244249278510);
kernel_4 = Kernel('type', 'sigmoid', 'gamma',0.177147886780899);
kernel_5 = Kernel('type', 'laplacian', 'gamma',8.11474963559049);
%kernelWeight =[0.00225920608995855,0.0117866663696470,0.00843011528192730,0.198674120718430];



% parameter optimization
opt.method = 'bayes'; % bayes, ga, pso
opt.display = 'on';
opt.iteration = 50;

% parameter
parameter = struct( 'display', 'on',...
                    'type', 'RVR',...
                    'kernelFunc', [kernel_1, kernel_2,kernel_4,kernel_5],...
                    'optimization', opt);
rvm = BaseRVM(parameter);

% RVM model training, testing, and visualization
rvm.train(trainData, trainLabel);
results = rvm.test(testData, testLabel);
% M(i)=results.performance.MAE;

rvm.draw(results)