function [xxx] = ttt( x,AAA)

% % trainData = AAA1(1:3000,1:6);
% % trainLabel = AAA1(1:3000,7);
% % testData = AAA1(3001:end,1:6);
% % testLabel = AAA1(3001:end,7);
% AAA_P1=zscore(AAA11(1:2080,1:7));AAA_T1=AAA11(1:2080,8);
% AAA_P1=sigmoid(AAA_P1);
% AAA_P2=zscore(AAA11(2081:2964,1:7));AAA_T2=AAA11(2081:2964,8);
% AAA_P2=sigmoid(AAA_P2);
% % [COEFF,AAA_P1,latent,tsquared,explained,mu]=pca(AAA_P1);
% % [COEFF1,AAA_P2,latent1,tsquared1,explained1,mu1]=pca(AAA_P2);
% 
% trainData = AAA_P1;
% trainLabel = AAA_T1;
% testData = AAA_P2;
% testLabel = AAA_T2;
%K
  k = randperm(size(AAA,1));
AAA_P1=AAA(k(1:1903),[1 2 3 4 5]);AAA_T1=AAA(k(1:1903),6);
AAA_P1=sigmoid(AAA_P1);
AAA_P2=AAA(k(1904:end),[1 2 3 4 5]);AAA_T2=AAA(k(1904:end),6);
AAA_P2=sigmoid(AAA_P2);
% AAA_P1=AAA(1:1903,[1 2 3 4 5]);AAA_T1=AAA(1:1903,6);
%  AAA_P1=sigmoid(AAA_P1);
% AAA_P2=AAA(1904:end,[1 2 3 4 5]);AAA_T2=AAA(1904:end,6);
%  AAA_P2=sigmoid(AAA_P2);

trainData = AAA_P1;
trainLabel = AAA_T1;
testData = AAA_P2;
testLabel = AAA_T2;
% kernel function
kernel_1 = Kernel('type', 'gaussian', 'gamma',  x(1) );
kernel_2 = Kernel('type', 'polynomial', 'degree',  x(2));
kernel_4 = Kernel('type', 'sigmoid', 'gamma',   x(3));
kernel_5 = Kernel('type', 'laplacian', 'gamma',  x(4));
kernelWeight = [x(5) x(6) x(7) x(8)];

% parameter optimization
opt.method = 'bayes'; % bayes, ga, pso
opt.display = 'on';
opt.iteration = 50;

% parameter
parameter = struct( 'display', 'on',...
                    'type', 'RVR',...
                    'kernelFunc', [kernel_1, kernel_2 ,kernel_4,kernel_5],...
                    'kernelWeight', kernelWeight);
rvm = BaseRVM(parameter);

% RVM model training, testing, and visualization
rvm.train(trainData, trainLabel);
results = rvm.test(testData, testLabel);
xxx=results.performance.RMSE;%+results.performance.RMSE;

end