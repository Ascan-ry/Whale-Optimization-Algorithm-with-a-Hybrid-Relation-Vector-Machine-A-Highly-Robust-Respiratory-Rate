%{
        A demo for regression using RVM with hybrid_kernel
%}

clc
clear 
close all
addpath(genpath(pwd))
k = randperm(size(AAA,1));
% AAA=sigmoid(AAA(:,[6 7 8 9 10]));
% [M,N]=find(AAA(1,5)==rvm.data(:,5));
AAA_P1=AAA(k(1:1993),[1 2 3 4 5]);AAA_T1=AAA(k(1:1993),6);
AAA_P1=sigmoid(AAA_P1);
AAA_P2=AAA(k(1994:end),[1 2 3 4 5]);AAA_T2=AAA(k(1994:end),6);
AAA_P2=sigmoid(AAA_P2);
% sinc funciton
trainData = AAA_P1;
trainLabel = AAA_T1;
testData = AAA_P2;
testLabel = AAA_T2;

% kernel function  WOA
% kernel_1 = Kernel('type', 'gaussian', 'gamma',50.7474);
% kernel_2 = Kernel('type', 'polynomial', 'degree', 1.5318);
% kernel_4 = Kernel('type', 'sigmoid', 'gamma', 58.6999);
% kernel_5 = Kernel('type', 'laplacian', 'gamma', 20.5505);
% kernelWeight = [0.9726  0.6631  0.0470 0.9224];


%%30次迭代  0.63
% kernel_1 = Kernel('type', 'gaussian', 'gamma',52.1255933292065);
% kernel_2 = Kernel('type', 'polynomial', 'degree',2.05900125886026);
% kernel_4 = Kernel('type', 'sigmoid', 'gamma',16.9885293441867);
% kernel_5 = Kernel('type', 'laplacian', 'gamma',13.7696098537998);
% kernelWeight =[0.803589385283221,0.0878842846139502,0.313086072698679,0.200511946361455];
% 
% % 100次迭代  0.7
% kernel_1 = Kernel('type', 'gaussian', 'gamma',67.828753602205);
% kernel_2 = Kernel('type', 'polynomial', 'degree',7.18311320418329);
% kernel_4 = Kernel('type', 'sigmoid', 'gamma',61.6068263258971);
% kernel_5 = Kernel('type', 'laplacian', 'gamma',11.2215311193506);
% kernelWeight =[0. 374533724563469, 0. 779954586622553, 0. 221981061421526, 0. 157506642958873];

% %200次迭代  
% kernel_1 = Kernel('type', 'gaussian', 'gamma',23.5707247410983 );
% kernel_2 = Kernel('type', 'polynomial', 'degree',1.14505414152630);
% kernel_4 = Kernel('type', 'sigmoid', 'gamma',48.0911064422381);
% kernel_5 = Kernel('type', 'laplacian', 'gamma',18.1161547795467);
% kernelWeight =[0.408921046828358	0.733833942957333	0	0.735183761169084];

% %%50次迭代 0.66
% kernel_1 = Kernel('type', 'gaussian', 'gamma',52.7193068869578);
% kernel_2 = Kernel('type', 'polynomial', 'degree',7.58244249278510);
% kernel_4 = Kernel('type', 'sigmoid', 'gamma',0.177147886780899);
% kernel_5 = Kernel('type', 'laplacian', 'gamma',8.11474963559049);
% kernelWeight =[0.444083100215762 0 0 0.0508424867254573];

%%50次迭代 0.68 无滤波
% kernel_1 = Kernel('type', 'gaussian', 'gamma',80);
% kernel_2 = Kernel('type', 'polynomial', 'degree',10);
% kernel_4 = Kernel('type', 'sigmoid', 'gamma',28.8192853458575);
% kernel_5 = Kernel('type', 'laplacian', 'gamma',24.8702176919702);
% kernelWeight =[1,1,0.473673807144141,0.306552304992037];

%50次迭代 1.0875mae
% kernel_1 = Kernel('type', 'gaussian', 'gamma',80);
% kernel_2 = Kernel('type', 'polynomial', 'degree',1.93456332564213);
% kernel_4 = Kernel('type', 'sigmoid', 'gamma',42.6144229843662);
% kernel_5 = Kernel('type', 'laplacian', 'gamma',80);
% kernelWeight =[0.930743020600100,1,0.123723433486323,0.236218247078536];
% 		[69.4259093688808,2.65647628883931,17.0326546218032,25.0072217399432]
% kernel_1 = Kernel('type', 'gaussian', 'gamma',66.1323);
% kernel_2 = Kernel('type', 'polynomial', 'degree',6.00947);
% kernel_5 = Kernel('type', 'laplacian', 'gamma',27.0815);
% kernelWeight = [ 0.928636     0.944747     0.633225];


kernel_1 = Kernel('type', 'gaussian', 'gamma',60.5731197888378);
kernel_2 = Kernel('type', 'polynomial', 'degree',6.13530492652714);
kernel_4 = Kernel('type', 'sigmoid', 'gamma',77.4416969776727);
kernel_5 = Kernel('type', 'laplacian', 'gamma',50.1869952143169);
kernelWeight =[0.192040369259671,0.0509971027609478,0.157254750748877,0.290367305535907];
% parameter optimization
opt.method = 'bayes'; % bayes, ga, pso
opt.display = 'on';
opt.iteration = 50;

% parameter
parameter = struct( 'display', 'on',...
                    'type', 'RVR',...
                    'kernelFunc', [kernel_1, kernel_2,kernel_4, kernel_5],...
                    'kernelWeight', kernelWeight);
rvm = BaseRVM(parameter);


%200次迭代  


% RVM model training, testing, and visualization
rvm.train(trainData, trainLabel);
results = rvm.test(testData, testLabel);

rvm.draw(results)


