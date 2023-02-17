%{
        A demo for regression using RVM
%}

clc
clear
close all
addpath(genpath(pwd))

% sinc funciton
load Z
trainData = Z(1:1000,1:4);
trainLabel = Z(1:1000,5);
testData = Z(1001:1500,1:4);
testLabel = Z(1001:1500,5);

% % kernel function
% kernel = Kernel('type', 'gaussian', 'gamma', 0.1);
kernel = Kernel('type', 'polynomial', 'degree', 1);

% parameter
parameter = struct( 'display', 'on',...
                    'type', 'RVR',...
                    'kernelFunc', kernel);
rvm = BaseRVM(parameter);

% RVM model training, testing, and visualization
rvm.train(trainData, trainLabel);
results = rvm.test(testData, testLabel);
rvm.draw(results)


