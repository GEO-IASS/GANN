% Test of a GA-trained NN

% clearvars
clear all
close all

% Read MNIST Data
[testimages, testlabels, trainimages, trainlabels] = readMNIST();

% Train a network using the GA
InputSize =  25*28*28 + 8*25*25 + 10*25;
Nvars = InputSize;

f = @(x)EconFunc(x, trainimages, trainlabels);
options = gaoptimset('Display', 'iter', 'Generations', 200, 'PopulationSize', 500, 'CrossoverFraction', 0.5, 'EliteCount', 20,...
    'TolFun', 1e-9, 'UseParallel', 'always', 'CreationFcn', @gacreationuniform);
[X, FVAL, EXITFLAG, OUTPUT, POPULATION, SCORES] = ga(f, Nvars, [], [], [], [], [], [], [], [], options);

% Compare the results on the training data.
hardtest(X, trainimages, trainlabels);

% Compare the results on the testing data.