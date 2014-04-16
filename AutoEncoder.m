clearvars
%clear all
close all

global tempimage;
tempimage = round(rand(10,10)*250);
tempimage = fspecial('gaussian',[10 10],1)*100;

% Read MNIST Data
[testimages, testlabels, trainimages, trainlabels] = readMNIST();

for layers = 1:1
    
    Nvars = 10*101 + 100*11;
    f = @(x)TrainFuncAuto(x, trainimages, trainlabels);
    options = gaoptimset('Display', 'iter', 'Generations', 1000, 'PopulationSize', 300, 'CrossoverFraction', 0.5, 'EliteCount', 35, 'TolFun', 1e-6)%,...
        %'UseParallel', 'always'); %, 'MutationFcn', {@mutationuniform, 0.05});
    [X, FVAL, EXITFLAG, OUTPUT, POPULATION, SCORES] = ga(f, Nvars, [], [], [], [], [], [], [], [], options);

  
    a = 10;
end