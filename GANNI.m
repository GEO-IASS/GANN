% Test of a GA-trained NN Iterative

clearvars
%clear all
close all

% Read MNIST Data
[testimages, testlabels, trainimages, trainlabels] = readMNIST();

% Train a network using the GA
inputsz = 28*28;
numhiddenlayers = 10;
outputelements = 10;
interweights = zeros(inputsz, inputsz, numhiddenlayers);
finalweights = zeros(outputelements, inputsz);

% Train the intermediate layers
for layers = 1:numhiddenlayers
    
    Nvars = inputsz * inputsz;
    lb = ones(1, Nvars)*(-1);
    ub = ones(1, Nvars);
    f = @(x)TrainFunc(x, trainimages, trainlabels, layers, interweights);
    options = gaoptimset('Display', 'iter', 'Generations', 500, 'PopulationSize', 200, 'CrossoverFraction', 0.5, 'EliteCount', 15, 'TolFun', 1e-9,...
        'UseParallel', 'always', 'PopulationType', 'bitstring', 'MutationFcn', {@mutationuniform, 0.02});
    [X, FVAL, EXITFLAG, OUTPUT, POPULATION, SCORES] = ga(f, Nvars, [], [], [], [], [], [], [], [], options);
    
    a = 10;
end

% Compare the results on the training data.
hardtest(X, trainimages, trainlabels);

% Compare the results on the testing data.

