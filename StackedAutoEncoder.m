clearvars
close all

global tempimage;
tempimage = round(fspecial('gaussian',[10 10],1)*250);

% Read MNIST Data
[testimages, testlabels, trainimages, trainlabels] = readMNIST();

nencoders = 3;
inputsz = 100;
hiddensz = 10;

% generate training data
for i = 1:100
    Data(i,:) = zeros(1,50);
    Position = round(rand(1,1)*30)+10;
    Data(i,Position) = 700;
    TrainingData(i,:) = convn(Data(i,:), fspecial('gaussian',[1 7],1), 'same');
end

Nvars = (hiddensz*(inputsz+1) + inputsz*(hiddensz+1));

Weights = zeros(nencoders, Nvars);

for layers = 1:nencoders
    
    f = @(x)TrainStackedFuncAuto(x, trainimages, trainlabels, layers);
    options = gaoptimset('Display', 'iter', 'Generations', 1000, 'PopulationSize', 200, 'CrossoverFraction', 0.5, 'EliteCount', 25, 'TolFun', 1e-12)%,...
        %'UseParallel', 'always'); %, 'MutationFcn', {@mutationuniform, 0.05});
    [X, FVAL, EXITFLAG, OUTPUT, POPULATION, SCORES] = ga(f, Nvars, [], [], [], [], [], [], [], [], options);
  
    a = 10;
end