function [ W, outputimgs ] = TrainGAAutoEncoder(inputimgs, hratio, sigmoidactivation)

    global numimages;

    [nimg, sz] = size(inputimgs);
    hsz = round(hratio*sz);
    
    inputimgs = inputimgs';

    Nvars = hsz*(sz+1) + sz; %+(sz)*(hsz+1);
    f = @(x)TrainInternal(x, inputimgs, hsz, sz, nimg, sigmoidactivation);
    options = gaoptimset('Display', 'iter', 'Generations', 1000, 'PopulationSize', 30, 'CrossoverFraction', 0.5, 'EliteCount', 2, 'TolFun', 1e-9, 'UseParallel', 'never',...
        'MutationFcn', {@mutationgaussian, 1, 1});%,...
    [W, FVAL, EXITFLAG, OUTPUT, POPULATION] = ga(f, Nvars, [], [], [], [], ones(1,Nvars)*-.05, ones(1,Nvars)*.05, [], [], options);
    
%     POPULATION = single(POPULATION);
%     
%     f = @(x)TrainInternal(x, inputimgs, hsz, sz, nimg);
%     options = gaoptimset('Display', 'iter', 'Generations', 5000, 'PopulationSize', 200, 'CrossoverFraction', 0.5, 'EliteCount', 15, 'TolFun', 1e-18, 'UseParallel', 'never',...
%         'MutationFcn', {@mutationgaussian, 25, 1}, 'InitialPopulation', POPULATION);
%     tic
%     [W] = ga(f, Nvars, [], [], [], [], [], [], [], [], options);
%     toc
    
    outputimgs = GenOutInternal(W, inputimgs, hsz, sz, nimg, sigmoidactivation);
    outputimgs = outputimgs';

end

function [Fitness] = TrainInternal(x, inputimgs, hsz, sz, nimg, sigmoidactivation)

    global numimages;   

    Fitness = 0;
    matrix1 = (x(1: hsz*(sz+1)));
    matrix1 = reshape(matrix1, hsz, sz+1);
%     matrix2 = (x(hsz*(sz+1)+1: end));
%     matrix2 = reshape(matrix2, sz, (hsz+1));
    matrix2 = matrix1(:,1:end-1);
    matrix2 = matrix2';
    matrix2 = [matrix2 (x(hsz*(sz+1)+1:end))'];
    
    input = inputimgs;
    input = [input; ones(1,numimages)];
    output = matrix1*input;
    if sigmoidactivation
        output = 1./(1+exp(-output));
    end
    output = [output; ones(1,numimages)];
    output = matrix2*output;
    Diff = input(1:end-1,:) - output;
    colnorm=sqrt(sum(Diff.^2,1));
    Fitness = sum(colnorm)/numimages;
    
%   Calculate the first norm
%   firstnorm=sum(abs(x));
%   Fitness = (0.99)*Fitness + (0.01)*firstnorm;
    
end

function [Output] = GenOutInternal(x, inputimgs, hsz, sz, nimg, sigmoidactivation)
 
    global numimages;

    matrix1 = (x(1: hsz*(sz+1)));
    matrix1 = reshape(matrix1, hsz, sz+1);
    matrix2 = matrix1(:,1:end-1);
    matrix2 = matrix2';
    matrix2 = [matrix2 (x(hsz*(sz+1)+1:end))'];
    
    input = inputimgs;
    input = [input; ones(1,numimages)];
    output = matrix1*input;
    if sigmoidactivation
        output = 1./(1+exp(-output)); 
    end
    Output = output;
    
end

