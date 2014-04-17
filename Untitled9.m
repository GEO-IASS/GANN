% test script

close all
clearvars

diary
diary on

global numimages;
numimages = 100;

[testimages, testlabels, trainimages, trainlabels] = readMNIST();

TempData = reshape(trainimages(:,:,1:numimages), 28*28, numimages);
TempData = TempData';
Data1 = TempData;

clear trainimages

% for i = 1:10000
%     temp = trainimages(:,:,i);
%     temp = reshape(temp, 1, 28*28);
%     Data(i,:) = temp;
% end
% Data = single(Data);

[ W1, Data2 ] = TrainGAAutoEncoder(Data1, 0.51); 
save C:\encode1.mat 
[ W2, Data3 ] = TrainGAAutoEncoder(Data2, .50);
save C:\encode2.mat 
[ W3, Data4 ] = TrainGAAutoEncoder(Data3, .20);
save C:\encode3.mat
[ W4, Data5 ] = TrainGAAutoEncoder(Data4, .25);
save C:\encode4.mat

