function [ testimages, testlabels, trainimages, trainlabels ] = readMNIST()
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

% file format described at the website http://yann.lecun.com/exdb/mnist/

% Read the training images.
fid = fopen('train-images.idx3-ubyte', 'r');
info = dir('train-images.idx3-ubyte');
numimages = (info.bytes - 8)/(28*28);
trainimages = uint8(zeros([28 28 uint32(numimages)]));
status = fseek(fid, 16, 'bof');
data = fread(fid, 28*28*uint32(numimages));
trainimages = reshape(data, [28, 28, uint32(numimages)]);
trainimages = permute(trainimages, [2 1 3]);
fclose(fid)

% Read the test images
fid = fopen('t10k-images.idx3-ubyte', 'r');
info = dir('t10k-images.idx3-ubyte');
numimages = (info.bytes - 8)/(28*28);
testimages = uint8(zeros([28 28 uint32(numimages)]));
status = fseek(fid, 16, 'bof');
data = fread(fid, 28*28*uint32(numimages));
testimages = reshape(data, [28, 28, uint32(numimages)]);
testimages = permute(testimages, [2 1 3]);
fclose(fid)

% Read the training labels
fid = fopen('train-labels.idx1-ubyte', 'r');
info = dir('train-labels.idx1-ubyte');
numlabels = (info.bytes - 8);
trainlabels = uint8(zeros([1 uint32(numlabels)]));
status = fseek(fid, 8, 'bof');
trainlabels = fread(fid, 1*uint32(numlabels));
fclose(fid)

% Read the test labels
fid = fopen('t10k-labels.idx1-ubyte', 'r');
info = dir('t10k-labels.idx1-ubyte');
numlabels = (info.bytes - 8);
testlabels = uint8(zeros([1 uint32(numlabels)]));
status = fseek(fid, 8, 'bof');
testlabels = fread(fid, 1*uint32(numlabels));
fclose(fid)

end

