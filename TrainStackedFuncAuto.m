
function [ Fitness] = TrainStackedFuncAuto( x, trainimages, trainlabelsm, l)
    global tempimage;
    Fitness = 0;
    for i = 1:1
        
        try
            input = tempimage;
        catch
            disp('paused');
            drawnow
            pause
        end
        
        matrix1 = (x(1: 10*101));
        matrix1 = reshape(matrix1, 10, 101);
        input = reshape(input, 10*10, 1);
        input = [input; 1];
        output = matrix1*input;
        
        output = 1./(1+exp(-output)); 
        
        matrix2 = (x(101*10+1: end));
        matrix2 = reshape(matrix2, 100, 11);
        output = [output; 1];
        output = matrix2*output;
        
        Fitness = Fitness + norm(output-input(1:end-1)) + abs(sum(output(find(output<0))));
        
    end
end

