function [ Fitness ] = TrainFunc( x, trainimages, trainlabels, layers, interweights )

%     Indices = round(rand(1,10000)*59999)+1;
    Fitness = 0;
    matrix = reshape(x, 28*28, 28*28);
    for i = 1:1
        
        try
%             input = trainimages(:,:,Indices(i)); 
        input = trainimages(:,:,i); 
        catch
            disp('paused');
            drawnow
            pause
        end
        
        input = reshape(input, 28*28, 1);
%         output = input;
        
        % Propogate till the last computed layers
%         for l = 1:layers-1
%             input = output;
%             matrix = reshape(interweights(l), 28*28, 28*28);
%             output = matrix*input;
%         end
        
        % Calculate the unpropogates intermediate layer
%         input = output;
        
        output = matrix*input;
        Fitness = Fitness + norm(output-input);
        
    end
%     disp('check...')
    
end