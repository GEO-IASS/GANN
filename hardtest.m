function [ ErrorN ] = hardtest ( x, trainimages, trainlabels )
    
    ErrorN = 0;
    for i = 1:length(trainlabels)
        
        % Multiple Input by First Weights
        try
            firstlayer = trainimages(:,:,i); 
        catch
            disp('paused');
            drawnow
            pause
        end
        firstlayer = reshape(firstlayer, 28*28, 1);
        weights = x(1: 15*28*28);
        weights = reshape(weights, 15, 28*28);
        tempout = weights*firstlayer;
        tempout = 1./(1+exp(-tempout)); 

        % Multiple Middle layers
        for hcount = 0:3
            inputlayer = tempout;
            weights = x(15*28*28 + (15*15)*hcount + 1: 15*28*28 + (15*15)*(hcount+1));
            weights = reshape(weights, 15, 15);
            tempout = weights*inputlayer;
            tempout = 1./(1+exp(-tempout)); 
        end
        
        % Multiple Last Layers
        inputlayer = tempout;
        weights = x(15*28*28 + (15*15)*4 + 1: end);
        weights = reshape(weights, 10, 15);
        tempout = weights*inputlayer;
        tempout = 1./(1+exp(-tempout)); 
        
        % Calculate Fitness
        ReferenceOutput = zeros(10,1);
        Position = trainlabels(i) + 1;
        ReferenceOutput(Position) = 1;
        
        Sum = abs(round(round(tempout) - ReferenceOutput));
        if sum(Sum) > 0
            ErrorN = ErrorN + 1;
        end
              
    
    end
    
end

