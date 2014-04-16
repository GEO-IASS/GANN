function [ Fitness ] = EconFunc( x, trainimages, trainlabels )

    Indices = round(rand(1,10000)*59999)+1;
    Fitness = 0;
    for i = 1:10000
        
        % Multiple Input by First Weights
        try
            firstlayer = trainimages(:,:,Indices(i)); 
        catch
            disp('paused');
            drawnow
            pause
        end
        firstlayer = reshape(firstlayer, 28*28, 1);
        weights = x(1: 25*28*28);
        weights = reshape(weights, 25, 28*28);
        tempout = weights*firstlayer;
        tempout = 1./(1+exp(-tempout)); 

        % Multiple Middle layers
        for hcount = 0:7
            inputlayer = tempout;
            weights = x(25*28*28 + (25*25)*hcount + 1: 25*28*28 + (25*25)*(hcount+1));
            weights = reshape(weights, 25, 25);
            tempout = weights*inputlayer;
            tempout = 1./(1+exp(-tempout)); 
        end
        
        % Multiple Last Layers
        inputlayer = tempout;
        weights = x(25*28*28 + (25*25)*8 + 1: end);
        weights = reshape(weights, 10, 25);
        tempout = weights*inputlayer;
        tempout = 1./(1+exp(-tempout)); 
        
        % Calculate Fitness
        ReferenceOutput = zeros(10,1);
        Position = trainlabels(Indices(i)) + 1;
        ReferenceOutput(Position) = 1;
        Fitness = Fitness + sum(abs((ReferenceOutput - tempout)));
    
    end
    
    Fitness;

end

