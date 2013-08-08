function [ weights, weightsC, biasesVis, biasesHid, biasesC, errsum ] = RBM_FIT(aVH,aHH, dH, numNodes1, numNodes2, restart, PARAMS, offset,varargin)
%RBM_FIT Use contrastic divergence to fit the last layer of a RBM to hard
%labels. Some magic going on here. Based on Andrej Karpathy's code: https://code.google.com/p/matrbm/
%  aVH - Handle to activations of the visible layer (last layer)
%  dH - Handle to data file containing the labels, Y.
%% Constants
maxEpoch                = PARAMS.maxEpoch;
learningRateW           = PARAMS.learningRateW;
learningRateBiasVis     = PARAMS.learningRateBiasVis;
learningRateBiasHid     = PARAMS.learningRateBiasHid;
learningRateBiasLabel   = PARAMS.learningRateBiasLabel;
weightCost              = PARAMS.weightCost;
initialMomentum         = PARAMS.initialMomentum;
finalMomentum           = PARAMS.finalMomentum;
numBatches              = PARAMS.numBatches;
batchSize               = PARAMS.batchSize;
epochToChangeMomentum   = PARAMS.epochToChangeMomentum;
numDimensions           = PARAMS.dataLength   
totalNum                = PARAMS.numBatches * PARAMS.batchSize;
numTargetClass          = PARAMS.numTargets;

%% Create figure

if PARAMS.displayVisualization == 1
    fig = figure;

end

%% update variables 
if restart == 1
    epoch=1;

    % Initializing symmetric weights and biases.
    weights     = 0.1*randn(numNodes1, numNodes2);
    weightsC    = 0.1*randn(numTargetClass,numNodes2);
    biasesHid  = zeros(1,numNodes2);
    biasesVis  = zeros(1,numNodes1);
    biasesC     = zeros(1,numTargetClass);
    deltaWeights  = zeros(numNodes1,numNodes2);
    deltaBiasesHid = zeros(1,numNodes2);
    deltaBiasesVis = zeros(1,numNodes1);
    deltaWeightsC  = zeros(numTargetClass,numNodes2);
    deltaBiasC     = zeros(1,numTargetClass);
else
   disp('Starting from a non-zero state');
   epoch = varargin{1};
   load([varargin{2} 'state' num2str(varargin{3})]);
   deltaWeights  = zeros(numNodes1,numNodes2);
    deltaBiasesHid = zeros(1,numNodes2);
    deltaBiasesVis = zeros(1,numNodes1);
    deltaWeightsC  = zeros(numTargetClass,numNodes2);
    deltaBiasC     = zeros(1,numTargetClass);
end

%% train RBM weights
data = NaN(batchSize,numNodes1);
for epoch = epoch:maxEpoch
    %fprintf(1,'epoch %d\r',epoch);
    errsum=0;
    
    idx = randperm(totalNum)+offset;
    for batch = 1:numBatches
        % Generate batch indices
        batchStart = (batch - 1)*batchSize + 1;
        batchEnd = batch*batchSize; 
        for kk = 1:batchSize 
            data(kk,1:numNodes1) = aVH.X(idx(batchStart+kk-1),1:numNodes1);   
            labels(kk,1:numTargetClass) = dH.Y(idx(batchStart+kk-1),1:numTargetClass);  %
        end
        %fprintf(1,'epoch %d batch %d\r',epoch,batch);
        
        %% START POSITIVE PHASE

        %S = load([pathBatch1 'batch' num2str(batch)]);
        %data = loadBatch(dH,numBatches,batchSize);
        posHidProbs = 1./(1 + exp(-data*weights - labels*weightsC - repmat(biasesHid,batchSize,1)));  %
        batchData = posHidProbs; % for next level input
        posProds    = data' * posHidProbs;
        posProdsC    = labels' * posHidProbs;       %
        poshidact   = sum(posHidProbs);
        posvisact = sum(data);
        posvisactC = sum(labels);           %
        
        poshidstates = posHidProbs > rand(batchSize,numNodes2);
        
        %% START NEGATIVE PHASE
        negdata = 1./(1 + exp(-poshidstates*weights' - repmat(biasesVis,batchSize,1)));
        
        negclasses = softmaxPmtk(poshidstates*weightsC' + repmat(biasesC,batchSize,1));     %
		negclassesstates = softmax_sample(negclasses);                                      %
        
        negHidProbs = 1./(1 + exp(-negdata*weights -negclassesstates*weightsC - repmat(biasesHid,batchSize,1)));
        negProds  = negdata'*negHidProbs;
        negProdsC  = negclassesstates'*negHidProbs;          %
        neghidact = sum(negHidProbs);
        negvisact = sum(negdata);
        
        negvisactC = sum(negclassesstates);           %
        
        %% Udate running error and momentum
        err = sum(sum( (data-negdata).^2 ));
        errsum = err + errsum;
        
        if epoch > epochToChangeMomentum
            momentum=finalMomentum;
        else
            momentum=initialMomentum;
        end;
        
        %%  UPDATE WEIGHTS AND BIASES 
        deltaWeights = momentum*deltaWeights + ...
            learningRateW*( (posProds-negProds)/batchSize - weightCost*weights);
        deltaBiasesVis = momentum*deltaBiasesVis + (learningRateBiasVis/batchSize)*(posvisact-negvisact);
        deltaBiasesHid = momentum*deltaBiasesHid + (learningRateBiasHid/batchSize)*(poshidact-neghidact);
        
        deltaWeightsC = momentum*deltaWeightsC + ...
            learningRateW*( (posProdsC-negProdsC)/batchSize - weightCost*weightsC);         %
        deltaBiasC = momentum*deltaBiasC + (learningRateBiasLabel/batchSize)*(posvisactC-negvisactC);   %
        
        weights = weights + deltaWeights;
        biasesVis = biasesVis + deltaBiasesVis;
        biasesHid = biasesHid + deltaBiasesHid;
        weightsC = weightsC + deltaWeightsC;                %
        biasesC = biasesC + deltaBiasC;               %
        
        % save layer 2 batch data as input to next layer
        if epoch == maxEpoch
            % $$$$ NOTE $$$$ - Might need to preinitialize the X array
            % first for efficiency??
            for kk = 1:batchSize 
                aHH.X(idx(batchStart+kk-1),1:numNodes2) = batchData(kk,1:numNodes2);
            end
            %save([pathBatch2 'batch' num2str(batch)], 'batchData');
        end
        %% Update display
        if PARAMS.displayVisualization == 1 && mod(batch,20) == 1
            
           updateDisplay(fig,weights,biasesHid,biasesVis,deltaWeights,posHidProbs,negdata); 
           drawnow;
           %pause;
        end
    end
    fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum);
end
