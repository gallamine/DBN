function [weights, biasesVis, biasesHid, errsum] = DBN_RBM(aVH,aHH, numNodes1, numNodes2, restart, PARAMS, offset, varargin)
% DBN_RBM ...
%   aVH - handle to visible activations
%   aHH - handle to hidden activations
%
%   Example
%   DBN_RBM

%   See also
%

%% AUTHOR    : Tushar Tank
%% $DATE     : 02-May-2013 13:30:36 $
%% $Revision : 1.00 $
%% DEVELOPED : 7.13.0.564 (R2011b)
%% FILENAME  : DBN_RBM.m
%% COPYRIGHT 2011 3 Phonenix Inc.

%% constants
maxEpoch                = PARAMS.maxEpoch;
learningRateW           = PARAMS.learningRateW;
learningRateBiasVis     = PARAMS.learningRateBiasVis;
learningRateBiasHid     = PARAMS.learningRateBiasHid;
weightCost              = PARAMS.weightCost;
initialMomentum         = PARAMS.initialMomentum;
finalMomentum           = PARAMS.finalMomentum;
numBatches              = PARAMS.numBatches;
batchSize               = PARAMS.batchSize;
epochToChangeMomentum   = PARAMS.epochToChangeMomentum;
numDimensions           = PARAMS.dataLength   
totalNum                = PARAMS.numBatches * PARAMS.batchSize;
binaryNode              = strcmp(PARAMS.nodeType,'binary');

%% Create figure

if PARAMS.displayVisualization == 1
    fig = figure;

end

%% update variables 
if restart == 1
    epoch=1;

    % Initializing symmetric weights and biases.
    weights     = 0.1*randn(numNodes1, numNodes2);
    biasesHid  = zeros(1,numNodes2);
    biasesVis  = zeros(1,numNodes1);
    deltaWeights  = zeros(numNodes1,numNodes2);
    deltaBiasesHid = zeros(1,numNodes2);
    deltaBiasesVis = zeros(1,numNodes1);
else
   disp('Starting from a non-zero state');
   epoch = varargin{1};
   load([varargin{2} 'state' num2str(varargin{3})]);
    deltaWeights  = zeros(numNodes1,numNodes2);
    deltaBiasesHid = zeros(1,numNodes2);
    deltaBiasesVis = zeros(1,numNodes1);
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
        end
        %fprintf(1,'epoch %d batch %d\r',epoch,batch);
        
        %% START POSITIVE PHASE

        %S = load([pathBatch1 'batch' num2str(batch)]);
        %data = loadBatch(dH,numBatches,batchSize);
        if binaryNode == 1
            posHidProbs = 1./(1 + exp(-data*weights - repmat(biasesHid,batchSize,1)));
        else
            posHidProbs = data*weights + repmat(biasesHid,batchSize,1);
        end
        batchData = posHidProbs; % for next level input
        posProds    = data' * posHidProbs;
        poshidact   = sum(posHidProbs);
        posvisact = sum(data);
        
        if binaryNode == 1
            poshidstates = posHidProbs > rand(batchSize,numNodes2);
        else
            poshidstates = posHidProbs + randn(batchSize,numNodes2);       % Sample from Normal(podHidProbs, 1)
        end
        
        %% START NEGATIVE PHASE
        negdata = 1./(1 + exp(-poshidstates*weights' - repmat(biasesVis,batchSize,1)));
        
        if binaryNode == 1
            negHidProbs = 1./(1 + exp(-negdata*weights - repmat(biasesHid,batchSize,1)));
        else
            negHidProbs = negdata*weights + repmat(biasesHid,batchSize,1);
        end
        negProds  = negdata'*negHidProbs;
        neghidact = sum(negHidProbs);
        negvisact = sum(negdata);
        
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
        
        weights = weights + deltaWeights;
        biasesVis = biasesVis + deltaBiasesVis;
        biasesHid = biasesHid + deltaBiasesHid;
        
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

