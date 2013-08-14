function [finalModelFileName] = DBN_BackProp(dH,pathDir, PARAMS,varargin)
% DBN_BACKPROP ...
%   DBN_BACKPROP
%
%   dH - Data handle to input data. X is NxD, Y is Nx1
%
%   Example
%   DBN_BackProp

%   See also
%

%% AUTHOR    : Tushar Tank
%% Edited    : William Cox
%% $DATE     : 02-May-2013 17:18:13 $
%% $Revision : 1.00 $
%% DEVELOPED : 7.13.0.564 (R2011b)
%% FILENAME  : DBN_BackProp.m
%% COPYRIGHT 2011 3 Phonenix Inc.

%% Constants
maxEpoch                = PARAMS.maxBackPropEpoch;
numNodes                = numel(PARAMS.nodes);
numTargetClass          = PARAMS.numTargets;
numBatches              = PARAMS.numBatches;
batchSize               = PARAMS.batchSize;
numDimensions           = PARAMS.dataLength;
numCombinedBatches      = PARAMS.numCombinedBatches;
maxIterations           = PARAMS.numberOfLineSearches;
numValidateBatches      = PARAMS.numValidate;
combo                   = PARAMS.combo;
totalTrainNum           = PARAMS.numBatches * PARAMS.batchSize;
totalValidatenNum       = PARAMS.numValidate * PARAMS.batchSize;
comboBatchSize          = combo*batchSize;
dropoutP                = PARAMS.dropOutRatio;
CHECK_INTERVAL          = PARAMS.backpropCheckInterval;
MEASUREMENT_PROP        = PARAMS.measurementProp_Backprop;
numReduBatches          = round(numBatches*MEASUREMENT_PROP);           % If we're only evaluating test/train performance on a smaller number of batches, how many?
numValReduBatches       = round(numValidateBatches*MEASUREMENT_PROP);
NUM_TOP_LAYER_BP        = PARAMS.numTopLayerBackpropEpochs;
%% Operating flag that should go away

TEST_AGAINST_HINTON = PARAMS.useFileBatches;

if numel(varargin) > 0
    pathBatch = varargin{1};
    pathValidate = varargin{2};
end
if TEST_AGAINST_HINTON == 1
    disp('Using non-random batch mixing');
end
%% Load and initilize states
% initialize weights and biases to dummy val
[w{1:numNodes}] = deal(eye(2));
[v{1:numNodes}] = deal(eye(2));
wIdx = zeros(numNodes,1);
for ii = 1:numNodes
    S = load([pathDir 'state' num2str(ii)]);
    w{ii} = [S.weights; S.biasesHid];
    v{ii} = S.biasesVis;
    wIdx(ii) = size(w{ii},1)-1;
 

end
if ~isfield('S','weightsC')
    w{numNodes+1} = 0.1*randn(size(w{end},2)+1,numTargetClass);
else
    w{numNodes+1} = [S.weightsC'; S.biasesC];
end
wIdx(numNodes+1) = size(w{numNodes+1},1)-1;
wIdx(numNodes+2) = numTargetClass;

testError              = nan(1,maxEpoch);
testErrorNormalized    = nan(1,maxEpoch);
trainError             = nan(1,maxEpoch);
trainErrorNormalized   = nan(1,maxEpoch);


fprintf(1,'%d training batches of %d samples\n', numBatches, batchSize);
fprintf(1,'%d validation batches of %d samples\n', numValidateBatches, batchSize);
trainBatchRanges = 1:numBatches; 
valBatchRanges = 1:numValidateBatches;

for epoch = 1:maxEpoch

    
    %% training misclassification rate
    error = 0;
    counter = 0;
    if mod(epoch,CHECK_INTERVAL) == 0
        idx =  1:totalTrainNum;
        label = zeros(batchSize,numTargetClass);
        if MEASUREMENT_PROP ~= 1
            tmp = randperm(numBatches);
            trainBatch = tmp(1:numReduBatches);
            tmp = randperm(numValidateBatches);
            valBatchRanges = tmp(1:numValReduBatches);
            
        end
        for batch = trainBatchRanges
            if TEST_AGAINST_HINTON == 1
                S = load([pathBatch '/batch' num2str(batch)]);
                data = [S.batchData ones(batchSize,1)];
                label = [S.batchLabel];
                clear S;
            else
                % We're reading all the data, so this doesn't have to be random
                batchStart = (batch - 1)*batchSize + 1;
                batchEnd = batch*batchSize;
                
                dataNoBias(1:batchSize,1:numDimensions) = dH.X(idx(batchStart:batchEnd),1:numDimensions);
                label(1:batchSize,1:numTargetClass) = dH.Y(idx(batchStart:batchEnd),1:numTargetClass);
                %%%
                
                data = [dataNoBias ones(batchSize,1)];
            end
            if mod(batch,50) == 1
                fprintf(1,'Epoch %d\tBatch %d\n', epoch, batch);
            end
            
            labelEst = nn_fwd(w,data,numNodes,dropoutP,batchSize,numTargetClass);
            
            [~, idxEst]= max(labelEst,[],2);
            [~, idxTrue]= max(label,[],2);
            
            counter = counter + sum(idxEst == idxTrue);                 % Count true labels
            
            error = error - sum(sum(label(:,1:end).*log(labelEst)));
        end
        trainError(epoch) = (batchSize*numReduBatches-counter);
        trainErrorNormalized(epoch)= error/numReduBatches;
        
        
        %% test misclassification rate
        error = 0;
        counter = 0;
        
        offset = totalTrainNum;
        idx = [1:totalValidatenNum] + offset;

        for batch = valBatchRanges
            if TEST_AGAINST_HINTON == 1
                S = load([pathValidate '/batch' num2str(batch)]);
                data = [S.batchData ones(batchSize,1)];
                label = [S.batchLabel];
                clear S;
            else    % Ideally we should figure out how much data we can load at once and process in that chunk size
                batchStart = (batch - 1)*batchSize + 1;
                batchEnd = batch*batchSize;
                dataNoBias(1:batchSize,1:numDimensions) = dH.X(idx(batchStart:batchEnd),1:numDimensions);
                label(1:batchSize,1:numTargetClass) = dH.Y(idx(batchStart:batchEnd),1:numTargetClass);
                data = [dataNoBias ones(batchSize,1)];
            end
            
            labelEst = nn_fwd(w,data,numNodes,dropoutP,batchSize,numTargetClass);
            
            [~, idxEst]= max(labelEst,[],2);
            [~, idxTrue]= max(label,[],2);
            counter = counter + sum(idxEst==idxTrue);
            error = error- sum(sum(label(:,1:end).*log(labelEst)));    % Cross-entropy error. See http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression for Softmax
        end
        testError(epoch) = (batchSize*numValReduBatches-counter);
        testErrorNormalized(epoch)= error/numValReduBatches;
        
        fprintf(1,'Before epoch %d\nTrain # misclassified: %d (from %d)\nTest # misclassified: %d (from %d) \t \t \n',...
            epoch,trainError(epoch),batchSize*numReduBatches,testError(epoch),batchSize*numValReduBatches);
    end
    %% gradient descent with three line searches
    
    idx = randperm(totalTrainNum);
    for batch = 1:numCombinedBatches
        if mod(batch,5) == 0
            disp(['CjGD on ' num2str(batch) ' of ' num2str(numCombinedBatches)]); 
        end
        % make a bigger minibatch
        if TEST_AGAINST_HINTON == 1
            data = [];
            label = [];
            for count = 0:combo-1
                S = load([pathBatch '/batch' num2str(batch+count)]);
                temp = [S.batchData ones(batchSize,1)];
                data = [data; temp;];
                label = [label; S.batchLabel;];
                %label(batch:batch+comb-1) = S.label;
                clear S;
            end
            comboData = data(:,1:end-1);
        else            
            batchStart = (batch - 1)*comboBatchSize + 1;
            comboData = zeros(comboBatchSize,numDimensions);
            for kk = 1:comboBatchSize
                comboData(kk,1:numDimensions) = dH.X(idx(batchStart+kk-1),1:numDimensions);
                label(kk,1:numTargetClass) = dH.Y(idx(batchStart+kk-1),1:numTargetClass);
            end
            data = [comboData ones(comboBatchSize,1)];
        end

        %% conjgate gradient descent with linesearches 
        if epoch < NUM_TOP_LAYER_BP  % First update top-level weights holding other weights fixed.
            
            [~,data] = nn_fwd(w,data,numNodes,dropoutP,comboBatchSize,numTargetClass);
            % remove bias
            data = data(:, 1:end-1);
            
            % vectorize final layer wieghts 
            wFinal = w{numNodes+1}(:);
            
            % CG and update weights
            dim = wIdx(numNodes+1:end);
            [X, ~] = minimize(wFinal,'DBN_ConjugateGradientInit',maxIterations,dim,data,label,PARAMS);
            w{end} = reshape(X,wIdx(end-1)+1,wIdx(end));
        
        else
            
            wFinal = [];
            for level = 1:numNodes+1
                wFinal = [wFinal; w{level}(:)];
            end
            
            dim = wIdx;
            [X, ~] = minimize(wFinal,'DBN_ConjugateGradient',maxIterations,dim,comboData,label,PARAMS);
            
            delta = 0;
            for ii = 1:numNodes+1
                row = wIdx(ii) + 1;
                col = wIdx(ii+1);
                w{ii} = reshape(X(delta+1:delta+(row*col)),row,col);
                delta = delta + row*col;
            end
        end
    end
    save(['finalState' num2str(epoch)], 'w', 'v', 'trainError', 'trainErrorNormalized', 'testError', 'testErrorNormalized')
end
finalModelFileName = ['finalState' num2str(epoch) '.mat'];