function [ testError,trainError,testErrorNormalized,trainErrorNormalized ] = ...
    DBN_UNFOLD_NOBACKPROP( dH,pathDir, PARAMS )
%DBN_UNFOLD_NOBACKPROP Unwrap a DBN that hasn't had backprop performed yet
%and test the performance just on the CD trained network
%   Detailed explanation goes here

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
w{numNodes+1} = [S.weightsC'; S.biasesC];
wIdx(numNodes+1) = size(w{numNodes+1},1)-1;
wIdx(numNodes+2) = numTargetClass;

%% Training data misclassification

%% training misclassification rate
error = 0;
counter = 0;

idx = 1:totalTrainNum;
label = zeros(batchSize,numTargetClass);
for batch = 1:numBatches
    % Generate batch indices
    batchStart = (batch - 1)*batchSize + 1; 
    batchEnd = batch*batchSize;
    
    dataNoBias(1:batchSize,1:numDimensions) = dH.X(idx(batchStart:batchEnd),1:numDimensions);
    label(1:batchSize,1:numTargetClass) = dH.Y(idx(batchStart:batchEnd),1:numTargetClass);
    
    %%%
    if mod(batch,50) == 1
        fprintf(1,'Find err in training, Batch %d\n', batch);
    end
    data = [dataNoBias ones(batchSize,1)];

    for level = 1:numNodes
        temp = 1./(1 + exp(-data*w{level}));
        data = [temp ones(batchSize, 1)];
    end

    labelEst = exp(data*w{numNodes+1});     % Fit to labels
    labelEst = labelEst./repmat(sum(labelEst,2), 1, numTargetClass);
    [~, idxEst]= max(labelEst,[],2);
    [~, idxTrue]= max(label,[],2);
    counter = counter + length(find(idxEst==idxTrue));
    error = error - sum(sum(label(:,1:end).*log(labelEst)));
end
trainError = (batchSize*numBatches-counter);
trainErrorNormalized = error/numBatches;


%% Testing data misclassification rate
error = 0;
counter = 0;

offset = totalTrainNum;
idx = [1:totalValidatenNum] + offset;
for batch = 1:numValidateBatches
    batchStart = (batch - 1)*batchSize + 1;
    batchEnd = batch*batchSize;

    dataNoBias(1:batchSize,1:numDimensions) = dH.X(idx(batchStart:batchEnd),1:numDimensions);
    label(1:batchSize,1:numTargetClass) = dH.Y(idx(batchStart:batchEnd),1:numTargetClass);
    data = [dataNoBias ones(batchSize,1)];


    for level = 1:numNodes
        temp = 1./(1 + exp(-data*w{level}));
        data = [temp ones(batchSize, 1)];
    end

    labelEst = exp(data*w{level+1});
    labelEst = labelEst./repmat(sum(labelEst,2), 1, numTargetClass);
    [~, idxEst]= max(labelEst,[],2);
    [~, idxTrue]= max(label,[],2);
    counter = counter + length(find(idxEst==idxTrue));
    error = error- sum(sum(label(:,1:end).*log(labelEst)));
end
testError = (batchSize*numValidateBatches-counter);
testErrorNormalized = error/numValidateBatches;

fprintf(1,'Train # misclassified: %d (from %d)\nTest # misclassified: %d (from %d) \t \t \n',...
        trainError,batchSize*numBatches,testError,batchSize*numValidateBatches);


end

