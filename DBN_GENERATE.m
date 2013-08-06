function [ output_args ] = DBN_GENERATE( aH, pathDir, useRMBLearnedLabels, backPropIter, PARAMS)
%DBN_GENERATE Use the DBN to generate data
%   Input the top layer hidden activations. Only one node should be turned
%   on corresponding to the label to generate. 

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
numIters = 250;
figure(1);
%% Load and initilize states
% initialize weights and biases to dummy val
[w{1:numNodes}] = deal(eye(2));
[v{1:numNodes}] = deal(eye(2));
wIdx = zeros(numNodes,1);

% Load the states of the various layers

for ii = 1:numNodes
    S = load([pathDir 'state' num2str(ii)]);
    w{ii} = [S.weights; S.biasesHid];
    v{ii} = S.biasesVis;
    wIdx(ii) = size(w{ii},1)-1;
    biasesVis{ii} = S.biasesVis;
    biasesHid{ii} = S.biasesHid;
end

w{numNodes+1} = [S.weightsC'; S.biasesC];


if useRMBLearnedLabels ~= 1

    S = load([pathDir(1:end-14) 'finalState' num2str(backPropIter)]);
    w = S.w;
end
biasesVis{numNodes+1} = zeros(1,size(w{numNodes},2));
wIdx(numNodes+1) = size(w{numNodes+1},1)-1;
wIdx(numNodes+2) = numTargetClass;
dataAvg = zeros(1,size(w{1},1)-1);
for i = 1:numIters
    %% Down propogate
    data = [aH];             % Input label to turn on and down pass
    for level = numNodes+1:-1:1
        currentWeights = w{level};
        currentWeights = currentWeights(1:end-1,:);     % Remove the hidden biases ... 

        temp = 1./(1 + exp(-data*currentWeights' - biasesVis{level}));
        %dataProbs = [temp ones(batchSize*combo, 1)];

        data = temp > rand(size(temp));
    end
    dataAvg = dataAvg + data;
    imagesc(reshape(dataAvg,48,[])');
    drawnow
end
dataAvg = dataAvg./numIters;
imagesc(reshape(dataAvg,48,[])');

end

