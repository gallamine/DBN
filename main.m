% main loop
%% tunable constants
PARAMS.trainSamples             = 28000;
PARAMS.batchSize                = 15;
PARAMS.validatePercentage       = .25;
PARAMS.maxEpoch                 = 25;
PARAMS.nodes                    = [500 250 125 50];
PARAMS.learningRateW            = 0.05;   % Learning rate for RBM weights
PARAMS.learningRateBiasVis      = 0.05;   % Learning rate for biases of visible units
PARAMS.learningRateBiasHid      = 0.05;   % Learning rate for biases of hidden units
PARAMS.learningRateBiasLabel    = 0.05;    % Learning rate to fit the labels at the end
PARAMS.weightCost               = 0.0002;
PARAMS.initialMomentum          = 0.5;
PARAMS.finalMomentum            = 0.9;
PARAMS.epochToChangeMomentum    = 5;
PARAMS.maxBackPropEpoch         = 10;
PARAMS.combo                    = 10; % for gradient descent
%PARAMS.numTargets               = 2;
PARAMS.numberOfLineSearches     = 3; % for conjugate gradient descent
PARAMS.displayVisualization     = 1;

%% path definitions
% set this one
dirpath = 'C:\Users\william.cox\Documents\DBN\emotions_data\';
dH = matfile([dirpath 'trainXY.mat'],'Writable',true);  % Should contain X NxD, Y Nx1
% these will get created
pathTrain = [dirpath 'processed/train/'];
pathTest = [dirpath 'processed/test/'];
pathBatch = [dirpath 'processed/batch'];
pathBatch1 = [pathBatch '1'];
pathValidate = [dirpath 'processed/validate/'];

% make output directories
if ~exist(pathTrain,'dir')
    mkdir(pathTrain);
end
if ~exist(pathTest,'dir')
    mkdir(pathTest);
end
if ~exist(pathValidate,'dir')
    mkdir(pathValidate);
end

% make output directories
for ii = 1:(numel(PARAMS.nodes)+1)
    newDir = [pathBatch num2str(ii)];
    if ~exist(newDir,'dir')
        mkdir(newDir);
    end
end

%% fixed constants
% batches and validation set split
trainPercentage = 1 - PARAMS.validatePercentage;
totalTrainSamples = floor(PARAMS.trainSamples * trainPercentage);
PARAMS.numBatches = floor(totalTrainSamples/PARAMS.batchSize);
totalTrainSamples = PARAMS.numBatches * PARAMS.batchSize;

totalValidateSamples = PARAMS.trainSamples - totalTrainSamples;
PARAMS.numValidate = floor(totalValidateSamples/PARAMS.batchSize);
totalValidateSamples = PARAMS.numValidate * PARAMS.batchSize;
PARAMS.numCombinedBatches = floor(PARAMS.numBatches / PARAMS.combo);

numberOfLayers = numel(PARAMS.nodes);

% read onfile to get dimensions of original image
dataSize = size(dH,'X');
labelSize = size(dH,'Y');
PARAMS.dataLength = dataSize(2);
PARAMS.numTargets = labelSize(2);
fprintf(1, 'Begin DBN Training \n');
fprintf(1, 'Data dimension is %d \n', dataSize(2));
fprintf(1, 'Creating output subdirectories if they do not exist \n');

%% Reformat and Preprocess data
fprintf(1, 'Preprocess Training Data.\nUsing %f of data for training\n', (1-PARAMS.validatePercentage));
%DBN_FormatData(dirpath, pathTrain, pathTest, PARAMS);

%% train rbm
% make batches
fprintf(1, 'Create Batchfiles\nEach batch will have %d samples \n\n', PARAMS.batchSize);
offset = 0;
%DBN_MakeBatches('train', totalTrainSamples, PARAMS.numBatches,offset, pathBatch1, pathTrain, PARAMS);

%% train RBM
numNodes = [PARAMS.dataLength PARAMS.nodes];
offset = 0;
for ii = 1:numberOfLayers
    fprintf(1,'Pretraining Layer %d with RBM: %d-%d \n',ii,numNodes(ii),numNodes(ii+1));
    restart=1;
    path1 = [pathBatch num2str(ii) '\'];
    path2 = [pathBatch num2str(ii+1) '\'];
    if ii == 1 % First layer activations are the input data
        actH1 = dH;
    else
        actH1 = matfile([path1 'data.mat'],'Writable',true);  % Visible layer activations file
    end
    actH2 = matfile([path2 'data.mat'],'Writable',true);  % Hidden layer activations file
    if ii < numberOfLayers
        [weights, biasesVis, biasesHid, errsum] = ...
        DBN_RBM(actH1, actH2, numNodes(ii), numNodes(ii+1), restart, PARAMS,offset);
        save([dirpath 'state' num2str(ii)], 'weights', 'biasesVis', 'biasesHid', 'errsum');
    else
        [weights, weightsC, biasesVis, biasesHid, biasesC, errsum] = ...
        RBM_FIT(actH1, actH2, dH, numNodes(ii), numNodes(ii+1), restart, PARAMS,offset);
        save([dirpath 'state' num2str(ii)], 'weights', 'biasesVis', 'biasesHid', 'errsum', 'weightsC','biasesC');
    end
    
end
fprintf(1,'RBM training complete \n\n');

%% Fit last layer to labels?

%[weights, biasesVis, biasesHid, errsum] = ...
        [testE,trainE,tEn,trainEN] = DBN_UNFOLD_NOBACKPROP(dH,dirpath,PARAMS);

%% backprop with labels
fprintf(1,'Create new batches for backprop training and validation\n');
% rebatch and validate data, for backprop
offset = 0;
%DBN_MakeBatches('train', totalTrainSamples, PARAMS.numBatches, offset, pathBatch1, pathTrain, PARAMS);
offset = totalTrainSamples;
%DBN_MakeBatches('train', totalValidateSamples, PARAMS.numValidate, offset, pathValidate, pathTrain, PARAMS);

%%
% backprop
fprintf(1,'Begin Backpropogation\n');
DBN_BackProp(dH,dirpath,PARAMS)

%% test
[pd, testout] = DBN_TEST(pathTest, PARAMS);
save('temp','pd');