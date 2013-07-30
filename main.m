% main loop
%% tunable constants
PARAMS.trainSamples             = 47841;
PARAMS.batchSize                = 100;
PARAMS.validatePercentage       = .25;
PARAMS.maxEpoch                 = 5;
PARAMS.nodes                    = [500 500 2000];
PARAMS.learningRateW            = 0.1;   % Learning rate for RBM weights
PARAMS.learningRateBiasVis      = 0.1;   % Learning rate for biases of visible units
PARAMS.learningRateBiasHid      = 0.1;   % Learning rate for biases of hidden units
PARAMS.weightCost               = 0.0002;
PARAMS.initialMomentum          = 0.5;
PARAMS.finalMomentum            = 0.9;
PARAMS.epochToChangeMomentum    = 5;
PARAMS.maxBackPropEpoch         = 200;
PARAMS.combo                    = 10; % for gradient descent
PARAMS.numTargets               = 2;
PARAMS.numberOfLineSearches     = 3; % for conjugate gradient descent

%% path definitions
% set this one
dirpath = '/home/administrator/bigdata/ML/Whales/data2/';
% these will get created
pathTrain = [dirpath 'processed/train/'];
pathTest = [dirpath 'processed/test/'];
pathBatch = [dirpath 'processed/batch'];
pathBatch1 = [pathBatch '1'];
pathValidate = [dirpath 'processed/validate/'];
trainFile = '20090328_000000_002s3ms_TRAIN0_0.aif';

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
data = double(aiffread([dirpath 'train/' trainFile]));
[data, row, col] = DBN_Preprocess(data);
PARAMS.dataLength = numel(data);

fprintf(1, 'Begin DBN Training \n');
fprintf(1, 'Dimensions of Input Image %d x %d \n', row, col);
fprintf(1, 'Creating output subdirectories if they do not exist \n');

%% Reformat and Preprocess data
fprintf(1, 'Preprocess Training Data.\nUsing %f of data for training\n', (1-PARAMS.validatePercentage));
DBN_FormatData(dirpath, pathTrain, pathTest, PARAMS);

%% train rbm
% make batches
fprintf(1, 'Create Batchfiles\nEach batch will have %d samples \n\n', PARAMS.batchSize);
offset = 0;
DBN_MakeBatches('train', totalTrainSamples, PARAMS.numBatches,offset, pathBatch1, pathTrain, PARAMS);

%% train RBM
numNodes = [PARAMS.dataLength PARAMS.nodes];

for ii = 1:numberOfLayers
    fprintf(1,'Pretraining Layer %d with RBM: %d-%d \n',ii,numNodes(ii),numNodes(ii+1));
    restart=1;
    path1 = [pathBatch num2str(ii) '\'];
    path2 = [pathBatch num2str(ii+1) '\'];
    [weights, biasesVis, biasesHid, errsum] = ...
        DBN_RBM(path1, path2, numNodes(ii), numNodes(ii+1), restart, PARAMS);
    save([dirpath 'state' num2str(ii)], 'weights', 'biasesVis', 'biasesHid', 'errsum');
end
fprintf(1,'RBM training complete \n\n');

%% backprop with labels
fprintf(1,'Create new batches for backprop training and validation\n');
% rebatch and validate data, for backprop
offset = 0;
DBN_MakeBatches('train', totalTrainSamples, PARAMS.numBatches, offset, pathBatch1, pathTrain, PARAMS);
offset = totalTrainSamples;
DBN_MakeBatches('train', totalValidateSamples, PARAMS.numValidate, offset, pathValidate, pathTrain, PARAMS);

%%
% backprop
fprintf(1,'Begin Backpropogation\n');
DBN_BackProp(dirpath,pathBatch1,pathValidate,PARAMS)

%% test
[pd, testout] = DBN_TEST(pathTest, PARAMS);
save('temp','pd');