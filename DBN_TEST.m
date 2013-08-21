function [pd, targetout] = DBN_TEST(dH,w,PARAMS)
% DBN_RBM ...
%   DBN_RBM
%
%   Example
%   DBN_RBM

%   See also
%

%% AUTHOR    : Tushar Tank
%% $DATE     : 02-May-2013 13:30:36 $
%% $Revision : 1.00 $
%% DEVELOPED : 7.13.0.564 (R2011b)
%% FILENAME  : DBN_TEST.m
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
%% constants
numNodes = numel(PARAMS.nodes);

%% find prob for each test data file

% load DBN weights
%S = load('finalState3.mat');
%w = S.w;
%clear S;

%filename = getAllFiles(pathTest);
numTestExamples = size(dH,'X');

pd = zeros(numTestExamples(1),1);
%targetout = zeros(numel(filename),2);

for ii = 1:numTestExamples(1)
    if (mod(ii,100) == 1)
       disp(['Processing ' num2str(ii)]); 
    end
%      [a, ~] = regexp(filename{ii}, 'test([0-9]*).mat', 'tokens');
%      idx = str2double(a{1}{1}) + 1;
%      
%      [~,name,~] = fileparts(filename{ii});
%      S = load(filename{ii});
%      dataNoBias = S.data;
%      clear S;
     dataNoBias = dH.X(ii,1:numDimensions);
     N = numel(dataNoBias);
     wprobs{1} = [dataNoBias 1/dropoutP];   % Ajust the biases so they become 1 after scaling by p_dropout
     
     
%      for jj = 1:numNodes
%          temp = 1./(1 + exp(-wprobs{jj}*w{jj}));
%          wprobs{jj+1} = [temp 1]; 
%      end
%      % Compute softmax
%      targetout(ii,1:numTargetClass) = exp(wprobs{jj+1}*w{jj+1});
     w = w .* dropoutP;
     [labelEst,~] = nn_fwd();
     pd(ii) = (targetout(ii,1) ./ sum(targetout(ii,:),2));        
end

end

function fileList = getAllFiles(dirName)
  dirData = dir(dirName);      %# Get the data for the current directory
  dirIndex = [dirData.isdir];  %# Find the index for directories
  fileList = {dirData(~dirIndex).name}';  %'# Get a list of the files
  if ~isempty(fileList)
    fileList = cellfun(@(x) fullfile(dirName,x),...  %# Prepend path to files
                       fileList,'UniformOutput',false);
  end
  subDirs = {dirData(dirIndex).name};  %# Get a list of the subdirectories
  validIndex = ~ismember(subDirs,{'.','..'});  %# Find index of subdirectories
                                               %#   that are not '.' or '..'
  for iDir = find(validIndex)                  %# Loop over valid subdirectories
    nextDir = fullfile(dirName,subDirs{iDir});    %# Get the subdirectory path
    fileList = [fileList; getAllFiles(nextDir)];  %# Recursively call getAllFiles
  end

end

