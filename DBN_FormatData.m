function DBN_FormatData(dirpath, outpathTrain, outpathTest, PARAMS)
% DBN_FORMATDATA ... 
%   DBN_FORMATDATA 
%  
%   Example 
%   DBN_FormatData 

%   See also 
% 

%% AUTHOR    : Tushar Tank 
%% $DATE     : 30-Apr-2013 13:54:04 $ 
%% $Revision : 1.00 $ 
%% DEVELOPED : 7.13.0.564 (R2011b) 
%% FILENAME  : DBN_FormatData.m 


%% read in data from file and save off each processed file

% We are assuming a directory strucutre such that under the dirpath
% directory we have two subdirectories: test and train. Both these
% directories will have a set of files that are the input data. The
% filename for these files will be prepended with either train or test and
% concatenated with a number. The number will be sequential from 1 to the
% number of files in that directory. For example the subdirectory train
% with have files: train1, train2, ... trainN. The subdirectory test will
% have test1, test2, ... testN. In this example (kaggle whale detection
% example) the files are aiff files. We also expect a comma seperated value
% file (csv) where one the first column is the file name and the second
% colunm is the class label. The code below will need to be
% modified for your paticular file type. The variable dirpath will have to
% be modified to where you have downloaded your data. 

% constants
% dirpath = 'C:\Users\tushar.tank\Downloads\whale_data\data\';
% outpathTrain = 'C:\Users\tushar.tank\Downloads\whale_data\data\processed\train\';
% outpathTest = 'C:\Users\tushar.tank\Downloads\whale_data\data\processed\test\';

% read labels
% dataCellArray = importdata([dirpath 'train.csv']);
% labelSingle = dataCellArray.data;
% filename = dataCellArray.textdata;  % read in as xls file
% filename = filename(2:end,1);                       % remove header
% % format labels in matrix of indicators
% labelSingle = labelSingle(1:PARAMS.trainSamples);
% label = zeros(numel(labelSingle), PARAMS.numTargets);
% for ii = 1:PARAMS.numTargets
%     label(:,ii) = labelSingle == ii-1;
% end
% 
% save([outpathTrain 'label'], 'label');
% 

% train data
filename = getAllFiles([dirpath 'train']);
labelSingle = ones(1,PARAMS.trainSamples)*-1;
data = double(aiffread(filename{1}));
PARAMS.trainLength = numel(data);

for ii = 1:PARAMS.trainSamples
     [~,name,~] = fileparts(filename{ii});
     data = double(aiffread(filename{ii}));
     if numel(data) < PARAMS.trainLength
         fprintf(1, 'Sample %d is truncated. Zero padding\n', ii);
         temp = zeros(1,PARAMS.trainLength);
         temp(1:numel(data)) = data;
         data = temp;
     else
         if numel(data) > PARAMS.trainLength
            fprintf(1, 'Sample %d has been truncated\n', ii);
         end         
         data = data(1:PARAMS.trainLength);
     end
     [data, ~, ~] = DBN_Preprocess(data);
     [a, ~] = regexp(filename{ii}, 'TRAIN([0-9])*_([0-9]).aif', 'tokens');
     fileNum = a{1}(1);
     labelSingle(str2double(fileNum{1})+1) = str2double(a{1}(2));
     save([outpathTrain 'train' fileNum{1}], 'data');
end

% format labels in matrix of indicators
label = zeros(numel(labelSingle), PARAMS.numTargets);
for ii = 1:PARAMS.numTargets
    label(:,ii) = labelSingle == ii-1;
end
save([outpathTrain 'label'], 'label');

%test data
filename = getAllFiles([dirpath 'test']);

data = double(aiffread(filename{1}));
PARAMS.testLength = numel(data);

for ii = 23218:numel(filename)
     [~,name,~] = fileparts(filename{ii});
     data = double(aiffread(filename{ii}));
     if numel(data) < PARAMS.testLength
         fprintf(1, 'Sample %d is truncated. Zero padding\n', ii);
         temp = zeros(1,PARAMS.testLength);
         temp(1:numel(data)) = data;
         data = temp;
     else
         if numel(data) > PARAMS.testLength
            fprintf(1, 'Sample %d has been truncated\n', ii);
         end
         data = data(1:PARAMS.testLength);
     end    
     [a, ~] = regexp(filename{ii}, 'Test([0-9])*.aif', 'tokens');
     fileNum = a{1}(1);
     
     if ~exist([outpathTest 'test' fileNum{1} '.mat'],'file')
        data = DBN_Preprocess(data);
        save([outpathTest 'test' fileNum{1}], 'data');
     end
     
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
