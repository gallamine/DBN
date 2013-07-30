function [label data test] = readWhaleData2_Tushar()
% READWHALEDATA ... 
%   READWHALEDATA 
%  
%   Example 
%   readWhaleData 

%   See also 
% 

%% AUTHOR    : Tushar Tank 
%% $DATE     : 09-Feb-2013 14:39:04 $ 
%% $Revision : 1.00 $ 
%% DEVELOPED : 7.13.0.564 (R2011b) 
%% FILENAME  : readWhaleData.m 
%% COPYRIGHT 2011 3 Phonenix Inc. 
dbstop if error
%% constants
L = 4000; % length of each sound file
dirpath = 'C:\Users\william.cox\Documents\Dropbox\ML\Whales\data2';

%% read labels
% [label filename] = xlsread([dirpath '\train.csv']);    % read in as xls file
% filename = filename(2:end,1);               % remove header

%% read in data
fileList = getAllFiles([dirpath '\train2']);
numData = numel(fileList);

% Get labels
idx = strfind(fileList,'.aif'); % String index of the label
for ii = 1:numData
    file = fileList{ii};
    label(ii) = str2num(file(idx{ii}-1));
end
assert(length(label) == numData);
data = zeros(L,numData);

for ii = 1:numData
    if mod(ii,100) == 1
        disp(['Progress: ' num2str(ii/numData*100) '%']);
    end
    readFile = double(aiffread([fileList{ii}]));
    data(1:length(readFile),ii) = readFile;
end

%% read in test data

fileList = getAllFiles([dirpath '\test2']);
numData = numel(fileList);
test = zeros(L,numData);

for ii = 1:numData
     disp(['Progress: ' num2str(ii/numData*100) '%']);
    %test(:,ii) = double(aiffread([dirpath '\test2\test' num2str(ii) '.aif']));
    readFile = double(aiffread([fileList{ii}]));
    test(1:length(readFile),ii) = readFile;
end

