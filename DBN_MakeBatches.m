function [batchData, batchLabel] = DBN_MakeBatches(dH, totalNum, numBatches, offset, pathBatch, pathData, PARAMS)
% DBN_MAKEBATCHES ... 
%   DBN_MAKEBATCHES 
%  
%	dH - Data handle, file handle to data storage. X is training data.
%
%   Example 
%   DBN_MakeBatches 

%   See also 
% 

%% AUTHOR    : Tushar Tank 
%% $DATE     : 29-Apr-2013 12:05:03 $ 
%% $Revision : 1.00 $ 
%% DEVELOPED : 7.13.0.564 (R2011b) 
%% FILENAME  : DBN_MakeBatches.m 
%% COPYRIGHT 2011 3 Phonenix Inc. 
%% constants
batchSize = PARAMS.batchSize;
numDimension  =  PARAMS.dataLength;
batchData = zeros(batchSize, numDimension);
batchLabel = zeros(batchSize,PARAMS.numTargets);

%% random order
randomOrder = reshape(randperm(totalNum)+offset, [numBatches batchSize]);
S = load([pathData 'label']);
label = S.label;

%% create each batch and save it off
for ii = 1:numBatches
    batchData = zeros(batchSize, PARAMS.dataLength);
    for jj = 1:batchSize
    	S = dH.X(randomOrder(ii,jj),1:numDimension);
        %S = load([pathData fileName num2str(randomOrder(ii,jj))]);
        batchData(jj, 1:numel(S.data)) = S.data';                   
        batchLabel(jj, :) = label(randomOrder(ii,jj), :);
    end
    save([pathBatch '\batch' num2str(ii)], 'batchData', 'batchLabel');
end
