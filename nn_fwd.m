function [labelEst,data] = nn_fwd(w,data,numNodes,dropoutP,batchSize,numTargetClass)

%% Do forward pass through neural network.
%% By William Cox, 8/14/2013
%% 3Phoenix, Inc.

% dropoutP - percentage of nodes to remove during forward pass, each
% dropout is different for each row of data (training example in the
% batch).

for level = 1:numNodes
    dropout = 1;
    if dropoutP > 0
        dropout = dropoutP < rand(size(w{level},2));
        %dropout = repmat(dropout,1,size(w{level},2));
    end
    temp = 1./(1 + exp(-data*(w{level}*dropout)));      % Dropout random selection of nodes, different for each example.
    data = [temp ones(batchSize, 1)];
end

labelEst = exp(data*w{level+1});
labelEst = labelEst./repmat(sum(labelEst,2), 1, numTargetClass);


end