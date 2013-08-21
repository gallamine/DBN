function [labelEst,data] = nn_fwd(w,data,numNodes,dropoutP,batchSize,numTargetClass,varargin)

%% Do forward pass through neural network.
%% By William Cox, 8/14/2013
%% 3Phoenix, Inc.

% dropoutP - percentage of nodes to remove during forward pass, each
% dropout is different for each row of data (training example in the
% batch).
% testing - flag indicating to scale the weights by the dropout fraction
% for use in full scale testing (after network is trained).

if length(varargin) > 0
    testing = varargin{1};
else
    testing = 0;
end

for level = 1:numNodes
    if testing == 1
        % Scale the non-bias weights by the dropout percentage
        wS = size(w{level});
        w{level}(1:ws(1),1:ws(2)-1) = w{level}(1:ws(1),1:ws(2)-1).*dropoutP;
    end
    temp = 1./(1 + exp(-data*(w{level})));      
    if dropoutP > 0
        dropout = dropoutP > rand(size(temp));
        temp = temp.*dropout;   % Dropout random selection of nodes, different for each example.
    end
    data = [temp ones(batchSize, 1)];
    
end

if testing == 1
    % Scale the non-bias weights by the dropout percentage
    wS = size(w{level+1});
    w{level+1}(1:ws(1),1:ws(2)-1) = w{level+1}(1:ws(1),1:ws(2)-1).*dropoutP;
end
labelEst = exp(data*w{level+1});
labelEst = labelEst./repmat(sum(labelEst,2), 1, numTargetClass);


end