function updateDisplay(fig,weights,biasesHid,biasesVis,deltaWeights,posHidProbs,negdata)
%UNTITLED Control display for the learning rates of a DBN
%   Populate views showing how learning is progressing

%% AUTHOR    : William Cox
%% $DATE     : 24-July-2013 15:25:00 $
%% $Revision : 1.00 $
%% FILENAME  : updateDisplay.m
%% COPYRIGHT 2013 3 Phonenix Inc.

% Histogram of weights
subplot(4,2,1);
hist(weights(:),30);
title('Weights');
% Histogram of weight_deltas
subplot(4,2,2);
hist(deltaWeights(:),30);
title('\Delta Weights');
% Histogram of vis biases
subplot(4,2,3);
hist(biasesVis,30);
title('v_b');
% Histogram of hid biases
subplot(4,2,4);
hist(biasesHid,30);
title('h_b');
% Hidden node probabilties over time
subplot(4,2,[5 6]);
imagesc(posHidProbs,[0 1]);
colormap gray
title('p(h^1|v)');
subplot(4,2,[7 8]);
imagesc(negdata,[0 1]);
colormap gray
title('p(v^2|h^1)');
end

