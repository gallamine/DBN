function [f, df] = DBN_ConjugateGradientInit(VV,Dim,w,target,PARAMS)
% DBN_CONJUGATEGRADIENTINIT ... 
%   DBN_CONJUGATEGRADIENTINIT 
%  
%   Example 
%   DBN_ConjugateGradientInit 

%   See also 
% 

%% AUTHOR    : Tushar Tank 
%% $DATE     : 17-May-2013 13:47:50 $ 
%% $Revision : 1.00 $ 
%% DEVELOPED : 7.13.0.564 (R2011b) 
%% FILENAME  : DBN_ConjugateGradientInit.m 
%% COPYRIGHT 2011 3 Phonenix Inc. 

numTargets = PARAMS.numTargets;


l1 = Dim(1);
l2 = Dim(2);
N = size(w,1);

% Do decomversion.
w_class = reshape(VV,l1+1,l2);
w = [w  ones(N,1)];  

% p(yi = j | x;w) = exp(xi*wj)/sum(exp(xl*w))_l=1:T, for T classes

targetout = exp(w*w_class);
targetout = targetout./repmat(sum(targetout,2),1,numTargets);   % Softmax output - exp(q)/(sum(e(q)))
f = -sum(sum( target(:,1:end).*log(targetout))) ;   % Softmax regression error fcn (see http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression)
IO = (targetout-target(:,1:end));   
Ix_class=IO; 
dw_class =  w'*Ix_class;        % Derivative of 'f' w.r.t weights

df = (dw_class(:)')'; 

 