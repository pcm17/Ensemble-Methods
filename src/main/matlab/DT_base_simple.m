%%% BASE Simple, one node Decision Tree Classifier
%%% learns on the training set, after learning applies the model to the
%%% test set

function [test_y]  = DT_base_simple(tr_x, tr_y, test_x, params)
% calls decision tree code to learn the model
minParentSize = size(tr_x,1);
mdl = fitctree(tr_x, tr_y,'MinParentSize',minParentSize);

%%% now we apply the model to test data and make a decision
test_y=predict(mdl,test_x);
end