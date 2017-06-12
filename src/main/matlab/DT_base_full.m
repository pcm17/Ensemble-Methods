%%% BASE Full Decision Tree Classifier
%%% learns on the training set, after learning applies the model to the
%%% test set

function [test_y]  = DT_base_full(tr_x, tr_y, test_x, params)

% calls decision tree machine code to learn the model
mdl = fitctree(tr_x, tr_y);
%%% now we apply the model to test data and make a decision
test_y=predict(mdl,test_x);
end