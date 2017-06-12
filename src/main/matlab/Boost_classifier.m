function [test_y, E] = Boost_classifier(tr_x, tr_y, test_x, params)

% Classify using the AdaBoost algorithm
% Arguments:    1. Train patterns
%               2. Train targets
%               3. Test patterns
%               4. params = [base_classifier,
%                           NumberOfIterations,
%                           Classifier_parameters]
%
% Returns:      1. Predicted targets
%               2. Errors through iterations


[base_classifier, k_max, alg_params] = process_params(params);

[M,~]			= size(tr_x);
W			 	= ones(M,1)/M;
IterDisp		= 10;
Nc              = length(unique(tr_y));

full_x   = [tr_x; test_x];
test_y    = zeros(size(test_x,1),1);

% AdaBoost loop
for k = 1:k_max
   %Train weak learner Ck using the data sampled according to W:
   %...so sample the data according to W
   randnum = rand(M,1);
   cW	   = cumsum(W);
   indices = zeros(M,1);
   for i = 1:M
      %Find which bin the random number falls into
      loc = find(randnum(i) > cW, 1, 'last' )+1;
      if isempty(loc)
         indices(i) = 1;
      else
         indices(i) = loc;
      end
   end
   
   %...and now train the classifier
   Ck 	= feval(base_classifier, tr_x(indices,:), tr_y(indices), full_x, alg_params);
   %%% 'out of base classifier'
   %Ek <- Training error of Ck 
   E(k) = sum(W.*(Ck(1:M) ~= tr_y));
   
   if (E(k) == 0)
      break
   end
   
   %alpha_k <- 1/2*ln(1-Ek)/Ek)
   alpha_k = 0.5*log((1-E(k))/E(k));
   
   %W_k+1 = W_k/Z*exp(+/-alpha)
   W  = W.*exp(alpha_k*(xor(Ck(1:M),tr_y)*2-1));
   W  = W./sum(W);
   
   %Update the test targets
   test_y  = test_y + alpha_k*(2*Ck(M+1:end)-1);
   
%   if (k/IterDisp == floor(k/IterDisp)),
%      disp(['Completed ' num2str(k) ' boosting iterations'])
%   end
   
end

%If there are only two classes, collapse the targets
if (length(unique(tr_y)) == 2)
    test_y = test_y > mean(unique(tr_y));
end