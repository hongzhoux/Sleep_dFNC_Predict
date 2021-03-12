function Prediction = SVR_LOOCV_dFNC(Subjects_Data, Subjects_Scores,C_Parameter,K, M)
[Subjects_Quantity, ~] = size(Subjects_Data);
Optimal_w_all = [];
Pred_scores = [];
Feature_Quantity = size(Subjects_Data,2);
Feature_Frequency = zeros(1, Feature_Quantity);
parfor i = 1:Subjects_Quantity
    fprintf('Subjects #%03d is being processed, left %03d subjects!\n',i,Subjects_Quantity - i);
    Training_data = Subjects_Data;
    Training_scores = Subjects_Scores;
    % Select training data and testing data
    test_data = Training_data(i, :,:);
    test_score = Training_scores(i);
    Training_data(i, :,:) = [];
    Training_scores(i) = [];
%     Corr_wsizes = [];
%     for wi = 2%:size(Training_data,3)
%         Ranked = relieff(Training_data(:,:,wi),Training_scores,K);
%         index = find(Ranked <= M);
%         Pred_wsize = RVR_LOOCV(Training_data(:,index),Training_scores);
%         Corr_wsize = corr(Pred_wsize.Score',Training_scores);
%         Corr_wsizes = [Corr_wsizes;Corr_wsize];
%     end
%     [~,Optimal_w] = max(Corr_wsizes);
    Optimal_w = 3;
    Ranked = relieff(Training_data(:,:,Optimal_w),Training_scores,K);
    index_op = find(Ranked <= M);
    Selected_Mask = zeros(1, Feature_Quantity);
    Selected_Mask(index_op) = 1;
    Feature_Frequency = Feature_Frequency + Selected_Mask;
    Optimal_w_all = [Optimal_w_all;Optimal_w];
    Training_data_new = Training_data(:,index_op,Optimal_w);
    test_data_new = test_data(:,index_op,Optimal_w);
     % SVR training & predicting
    model = svmtrain(Training_scores,Training_data_new, ['-s 3 -t 0 -c ' num2str(C_Parameter)]);
    % Predict test data
    [Pred_score_final, ~, ~] = svmpredict(test_score,test_data_new, model);
    Pred_scores = [Pred_scores;Pred_score_final];
end
Prediction.Score = Pred_scores;
[Prediction.Corr, Prediction.P] = corr(Pred_scores, Subjects_Scores);
Prediction.RMSE = mean((Pred_scores - Subjects_Scores).^2);
Prediction.MAE = mean(abs((Pred_scores - Subjects_Scores)));
Prediction.Optimal_w_all = Optimal_w_all;
Prediction.Feature_Frequency = Feature_Frequency;
end