function Prediction = SVR_LOOCV_Corr(Subjects_Data, Subjects_Scores,M,C_Parameter)
[Subjects_Quantity, ~] = size(Subjects_Data);
Pred_scores = [];
Feature_Quantity = size(Subjects_Data,2);
Feature_Frequency = zeros(1, Feature_Quantity);
all_weights = [];
parfor i = 1:Subjects_Quantity
    fprintf('Subjects #%03d is being processed, left %03d subjects!\n',i,Subjects_Quantity - i);
    Training_data = Subjects_Data;
    Training_scores = Subjects_Scores;
    % Select training data and testing data
    test_data = Training_data(i, :,:);
    test_score = Training_scores(i);
    Training_data(i, :,:) = [];
    Training_scores(i) = [];
    r_mat = [];
    for ii = 1:Feature_Quantity
        [r,~] = corr(Training_data(:,ii),Training_scores);
        r_mat = [r_mat,r];
    end
    [~,Ranked] = maxk(abs(r_mat),Feature_Quantity);
    index_op = find(Ranked <= M);
    Selected_Mask = zeros(1, Feature_Quantity);
    Selected_Mask(index_op) = 1;
    Feature_Frequency = Feature_Frequency + Selected_Mask;
    Training_data_new = Training_data(:,index_op);
    test_data_new = test_data(:,index_op);
    [Pred_Score,Weights] = SVR_Pred(Training_data_new,Training_scores,test_data_new,test_score,index_op,Feature_Quantity,C_Parameter);
    Pred_scores = [Pred_scores;Pred_Score];
    all_weights = [all_weights;Weights];
end
Prediction.Score = Pred_scores;
[Prediction.Corr, Prediction.P] = corr(Pred_scores, Subjects_Scores);
Prediction.RMSE = sqrt(mean((Pred_scores - Subjects_Scores).^2));
Prediction.NRMSE = sqrt(mean((Pred_scores - Subjects_Scores).^2))/mean(Pred_scores);
Prediction.MAE = mean(abs((Pred_scores - Subjects_Scores)));
Prediction.Feature_Frequency = Feature_Frequency;
Prediction.Mean_Weights = mean(all_weights);
end