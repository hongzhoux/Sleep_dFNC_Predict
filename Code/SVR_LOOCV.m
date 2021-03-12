function Prediction = SVR_LOOCV(Subjects_Data,Subjects_Scores,C_Parameter,Cov)
NSubj = size(Subjects_Data,1);
for i = 1:NSubj 
    fprintf('Subject #%03d is being processed, left %03d subjects!\n',i,NSubj-i)
    Training_data = Subjects_Data;
    Training_scores = Subjects_Scores; 
    % Select training data and testing data
    test_data = Training_data(i, :);
    test_score = Training_scores(i);
    Training_data(i, :) = [];
    Training_scores(i) = [];
    %Normalizing
    MeanValue = mean(Training_data);
    StandardDeviation = std(Training_data);
    [~, columns_quantity] = size(Training_data);
    for j = 1:columns_quantity
        Training_data(:, j) = (Training_data(:, j) - MeanValue(j)) / StandardDeviation(j);
    end
    Training_data_final = double(Training_data);
    test_data = (test_data - MeanValue) ./ StandardDeviation;
    test_data_final = double(test_data);
    % SVR training
    model = svmtrain(Training_scores, Training_data_final, ['-s 3 -t 0 -c ' num2str(C_Parameter)]);
    % Predict test data
    [Predicted_Score, ~, ~] = svmpredict(test_score, test_data_final, model);
    Predicted_Scores(i) = Predicted_Score;
end
Prediction.Score = Predicted_Scores;
if nargin > 3
    [Prediction.Corr, Prediction.P] = partialcorr(Predicted_Scores', Subjects_Scores,Cov);
else
    [Prediction.Corr, Prediction.P] = corr(Predicted_Scores', Subjects_Scores);
end
Prediction.MAE = mean(abs((Predicted_Scores' - Subjects_Scores)));
Prediction.RMSE = sqrt(mean(((Predicted_Scores' - Subjects_Scores).^2)));
Prediction.NRMSE = sqrt(mean(((Predicted_Scores' - Subjects_Scores).^2)))/mean(Predicted_Scores);
