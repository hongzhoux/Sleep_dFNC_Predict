function Prediction = SVR_Prediction(Subjects_Data,Subjects_Scores,NFold,C_Parameter)
NSubj = size(Subjects_Data,1);
for i = 1:NSubj 
    disp(['The ' num2str(i) ' subject!']);
    Training_data = Subjects_Data;
    Training_scores = Subjects_Scores; 
    % Select training data and testing data
    test_data_loocv = Training_data(i, :);
    test_score_loocv = Training_scores(i);
    Training_data(i, :) = [];
    Training_scores(i) = [];
    %Normalizing
    MeanValue = mean(Training_data);
    StandardDeviation = std(Training_data);
    [~, columns_quantity] = size(Training_data);
    for j = 1:columns_quantity
        Training_data(:, j) = (Training_data(:, j) - MeanValue(j)) / StandardDeviation(j);
    end
    Training_data_norm = double(Training_data);
    test_data_loocv = (test_data_loocv - MeanValue) ./ StandardDeviation;
    test_data_final = double(test_data_loocv);
    % Split into N folds randomly
    Train_NSubj = size(Training_data,1);
    NEachPart = fix(Train_NSubj/NFold);
    RandID = randperm(Train_NSubj);
    for j = 1:NFold
        Origin_ID{j} = RandID([(j - 1) * NEachPart + 1: j * NEachPart])';
    end
    Reamin = mod(Train_NSubj,NFold);
    for j = 1:Reamin
        Origin_ID{j} = [Origin_ID{j} ; RandID(NFold * NEachPart + j)];
    end
    for j = 1:NFold
        % Select training data and testing data
        Training_data_kfold = Training_data_norm;
        Training_data_kfold(Origin_ID{j}, :) = [];
        Training_scores_kfold = Training_scores;
        Training_scores_kfold(Origin_ID{j}) = [];
        % SVR training
        Training_data_final = double(Training_data_kfold);
        model = svmtrain(Training_scores_kfold, Training_data_final, ['-s 3 -t 0 -c ' num2str(C_Parameter)]);
    end
    % Predict test data
    [Predicted_Score, ~, ~] = svmpredict(test_score_loocv, test_data_final, model);
    Predicted_Scores(i) = Predicted_Score;
end
Prediction.Score = Predicted_Scores;
[Prediction.Corr, ~] = corr(Predicted_Scores', Subjects_Scores);
Prediction.MAE = mean(abs((Predicted_Scores' - Subjects_Scores)));
       
