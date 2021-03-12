function [Predicted_Score,Weights] = SVR_Pred(Training_data,Training_scores,test_data,test_score,index_op,Feature_Quantity,C_Parameter)
    MeanValue = mean(Training_data);
    StandardDeviation = std(Training_data);
    [~, columns_quantity] = size(Training_data);
    for j = 1:columns_quantity
        Training_data(:, j) = (Training_data(:, j) - MeanValue(j)) / StandardDeviation(j);
    end
    Training_data_final = double(Training_data);
    test_data = (test_data - MeanValue) ./ StandardDeviation;
    test_data_final = double(test_data);
    % RVR training & predicting
    model = svmtrain(Training_scores, Training_data_final, ['-s 3 -t 0 -c ' num2str(C_Parameter)]);
    % Predict test data
    [Predicted_Score, ~, ~] = svmpredict(test_score, test_data_final, model);
    w_Brain = zeros(1,length(index_op));
    for j = 1 : model.totalSV
        w_Brain = w_Brain + model.sv_coef(j) * model.SVs(j, :);
    end
    w_Brain = w_Brain / norm(w_Brain);
    Weights = NaN*zeros(1,Feature_Quantity);
    Weights(1,index_op) = w_Brain;
end