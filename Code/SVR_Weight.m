function Prediction = SVR_Weight(Subjects_Data,Subjects_Scores,C_Parameter)
% Calculate the weight
%Normalizing
MeanValue = mean(Subjects_Data);
StandardDeviation = std(Subjects_Data);
[~, columns_quantity] = size(Subjects_Data);
for j = 1:columns_quantity
    Subjects_Data(:, j) = (Subjects_Data(:, j) - MeanValue(j)) / StandardDeviation(j);
end
% SVR
Subjects_Data = double(Subjects_Data);
model_All = svmtrain(Subjects_Scores, Subjects_Data,['-s 3 -t 0 -c ' num2str(C_Parameter)]);
Features_Quantity = size(Subjects_Data,2);
True_Weight = zeros(1, Features_Quantity);
for j = 1 : model_All.totalSV
    True_Weight = True_Weight + model_All.sv_coef(j) * model_All.SVs(j, :);
end
True_Weight = True_Weight / norm(True_Weight);

Prediction.Weight = True_Weight;