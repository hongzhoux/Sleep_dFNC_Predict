clear;clc;
root = 'E:\Sleep_analysis\Manuscript\Machin_Learning\Data';
%% Load data
load([root,filesep,'all_subjects_sFNC.mat']);
load([root,filesep,'all_subjects_MOCA.mat'])
load([root,filesep,'Group_labels.mat']);
%% Initialization parameters
Group = {BS,GS};
mList = 10:20:550;
C_Parameter = 1;
Permutation_times1 = 1000;
Permutation_times2 = 10000;
for v = 1:length(Group)
    Subjects_Data = sFNC(Group{v},:);
    Subjects_Scores = MOCA(Group{v},1);
    for mi = 1:length(mList)
        Prediction = SVR_LOOCV_Corr(Subjects_Data, Subjects_Scores,mList(mi),C_Parameter );
        Corr_Irer(mi,1) = Prediction.Corr;
    end
    [~,Optimal_m] = max(Corr_Irer);
    Pred_final = SVR_LOOCV_Corr(Subjects_Data,Subjects_Scores,mList(Optimal_m),C_Parameter );  
    Results.Corr(v,1) = Pred_final.Corr;
    Results.Pred_Score{v,1} = Pred_final.Score;
    Results.P_value(v,1) = Pred_final.P;
    Results.RMSE(v,1) = Pred_final.RMSE;
    Results.NRMSE(v,1) = Pred_final.NRMSE;
    Results.Optimal_m(v,1) = mList(Optimal_m(1));
    Results.Feature_Frequency{v,1} = Pred_final.Feature_Frequency;
    Results.Mean_Weights{v,1} = Pred_final.Mean_Weights;
end
Results.Final_Pred = table(Results.Corr,Results.P_value,Results.RMSE,Results.NRMSE, ...
Results.Optimal_m,'RowNames',{'Poor sleep','Good sleep'},'VariableNames',{'Correlation','P_value','RMSE','NRMSE','Optimal_m'});
disp('_________________________________________________________');
disp('The prediction results are as follows:'); 
disp(Results.Final_Pred);
save([root,filesep,'Group_sFNC_prediction_corr.mat'],'Results')