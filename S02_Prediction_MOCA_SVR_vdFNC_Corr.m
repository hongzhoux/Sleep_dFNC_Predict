clear;clc;
root = 'E:\Sleep_analysis\Machin_Learning\Data';
%% Load data
load([root,filesep,'all_subjects_vdFNC.mat']);
load([root,filesep,'all_subjects_MOCA.mat'])
load([root,filesep,'Group_labels.mat']);
%% Initialization parameters
Group = {BS,GS};
mList = 10:20:550;
C_Parameter = 1;
Permutation_times1 = 1000;
Permutation_times2 = 10000;
for v = 1:length(Group)
    Subjects_Data = vdFNC(Group{v},:);
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
   %% Weights calculate
   idx = find(Results.Feature_Frequency{v, 1}==length(Subjects_Scores));
   SVRWeight = SVR_Weight(Subjects_Data(:,idx),Subjects_Scores,C_Parameter);
   %% Permutation test for weights
    PermWeights = [];
    PermWeights = [PermWeights; SVRWeight.Weight];
    parfor permi = 1:Permutation_times2
        PermIdx = randperm(size(Subjects_Data,1));
        PermWeight = SVR_Weight(Subjects_Data(:,idx),Subjects_Scores(PermIdx,1),C_Parameter);
        PermWeights = [PermWeights;PermWeight.Weight];
    end
    PermWeights = sort(PermWeights,'descend');
    for ii = 1:length(idx)
          xidx = find(SVRWeight.Weight(1,ii) == PermWeights(:,ii));
          P_value = xidx/Permutation_times2;
          if P_value < 0.05 || P_value > 0.95
              Sig_Weight(1,ii) = SVRWeight.Weight(1,ii);
          else
              Sig_Weight(1,ii) = NaN;
          end
    end                
    Weight_mask = NaN * zeros(1,size(Subjects_Data,2));
    Weight_mask(1,idx) = SVRWeight.Weight;
    Results.Weight{v,:} = Weight_mask;
    SigWeight_mask = NaN * zeros(1,size(Subjects_Data,2));
    SigWeight_mask(1,idx) = Sig_Weight;
    Results.SigWeight{v,:} = SigWeight_mask;
    clear Sig_Weight
end
Results.Final_Pred = table(Results.Corr,Results.P_value,Results.RMSE,Results.NRMSE, ...
Results.Optimal_m,'RowNames',{'Poor sleep','Good sleep'},'VariableNames',{'Correlation','P_value','RMSE','NRMSE','Optimal_m'});
disp('_________________________________________________________');
disp('The prediction results are as follows:'); 
disp(Results.Final_Pred);
save([root,filesep,'Group_vdFNC_prediction_corr.mat'],'Results')