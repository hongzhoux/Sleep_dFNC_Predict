clear;clc;
root = 'E:\Sleep_analysis\Machin_Learning\Data';
%% Load data
load('E:\Sleep_analysis\dFNC_22TR\Sleep_dfnc_cluster_stats.mat');
load([root,filesep,'all_subjects_PSQI.mat']);
load([root,filesep,'Group_labels.mat']);
%% Initialization parameters
dFNC = squeeze(dfnc_corrs);
Group = BS;
mList = 10:20:550;
C_Parameter = 1;
Permutation_times = 10000;
for v = 1:size(dFNC,3)
    Subjects_Data = dFNC(Group,:,v);
    Subjects_Scores = PSQI(Group,1);
    nanidx = isnan(Subjects_Data(:,1));
    idx = find(nanidx == 0);
    Subjects_Data = Subjects_Data(idx,:);
    Subjects_Scores = Subjects_Scores(idx,1);
    for mi = 1:length(mList)
        Prediction = SVR_LOOCV_Corr(Subjects_Data, Subjects_Scores,mList(mi),C_Parameter);
        Corr_Iter(mi,v) = Prediction.Corr;
    end
   [~,Optimal_m] = max(Corr_Iter(:,v));
   Pred_final = SVR_LOOCV_Corr(Subjects_Data,Subjects_Scores,mList(Optimal_m),C_Parameter);
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
   if ~isempty(idx)
       SVRWeight = SVR_Weight(Subjects_Data(:,idx),Subjects_Scores,C_Parameter);
       %% Permutation test for weights
        PermWeights = [];
        PermWeights = [PermWeights; SVRWeight.Weight];
        parfor permi = 1:Permutation_times
            PermIdx = randperm(size(Subjects_Data,1));
            PermWeight = SVR_Weight(Subjects_Data(:,idx),Subjects_Scores(PermIdx,1),C_Parameter);
            PermWeights = [PermWeights;PermWeight.Weight];
        end
        PermWeights = sort(PermWeights,'descend');
        for ii = 1:length(idx)
             xidx = find(SVRWeight.Weight(1,ii)==PermWeights(:,ii));
             P_value = xidx/Permutation_times2;
             if P_value(1) < 0.05 || P_value(1) > 0.95
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
   else
        Weight_mask = NaN * zeros(1,size(Subjects_Data,2));
        Results.Weight{v,:} = Weight_mask;
        SigWeight_mask = NaN * zeros(1,size(Subjects_Data,2));
        Results.SigWeight{v,:} = SigWeight_mask;
        clear Sig_Weight
   end
end
Results.Final_Pred = table(Results.Corr,Results.P_value,Results.RMSE,Results.NRMSE, ...
Results.Optimal_m,'RowNames',{'State1','State2','State3','State4'},'VariableNames',{'Correlation','P_value','RMSE','NRMSE','Optimal_m'});
disp('_________________________________________________________');
disp('The prediction results are as follows:'); 
disp(Results.Final_Pred);
% save([root,filesep,'BS_dFNC_kmeans_results_corr.mat'],'Results')