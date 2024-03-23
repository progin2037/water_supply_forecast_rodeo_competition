import joblib

#Set general LightGBM hyperparameters the same for each month
BAGGING_FREQ = 50
OBJECTIVE = 'quantile'
METRIC = 'quantile'
VERBOSE = -1
REG_ALPHA = 0
MIN_GAIN_TO_SPLIT = 0.0
MIN_SUM_HESSIAN_IN_LEAF = 0.001
FEATURE_FRACTION_SEED = 22
SEED = 22
#Keep all repetitive hyperparams in one dictionary
joblib.dump([BAGGING_FREQ,
             OBJECTIVE,
             METRIC,
             VERBOSE,
             REG_ALPHA,
             MIN_GAIN_TO_SPLIT,
             MIN_SUM_HESSIAN_IN_LEAF,
             FEATURE_FRACTION_SEED,
             SEED],
            'data\general_hyperparams_final.pkl')

#Read hyperparameters tuning results. Commented as this step is not necessary,
#values from hyperparamaters tuning are already hardcoded below
'''
#Read params from files
print('Jan')
study = joblib.load('results/hyperparams_tuning/study_2024_01_11_16_20_18_month_1.pkl')
display(study.best_params)

print('Feb')
study = joblib.load('results/hyperparams_tuning/study_2024_01_11_19_23_41_month_2.pkl')
display(study.best_params)

print('Mar')
#Month 3 was the first to optimize and it was done with 100 iterations instead of 80
study = joblib.load('results/hyperparams_tuning/study_2024_01_08_17_21_37_month_3.pkl')
display(study.best_params)

print('Apr')
study = joblib.load('results/hyperparams_tuning/study_2024_01_10_12_46_25_month_4.pkl')
display(study.best_params)

print('May')
study = joblib.load('results/hyperparams_tuning/study_2024_01_11_17_13_45_month_5.pkl')
display(study.best_params)

print('June')
study = joblib.load('results/hyperparams_tuning/study_2024_01_11_00_10_08_month_6.pkl')
display(study.best_params)

print('July')
#There are only first 5 saved iterations for July due to some crashes in the
#processing. The optimization was performed again and after ~60 iterations the
#5th one was still the best, so it was kept as the final one for the Forecast
#Stage due to the approaching deadline
study = joblib.load('results/hyperparams_tuning/study_2024_01_10_13_23_44_month_7.pkl')
display(study.best_params)
'''

#Set different params for each month. These are best params from hyperparameters
#optimization

#Jan
params_1 = {'objective': OBJECTIVE,
            'metric': METRIC,
            'learning_rate': 0.08668237528700982,
            'max_depth': 4,
            'num_leaves': 50,
            'lambda_l1': REG_ALPHA,
            'lambda_l2': 7.921707376269314,
            'min_gain_to_split': MIN_GAIN_TO_SPLIT,
            'subsample': 0.7802262313131088,
            'bagging_freq': BAGGING_FREQ,
            'bagging_seed': FEATURE_FRACTION_SEED,
            'feature_fraction': 0.8,
            'feature_fraction_seed': FEATURE_FRACTION_SEED,
            'max_bin': 120,
            'min_data_in_leaf': 21,
            'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
            'verbose': VERBOSE,
            'seed': SEED}

#Feb
params_2 = {'objective': OBJECTIVE,
            'metric': METRIC,
            'learning_rate': 0.122248501301184,
            'max_depth': 6,
            'num_leaves': 63,
            'lambda_l1': REG_ALPHA,
            'lambda_l2': 0.17616556280119494,
            'min_gain_to_split': MIN_GAIN_TO_SPLIT,
            'subsample': 0.7005458097707512,
            'bagging_freq': BAGGING_FREQ,
            'bagging_seed': FEATURE_FRACTION_SEED,
            'feature_fraction': 0.8571428571428571,
            'feature_fraction_seed': FEATURE_FRACTION_SEED,
            'max_bin': 143,
            'min_data_in_leaf': 30,
            'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
            'verbose': VERBOSE,
            'seed': SEED}

#Mar
params_3 = {'objective': OBJECTIVE,
            'metric': METRIC,
            'learning_rate': 0.06185209436616753,
            'max_depth': 5,
            'num_leaves': 35,
            'lambda_l1': REG_ALPHA,
            'lambda_l2': 0.022069278840869757,
            'min_gain_to_split': MIN_GAIN_TO_SPLIT,
            'subsample': 0.7961156775816042,
            'bagging_freq': BAGGING_FREQ,
            'bagging_seed': FEATURE_FRACTION_SEED,
            'feature_fraction': 0.8571428571428571,
            'feature_fraction_seed': FEATURE_FRACTION_SEED,
            'max_bin': 200,
            'min_data_in_leaf': 29,
            'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
            'verbose': VERBOSE,
            'seed': SEED}

#Apr
params_4 = {'objective': OBJECTIVE,
            'metric': METRIC,
            'learning_rate': 0.05711692601228281,
            'max_depth': 6,
            'num_leaves': 91,
            'lambda_l1': REG_ALPHA,
            'lambda_l2': 0.010781401143472777,
            'min_gain_to_split': MIN_GAIN_TO_SPLIT,
            'subsample': 0.7146513043023093,
            'bagging_freq': BAGGING_FREQ,
            'bagging_seed': FEATURE_FRACTION_SEED,
            'feature_fraction': 0.875,
            'feature_fraction_seed': FEATURE_FRACTION_SEED,
            'max_bin': 182,
            'min_data_in_leaf': 17,
            'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
            'verbose': VERBOSE,
            'seed': SEED}

#May
params_5 = {'objective': OBJECTIVE,
            'metric': METRIC,
            'learning_rate': 0.036413549774305234,
            'max_depth': 9,
            'num_leaves': 59,
            'lambda_l1': REG_ALPHA,
            'lambda_l2': 0.00010009943502276089,
            'min_gain_to_split': MIN_GAIN_TO_SPLIT,
            'subsample': 0.7708049704957225,
            'bagging_freq': BAGGING_FREQ,
            'bagging_seed': FEATURE_FRACTION_SEED,
            'feature_fraction': 0.875,
            'feature_fraction_seed': FEATURE_FRACTION_SEED,
            'max_bin': 226,
            'min_data_in_leaf': 25,
            'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
            'verbose': VERBOSE,
            'seed': SEED}
#Jun
params_6 = {'objective': OBJECTIVE,
            'metric': METRIC,
            'learning_rate': 0.03710481544436431,
            'max_depth': 10,
            'num_leaves': 109,
            'lambda_l1': REG_ALPHA,
            'lambda_l2': 2.377774250624091,
            'min_gain_to_split': MIN_GAIN_TO_SPLIT,
            'subsample': 0.7631179547240611,
            'bagging_freq': BAGGING_FREQ,
            'bagging_seed': FEATURE_FRACTION_SEED,
            'feature_fraction': 1.0,
            'feature_fraction_seed': FEATURE_FRACTION_SEED,
            'max_bin': 295,
            'min_data_in_leaf': 24,
            'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
            'verbose': VERBOSE,
            'seed': SEED}
#Jul
params_7 = {'objective': OBJECTIVE,
            'metric': METRIC,
            'learning_rate': 0.025193770366946387,
            'max_depth': 8,
            'num_leaves': 102,
            'lambda_l1': REG_ALPHA,
            'lambda_l2': 0.0005725785588927758,
            'min_gain_to_split': MIN_GAIN_TO_SPLIT,
            'subsample': 0.9101358458638595,
            'bagging_freq': BAGGING_FREQ,
            'bagging_seed': FEATURE_FRACTION_SEED,
            'feature_fraction': 0.8888888888888888,
            'feature_fraction_seed': FEATURE_FRACTION_SEED,
            'max_bin': 290,
            'min_data_in_leaf': 15,
            'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
            'verbose': VERBOSE,
            'seed': SEED}

#Keep all params in one dictionary
params_dict = {1: params_1,
               2: params_2,
               3: params_3,
               4: params_4,
               5: params_5,
               6: params_6,
               7: params_7}

#Save params to .pkl
joblib.dump(params_dict, 'data\lgbm_model_params_final.pkl')

#Set different features for each month
#Jan
train_feat_1 = ['site_id',
                'WTEQ_DAILY_prev',
                'issue_date_no_year',
                'pdsi_prev',
                'sd_forecasts']
#Feb
train_feat_2 = ['site_id',
                'WTEQ_DAILY_prev',
                'issue_date_no_year',
                'discharge_cfs_mean_since_Oct_std',
                'longitude',
                'pdsi_prev',
                'sd_forecasts']
#Mar
train_feat_3 = ['site_id',
                'nat_flow_prev',
                'WTEQ_DAILY_prev',
                'issue_date_no_year',
                'discharge_cfs_mean_since_Oct_std',
                'longitude',
                'sd_forecasts_with_jun']
#Apr
train_feat_4 = ['site_id',
                'WTEQ_DAILY_prev',
                'issue_date_no_year',
                'discharge_cfs_mean_since_Oct_std',
                'longitude', 
                'pdsi_prev_to_last_month_diff',
                'pdsi_prev',
                'sd_forecasts_with_jun']
#May
train_feat_5 = ['site_id', 'nat_flow_prev',
                'WTEQ_DAILY_prev',
                'issue_date_no_year',
                'discharge_cfs_mean_since_Oct_std',
                'longitude',
                'WTEQ_DAILY_Apr_mean',
                'discharge_cfs_mean_Apr_mean',
                'pdsi_prev']
#Jun
train_feat_6 = ['site_id',
                'nat_flow_prev',
                'WTEQ_DAILY_prev',
                'issue_date_no_year',
                'discharge_cfs_mean_since_Oct_std',
                'longitude',
                'WTEQ_DAILY_Apr_mean',
                'discharge_cfs_mean_Apr_mean',
                'nat_flow_Apr_mean',
                'pdsi_prev']
#Jul
train_feat_7 = ['site_id',
                'nat_flow_prev',
                'WTEQ_DAILY_prev',
                'issue_date_no_year',
                'discharge_cfs_mean_std',
                'longitude',
                'nat_flow_Apr_mean',
                'WTEQ_DAILY_Apr_mean',
                'discharge_cfs_mean_Apr_mean',
                'pdsi_prev']

#Keep all features in one dictionary
train_feat_dict = {1: train_feat_1,
                   2: train_feat_2,
                   3: train_feat_3,
                   4: train_feat_4,
                   5: train_feat_5,
                   6: train_feat_6,
                   7: train_feat_7}
#Save features to .pkl
joblib.dump(train_feat_dict, 'data\lgbm_model_feats_final.pkl')
