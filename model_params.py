import joblib

#Set general LightGBM hyperparameters the same for each month
BAGGING_FREQ = 50
OBJECTIVE = 'quantile'
METRIC = 'quantile'
VERBOSE = -1
REG_ALPHA = 0
MIN_GAIN_TO_SPLIT = 0.0
MIN_SUM_HESSIAN_IN_LEAF = 0.001
FEATURE_FRACTION_SEED = 2112
SEED = 2112
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
            'data\general_hyperparams.pkl')

#Read hyperparameters tuning results. Commented as this step is not necessary,
#values from hyperparamaters tuning are already hardcoded below
'''
#Read params from files
print('Jan')
study = joblib.load('results/hyperparams_tuning/study_2023_12_19_14_25_54_month_1.pkl')
display(study.best_params)

print('Feb')
study = joblib.load('results/hyperparams_tuning/study_2023_12_19_16_25_25_month_2.pkl')
display(study.best_params)

print('Mar')
study = joblib.load('results/hyperparams_tuning/study_2023_12_19_20_40_25_month_3.pkl')
display(study.best_params)

print('Apr')
study = joblib.load('results/hyperparams_tuning/study_2023_12_20_00_18_08_month_4.pkl')
display(study.best_params)

print('May')
study = joblib.load('results/hyperparams_tuning/study_2023_12_20_18_33_05_month_5.pkl')
display(study.best_params)

print('June')
study = joblib.load('results/hyperparams_tuning/study_2023_12_20_19_05_48_month_6.pkl')
display(study.best_params)

print('July')
study = joblib.load('results/hyperparams_tuning/study_2023_12_21_17_58_19_month_7.pkl')
display(study.best_params)
'''

#Set different params for each month. These are best params from hyperparameters
#optimization
#Jan
params_1 = {'objective': OBJECTIVE,
            'metric': METRIC,
            'learning_rate': 0.07080932034021628,
            'max_depth': 6,
            'num_leaves': 108,
            'lambda_l1': REG_ALPHA,
            'lambda_l2': 0.02544353791657603,
            'min_gain_to_split': MIN_GAIN_TO_SPLIT,
            'subsample': 0.801020595961324,
            'bagging_freq': BAGGING_FREQ,
            'bagging_seed': FEATURE_FRACTION_SEED,
            'feature_fraction': 1.0,
            'feature_fraction_seed': FEATURE_FRACTION_SEED,
            'max_bin': 248,
            'min_data_in_leaf': 22,
            'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
            'verbose': VERBOSE,
            'seed': SEED}
#Feb
params_2 = {'objective': OBJECTIVE,
            'metric': METRIC,
            'learning_rate': 0.07431327342386611,
            'max_depth': 6,
            'num_leaves': 58,
            'lambda_l1': REG_ALPHA,
            'lambda_l2': 0.08035611763711388,
            'min_gain_to_split': MIN_GAIN_TO_SPLIT,
            'subsample': 0.8012846276823483,
            'bagging_freq': BAGGING_FREQ,
            'bagging_seed': FEATURE_FRACTION_SEED,
            'feature_fraction': 0.9,
            'feature_fraction_seed': FEATURE_FRACTION_SEED,
            'max_bin': 238,
            'min_data_in_leaf': 25,
            'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
            'verbose': VERBOSE,
            'seed': SEED}
#Mar
params_3 = {'objective': OBJECTIVE,
            'metric': METRIC,
            'learning_rate': 0.05391608535819001,
            'max_depth': 6,
            'num_leaves': 76,
            'lambda_l1': REG_ALPHA,
            'lambda_l2': 0.01991847988533871,
            'min_gain_to_split': MIN_GAIN_TO_SPLIT,
            'subsample': 0.8960761555373657,
            'bagging_freq': BAGGING_FREQ,
            'bagging_seed': FEATURE_FRACTION_SEED,
            'feature_fraction': 0.9,
            'feature_fraction_seed': FEATURE_FRACTION_SEED,
            'max_bin': 294,
            'min_data_in_leaf': 16,
            'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
            'verbose': VERBOSE,
            'seed': SEED}
#Apr
params_4 = {'objective': OBJECTIVE,
            'metric': METRIC,
            'learning_rate': 0.04957368927648065,
            'max_depth': 6,
            'num_leaves': 64,
            'lambda_l1': REG_ALPHA,
            'lambda_l2': 0.0004490693730225264,
            'min_gain_to_split': MIN_GAIN_TO_SPLIT,
            'subsample': 0.8126256151956979,
            'bagging_freq': BAGGING_FREQ,
            'bagging_seed': FEATURE_FRACTION_SEED,
            'feature_fraction': 1.0,
            'feature_fraction_seed': FEATURE_FRACTION_SEED,
            'max_bin': 218,
            'min_data_in_leaf': 25,
            'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
            'verbose': VERBOSE,
            'seed': SEED}
#May
params_5 = {'objective': OBJECTIVE,
            'metric': METRIC,
            'learning_rate': 0.07131428559622419,
            'max_depth': 10,
            'num_leaves': 125,
            'lambda_l1': REG_ALPHA,
            'lambda_l2': 0.20632642059931852,
            'min_gain_to_split': MIN_GAIN_TO_SPLIT,
            'subsample': 0.816939322034667,
            'bagging_freq': BAGGING_FREQ,
            'bagging_seed': FEATURE_FRACTION_SEED,
            'feature_fraction': 0.875,
            'feature_fraction_seed': FEATURE_FRACTION_SEED,
            'max_bin': 239,
            'min_data_in_leaf': 23,
            'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
            'verbose': VERBOSE,
            'seed': SEED}
#Jun
params_6 = {'objective': OBJECTIVE,
            'metric': METRIC,
            'learning_rate': 0.0720252171436197,
            'max_depth': 7,
            'num_leaves': 63,
            'lambda_l1': REG_ALPHA,
            'lambda_l2': 0.007869332952423388,
            'min_gain_to_split': MIN_GAIN_TO_SPLIT,
            'subsample': 0.8713846673500828,
            'bagging_freq': BAGGING_FREQ,
            'bagging_seed': FEATURE_FRACTION_SEED,
            'feature_fraction': 0.8888888888888888,
            'feature_fraction_seed': FEATURE_FRACTION_SEED,
            'max_bin': 234,
            'min_data_in_leaf': 23,
            'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
            'verbose': VERBOSE,
            'seed': SEED}
#Jul
params_7 = {'objective': OBJECTIVE,
            'metric': METRIC,
            'learning_rate': 0.0670977949575001,
            'max_depth': 9,
            'num_leaves': 71,
            'lambda_l1': REG_ALPHA,
            'lambda_l2': 0.7613143327228161,
            'min_gain_to_split': MIN_GAIN_TO_SPLIT,
            'subsample': 0.9523355682312036,
            'bagging_freq': BAGGING_FREQ,
            'bagging_seed': FEATURE_FRACTION_SEED,
            'feature_fraction': 1.0,
            'feature_fraction_seed': FEATURE_FRACTION_SEED,
            'max_bin': 283,
            'min_data_in_leaf': 19,
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
joblib.dump(params_dict, 'data\lgbm_model_params.pkl')

#Set different features for each month
#Jan
train_feat_1 = ['site_id', 'nat_flow_prev', 'WTEQ_DAILY_prev', 'issue_date_no_year',
                'nat_flow_11_to_10_ratio']
#Feb
train_feat_2 = ['site_id', 'nat_flow_prev', 'WTEQ_DAILY_prev', 'issue_date_no_year',
                'discharge_cfs_mean_std', 'longitude']
#Mar
train_feat_3 = ['site_id', 'nat_flow_prev', 'WTEQ_DAILY_prev', 'issue_date_no_year',
                'discharge_cfs_mean_std', 'longitude']
#Apr
train_feat_4 = ['site_id', 'nat_flow_prev', 'WTEQ_DAILY_prev', 'issue_date_no_year',
                'discharge_cfs_mean_std', 'longitude']
#May
train_feat_5 = ['site_id', 'nat_flow_prev', 'WTEQ_DAILY_prev', 'issue_date_no_year',
                'discharge_cfs_mean_std', 'longitude', 'WTEQ_DAILY_Apr_mean',
                'discharge_cfs_mean_Apr_mean']
#Jun
train_feat_6 = ['site_id', 'nat_flow_prev', 'WTEQ_DAILY_prev', 'issue_date_no_year',
                'discharge_cfs_mean_std', 'longitude', 'WTEQ_DAILY_Apr_mean',
                'discharge_cfs_mean_Apr_mean', 'nat_flow_Apr_mean']
#Jul
train_feat_7 = ['site_id', 'nat_flow_prev', 'WTEQ_DAILY_prev', 'issue_date_no_year',
                'discharge_cfs_mean_std', 'longitude', 'nat_flow_Apr_mean',
                'WTEQ_DAILY_Apr_mean', 'discharge_cfs_mean_Apr_mean']

#Keep all features in one dictionary
train_feat_dict = {1: train_feat_1,
                   2: train_feat_2,
                   3: train_feat_3,
                   4: train_feat_4,
                   5: train_feat_5,
                   6: train_feat_6,
                   7: train_feat_7}
#Save features to .pkl
joblib.dump(train_feat_dict, 'data\lgbm_model_feats.pkl')
