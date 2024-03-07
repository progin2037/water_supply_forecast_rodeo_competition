import pandas as pd
import numpy as np
import joblib

from train_utils import load_models, to_categorical
from utils import get_quantiles_from_distr, all_distr_dict, distr_to_change

import time

start = time.time()

issue_months = [1, 2, 3, 4, 5, 6, 7]
#Load models. For loading models not created today, use read_models_from_repo
#or specific_date parameters
lgb_models_10, lgb_models_50, lgb_models_90 = load_models(issue_months)
#Read data
data = pd.read_pickle('data/train_test_forecast.pkl')
#Read model features
train_feat_dict = joblib.load('data\lgbm_model_feats.pkl')
#Read min-max values of volume per site_id
min_max_site_id = pd.read_pickle('data\min_max_site_id.pkl')

#Read submission example
submission_format = pd.read_csv('data/submission_format.csv')

#Path to different distributions for different site_id
PATH_DISTR = 'data\distr_per_site_forecast_50_outliers_2_5_best'
#Set importance for distribution in quantiles 0.1 and 0.9 calculation for
#different months
distr_perc_dict = {1: 0.6,
                   2: 0.45,
                   3: 0.4,
                   4: 0.3,
                   5: 0.25,
                   6: 0.2,
                   7: 0.1}

#Keep only test data
test = data[data.set == 'test'].reset_index(drop = True)
#Change categorical data types
test = to_categorical(['site_id'], test)

#Initialize columns to predict
#Final predictions
test['volume_10'] = np.nan
test['volume_50'] = np.nan
test['volume_90'] = np.nan
#Predictions from models. Only for 0.1 and 0.9 as 0.5Q use only model results
test['volume_10_lgbm'] = np.nan
test['volume_90_lgbm'] = np.nan
#Predictions from distribution
test['volume_10_distr'] = np.nan
test['volume_90_distr'] = np.nan

#Fill LightGBM model values
for idx, month in enumerate(issue_months):
    test.loc[test.month == month, 'volume_50'] =\
        lgb_models_50[idx].predict(test.loc[test.month == month, train_feat_dict[month]])
    test.loc[test.month == month, 'volume_10_lgbm'] =\
        lgb_models_10[idx].predict(test.loc[test.month == month, train_feat_dict[month]])
    test.loc[test.month == month, 'volume_90_lgbm'] =\
        lgb_models_90[idx].predict(test.loc[test.month == month, train_feat_dict[month]])

#Get predictions from distributions
test = get_quantiles_from_distr(test,
                                min_max_site_id,
                                all_distr_dict,
                                PATH_DISTR,
                                distr_to_change)

#Add min and max for site_id as 'max' and 'min' columns
test = pd.merge(test,
                min_max_site_id,
                how = 'left',
                left_on = 'site_id',
                right_index = True)

#Change volume values values greater than min (max) for site_id to that min (max) value.
#Do it also for distribution volume, though it shouldn't exceed maximum values,
#just to be certain.
test.loc[test['volume_50'] < test['min'], 'volume_50'] = test['min']
test.loc[test['volume_50'] > test['max'], 'volume_50'] = test['max']

test.loc[test['volume_10_lgbm'] < test['min'], 'volume_10_lgbm'] = test['min']
test.loc[test['volume_10_distr'] < test['min'], 'volume_10_distr'] = test['min']

test.loc[test['volume_90_lgbm'] > test['max'], 'volume_90_lgbm'] = test['max']
test.loc[test['volume_90_distr'] > test['max'], 'volume_90_distr'] = test['max']

#Clipping:
#   if volume_90 < volume_50, change volume_90 to volume_50
#   if volume_50 < volume_10, change volume_10 to volume_50            
#Do it also for distribution estimates to be certain that it's being used.
test.loc[test.volume_90_lgbm < test.volume_50, 'volume_90_lgbm'] = test.volume_50
test.loc[test.volume_50 < test.volume_10_lgbm, 'volume_10_lgbm'] = test.volume_50

test.loc[test.volume_90_distr < test.volume_50, 'volume_90_distr'] = test.volume_50
test.loc[test.volume_50 < test.volume_10_distr, 'volume_10_distr'] = test.volume_50

#Get weighted average from distribution and models for Q0.1 and Q0.9. Do it
#separately for different months as distribution percentage varies depending on
#months
for month in issue_months:
    distr_perc = distr_perc_dict[month]
    test.loc[test.month == month, 'volume_10'] =\
        distr_perc * test.volume_10_distr + (1 - distr_perc) * test.volume_10_lgbm
    test.loc[test.month == month, 'volume_90'] =\
        distr_perc * test.volume_90_distr + (1 - distr_perc) * test.volume_90_lgbm

#Append predicted volumes to submission_format and save results
results = submission_format.copy()
results = pd.merge(results.drop(['volume_10', 'volume_50', 'volume_90'], axis = 1),
                   test[['site_id', 'issue_date', 'volume_10', 'volume_50', 'volume_90']])
results.to_csv('results/results.csv', index = False)

end = time.time()
elapsed = end - start
