import pandas as pd
import joblib

from train_utils import to_categorical, train_pipeline, save_models

import time

start = time.time()

#Read data
data = pd.read_pickle('data/train_test.pkl')
params_dict = joblib.load('data\lgbm_model_params.pkl')
feat_dict = joblib.load('data\lgbm_model_feats.pkl')

#Different models are created for different months
issue_months = [1, 2, 3, 4, 5, 6, 7]
#Number of boost rounds for following months. Taken from hyparparameters optimization
num_rounds_months = [220, 160, 300, 420, 480, 1000, 900]
#Set eval to show training results every EVAL step
EVAL = 100
#Initiate lists for storing models for given quantile for different months
lgb_models_10 = []
lgb_models_50 = []
lgb_models_90 = []

#Change types to 'category' in categorical columns
categorical = ['site_id']
data = to_categorical(['site_id'],
                      data)

#Divide data into different subsets
test = data[data.set == 'test'].reset_index(drop = True)
train = data[data.set == 'train'].reset_index(drop = True)
labels = train['volume']

#Train models for each month and keep them in different lists according to
#quantiles
for month_idx, month in enumerate(issue_months):
    print(f'Month: {month}')
    #Choose features from given month
    train_feat = feat_dict[month]
    #Choose params from given month
    params = params_dict[month]
    
    print('0.1 quantile')
    lgb_models_10 = train_pipeline(train,
                                   labels,
                                   month,
                                   month_idx,
                                   params,
                                   train_feat,
                                   num_rounds_months,
                                   categorical,
                                   0.1,
                                   lgb_models_10,
                                   EVAL)
    print('0.5 quantile')
    lgb_models_50 = train_pipeline(train,
                                   labels,
                                   month,
                                   month_idx,
                                   params,
                                   train_feat,
                                   num_rounds_months,
                                   categorical,
                                   0.5,
                                   lgb_models_50,
                                   EVAL)
    print('0.9 quantile')
    lgb_models_90 = train_pipeline(train,
                                   labels,
                                   month,
                                   month_idx,
                                   params,
                                   train_feat,
                                   num_rounds_months,
                                   categorical,
                                   0.9,
                                   lgb_models_90,
                                   EVAL)

#Save models
save_models(issue_months,
            lgb_models_10,
            lgb_models_50,
            lgb_models_90)

end = time.time()
elapsed = end - start
