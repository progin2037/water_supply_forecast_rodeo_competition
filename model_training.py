import pandas as pd
import joblib

from train_utils import to_categorical, train_pipeline, save_models
from cv_and_hyperparams_opt import get_years_cv, lgbm_cv

import time

start = time.time()

#Set types of training to run
RUN_CV = True
RUN_HYPERPARAMS_TUNING = False
RUN_TRAINING = True

#Read data
data = pd.read_pickle('data/train_test.pkl')
params_dict = joblib.load('data\lgbm_model_params.pkl')
feat_dict = joblib.load('data\lgbm_model_feats.pkl')
#Read min-max values of volume per site_id
min_max_site_id = pd.read_pickle('data\min_max_site_id.pkl')

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

#CV params
if RUN_CV == True:
    #Use many years in a fold or not
    YEAR_RANGE = True
    #Maximum number of LightGBM model iterations
    NUM_BOOST_ROUND = 1000
    #Initiate early stopping from this iteration
    NUM_BOOST_ROUND_START = 100
    #How many times in a row early stopping can be met before stopping training
    EARLY_STOPPING_ROUNDS = 2
    #Number of iterations when Early stopping is performed. 20 means that it's
    #done every 20 iterations, i,e. after 100, 120, 140, ... iters
    EARLY_STOPPING_STEP = 20
    #Path to different distributions for different site_id
    PATH_DISTR = 'data\distr_per_site_50_outliers_2_5_best'
    #Set importance for distribution in quantiles 0.1 and 0.9 calculation
    DISTR_PERC = 0.4

    #Get years for different CV folds
    years_cv = get_years_cv(YEAR_RANGE)

    #Run CV
    best_cv_per_month, best_cv_avg, num_rounds_months =\
        lgbm_cv(train,
                labels,
                NUM_BOOST_ROUND,
                NUM_BOOST_ROUND_START,
                EARLY_STOPPING_ROUNDS,
                EARLY_STOPPING_STEP,
                issue_months,
                years_cv,
                YEAR_RANGE,
                feat_dict,
                params_dict,
                categorical,
                min_max_site_id,
                PATH_DISTR,
                DISTR_PERC)
    print('CV result avg over months:', best_cv_avg)
    print('CV results per month with number of iters:', best_cv_per_month)
    print('Number of iters per month:', num_rounds_months)

#Train models for each month and keep them in different lists according to
#quantiles
if RUN_TRAINING == True:
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
