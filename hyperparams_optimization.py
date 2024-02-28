import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_pinball_loss
import joblib
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm

from utils import get_quantiles_from_distr, all_distr_dict, distr_to_change
from train_utils import get_years_cv, get_cv_folds, to_categorical
from model_params import train_feat_dict

import time


def objective(trial,
              train,
              month,
              train_feat,
              min_max_site_id,
              path_distr,
              distr_perc):

    ###########################################################################
    #HYPERPARAMETERS
    ###########################################################################
    #Set repetitive parameters
    BAGGING_FREQ = 50
    OBJECTIVE = 'quantile'
    METRIC = 'quantile'
    VERBOSE = -1 
    REG_ALPHA = 0
    MIN_GAIN_TO_SPLIT = 0.0
    FEATURE_FRACTION_SEED = 2112
    MIN_SUM_HESSIAN_IN_LEAF = 0.001
    SEED = 2112
    
    #Set minimial number of columns to one less than the number of columns.
    #0.001 is added to the result to to deal with optuna approximation.
    #Months 2, 3 and 4 were trained with static 0.9/1 feature fraction
    if month in [2, 3 ,4]:
        feature_fraction_min = 0.9
    else:
        feature_fraction_min = ((len(train_feat) - 1) / len(train_feat)) + 0.001
    
    #Set range of values for different hyperparameters
    params = {'objective': OBJECTIVE,
              'metric': METRIC,
              'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
              'max_depth': trial.suggest_int('max_depth', 5, 10),
              'num_leaves': trial.suggest_int('num_leaves', 16, 128),
              'lambda_l1': REG_ALPHA,
              'lambda_l2': trial.suggest_float('lambda_l2', 0.0001, 10.0, log = True),
              'min_gain_to_split': MIN_GAIN_TO_SPLIT,
              'subsample': trial.suggest_float('subsample', 0.7, 1.0),
              'bagging_freq': BAGGING_FREQ,
              'bagging_seed': FEATURE_FRACTION_SEED,
              'feature_fraction': trial.suggest_float('feature_fraction',
                                                      feature_fraction_min, 1.0,
                                                      step = 1-feature_fraction_min),
                                                      #0.9, 1.0, step = 0.1),
              'feature_fraction_seed': FEATURE_FRACTION_SEED,
              'max_bin': trial.suggest_int('max_bin', 200, 300),
              'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 15, 25),
              'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
              'verbose': VERBOSE,
              'seed': SEED}

    ###########################################################################
    #PARAMETERS AND VARIABLES INITIALIZATION
    ###########################################################################
    NUM_BOOST_ROUND = 600
    NUM_BOOST_ROUND_START = 100 #Start early stopping from 100 iterations
    EARLY_STOPPING_ROUNDS = 2 #How many times in a row early stopping can be
                              #met before stopping training
    EARLY_STOPPING_STEP = 20 #Number of iterations when Early stopping is
                             #performed. 20 means it's done every 20 iterations,
                             #i,e. after 100, 120, 140, ... iters
    YEAR_RANGE = True

    #Initialize empty variables
    results_10_clipped = []
    results_50_clipped = []
    results_90_clipped = []
    cv_results = []
    cv_results_all_months = dict()

    results_10_clipped_all_months = dict()
    results_50_clipped_all_months = dict()
    results_90_clipped_all_months = dict()

    results_10 = []
    results_50 = []
    results_90 = []

    cv_results_avg_fold = []

    results_10_clipped_all = []
    results_50_clipped_all = []
    results_90_clipped_all = []

    lgb_models_50 = dict()
    lgb_models_10 = dict()
    lgb_models_90 = dict()

    #Initialize starting values
    NUM_BOOST_ROUND_MONTH = NUM_BOOST_ROUND_START
    cv_result_avg_fold_prev = np.inf
    num_prev_iters_better = 0

    ###########################################################################
    #TRAINING
    ###########################################################################
    while (num_prev_iters_better < EARLY_STOPPING_ROUNDS) & (NUM_BOOST_ROUND_MONTH <= NUM_BOOST_ROUND):
        results_10_clipped = []
        results_50_clipped = []
        results_90_clipped = []
        cv_results = []
        
        #Get years for test folds
        years_cv = get_years_cv(YEAR_RANGE)
        
        for fold, year in enumerate(years_cv):
            #Get folds' indexes
            train_cv_idxs, test_cv_idxs = get_cv_folds(train,
                                                       month,
                                                       years_cv,
                                                       YEAR_RANGE)

            train_data = lgb.Dataset(data = train.loc[train_cv_idxs[fold], train_feat],
                                     label = labels[train_cv_idxs[fold]],
                                     categorical_feature = categorical)
            test_data = lgb.Dataset(data = train.loc[test_cv_idxs[fold], train_feat],
                                    label = labels[test_cv_idxs[fold]],
                                    reference = train_data)                
            params['alpha'] = 0.5
            
            if NUM_BOOST_ROUND_MONTH == NUM_BOOST_ROUND_START:
                lgb_model_50 = lgb.train(params,
                                         train_data,
                                         valid_sets=[train_data, test_data],
                                         num_boost_round = NUM_BOOST_ROUND_START,
                                         keep_training_booster = True)
                lgb_models_50[fold] = lgb_model_50
            else:
                while lgb_models_50[fold].current_iteration() < NUM_BOOST_ROUND_MONTH:
                    lgb_models_50[fold].update()

            preds_50 = lgb_models_50[fold].predict(train.loc[test_cv_idxs[fold], train_feat])

            params['alpha'] = 0.1

            train_data = lgb.Dataset(data = train.loc[train_cv_idxs[fold], train_feat],
                                     label = labels[train_cv_idxs[fold]],
                                     categorical_feature = categorical)
            test_data = lgb.Dataset(data = train.loc[test_cv_idxs[fold], train_feat],
                                    label = labels[test_cv_idxs[fold]],
                                    reference = train_data)

            if NUM_BOOST_ROUND_MONTH == NUM_BOOST_ROUND_START:
                lgb_model_10 = lgb.train(params,
                                         train_data,
                                         valid_sets=[train_data, test_data],
                                         num_boost_round = NUM_BOOST_ROUND_START,
                                         keep_training_booster = True)
                lgb_models_10[fold] = lgb_model_10
            else:
                while lgb_models_10[fold].current_iteration() < NUM_BOOST_ROUND_MONTH:
                    lgb_models_10[fold].update()

            preds_10_lgbm = lgb_models_10[fold].predict(train.loc[test_cv_idxs[fold], train_feat])

            params['alpha'] = 0.9

            train_data = lgb.Dataset(data = train.loc[train_cv_idxs[fold], train_feat],
                                     label = labels[train_cv_idxs[fold]],
                                     categorical_feature = categorical)
            test_data = lgb.Dataset(data = train.loc[test_cv_idxs[fold], train_feat],
                                    label = labels[test_cv_idxs[fold]],
                                    reference = train_data)

            if NUM_BOOST_ROUND_MONTH == NUM_BOOST_ROUND_START:
                lgb_model_90 = lgb.train(params,
                                         train_data,
                                         valid_sets=[train_data, test_data],
                                         num_boost_round = NUM_BOOST_ROUND_START,
                                         keep_training_booster = True)
                lgb_models_90[fold] = lgb_model_90
            else:
                while lgb_models_90[fold].current_iteration() < NUM_BOOST_ROUND_MONTH:
                    lgb_models_90[fold].update()

            preds_90_lgbm = lgb_models_90[fold].predict(train.loc[test_cv_idxs[fold], train_feat])

            result_df = train.loc[test_cv_idxs[fold], train_feat]
            result_df['volume_50'] = preds_50

            result_df['volume_10_lgbm'] = preds_10_lgbm
            result_df['volume_90_lgbm'] = preds_90_lgbm

            result_df = get_quantiles_from_distr(result_df,
                                                 min_max_site_id,
                                                 all_distr_dict,
                                                 PATH_DISTR,
                                                 distr_to_change)

            #Add min and max for site_id as 'max' and 'min' columns
            result_df = pd.merge(result_df,
                                 min_max_site_id,
                                 how = 'left',
                                 left_on = 'site_id',
                                 right_index = True)

            #Change volume values values greater than min (max) for site_id to that min (max) value.
            #Do it also for distribution volume, though it shouldn't exceed maximum values, just to be certain.
            result_df.loc[result_df['volume_50'] < result_df['min'], 'volume_50'] = result_df['min']
            result_df.loc[result_df['volume_50'] > result_df['max'], 'volume_50'] = result_df['max']

            result_df.loc[result_df['volume_10_lgbm'] < result_df['min'], 'volume_10_lgbm'] = result_df['min']
            result_df.loc[result_df['volume_10_distr'] < result_df['min'], 'volume_10_distr'] = result_df['min']

            result_df.loc[result_df['volume_90_lgbm'] > result_df['max'], 'volume_90_lgbm'] = result_df['max']
            result_df.loc[result_df['volume_90_distr'] > result_df['max'], 'volume_90_distr'] = result_df['max']


            #Clipping, if volume_90 < volume_50 -> change volume_90 to volume_50
            #          if volume_50 < volume_10 -> change volume_10 to volume_50            
            #Do it also for distribution estimate to be certain that it's used.
            result_df.loc[result_df.volume_90_lgbm < result_df.volume_50,
                          'volume_90_lgbm'] = result_df.volume_50
            result_df.loc[result_df.volume_50 < result_df.volume_10_lgbm,
                          'volume_10_lgbm'] = result_df.volume_50

            result_df.loc[result_df.volume_90_distr < result_df.volume_50,
                          'volume_90_distr'] = result_df.volume_50
            result_df.loc[result_df.volume_50 < result_df.volume_10_distr,
                          'volume_10_distr'] = result_df.volume_50

            #Get weighted average from distribution and models for Q0.1 and Q0.9
            result_df['volume_10'] = distr_perc * result_df.volume_10_distr +\
                (1 - distr_perc) * result_df.volume_10_lgbm
            result_df['volume_90'] = distr_perc * result_df.volume_90_distr +\
                (1 - distr_perc) * result_df.volume_90_lgbm

            result_10 = mean_pinball_loss(labels[test_cv_idxs[fold]],
                                    result_df.volume_10,
                                    alpha = 0.1)
            result_50 = mean_pinball_loss(labels[test_cv_idxs[fold]],
                                    result_df.volume_50,
                                    alpha = 0.5)
            result_90 = mean_pinball_loss(labels[test_cv_idxs[fold]],
                                    result_df.volume_90,
                                    alpha = 0.9)

            #Append results from this fold
            results_10.append(result_10)
            results_50.append(result_50)
            results_90.append(result_90)

            cv_result = 2 * (result_10 +
                         result_50 +
                         result_90) / 3

            results_clipped = result_df.copy()
            #Do the final clipping to make sure that the restriction is fulfilled
            #after taking weighted average for volume_10 and volume_90
            results_clipped.loc[results_clipped.volume_90 < results_clipped.volume_50,
                                'volume_50'] = results_clipped.volume_90
            results_clipped.loc[results_clipped.volume_50 < results_clipped.volume_10,
                                'volume_10'] = results_clipped.volume_50

            result_10_clipped = mean_pinball_loss(labels[test_cv_idxs[fold]],
                                    results_clipped.volume_10,
                                    alpha = 0.1)
            result_50_clipped = mean_pinball_loss(labels[test_cv_idxs[fold]],
                                    results_clipped.volume_50,
                                    alpha = 0.5)
            result_90_clipped = mean_pinball_loss(labels[test_cv_idxs[fold]],
                                    results_clipped.volume_90,
                                    alpha = 0.9)

            #Append results from this fold
            results_10_clipped.append([fold, result_10_clipped, NUM_BOOST_ROUND_MONTH])
            results_50_clipped.append([fold, result_50_clipped, NUM_BOOST_ROUND_MONTH])
            results_90_clipped.append([fold, result_90_clipped, NUM_BOOST_ROUND_MONTH])

            cv_result = 2 * (result_10_clipped +
                             result_50_clipped +
                             result_90_clipped) / 3
            cv_results.append([cv_result])
        cv_result_avg_fold = np.mean(cv_results)
        #Keep results from all quantiles and num boost rounds
        results_10_clipped_all.append(results_10_clipped)
        results_50_clipped_all.append(results_50_clipped)
        results_90_clipped_all.append(results_90_clipped)

        if cv_result_avg_fold > cv_result_avg_fold_prev:
            num_prev_iters_better += 1
            #cv_result_avg_fold_prev doesn't change, as the comparison is still to 
            #this value that is the best value so far
        else:
            cv_result_avg_fold_prev = cv_result_avg_fold
            num_prev_iters_better = 0

        cv_results_avg_fold.append([cv_result_avg_fold, NUM_BOOST_ROUND_MONTH])
        NUM_BOOST_ROUND_MONTH += EARLY_STOPPING_STEP
    #Take into account if last values weren't the best ones
    if num_prev_iters_better != 0:
        cv_results_avg_fold = cv_results_avg_fold[:-num_prev_iters_better]
    cv_results_all_months[month] = cv_results_avg_fold
    results_10_clipped_all_months[month] = results_10_clipped_all
    results_50_clipped_all_months[month] = results_50_clipped_all
    results_90_clipped_all_months[month] = results_90_clipped_all
    
    best_score = cv_results_all_months[month][-1][0]
    trial.set_user_attr("num_boost_rounds_best", cv_results_all_months[month][-1][1]) 
    
    return best_score

start = time.time()

#Set number of hyperparameters optimization iterations
N_TRIALS = 2#150
#Different models are created for different months
issue_months = [1, 2, 3, 4, 5, 6, 7]
#Path to different distributions for different site_id
PATH_DISTR = 'data\distr_per_site_50_outliers_2_5_best'
#Set importance for distribution in quantiles 0.1 and 0.9 calculation
DISTR_PERC = 0.4

#Read data
data = pd.read_pickle('data/train_test.pkl')
feat_dict = joblib.load('data\lgbm_model_feats.pkl')
#Read min-max values of volume per site_id
min_max_site_id = pd.read_pickle('data\min_max_site_id.pkl')

#Change types to 'category' in categorical columns
categorical = ['site_id']
data = to_categorical(['site_id'],
                      data)

#Divide data into different subsets
train = data[data.set == 'train'].reset_index(drop = True)
labels = train['volume']

for issue_month in tqdm(issue_months):
    #Read features from given month
    train_feat = train_feat_dict[issue_month]
    sampler = TPESampler(seed = 2112) #to get the same results all the time
    #Perform hyperparameters tuning
    study = optuna.create_study(direction = 'minimize', sampler = sampler)
    study.optimize(lambda trial: objective(trial,
                                           train,
                                           issue_month,
                                           train_feat,
                                           min_max_site_id,
                                           PATH_DISTR,
                                           DISTR_PERC),
                   n_trials = N_TRIALS)
    
    #Save study
    joblib.dump(study,
                f"results/hyperparams_tuning/study_{pd.to_datetime('today').strftime('%Y_%m_%d_%H_%M_%S')}_month_{issue_month}.pkl")

end = time.time()
elapsed = end - start
