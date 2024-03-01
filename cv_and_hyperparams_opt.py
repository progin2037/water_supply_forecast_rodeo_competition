import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_pinball_loss
from tqdm import tqdm

from utils import get_quantiles_from_distr, all_distr_dict, distr_to_change

def get_cv_folds(train: pd.DataFrame,
                 month: int,
                 years_cv: list,
                 year_range: bool) -> tuple[list, list]:
    """
    Create cross-validation folds. Get train and test indices for different
    folds.
    
    Args:
        train (pd.DataFrame): Train data
        month (int): Month to create folds for
        years_cv (list): A list with test years for different CV folds.
            In case of many years in one test fold, those years are specified
            in consecutive lists inside the years_cv list
        year_range (bool): Specifies if there could be many years in test data
            (True) or just one test data per fold (False)
    Returns:
        train_cv_idxs (list): Indexes of train data for consecutive folds
        test_cv_idxs (list): Indexes of test data for consecutive folds
    """
    train_cv_idxs = []
    test_cv_idxs = []

    if year_range == True:
        #Store many years in test sets
        for year in years_cv:
            train_cv_idxs.append(list(train[(~(train.year.between(year[0], year[1]))) &
                                            (train.month == month)].index))
            test_cv_idxs.append(list(train[(train.year.between(year[0], year[1])) &
                                           (train.month == month)].index))
    else:
        #Store only one year in test sets
        for year in years_cv:
            train_cv_idxs.append(list(train[(train.year != year) &
                                            (train.month == month)].index))
            test_cv_idxs.append(list(train[(train.year == year) &
                                            (train.month == month)].index))
    return train_cv_idxs, test_cv_idxs


def get_years_cv(year_range: bool) -> list:
    """
    Get years for CV test folds. There are 2 options: 2 years in a test fold
    or 1 year in a test fold.
    Keep in mind that odd years since 2005 are in the test set, so in such
    cases for 2-year test folds, they aren't included in 2 years range
    calculation (2020-2022 range is treated as 2 years as 2021 is missing).
    
    Args:
        year_range (bool): Specifies if there could be many years in test data
            (True) or just one test data per fold (False)
    Returns:
        years_cv (list): A list with test years for different CV folds.
            In case of many years in one test fold, those years are specified
            in consecutive lists inside the years_cv list
    """
    #3 years in test
    if year_range == True:
        years_cv = [[1994, 1995],
                    [1996, 1997],
                    [1998, 1999],
                    [2000, 2001],
                    [2002, 2003],
                    [2004, 2006],
                    [2008, 2010],
                    [2012, 2014],
                    [2016, 2018],
                    [2020, 2022]]
    else:
        #One year at a time
        years_cv = [1999,
                    2000,
                    2001,
                    2002,
                    2003,
                    2004,
                    2006,
                    2008,
                    2010,
                    2012,
                    2014,
                    2016,
                    2018,
                    2020,
                    2022]
    return years_cv


def lgbm_cv(train: pd.DataFrame,
            labels: pd.Series,
            num_boost_round: int,
            num_boost_round_start: int,
            early_stopping_rounds: int,
            early_stopping_step: int,
            issue_months: list,
            years_cv: list,
            year_range: bool,
            train_feat_dict: dict,
            params_dict: dict,
            categorical: list,
            min_max_site_id: pd.DataFrame,
            path_distr: str,
            distr_perc: float) -> tuple[np.array, float, list]:
    """
    Run LightGBM CV with early stopping, get distribution estimates and
    average the results. Perform additional clipping to model predictions.
    
    Args:
        train (pd.DataFrame): Whole training data before dividing into CV folds
        labels (pd.Series): Labels corresponding to train data
        num_boost_round (int): Maximum number of estimators used in LightGBM
            models. Model could be stopped earlier if early stopping is met
        num_boost_round_start (int): Number of estimators used in LightGBM
            model after which early stopping criterion starts (could be seen
            as the minimum number of model iterations)
        early_stopping_rounds (int): How many times in a row early stopping can
            be met before stopping training
        early_stopping_step (int): Number of iterations when early stopping is
            performed. 20 means that it's done every 20 iterations, i,e. after
            100, 120, 140, ... iters
        issue_months (list): Months to iterate over. Every month has its own
            model
        years_cv (list): A list with test years for different CV folds
        year_range (bool): Specifies if there could be many years in test data
            (True) or just one test data per fold (False)
        train_feat_dict (dict): Features to use for different months. Key
            indicates month and value the features.
        params_dict (dict): LightGBM hyperparameters to use for different months.
            Key indicates month and value a dictionary of hyperparameters
        categorical (list): Categorical features in the model
        min_max_site_id (pd.DataFrame): Minimum and maximum historical volumes
            for given site_id
        path_distr (str): Path to values of distribution estimate parameters
            per each site_id (without amendments to distributions. Amendments
            are imported from utils)
        distr_perc (float): How much importance is given for distribution
            estimate. For 0.4 value, it's 40% (while LightGBM model is 60%)
        
    Returns:
        best_cv_early_stopping (np.array): CV results from different months with
            number of model iterations for each month
        result_final_avg (float): CV results averaged over different months
        num_rounds_months (list): Number of model iterations for each month
    """
    ###########################################################################
    #PARAMETERS AND VARIABLES INITIALIZATION
    ###########################################################################

    
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
    
    #Iterate first over months, so training from one month is optimized first
    for month_idx, month in tqdm(enumerate(issue_months)):
        num_boost_round_month = num_boost_round_start
        cv_results_avg_fold = []
        
        results_10_clipped_all = []
        results_50_clipped_all = []
        results_90_clipped_all = []
        
        cv_result_avg_fold_prev = np.inf
        num_prev_iters_better = 0
        
        lgb_models_50 = dict()
        lgb_models_10 = dict()
        lgb_models_90 = dict()
        
        print(f'\nmonth {month}')
        #######################################################################
        #TRAINING
        #######################################################################
        while (num_prev_iters_better < early_stopping_rounds) & (num_boost_round_month <= num_boost_round):
            results_10_clipped = []
            results_50_clipped = []
            results_90_clipped = []
            cv_results = []
            
            for fold, year in enumerate(years_cv):
                train_cv_idxs, test_cv_idxs = get_cv_folds(train,
                                                           month,
                                                           years_cv,
                                                           year_range)
                train_feat = train_feat_dict[month]
    
                #Get params from given month
                params = params_dict[month]

                #Add alpha
                params['alpha'] = 0.5

                train_data = lgb.Dataset(data = train.loc[train_cv_idxs[fold], train_feat],
                                         label = labels[train_cv_idxs[fold]],
                                         categorical_feature = categorical)
                test_data = lgb.Dataset(data = train.loc[test_cv_idxs[fold], train_feat],
                                        label = labels[test_cv_idxs[fold]],
                                        reference = train_data)
    
                if num_boost_round_month == num_boost_round_start:
                    lgb_model_50 = lgb.train(params,
                                             train_data,
                                             valid_sets=[train_data, test_data],
                                             num_boost_round = num_boost_round_start,
                                             keep_training_booster = True)
                    lgb_models_50[fold] = lgb_model_50
                else:
                    while lgb_models_50[fold].current_iteration() < num_boost_round_month:
                        lgb_models_50[fold].update()


                preds_50 = lgb_models_50[fold].predict(train.loc[test_cv_idxs[fold], train_feat])

                #Get params from given month
                params = params_dict[month]
                #Add alpha
                params['alpha'] = 0.1

                train_data = lgb.Dataset(data = train.loc[train_cv_idxs[fold], train_feat],
                                         label = labels[train_cv_idxs[fold]],
                                         categorical_feature = categorical)
                test_data = lgb.Dataset(data = train.loc[test_cv_idxs[fold], train_feat],
                                        label = labels[test_cv_idxs[fold]],
                                        reference = train_data)


                if num_boost_round_month == num_boost_round_start:
                    lgb_model_10 = lgb.train(params,
                                             train_data,
                                             valid_sets=[train_data, test_data],
                                             num_boost_round = num_boost_round_start,
                                             keep_training_booster = True
                                            )
                    lgb_models_10[fold] = lgb_model_10
                else:
                    while lgb_models_10[fold].current_iteration() < num_boost_round_month:
                        lgb_models_10[fold].update()


                preds_10_lgbm = lgb_models_10[fold].predict(train.loc[test_cv_idxs[fold], train_feat])

                #Get params from given month
                params = params_dict[month]
                #Add alpha
                params['alpha'] = 0.9

                train_data = lgb.Dataset(data = train.loc[train_cv_idxs[fold], train_feat],
                                         label = labels[train_cv_idxs[fold]],
                                         categorical_feature = categorical)
                test_data = lgb.Dataset(data = train.loc[test_cv_idxs[fold], train_feat],
                                        label = labels[test_cv_idxs[fold]],
                                        reference = train_data)

                if num_boost_round_month == num_boost_round_start:
                    lgb_model_90 = lgb.train(params,
                                             train_data,
                                             valid_sets=[train_data, test_data],
                                             num_boost_round = num_boost_round_start,
                                             keep_training_booster = True)
                    lgb_models_90[fold] = lgb_model_90
                else:
                    while lgb_models_90[fold].current_iteration() < num_boost_round_month:
                        lgb_models_90[fold].update()

                preds_90_lgbm = lgb_models_90[fold].predict(train.loc[test_cv_idxs[fold], train_feat])

                result_df = train.loc[test_cv_idxs[fold], train_feat]
                result_df['volume_50'] = preds_50

                result_df['volume_10_lgbm'] = preds_10_lgbm
                result_df['volume_90_lgbm'] = preds_90_lgbm

                result_df = get_quantiles_from_distr(result_df,
                                                     min_max_site_id,
                                                     all_distr_dict,
                                                     path_distr,
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
                result_df['volume_10'] =\
                    distr_perc * result_df.volume_10_distr + (1 - distr_perc) * result_df.volume_10_lgbm
                result_df['volume_90'] =\
                    distr_perc * result_df.volume_90_distr + (1 - distr_perc) * result_df.volume_90_lgbm


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
                results_10_clipped.append([fold, result_10_clipped, num_boost_round_month])
                results_50_clipped.append([fold, result_50_clipped, num_boost_round_month])
                results_90_clipped.append([fold, result_90_clipped, num_boost_round_month])
    
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

            print(f'Avg result all folds for {num_boost_round_month} trees:', cv_result_avg_fold)
            cv_results_avg_fold.append([cv_result_avg_fold, num_boost_round_month])
            num_boost_round_month += early_stopping_step
        #Take into account if last values weren't the best ones
        if num_prev_iters_better != 0:
            cv_results_avg_fold = cv_results_avg_fold[:-num_prev_iters_better]
        cv_results_all_months[month] = cv_results_avg_fold

        results_10_clipped_all_months[month] = results_10_clipped_all
        results_50_clipped_all_months[month] = results_50_clipped_all
        results_90_clipped_all_months[month] = results_90_clipped_all

    #Get best fit per month
    best_cv_early_stopping = []
    for month in cv_results_all_months.keys():    
        best_cv_early_stopping.append(cv_results_all_months[month][-1])
    best_cv_early_stopping = np.array(best_cv_early_stopping)
    #Average best fits over months
    result_final_avg = np.mean(best_cv_early_stopping[:, 0])
    #Get optimal number of rounds for each month
    num_rounds_months = list(best_cv_early_stopping[:, 1].astype('int'))
    return best_cv_early_stopping, result_final_avg, num_rounds_months
