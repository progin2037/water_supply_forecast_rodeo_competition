import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_pinball_loss
import optuna
import joblib
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
    if year_range == True:
        #2 years in one fold
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

def train_cv(train: pd.DataFrame,
             labels: pd.Series,
             train_cv_idxs: list,
             test_cv_idxs: list,
             train_feat: list,
             params: dict,
             categorical: list,
             num_boost_round_start: int,
             num_boost_round_month: int,
             alpha: float,
             fold: int,
             lgb_models: dict) -> tuple[np.array, dict]:
    """
    Training pipeline for given fold-quantile combination. Creates model for
    given fold for the first time or continues training previous model until
    threshold is met.
    
    Args:
        train (pd.DataFrame): Whole training data before dividing into CV folds
        labels (pd.Series): Labels corresponding to train data
        train_cv_idxs (list): Indexes of train data for all folds
        test_cv_idxs (list): Indexes of test data for all folds
        train_feat (list): Features to use for this month
        params (dict): LightGBM hyperparameters to use for this month
        categorical (list): Categorical features in the model
        num_boost_round_start (int): Number of estimators used in LightGBM
            model after which early stopping criterion starts (could be seen
            as the minimum number of model iterations)
        num_boost_round_month (int): Maximum number of estimators used in
            LightGBM model for this training iteration. Model is trained until
            num_boost_round_month is reached
        alpha (float): Informs on which quantile should be calculated. In this
            case, it is a value from 0.1, 0.5 or 0.9
        fold (int): Fold from the given iteration. Used to get correct train
            and test indexes and key for LightGBM model for lgb_models
        lgb_models (dict): A dictionary of LightGBM models. One of its keys
            (folds) have to be updated in this iteration
    Returns:
        preds (np.array): Predictions from given quantile and fold
        lgb_models (dict): A dictionary of LightGBM models with updated fold
            values
    """
    #Add alpha to params (specify quantile)
    params['alpha'] = alpha
    #Crate lgb.Datasets for given fold
    train_data = lgb.Dataset(data = train.loc[train_cv_idxs[fold],
                                              train_feat],
                             label = labels[train_cv_idxs[fold]],
                             categorical_feature = categorical)
    test_data = lgb.Dataset(data = train.loc[test_cv_idxs[fold],
                                             train_feat],
                            label = labels[test_cv_idxs[fold]],
                            reference = train_data)
    #Train model
    if num_boost_round_month == num_boost_round_start:
        #Use lgb.train for the first CV iteration (for num_boost_round_start 
        #number of LightGBM boosting iters)
        lgb_model = lgb.train(params,
                              train_data,
                              valid_sets=[train_data, test_data],
                              num_boost_round = num_boost_round_start,
                              keep_training_booster = True)
        #Update dictionary with first fold result
        lgb_models[fold] = lgb_model
    else:
        #For other CV iters, update LightGBM model for early_stopping_step
        #number of itearations
        while lgb_models[fold].current_iteration() < num_boost_round_month:
            lgb_models[fold].update()
    #Get predictions
    preds = lgb_models[fold].predict(train.loc[test_cv_idxs[fold], train_feat])
    return preds, lgb_models

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
            distr_perc_dict: dict) -> tuple[np.array, float, list, np.array]:
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
            indicates month and value the features
        params_dict (dict): LightGBM hyperparameters to use for different months.
            Key indicates month and value, a dictionary of hyperparameters
        categorical (list): Categorical features in the model
        min_max_site_id (pd.DataFrame): Minimum and maximum historical volumes
            for given site_id
        path_distr (str): Path to values of distribution estimate parameters
            per each site_id (without amendments to distributions. Amendments
            are imported from utils)
        distr_perc_dict (dict): How much importance is given for distribution
            estimate. For 0.4 value, it's 40% (while LightGBM model is 60%).
            Different months use different distribution percentage (1 for
            January, 2 for February, ..., 7 for July)
    Returns:
        best_cv_early_stopping (np.array): CV results from different months with
            number of model iterations for each month
        result_final_avg (float): CV results averaged over different months
        num_rounds_months (list): Number of model iterations for each month
        best_interval_early_stopping (np.array): interval coverage results from
            different month for num_rounds_months iterations
    """
    ###########################################################################
    #Global parameters and variables initialization
    ###########################################################################
    #Initialize empty variables
    results_10_clipped = []
    results_50_clipped = []
    results_90_clipped = []
    cv_results_all_months = dict() #CV results from different months
    results_coverage_all_months = dict() #interval coverage from different months
    results_10 = []
    results_50 = []
    results_90 = []

    ###########################################################################
    #Iterate over months. Train one month at a time
    ###########################################################################    
    for month_idx, month in tqdm(enumerate(issue_months)):
        print(f'\nMonth: {month}')
        #Get distribution percentage to use for given month
        distr_perc = distr_perc_dict[month]
        #Initialize variables for the month
        #First evaluation done after num_boost_round_start iters
        num_boost_round_month = num_boost_round_start
        #Set previous value of averaged CV to infinity for first evaluation,
        #so the first evaluated value is always less
        cv_result_avg_fold_prev = np.inf
        #Initialize number of early stopping conditions met so far with 0
        num_prev_iters_better = 0
        #All [avg fold result-number of LightGBM iterations] from given month.
        #All LGBM iters after each early_stopping_step have a row in the list
        cv_results_avg_fold = []
        #Similarly for interval coverage
        results_coverage_avg_fold = []
        #Quantile 0.5 models for newest fold results
        lgb_models_50 = dict()
        #Quantile 0.1 models for newest fold results
        lgb_models_10 = dict()
        #Quantile 0.9 models for newest fold results
        lgb_models_90 = dict()

        #######################################################################
        #Start training. Train until early stopping/maximum number of iters met
        #######################################################################
        while (num_prev_iters_better < early_stopping_rounds) &\
            (num_boost_round_month <= num_boost_round):
            #Initialize variables for given iter
            results_10_clipped = []
            results_50_clipped = []
            results_90_clipped = []
            results_coverage = []
            cv_results = []

            ###################################################################
            #Iterate over different folds
            ###################################################################
            for fold, year in enumerate(years_cv):
                #Get indexes from train DataFrame for given fold's train and test
                train_cv_idxs, test_cv_idxs = get_cv_folds(train,
                                                           month,
                                                           years_cv,
                                                           year_range)
                #Choose features from given month
                train_feat = train_feat_dict[month]
                #Get params from given month
                params = params_dict[month]
                #Train/continue training model from given fold for Q0.5
                preds_50, lgb_models_50 = train_cv(train,
                                                   labels,
                                                   train_cv_idxs,
                                                   test_cv_idxs,
                                                   train_feat,
                                                   params,
                                                   categorical,
                                                   num_boost_round_start,
                                                   num_boost_round_month,
                                                   0.5,
                                                   fold,
                                                   lgb_models_50)
                #Train/continue training model from given fold for Q0.1.
                #Named preds_10_lgbm, as final preds_10 will use a weighted
                #average with distribution estimates
                preds_10_lgbm, lgb_models_10 = train_cv(train,
                                                        labels,
                                                        train_cv_idxs,
                                                        test_cv_idxs,
                                                        train_feat,
                                                        params,
                                                        categorical,
                                                        num_boost_round_start,
                                                        num_boost_round_month,
                                                        0.1,
                                                        fold,
                                                        lgb_models_10)
                #Train/continue training model from given fold for Q0.9.
                #Named preds_90_lgbm, as final preds_10 will use a weighted
                #average with distribution estimates
                preds_90_lgbm, lgb_models_90 = train_cv(train,
                                                        labels,
                                                        train_cv_idxs,
                                                        test_cv_idxs,
                                                        train_feat,
                                                        params,
                                                        categorical,
                                                        num_boost_round_start,
                                                        num_boost_round_month,
                                                        0.9,
                                                        fold,
                                                        lgb_models_90)
                #Get test rows from given fold with predictions
                result_df = train.loc[test_cv_idxs[fold], train_feat]
                result_df['volume_50'] = preds_50
                result_df['volume_10_lgbm'] = preds_10_lgbm
                result_df['volume_90_lgbm'] = preds_90_lgbm
                #Append quantile results
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
                #Change volume values greater than min (max) for site_id to
                #that min (max) value. Do it also for distribution volume.
                #Though distribution estimates shouldn't exceed maximum values,
                #do it just to be certain
                result_df.loc[result_df['volume_50'] < result_df['min'],
                              'volume_50'] = result_df['min']
                result_df.loc[result_df['volume_50'] > result_df['max'],
                              'volume_50'] = result_df['max']
                result_df.loc[result_df['volume_10_lgbm'] < result_df['min'],
                              'volume_10_lgbm'] = result_df['min']
                result_df.loc[result_df['volume_10_distr'] < result_df['min'],
                              'volume_10_distr'] = result_df['min']
                result_df.loc[result_df['volume_90_lgbm'] > result_df['max'],
                              'volume_90_lgbm'] = result_df['max']
                result_df.loc[result_df['volume_90_distr'] > result_df['max'],
                              'volume_90_distr'] = result_df['max']
                #Clipping:
                    #if volume_90 < volume_50 -> change volume_90 to volume_50
                    #if volume_50 < volume_10 -> change volume_10 to volume_50            
                result_df.loc[result_df.volume_90_lgbm < result_df.volume_50,
                              'volume_90_lgbm'] = result_df.volume_50
                result_df.loc[result_df.volume_50 < result_df.volume_10_lgbm,
                              'volume_10_lgbm'] = result_df.volume_50
                result_df.loc[result_df.volume_90_distr < result_df.volume_50,
                              'volume_90_distr'] = result_df.volume_50
                result_df.loc[result_df.volume_50 < result_df.volume_10_distr,
                              'volume_10_distr'] = result_df.volume_50
                #Get weighted average from distributions and models for Q0.1
                #and Q0.9
                result_df['volume_10'] =\
                    distr_perc * result_df.volume_10_distr +\
                        (1 - distr_perc) * result_df.volume_10_lgbm
                result_df['volume_90'] =\
                    distr_perc * result_df.volume_90_distr +\
                        (1 - distr_perc) * result_df.volume_90_lgbm
                #Get quantile loss for given fold
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
                #Get competition metric
                cv_result = 2 * (result_10 + result_50 + result_90) / 3
                #Do the final clipping to make sure that the restrictions are
                #met after taking weighted average for volume_10 and volume_90
                results_clipped = result_df.copy()
                results_clipped.loc[results_clipped.volume_90 < results_clipped.volume_50,
                                    'volume_50'] = results_clipped.volume_90
                results_clipped.loc[results_clipped.volume_50 < results_clipped.volume_10,
                                    'volume_10'] = results_clipped.volume_50
                #Get quantile loss for given fold
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
                #Get competition metric
                cv_result = 2 * (result_10_clipped +
                                 result_50_clipped +
                                 result_90_clipped) / 3
                #Append the result from given fold-model iteration
                cv_results.append([cv_result])
                #Get interval coverage for given month
                different_volumes = ['volume_10', 'volume_50', 'volume_90']
                result_coverage =\
                    interval_coverage(np.array(labels[test_cv_idxs[fold]]),
                                      np.array(results_clipped[different_volumes]))
                results_coverage.append(result_coverage)
            #Average results over different folds for given model iteration
            cv_result_avg_fold = np.mean(cv_results)
            #Do the same for interval coverage
            result_coverage_avg_fold = np.mean(results_coverage)
            #Keep track of early stopping condition if result is poorer than
            #in the previous early stopping check (early_stopping_step before)
            if cv_result_avg_fold > cv_result_avg_fold_prev:
                num_prev_iters_better += 1
            else:
                #If new result is better, use new result in next early stopping
                #check. Reset num_prev_iters_better to 0
                cv_result_avg_fold_prev = cv_result_avg_fold
                num_prev_iters_better = 0
            print(f'Avg result all folds for {num_boost_round_month} trees:',
                  cv_result_avg_fold)
            print(f'Avg interval coverage all folds for {num_boost_round_month} trees:',
                  result_coverage_avg_fold)
            #Append number of boosting iterations to average results
            cv_results_avg_fold.append([cv_result_avg_fold,
                                        num_boost_round_month])
            #Do the same for interval coverage
            results_coverage_avg_fold.append([result_coverage_avg_fold,
                                              num_boost_round_month])
            #Update information when next early stopping will be evaluated
            num_boost_round_month += early_stopping_step

        #######################################################################
        #Early stopping/maximum number of iterations (num_boost_round) met
        #for the selected month
        #######################################################################
        #Update results if last values weren't the best ones. Maximum value
        #is chosen as the final value
        if num_prev_iters_better != 0:
            cv_results_avg_fold = cv_results_avg_fold[:-num_prev_iters_better]
            results_coverage_avg_fold =\
                results_coverage_avg_fold[:-num_prev_iters_better]
        cv_results_all_months[month] = cv_results_avg_fold
        results_coverage_all_months[month] = results_coverage_avg_fold
    ###########################################################################
    #All months were trained. Get final results to return
    ###########################################################################
    #Get best fit per month with best number of iterations
    best_cv_early_stopping = []
    for month in cv_results_all_months.keys():    
        best_cv_early_stopping.append(cv_results_all_months[month][-1])
    best_cv_early_stopping = np.array(best_cv_early_stopping)
    #Average best fits over months
    result_final_avg = np.mean(best_cv_early_stopping[:, 0])
    #Get optimal number of rounds for each month separately
    num_rounds_months = list(best_cv_early_stopping[:, 1].astype('int'))
    #Get interval coverage from the best iteration
    best_interval_early_stopping = []
    for month in cv_results_all_months.keys():
        best_interval_early_stopping.append(results_coverage_all_months[month][-1])    
    best_interval_early_stopping = np.array(best_interval_early_stopping)[:, 0]
    return best_cv_early_stopping, result_final_avg, num_rounds_months,\
        best_interval_early_stopping

def objective(trial: optuna.trial.Trial,
              train: pd.DataFrame,
              labels: pd.Series,
              month: int,
              years_cv: list,
              year_range: bool,
              train_feat: list,
              categorical: list,
              min_max_site_id: pd.DataFrame,
              path_distr: str,
              distr_perc_dict: dict,
              num_boost_round: int,
              num_boost_round_start: int,
              early_stopping_rounds: int,
              early_stopping_step: int) -> float:
    """
    Set logic for optuna hyperparameters tuning, set range of values for
    different hyperparameters, append CV evaluation.
    
    Args:
        trial (optuna.trial.Trial): A process of evaluating an objective
            function. This object is passed to an objective function and
            provides interfaces to get parameter suggestion, manage the trial’s
            state, and set/get user-defined attributes of the trial
            (https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial)
        train (pd.DataFrame): Whole training data before dividing into CV folds
        labels (pd.Series): Labels corresponding to train data
        month (int): Month to optimize hyperparameters for
        years_cv (list): A list with test years for different CV folds
        year_range (bool): Specifies if there could be many years in test data
            (True) or just one test data per fold (False)
        train_feat (list): Features to use for this month
        categorical (list): Categorical features in the model
        min_max_site_id (pd.DataFrame): Minimum and maximum historical volumes
            for given site_id
        path_distr (str): Path to values of distribution estimate parameters
            per each site_id (without amendments to distributions. Amendments
            are imported from utils)
        distr_perc_dict (dict): How much importance is given for distribution
            estimate. For 0.4 value, it's 40% (while LightGBM model is 60%).
            Different months use different distribution percentage (1 for
            January, 2 for February, ..., 7 for July)
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
    Returns:
        best_cv_avg (float): Score for this optimization iteration
    """
    #Set repetitive parameters created in model_params.py
    BAGGING_FREQ, OBJECTIVE, METRIC, VERBOSE, REG_ALPHA, MIN_GAIN_TO_SPLIT,\
        MIN_SUM_HESSIAN_IN_LEAF, FEATURE_FRACTION_SEED, SEED =\
            joblib.load('data\general_hyperparams_forecast.pkl')
    #Set minimial number of columns to one less than the number of columns.
    #0.001 is added to the result to deal with optuna approximation.
    #Months 2, 3 and 4 were trained with a static 0.9 feature fraction
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
                                                      feature_fraction_min,
                                                      1.0,
                                                      step = 1 - feature_fraction_min),
              'feature_fraction_seed': FEATURE_FRACTION_SEED,
              'max_bin': trial.suggest_int('max_bin', 200, 300),
              'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 15, 25),
              'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
              'verbose': VERBOSE,
              'seed': SEED}
    #Change params to params_dict dictionary
    params_dict = {month: params}
    #Change features to dictionary
    train_feat_dict = {month: train_feat}
    #Change month type to list
    month = [month]
    #Perform CV calculation
    best_cv_per_month, best_cv_avg, num_rounds_months,\
        best_interval_early_stopping = lgbm_cv(train,
                                               labels,
                                               num_boost_round,
                                               num_boost_round_start,
                                               early_stopping_rounds,
                                               early_stopping_step,
                                               month,
                                               years_cv,
                                               year_range,
                                               train_feat_dict,
                                               params_dict,
                                               categorical,
                                               min_max_site_id,
                                               path_distr,
                                               distr_perc_dict)
    trial.set_user_attr("num_boost_rounds_best", num_rounds_months[0])
    trial.set_user_attr("interval_coverage", best_interval_early_stopping[0])
    return best_cv_avg

def interval_coverage(actual: np.ndarray,
                      predicted: np.ndarray) -> float:
    """
    Calculates interval coverage for quantile predictions. Assumes at least two
    columns in `predicted`, and that the first column is the lower bound of
    the interval, and the last column is the upper bound of the interval.
    Taken from https://github.com/drivendataorg/water-supply-forecast-rodeo-runtime/blob/main/scoring/score.py.

    Args:
        actual (np.ndarray): Array of actual values (labels)
        predicted (np.ndarray): Array of predicted values
    Returns:
        interval_result (float): Interval coverage (proportion of predictions
            that fall within lower and upper bound)
    """
    # Use ravel to reshape to 1D arrays.
    lower = predicted[:, 0].ravel()
    upper = predicted[:, -1].ravel()
    actual = actual.ravel()
    interval_result = np.average((lower <= actual) & (actual <= upper))
    return interval_result
