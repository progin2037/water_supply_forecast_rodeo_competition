import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from pathlib import Path
import joblib
from tqdm import tqdm
from sklearn.metrics import mean_pinball_loss

from utils import get_quantiles_from_distr, all_distr_dict, distr_to_change

def train_pipeline(train_df: pd.DataFrame,
                   labels: pd.Series,
                   month: int,
                   month_idx: int,
                   params: list,
                   train_feat: list,
                   num_rounds_months: list,
                   categorical: list,
                   alpha: float,
                   lgb_models: list,
                   evals: int) -> list:
    """
    Train LightGBM model and append it to a list. It is a single iteration of
    given month and quantile.
    
    Args:
        train_df (pd.DataFrame): Train data
        labels (pd.Series): The response variable
        month (int): Month from a given issue date (starts with 1 for Jan)
        month_idx (int): Index of a given month (indexes start with 0)
        params (list): A list of model parameters for a selected month
        train_feat (list): A list of features to use for a selected month
        num_rounds_months (list): A number of estimators used in LightGBM models
            from different months. Only value from a selected month will be used
        categorical (list): A list of categorical columns
        alpha (float): Informs on which quantile should be calculated. In this
            case, it is a value from 0.1, 0.5 or 0.9
        lgb_models (list): A list of LightGBM models created so far for a given
            quantile
        evals (int): A step of number of LightGBM iterations when training
            progress is shown
    Returns:
        lgb_models (list): A list of LightGBM models created so far, including
            this iteration
    """
    train_data = lgb.Dataset(train_df.loc[train_df.month == month, train_feat],
                             label = labels[train_df[train_df.month == month].index],
                             categorical_feature = categorical)
    #Add alpha. 3 different models will be run for 0.1, 0.5 and 0.9 quantiles
    #in the training script.
    params['alpha'] = alpha

    lgb_model = lgb.train(params,
                          train_data,
                          valid_sets=[train_data],
                          num_boost_round = num_rounds_months[month_idx],
                          callbacks = [lgb.log_evaluation(evals)])
    lgb_models.append(lgb_model)
    return lgb_models


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

def cv(train: pd.DataFrame,
       labels: pd.Series,
       issue_months: list,
       years_cv: list,
       year_range: bool,
       train_feat_dict: dict,
       params_dict: dict,
       categorical: list,
       min_max_site_id: pd.DataFrame,
       path_distr: str,
       distr_perc: float) -> tuple[np.array, float, list]:
    ###########################################################################
    #PARAMETERS AND VARIABLES INITIALIZATION
    ###########################################################################
    NUM_BOOST_ROUND = 1000
    NUM_BOOST_ROUND_START = 100 #Start early stopping from 100 iterations
    EARLY_STOPPING_ROUNDS = 2 #How many times in a row early stopping can be
                              #met before stopping training
    EARLY_STOPPING_STEP = 20 #Number of iterations when Early stopping is
                             #performed. 20 means it's done every 20 iterations,
                             #i,e. after 100, 120, 140, ... iters
    
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
        NUM_BOOST_ROUND_MONTH = NUM_BOOST_ROUND_START
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
        while (num_prev_iters_better < EARLY_STOPPING_ROUNDS) & (NUM_BOOST_ROUND_MONTH <= NUM_BOOST_ROUND):
            results_10_clipped = []
            results_50_clipped = []
            results_90_clipped = []
            cv_results = []
            
            for fold, year in enumerate(years_cv):
                train_cv_idxs, test_cv_idxs = get_cv_folds(month, years_cv, year_range)
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


                if NUM_BOOST_ROUND_MONTH == NUM_BOOST_ROUND_START:
                    lgb_model_10 = lgb.train(params,
                                             train_data,
                                             valid_sets=[train_data, test_data],
                                             num_boost_round = NUM_BOOST_ROUND_START,
                                             keep_training_booster = True
                                            )
                    lgb_models_10[fold] = lgb_model_10
                else:
                    while lgb_models_10[fold].current_iteration() < NUM_BOOST_ROUND_MONTH:
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
                result_df.loc[result_df.volume_90_lgbm < result_df.volume_50, 'volume_90_lgbm'] = result_df.volume_50
                result_df.loc[result_df.volume_50 < result_df.volume_10_lgbm, 'volume_10_lgbm'] = result_df.volume_50
    
                result_df.loc[result_df.volume_90_distr < result_df.volume_50, 'volume_90_distr'] = result_df.volume_50
                result_df.loc[result_df.volume_50 < result_df.volume_10_distr, 'volume_10_distr'] = result_df.volume_50


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

            print(f'Avg result all folds for {NUM_BOOST_ROUND_MONTH} trees:', cv_result_avg_fold)
            cv_results_avg_fold.append([cv_result_avg_fold, NUM_BOOST_ROUND_MONTH])
            NUM_BOOST_ROUND_MONTH += EARLY_STOPPING_STEP
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


def to_categorical(categorical: list,
                   df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep categorical features in a list. LightGBM accepts only 'category'
    features for categorical columns.
    
    Args:
        categorical (list): Columns that should be changed to 'category' type
        df (pd.DataFrame): A DataFrame with columns to change
    Returns:
        df (pd.DataFrame): Df with columns from categorical changed to 'category'
    """
    for cat in categorical:
        df[cat] = df[cat].astype('category')
    return df


def save_models(issue_months: list,
                lgb_models_10: list,
                lgb_models_50: list,
                lgb_models_90: list):
    """
    Save LightGBM models. Models are saved separately for different quantiles
    and months.
    
    Args:
        issue_months (list): Different months to create models for. The months
            are based on issue dates.
        lgb_models_10 (list): LightGBM models for 0.1 quantile. Should be in
            the same order as issue_months
        lgb_models_50 (list): LightGBM models for 0.5 quantile. Should be in
            the same order as issue_months
        lgb_models_90 (list): LightGBM models for 0.9 quantile. Should be in
            the same order as issue_months
    """
    #Create models\ directory if it doesn't exist
    models_path = Path('models')
    models_path.mkdir(parents = True, exist_ok = True)
    
    #Get today's date
    today = pd.to_datetime('today').strftime('%Y_%m_%d')
    path_save_models = os.path.join('models', today)
    #Create a folder in models\ with today's date where results will be saved.
    #If the folder was created already today, FileExistsError will be thrown. It
    #is created this way for safety, so already created models won't be replaced.
    os.mkdir(path_save_models)
    
    #Save models
    for month, lgb_model in zip(issue_months, lgb_models_10):
        joblib.dump(lgb_model,
                    f'{path_save_models}\lgbm_10_{month}_month_{today}.pkl')
        print(f'LightGBM model for month {month} and quantile 0.1 saved')
    for month, lgb_model in zip(issue_months, lgb_models_50):
        joblib.dump(lgb_model,
                    f'{path_save_models}\lgbm_50_{month}_month_{today}.pkl')
        print(f'LightGBM model for month {month} and quantile 0.5 saved')
    for month, lgb_model in zip(issue_months, lgb_models_90):
        joblib.dump(lgb_model,
                    f'{path_save_models}\lgbm_90_{month}_month_{today}.pkl')
        print(f'LightGBM model for month {month} and quantile 0.9 saved')


def load_models(issue_months: list,
                read_models_from_repo: bool=False,
                specific_date: str='today') -> tuple[list, list, list]:
    """
    Load LightGBM models. Models are read separately for different quantiles
    and months.
    
    Args:
        issue_months (list): Different months for models to load. The months
            are based on issue dates
        read_models_from_repo (bool): Condition if read 2023_12_21 models from
            the repo used in the Hindcast Stage of the competition or not.
            Defaults to False
        specific_date (str): a date for models to be loaded. 
            Defaults to 'today'. For other values, use date in the format of
            YYYY_MM_DD
    Returns:
        lgb_models_10 (list): LightGBM models for 0.1 quantile. Should be in
            the same order as issue_months
        lgb_models_50 (list): LightGBM models for 0.5 quantile. Should be in
            the same order as issue_months
        lgb_models_90 (list): LightGBM models for 0.9 quantile. Should be in
            the same order as issue_months
    """
    lgb_models_10 = []
    lgb_models_50 = []
    lgb_models_90 = []

    if read_models_from_repo == True:  
        today = '2023_12_21'
    elif specific_date != 'today':
        today = specific_date
    else:
        today = pd.to_datetime('today').strftime('%Y_%m_%d')

    #Get path
    path_save_models = os.path.join('models', today)
    #Iterate over paths from different months and append models to lists
    for month in issue_months:
        print(f'Load models from month {month}')
        lgb_models_10.append\
            (joblib.load(f'{path_save_models}\lgbm_10_{month}_month_{today}.pkl'))
        lgb_models_50.append\
            (joblib.load(f'{path_save_models}\lgbm_50_{month}_month_{today}.pkl'))
        lgb_models_90.append\
            (joblib.load(f'{path_save_models}\lgbm_90_{month}_month_{today}.pkl'))
    return lgb_models_10, lgb_models_50, lgb_models_90
