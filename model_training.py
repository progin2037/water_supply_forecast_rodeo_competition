import pandas as pd
import joblib
import pickle
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm

from train_utils import to_categorical, train_pipeline, save_models
from cv_and_hyperparams_opt import get_years_cv, lgbm_cv, objective

import time

start = time.time()

#Set types of training to run
RUN_CV = True
RUN_HYPERPARAMS_TUNING = False
RUN_TRAINING = True

#Read data
train = pd.read_pickle('data/train_test_final.pkl')
params_dict = joblib.load('data\lgbm_model_params_final.pkl')
feat_dict = joblib.load('data\lgbm_model_feats_final.pkl')

#Different models are created for different months
issue_months = [1, 2, 3, 4, 5, 6, 7]
#Number of boost rounds for following months. Taken from hyparparameters
#optimization. IF CV is ran, it is overwritten with CV results
num_rounds_months = [140, 240, 400, 340, 800, 740, 1000]
#Set eval to show training results every EVAL step
EVAL = 100

#Initiate and read data for CV/hyperparams tuning
if RUN_CV == True or RUN_HYPERPARAMS_TUNING == True:
    #Read min-max values of volume per site_id
    min_max_site_id = joblib.load('data\min_max_site_id_dict_final.pkl')
    #Path to different distributions for different site_ids. Distributions
    #for different LOOCV years are in different files with year suffix
    PATH_DISTR = 'data\distr\distr_final_'
    #Set importance for distribution in quantiles 0.1 and 0.9 calculation for
    #different months
    distr_perc_dict = {1: 0.6,
                       2: 0.45,
                       3: 0.4,
                       4: 0.3,
                       5: 0.25,
                       6: 0.2,
                       7: 0.1}
    #Use many years in a fold or not
    YEAR_RANGE = False
    #Initiate early stopping from this iteration
    NUM_BOOST_ROUND_START = 100
    #How many times in a row early stopping can be met before stopping training
    EARLY_STOPPING_ROUNDS = 3
    #Number of iterations when Early stopping is performed. 20 means that it's
    #done every 20 iterations, i,e. after 100, 120, 140, ... iters
    EARLY_STOPPING_STEP = 10
    #Get years for different CV folds
    years_cv = get_years_cv(YEAR_RANGE)

#Initiate lists for storing models for given quantile for different months
lgb_models_10 = []
lgb_models_50 = []
lgb_models_90 = []
#Change types to 'category' in categorical columns
categorical = ['site_id']
train = to_categorical(['site_id'],
                      train)
#Get labels
labels = train['volume']

#CV
if RUN_CV == True:
    #Maximum number of LightGBM model iterations
    NUM_BOOST_ROUND = 2000
    #Run CV
    best_cv_per_month, best_cv_avg_rms, best_cv_avg, num_rounds_months,\
        interval_coverage_all_months, best_interval_early_stopping, preds =\
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
                    distr_perc_dict)
    print('CV result avg over months:', best_cv_avg)
    print('CV results per month with number of iters:', best_cv_per_month)
    print('Number of iters per month:', num_rounds_months)
    print('Interval coverage per month:', interval_coverage_all_months)
    print('Average interval coverage:', best_interval_early_stopping)

    ###########################################################################
    #Submission pipeline
    ###########################################################################
    #Get cross-validation LB format for predictions
    submission = pd.read_csv('data\cross_validation_submission_format.csv')
    #Get issue date encoding
    with open("data\issue_date_encoded", "rb") as fp:
        issue_date_encoded = pickle.load(fp)
    #Get month_day and year
    submission['month_day'] = submission.issue_date.str[5:]
    submission['year'] = submission.issue_date.str[:4]
    submission['year'] = submission.year.astype('int')
    #Encode issue dates to get issue_date_no_year from training
    submission['issue_date_no_year'] = submission.month_day.map(issue_date_encoded)
    #Drop volume columns that will be replaced with predictions
    submission.drop(['volume_10', 'volume_50', 'volume_90'],
                    axis = 1, inplace = True)
    #Merge submission format with predictions
    submission = pd.merge(
        submission[['site_id', 'issue_date', 'issue_date_no_year', 'year']],
        preds[['site_id', 'issue_date_no_year', 'volume_10', 'volume_50', 'volume_90', 'year']],
        how = 'left',
        on = ['site_id', 'issue_date_no_year', 'year'])
    #Keep only columns for submission
    submission.drop(['issue_date_no_year', 'year'], axis = 1, inplace = True)
    #Save submission
    submission.to_csv('submission.csv', index = False)

#Hyperparameters tuning
if RUN_HYPERPARAMS_TUNING == True:
    #Set if that's a final tuning. There are 2 types, initial one, where wide
    #range of hyperparams is used to determine area where results are optimal
    #and final one, where based on the results of the initial tuning,
    #hyperparams space is narrowed
    FINAL_TUNING = False
    if FINAL_TUNING == True:
        path_suff = 'final'
    else:
        path_suff = 'initial'
    #Maximum number of LightGBM model iterations
    NUM_BOOST_ROUND = 2000
    #All months should take 15-40 hours to run. It is recommended to optimize
    #one year at a time
    for issue_month in tqdm(issue_months):
        #Choose features from given month
        if issue_month in [1, 2]:
            #Use a smaller number of iterations for final tuning
            if FINAL_TUNING == True:
                #Use a smaller number of iterations for final tuning. More
                #iterations than in later months as it is quicker to execute
                N_TRIALS = 80
            else:
                #Use a larger number of iterations to determine best hyperparams space
                N_TRIALS = 150
        elif issue_month in [3, 4]:
            #Use a smaller number of iterations for final tuning
            if FINAL_TUNING == True:
                #Use 20 iters less for initial and 10 less for final, as it
                #takes more time to run
                N_TRIALS = 70
            else:
                #Use a larger number of iterations to determine best hyperparams space
                N_TRIALS = 130
        else:
            #Use a smaller number of iterations for final tuning. 5, 6 and 7
            #months have strong prediction power, so it isn't that important
            #to train them for long. They take also more time for 1 iteration
            #to run
            if FINAL_TUNING == True:
                #Use a smaller number of iterations for final tuning
                N_TRIALS = 60
            else:
                #Use a larger number of iterations to determine best hyperparams space
                N_TRIALS = 100
        train_feat = feat_dict[issue_month]
        sampler = TPESampler(seed = 22) #to get the same results all the time
        #Perform hyperparameters tuning. Ranges of values to select from are
        #already chosen in the function
        study = optuna.create_study(direction = 'minimize', sampler = sampler)
        study.optimize(lambda trial: objective(trial,
                                               train,
                                               labels,
                                               issue_month,
                                               years_cv,
                                               YEAR_RANGE,
                                               train_feat,
                                               categorical,
                                               min_max_site_id,
                                               PATH_DISTR,
                                               distr_perc_dict,
                                               NUM_BOOST_ROUND,
                                               NUM_BOOST_ROUND_START,
                                               EARLY_STOPPING_ROUNDS,
                                               EARLY_STOPPING_STEP,
                                               FINAL_TUNING),
                       n_trials = N_TRIALS)
        #Save study from given month
        joblib.dump(study,
                    f"results/hyperparams_tuning/study_{path_suff}_{pd.to_datetime('today').strftime('%Y_%m_%d_%H_%M_%S')}_month_{issue_month}.pkl")

#Train models for each month and keep them in different lists according to
#quantiles
if RUN_TRAINING == True:
    for month_idx, month in enumerate(issue_months):
        print(f'Month: {month}')
        #Choose features from given month
        train_feat = feat_dict[month]
        #Choose params from given month
        params = params_dict[month]
        #Train models for different quantiles
        print('0.1 quantile')
        lgb_models_10 = train_pipeline(train,
                                       labels,
                                       month,
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
