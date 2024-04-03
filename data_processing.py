import pandas as pd
import numpy as np
import pickle
import joblib
import os

import time
start = time.time()

from utils import ReadAllData, years_cv, get_outliers, create_cds_dataframe
#ReadAllData should take about 30-45 minutes to execute if run for the first time

from feature_engineering import get_aggs_month_day, get_aggs_month,\
preprocess_monthly_naturalized_flow, preprocess_snotel, get_prev_monthly,\
get_prev_daily, nat_flow_sum_cumul_since_apr, get_prev_cds_data,\
get_prev_cds_forecasts_data


#Set threshold for outliers removal. A good practise is to set z-score threshold
#to 3 but based on data exploration, it's save to remove more outliers within
#2.5 threshold
OUT_THRES = 2.5
#Keep data only since given interval
YEAR_SINCE = 1965


###############################################################################
#Read and preprocess data
###############################################################################
#Read all data from utils
dfs = ReadAllData()
#Get unique site_ids
site_ids_unique = dfs.site_ids_unique.copy()
#Get train df from dfs list
train = dfs.train.copy()
#Remove missing volume
train = train[train.volume.notna()].reset_index(drop = True)

#Get and save min and max values for given site_id. It's done after outliers
#removal to exclude too small/big values from site_id's volume range. It's used
#for all years as there isn't much data, so even potentially inacurate old
#volumes before YEAR_SINCE are used. Min/max values are created separetely
#without each LOOCV year
min_max_site_id_dict = dict()
#Set threshold for removing outliers in min/max calculations
OUT_THRES = 2.5
for year_cv in years_cv:
    #Remove data from given CV fold and store it in a different DataFrame
    #Don't reset index, so index is kept
    train_without_cv = train[train.year != year_cv]
    #Remove outliers. Remove them based on train_without_cv and append those
    #results to train
    zscores_outliers = get_outliers(train_without_cv, OUT_THRES)
    train_without_cv = train_without_cv.\
        loc[~train_without_cv.index.isin(zscores_outliers.index)].reset_index(drop = True)
    #Get volumes after removing outliers. They could be used later to get full data
    #after removing data from specific years
    volume_full_data = train_without_cv[['site_id', 'year', 'volume']]
    #Get min and max values without given LOOCV year
    min_max_site_id =\
        volume_full_data.groupby(['site_id'])['volume'].agg(['min', 'max'])
    #Append volume_full_data with this year_cv outliers removal
    min_max_site_id_dict[year_cv] = min_max_site_id    
#Save min/max volumes to .pkl
joblib.dump(min_max_site_id_dict, 'data\min_max_site_id_dict_final.pkl')

#Sort by site_id and year
train = train.sort_values(['site_id', 'year']).reset_index(drop = True)

#Create issue dates for each row. Predictions in this challenge are made on
#28 different weekly issue dates per year and site_id, starting with 1 Jan,
#8 Jan, 15 Jan, ... , 15 Jul, 22 Jul (detroit_lake_inflow is an exception and
#ends on 22 Jun, it will be addressed later).

#Read submission_format to get available issue_dates
submission_format = dfs.submission_format.copy()
#Get unique combinations of month_day, month and day from submission_format.
#Month and day variable will be helpful later to faciliate processing.
issue_dates = submission_format['issue_date'].str[5:].unique()
#issue_months has the same rows as issue_days. issue_months isn't a unique
#list of months, it's just month from issue_dates' rows
issue_months = pd.Series(issue_dates).str.split('-').str[0].astype('int')
issue_days = pd.Series(issue_dates).str.split('-').str[1].astype('int')
issue_days_unique = issue_days.unique()
#Get length of train to facilitate rows creation with issue_date
train_len_old = len(train)
#Copy rows in train len(issue_dates) times to get number of rows ready for
#issue dates filling
train = pd.concat([train]*len(issue_dates), ignore_index = True)
#Sort values
train = train.sort_values(['site_id', 'year']).reset_index(drop = True)
#Fill issue_dates
train['issue_date'] = pd.concat([pd.Series(issue_dates)] * train_len_old).\
    reset_index(drop = True)
train['issue_date'] = train.year.astype('str') + '-' + train.issue_date

###############################################################################
#Feature engineering
###############################################################################
#Append 'longitude' column from metadata
cols_meta_to_keep = ['site_id', 'longitude']
train = pd.merge(train,
                 dfs.meta[cols_meta_to_keep],
                 on = 'site_id',
                 how = 'left')

#Get month and day
train['month'] = train.issue_date.str.split('-').str[1]
train['day'] = train.issue_date.str.split('-').str[2]
#change year, month and day to int
train['month'] = train.month.astype('int')
train['day'] = train.day.astype('int')
train['year'] = train.year.astype('int')

#Get USGS streamflow features
streamflow = dfs.streamflow.reset_index(drop = True)

#Add, 'month' and 'day' columns to streamflow.
#Year is already represented as water year (https://water.usgs.gov/nwc/explain_data.html),
#so there's no need to make changes to year
streamflow['datetime'] = streamflow.datetime.dt.strftime('%Y-%m-%d')
streamflow_dates_split = streamflow.datetime.str.split('-')
streamflow['month'] = streamflow_dates_split.str[1].astype('int')
streamflow['day'] = streamflow_dates_split.str[2].astype('int')

#discharge_cfs_mean_std
train = get_aggs_month_day(streamflow,
                           train,
                           ['discharge_cfs_mean'],
                           ['std'],
                           issue_months,
                           issue_days,
                           'year',
                           suffix = '',
                           month_since = 12)
#discharge_cfs_mean_since_Oct_std
train = get_aggs_month_day(streamflow,
                           train,
                           ['discharge_cfs_mean'],
                           ['std'],
                           issue_months,
                           issue_days,
                           'year',
                           suffix = '_since_Oct',
                           month_since = 10)
#discharge_cfs_mean_Apr_mean
train = get_aggs_month_day(streamflow,
                           train,
                           ['discharge_cfs_mean'],
                           ['mean'],
                           issue_months,
                           issue_days,
                           'year',
                           suffix = '_Apr',
                           month_since = 4)

#Get monthly naturalized flow features
train_monthly_naturalized_flow = preprocess_monthly_naturalized_flow(dfs.train_monthly_naturalized_flow)

#Get naturalized flow from previous month
train = get_prev_monthly(df_aggs = train_monthly_naturalized_flow,
                         df_main = train,
                         cols = ['volume'],
                         new_col_names = ['nat_flow_prev'],
                         date_col = 'issue_date',
                         site_id_col = 'site_id',
                         month_offset = True,
                         day_start = 1)

#Rename columns to make the processing easier
train_monthly_naturalized_flow.rename({'volume': 'nat_flow'}, axis = 1, inplace = True)
#Get average naturalized flow since April, before issue date (nat_flow_Apr_mean)
train = get_aggs_month(train_monthly_naturalized_flow,
                       train,
                       ['nat_flow'],
                       ['mean'],
                       issue_months,
                       'forecast_year',
                       suffix = '_Apr',
                       month_since = 4)

#Get cumulative naturalized flows since April. It will be used in postprocessing.
#Get naturalized flow sum from April
train = nat_flow_sum_cumul_since_apr(
    train_monthly_naturalized_flow = train_monthly_naturalized_flow,
    df_main = train,
    new_col_name = 'nat_flow_sum_Apr_Apr',
    month_end = 4)
#Get naturalized flow sum between April and May
train = nat_flow_sum_cumul_since_apr(
    train_monthly_naturalized_flow = train_monthly_naturalized_flow,
    df_main = train,
    new_col_name = 'nat_flow_sum_Apr_May',
    month_end = 5)
#Get naturalized flow sum between April and June
train = nat_flow_sum_cumul_since_apr(
    train_monthly_naturalized_flow = train_monthly_naturalized_flow,
    df_main = train,
    new_col_name = 'nat_flow_sum_Apr_Jun',
    month_end = 6)

#Create nat_flow_10 and nat_flow_11 for nat_flow_11_to_10_ratio creation
cols = ['site_id', 'forecast_year', 'nat_flow']
agg_col = 'nat_flow'

for month in [10, 11]:
    month_df = train_monthly_naturalized_flow.loc[train_monthly_naturalized_flow.month == month, cols].\
        reset_index(drop = True)
    month_df = month_df.rename({agg_col: f'{agg_col}_{month}'}, axis = 1)
    train = pd.merge(train,
                     month_df,
                     how = 'left',
                     left_on = ['site_id', 'year'],
                     right_on = ['site_id', 'forecast_year'])
    train.drop('forecast_year', axis = 1, inplace = True)
    #Fill data with NaNs where issue month the same or less than this iter
    #issue date starts with January, so no need to deal with Oct-Dec
    train.loc[(train.month <= month) &
              (month < 10), #values regarding Oct, Nov, Dec shouldn't be concerned
                            #as the earliest issue date is Jan 1
              f'{agg_col}_{month}'] = np.nan

#Snotel features
snotel = preprocess_snotel(dfs.snotel,
                           dfs.sites_to_snotel_stations)

#Get previous value of WTEQ_DAILY
train = pd.merge(train,
                 snotel[['WTEQ_DAILY', 'site_id', 'issue_date']],
                 how = 'left',
                 on = ['site_id', 'issue_date'])
train.rename({'WTEQ_DAILY': 'WTEQ_DAILY_prev'}, axis = 1, inplace = True)

#WTEQ_DAILY_Apr_mean
train = get_aggs_month_day(snotel,
                           train,
                           ['WTEQ_DAILY'],
                           ['mean'],
                           issue_months,
                           issue_days,
                           'year_forecast',
                           suffix = '_Apr',
                           month_since = 4)

#Get previous value of PREC_DAILY
train = pd.merge(train,
                 snotel[['PREC_DAILY', 'site_id', 'issue_date']],
                 how = 'left',
                 on = ['site_id', 'issue_date'])
train.rename({'PREC_DAILY': 'PREC_DAILY_prev'}, axis = 1, inplace = True)

#PREC_DAILY_Apr_mean
train = get_aggs_month_day(snotel,
                           train,
                           ['PREC_DAILY'],
                           ['mean'],
                           issue_months,
                           issue_days,
                           'year_forecast',
                           suffix = '_Apr',
                           month_since = 4)

#Get PREC_DAILY_Apr_prev_diff. Will have to put it into function

#Necessary snotel processing again
snotel = dfs.snotel.copy()

#Add site_id to snotel
sites_to_snotel_stations = dfs.sites_to_snotel_stations.copy()
sites_to_snotel_stations['stationTriplet'] = sites_to_snotel_stations.stationTriplet.str.replace(':', '_')
snotel = pd.merge(snotel,
                  sites_to_snotel_stations,
                  how = 'left',
                  left_on = 'STATION',
                  right_on = 'stationTriplet')
#Get rid of redundant features
snotel.drop(['STATION', 'in_basin', 'stationTriplet'], axis = 1, inplace = True)
#Get average values for site_id and date (exclude STATIONS)
snotel = snotel.groupby(['date', 'site_id']).mean().reset_index()

#Add year, month, day
snotel_dates_split = snotel.date.str.split('-')
snotel['year'] = snotel_dates_split.str[0].astype('int')
snotel['month'] = snotel_dates_split.str[1].astype('int')
snotel['day'] = snotel_dates_split.str[2].astype('int')

#Takes about ~5 seconds per day-month iteration due to handling NaNs in quantiles
#Set October, November, December to next year for year_forecast
snotel['year_forecast'] = snotel.year
snotel.loc[snotel['month'].astype(int).between(10, 12), 'year_forecast'] = snotel.year_forecast + 1
#Get data from Mar 30 (available 1 day before Apr)
snotel_Mar_31 = snotel[(snotel.month == 3) & (snotel.day == 30)]
#Keep only columns to merge
snotel_Mar_31 = snotel_Mar_31[['site_id', 'PREC_DAILY', 'year']]
snotel_Mar_31 = snotel_Mar_31[snotel_Mar_31.PREC_DAILY.notna()]

snotel_Mar_31.columns = ['site_id', 'PREC_DAILY_Mar_31', 'year']

#Merge with train
train = pd.merge(train,
         snotel_Mar_31,
         how = 'left',
         on = ['site_id', 'year'])

#Fill with NaNs if months before April
train.loc[~train.month.isin([4, 5, 6, 7]), 'PREC_DAILY_Mar_31'] = np.nan

#Get difference between last day PREC and Mar 31
train['PREC_DAILY_Apr_prev_diff'] = train.PREC_DAILY_prev - train.PREC_DAILY_Mar_31


#Get WTEQ_DAILY_Jul_prev_diff
#Get data from Jun 29 (available one day before Jul)
snotel_Jun_30 = snotel[(snotel.month == 6) & (snotel.day == 29)]
#Keep only columns to merge
snotel_Jun_30 = snotel_Jun_30[['site_id', 'WTEQ_DAILY', 'year']]
snotel_Jun_30.columns = ['site_id', 'WTEQ_DAILY_Jun_30', 'year']
#Merge with train
train = pd.merge(train,
         snotel_Jun_30,
         how = 'left',
         on = ['site_id', 'year'])
#Fill with NaNs if months before Jul
train.loc[train.month != 7, 'WTEQ_DAILY_Jun_30'] = np.nan

#Get difference between last day and Jun 29
train['WTEQ_DAILY_Jul_prev_diff'] = train.WTEQ_DAILY_prev - train.WTEQ_DAILY_Jun_30

#Get WTEQ_DAILY_Jun_prev_diff
#Get data from May 30 (available 1 day before Jun)
snotel_May_31 = snotel[(snotel.month == 5) & (snotel.day == 30)]
#Keep only columns to merge
snotel_May_31 = snotel_May_31[['site_id', 'WTEQ_DAILY', 'year']]
snotel_May_31.columns = ['site_id', 'WTEQ_DAILY_May_31', 'year']
#Merge with train
train = pd.merge(train,
         snotel_May_31,
         how = 'left',
         on = ['site_id', 'year'])
#Fill with NaNs if months isn't Jun
train.loc[train.month != 6, 'WTEQ_DAILY_May_31'] = np.nan

#Get difference between last day and May 30
train['WTEQ_DAILY_Jun_prev_diff'] = train.WTEQ_DAILY_prev - train.WTEQ_DAILY_May_31


#Append PDSI_prev (latest available average PDSI value)
pdsi = dfs.pdsi.copy()
train = get_prev_daily(df_aggs = pdsi,
                       df_main = train,
                       cols = ['pdsi_mean'],
                       new_col_names = ['pdsi_prev'],
                       date_col = 'pdsi_date',
                       site_id_col = 'site_id',
                       issue_days_unique = issue_days_unique,
                       days_lag = 5)
#Append pdsi_prev_30_days (30 days before prev)
pdsi = dfs.pdsi.copy()
train = get_prev_daily(df_aggs = pdsi,
                       df_main = train,
                       cols = ['pdsi_mean'],
                       new_col_names = ['pdsi_prev_30_days'],
                       date_col = 'pdsi_date',
                       site_id_col = 'site_id',
                       issue_days_unique = issue_days_unique,
                       days_lag = 5+30)

#Create pdsi_prev_to_last_month_diff to get a difference between pdsi_prev
#and pdsi_prev_30_days
train['pdsi_prev_to_last_month_diff'] = train.pdsi_prev - train.pdsi_prev_30_days

#CDS data

#Get sd_prev
#Get CDS file name to add to train
cds_file = 'cds_monthly_snow'
#If the script is run for the first time, create a .pkl file for a given CDS
#data, as the file wasn't yet created from .nc
if os.path.isfile(f'data/cds/{cds_file}.pkl') == False:
    create_cds_dataframe(f'data/cds/{cds_file}.nc',
                         dfs.geospatial,
                         site_ids_unique,
                         cds_file,
                         False)
#Get sd_prev latest data before issue_date
train = get_prev_cds_data(f'data/cds/{cds_file}.pkl',
                           train,
                           5,
                           ['sd'],
                           ['sd_prev'])

#CDS Copernicus forecasts
#Set a dictionary with file data names and months on which predictions were
#issued
cds_forecasts_files = {'seasonal_dec': 12,
                       'seasonal_jan': 1,
                       'seasonal_feb': 2,
                       'seasonal_mar': 3,
                       'seasonal_apr': 4}
#Set columns with forecasts to append
cols_forecasts = ['sd']
#Set column suffixes, all created columns will have it appended
suffix = '_forecasts'
#Iterate over dict elements and assign forecast averages to train
for cds_file, issue_month in cds_forecasts_files.items():    
    if os.path.isfile(f'data/cds/{cds_file}.pkl') == False:
        create_cds_dataframe(f'data/cds/{cds_file}.nc',
                             dfs.geospatial,
                             site_ids_unique,
                             cds_file,
                             True)
    train = get_prev_cds_forecasts_data(path = f'data/cds/{cds_file}.pkl',
                                        df_main = train,
                                        issue_month = issue_month,
                                        issue_day = 6,
                                        cols = cols_forecasts,
                                        remove_end_jun = True,
                                        suffix = suffix)
#Create one column with values from different issue months. There shouldn't be
#any overlap in values, issues from January should be appended to different
#rows than issues from February
for col in cols_forecasts:
    cds_months = cds_forecasts_files.values()
    train[f'{col}{suffix}'] = np.nan
    for cds_month in cds_months:
        train.loc[train[f'{col}{suffix}_{cds_month}'].notna(),
                  f'{col}{suffix}'] = train[f'{col}{suffix}_{cds_month}']


#Get the same but with end of June/July data. For some months, June forecasts
#boost the predictive power of the model. It's needed to be run again, as for
#1st day of the issue date, previous issue month forecasts are used

#Set columns with forecasts to append
cols_forecasts = ['sd']
#Set column suffixes, all created columns will have it appended
suffix = '_forecasts_with_jun'
#Iterate over dict elements and assign forecast averages to train
for cds_file, issue_month in cds_forecasts_files.items():    
    if os.path.isfile(f'data/cds/{cds_file}.pkl') == False:
        create_cds_dataframe(f'data/cds/{cds_file}.nc',
                             dfs.geospatial,
                             site_ids_unique,
                             cds_file,
                             True)
    train = get_prev_cds_forecasts_data(path = f'data/cds/{cds_file}.pkl',
                                        df_main = train,
                                        issue_month = issue_month,
                                        issue_day = 6,
                                        cols = cols_forecasts,
                                        remove_end_jun = False,
                                        suffix = suffix)
#Create one column with values from different issue months. There shouldn't be
#any overlap in values, issues from January should be appended to different
#rows than issues from February
for col in cols_forecasts:
    cds_months = cds_forecasts_files.values()
    train[f'{col}{suffix}'] = np.nan
    for cds_month in cds_months:
        train.loc[train[f'{col}{suffix}_{cds_month}'].notna(),
                  f'{col}{suffix}'] = train[f'{col}{suffix}_{cds_month}']

###############################################################################
#Final operations on data
###############################################################################
#Remove July issue_date for detroit_lake_inflow. This site's volume is
#calculated for March-June period
train = train[~((train.site_id == 'detroit_lake_inflow') & (train.month == 7))].\
    reset_index(drop = True)

#Keep only data since given interval
train = train[train.year >= YEAR_SINCE].reset_index(drop = True)

#Create issue_date_no_year variable - issue_date encoded into integers
train['issue_date_no_year'] = train.issue_date.str[5:]
issue_date_encoded = dict(zip(issue_dates, range(0, 28)))
train['issue_date_no_year'] = train.issue_date_no_year.map(issue_date_encoded)

###############################################################################
#Save outputs
###############################################################################
#Save issue date encoding to be used in test pipeline
with open("data\issue_date_encoded", "wb") as fp:
    pickle.dump(issue_date_encoded, fp)

#Save train and test data ready for modelling
train.to_pickle('data/train_test_final.pkl')

end = time.time()
elapsed = end - start
print(elapsed)
