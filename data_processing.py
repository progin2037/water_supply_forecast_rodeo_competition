import pandas as pd
import numpy as np
import pickle

from utils import ReadAllData, get_outliers

from feature_engineering import get_aggs_month_day, get_aggs_month,\
preprocess_monthly_naturalized_flow, preprocess_snotel

import time

start = time.time()

#Set threshold for outliers removal. A good practise is to set z-score threshold
#to 3 but based on data exploration, it's save to remove more outliers within
#2.5 threshold
OUT_THRES = 2.5
#Keep data only since given interval
YEAR_SINCE = 1965


###############################################################################
#READ AND PREPROCESS DATA
###############################################################################
#Read all data from utils
dfs = ReadAllData()

#Get train df from dfs list
train = dfs.train.copy()

#Remove missing volume
train = train[train.volume.notna()].reset_index(drop = True)

#Remove outliers from the dataframe for values exceeding OUT_THRES's z-score
zscores_outliers = get_outliers(train, OUT_THRES)
train = train.iloc[~train.index.isin(zscores_outliers.index)].reset_index(drop = True)

#Get and save min and max values for given site_id. It's done after outliers
#removal to exclude too small/big values from site_id's volume range.
#It's used for all years with data as there isn't much data, so even
#potentially inacurate old volumes before YEAR_SINCE are used.
min_max_site_id = train.groupby(['site_id'])['volume'].agg(['min', 'max'])
min_max_site_id.to_pickle('data\min_max_site_id_forecast.pkl')

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
#FEATURE ENGINEERING
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
train['nat_flow_prev'] = train.apply(lambda x: train_monthly_naturalized_flow.loc\
                                     [(train_monthly_naturalized_flow.issue_date <= x.issue_date) &
                                      (train_monthly_naturalized_flow.site_id == x.site_id) &
                                      (train_monthly_naturalized_flow.forecast_year == x.year),
                                      'volume'].\
                                     tail(1).mean(), axis = 1)

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

#Get ratio between month 11 and 10 (nat_flow_11_to_10_ratio)
for month in [11]:
    prev_month = 10
    print(f'Ratio month {month} to {prev_month}')
    train[f'nat_flow_{month}_to_{prev_month}_ratio'] =\
        train[f'nat_flow_{month}'] / train[f'nat_flow_{prev_month}']

#Snotel features
snotel = preprocess_snotel(dfs.snotel,
                           dfs.sites_to_snotel_stations)

#Get previous value of WTEQ_DAILY
snotel['WTEQ_DAILY_prev'] = snotel.groupby(['site_id', 'year_forecast'])['WTEQ_DAILY'].shift()
#Merge with train
train = pd.merge(train,
                 snotel[['WTEQ_DAILY_prev', 'site_id', 'year', 'month', 'day']],
                 how = 'left',
                 on = ['site_id', 'year', 'month', 'day'])

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
#SAVE OUTPUTS
###############################################################################
#Save issue date encoding to be used in test pipeline
with open("data\issue_date_encoded", "wb") as fp:
    pickle.dump(issue_date_encoded, fp)

#Save train and test data ready for modelling
train.to_pickle('data/train_test_forecast.pkl')

end = time.time()
elapsed = end - start
