import pandas as pd
import numpy as np
from utils import flatten_pandas_agg

def get_aggs_month_day(df_aggs: pd.DataFrame,
                       df_main: pd.DataFrame,
                       cols: list,
                       aggs: list,
                       issue_months: pd.Series,
                       issue_days: pd.Series,
                       year_col: str,
                       suffix: str,
                       month_since: int) -> pd.DataFrame:
    """
    Perform aggregates on selected columns and merge them with the main
    DataFrame without looking into future. It is intended for daily data.
    
    Args:
        df_aggs (pd.DataFrame): A DataFrame to get aggregates from
        df_main (pd.DataFrame): The main DataFrame to be merged with df_aggs
        cols (list): Columns to aggregate
        aggs (list): Aggregations to be made
        issue_months (pd.Series): Months to iterate over. It is consistent with
            issue dates - it is just month taken from all MM-DD issue date
            combinations (should contain 28 values)
        issue_days (pd.Series): Days to iterate over. It is consistent with
            issue dates - it is just day taken from all MM-DD issue date
            combinations (should contain 28 values)
        year_col (str): Year column to aggregate on. Keep in mind that it only
            infulences column indicating year from df_aggs. It is used to
            be able to distinguish between normal year and water year.
            For df_main 'year' is always used.
        suffix (str): Suffix added to name of created columns
        month_since (int): The earliest month used for data aggregation. It is
            based on water years, so Oct (10) is the earliest possible month.
            This functionality was added as it is sometimes beneficial to not
            aggregate on first months.
    Returns:
        df_main (pd.DataFrame): The main DataFrame with appended columns
    """
    df_aggs = df_aggs.copy()
    #Add suffix to columns to aggregate on
    for feat in cols:
        df_aggs[f'{feat}{suffix}'] = df_aggs[feat]
    cols = [x + suffix for x in cols]
    
    aggr_values = pd.DataFrame()
    for month, day in zip(issue_months, issue_days):
        if month_since >= 10:
            #If the earliest month is Oct/Nov/Dec, get all of the below:
            #   1. all data between month_since and Dec
            #   2. all data from months before issue month
            #   3. for the same month as issue month, get data from this month
            #       up to the day before issue day
            df_aggs_before_issue =\
                df_aggs[(df_aggs.month >= month_since) |
                           (df_aggs.month < month) |
                           ((df_aggs.month == month) & (df_aggs.day < day))]
        else:
            #If the earliest month is Jan/Feb/Mar/Apr/May/Jun/Jul, get data
            #that meets both the criteria below:
            #   1. data from month since month_since 
            #   2. data from months before issue month or data from given
            #       issue month up to the day before issue day            
            df_aggs_before_issue =\
                df_aggs[(df_aggs.month >= month_since) &
                           ((df_aggs.month < month) |
                            ((df_aggs.month == month) & (df_aggs.day < day)))]
        #Per each site_id-year combination, get aggregated values
        to_add =\
            df_aggs_before_issue.groupby(['site_id', year_col])[cols].agg(aggs)
        to_add['day'] = day
        to_add['month'] = month
        aggr_values = pd.concat([aggr_values, to_add])
    aggr_values = aggr_values.reset_index()
    #Flatten column names to get rid of MultiIndex
    aggr_values = flatten_pandas_agg(aggr_values)
    #Merge with the main DataFrame on site_id, year, month, day.
    #This way, no looking into future is introduced.
    df_main = pd.merge(df_main,
                       aggr_values,
                       how = 'left',
                       left_on = ['site_id', 'year', 'month', 'day'], #it is always 'year' in the main DataFrame
                       right_on = ['site_id', year_col, 'month', 'day'])
    #Drop additionally created year_col if it isn't 'year'
    if year_col != 'year':
        df_main.drop(year_col, axis = 1, inplace = True)
    return df_main


def get_aggs_month(df_aggs: pd.DataFrame,
                   df_main: pd.DataFrame,
                   cols: list,
                   aggs: list,
                   issue_months: pd.Series,
                   year_col: str,
                   suffix: str,
                   month_since: int) -> pd.DataFrame:
    """
    Perform aggregates on selected columns and merge them with the main
    DataFrame without looking into future. It is intended for monthly data.
    
    Args:
        df_aggs (pd.DataFrame): A DataFrame to get aggregates from
        df_main (pd.DataFrame): The main DataFrame to be merged with df_aggs
        cols (list): Columns to aggregate
        aggs (list): Aggregations to be made
        issue_months (pd.Series): Months to iterate over. It is consistent with
            issue dates - it is just month taken from all MM-DD issue date
            combinations (should contain 28 values)
        year_col (str): Year column to aggregate on. Keep in mind that it only
            infulences column indicating year from df_aggs. It is used to
            be able to distinguish between normal year and water year.
            For df_main 'year' is always used.
        suffix (str): Suffix added to name of created columns
        month_since (int): The earliest month used for data aggregation. It is
            based on water years, so Oct (10) is the earliest possible month.
            This functionality was added as it is sometimes beneficial to not
            aggregate on first months.
    Returns:
        df_main (pd.DataFrame): The main DataFrame with appended columns
    """
    df_aggs = df_aggs.copy()
    #Add suffix to columns to aggregate on
    for feat in cols:
        df_aggs[f'{feat}{suffix}'] = df_aggs[feat]
    cols = [x + suffix for x in cols]
    
    aggr_values = pd.DataFrame()
    for month in issue_months.drop_duplicates().reset_index(drop = True):
        if month_since >= 10:
            #If the earliest month is Oct/Nov/Dec, get all of the below:
            #   1. all data between month_since and Dec
            #   2. all data from months before issue month
            df_aggs_before_issue =\
                df_aggs[(df_aggs.month >= month_since) |
                           (df_aggs.month < month)]
        else:
            #If the earliest month is Jan/Feb/Mar/Apr/May/Jun/Jul, get data
            #that meets both the criteria below:
            #   1. data from month since month_since
            #   2. data from months before issue month
            df_aggs_before_issue =\
                df_aggs[(df_aggs.month >= month_since) &
                           ((df_aggs.month < month))]
        #Per each site_id-year combination, get aggregated values
        to_add =\
            df_aggs_before_issue.groupby(['site_id', year_col])[cols].agg(aggs)
        to_add['month'] = month
        aggr_values = pd.concat([aggr_values, to_add])
    aggr_values = aggr_values.reset_index()
    #Flatten column names to get rid of MultiIndex
    aggr_values = flatten_pandas_agg(aggr_values)
    #Merge with the main DataFrame on site_id, year and month. This way, no
    #looking into future is introduced.
    df_main = pd.merge(df_main,
                       aggr_values,
                       how = 'left',
                       left_on = ['site_id', 'year', 'month'], #it is always 'year' in the main DataFrame
                       right_on = ['site_id', year_col, 'month'])
    #Drop additionally created year_col if it isn't 'year'
    if year_col != 'year':
        df_main.drop(year_col, axis = 1, inplace = True)
    return df_main


def get_prev_monthly(df_aggs: pd.DataFrame,
                     df_main: pd.DataFrame,
                     cols: list,
                     new_col_names: list,
                     date_col: str,
                     site_id_col: str,
                     month_offset: bool,
                     day_start: int) -> pd.DataFrame:
    """
    Get previous available value before issue date from a monthly DataFrame.
    
    Args:
        df_aggs (pd.DataFrame): A DataFrame to get aggregates from
        df_main (pd.DataFrame): The main DataFrame to be merged with df_aggs
        cols (list): Columns to get previous available values from (from df_aggs)
        new_col_names (list): New names of columns to append
        date_col (str): df_aggs date column to aggregate on
        site_id_col (str): A column from df_aggs with site_id information
        month_offset (bool): Indicates if date_col was created just based on
            in which month the feature occurred (True) or if offset of
            a month was already added (False)
        day_start (int): Earliest day when monthly data is available (when
            feature could be merged with the main DataFrame). It should be
            the same for each month
    Returns:
        df_main (pd.DataFrame): The main DataFrame with appended prev columns
    """
    df_aggs = df_aggs.copy()
    #Add suffix to columns to aggregate on
    for feat, new_name in zip(cols, new_col_names):
        df_aggs[new_name] = df_aggs[feat]
    #Make sure that site_id column from df_aggs is called site_id
    df_aggs.rename({site_id_col: 'site_id'}, axis = 1, inplace = True)
    #Create issue date one month and 5 days later to not look into future
    df_aggs[date_col] = pd.to_datetime(df_aggs[date_col])
    #Get year, month, day
    df_aggs['year'] = df_aggs[date_col].dt.year
    df_aggs['month'] = df_aggs[date_col].dt.month
    df_aggs['day'] = df_aggs[date_col].dt.day
    #Get date with day_start
    df_aggs['issue_date_var'] = pd.to_datetime(dict(year = df_aggs.year,
                                                    month = df_aggs.month,
                                                    day = day_start))
    #Add one month as a starting point only if it hasn't been yet appended
    if month_offset == False:
        df_aggs['issue_date_var'] = df_aggs.issue_date_var +\
            pd.DateOffset(months = 1)
    df_aggs['issue_date_var'] = df_aggs.issue_date_var.astype('str')
        
    #Get date from the first available day to merge on (day_start)
    df_main['issue_date_var'] = pd.to_datetime(df_main.year.astype('str') + '-' +\
                                               df_main.month.astype('str') + '-' +\
                                               str(day_start))
    #Subtract one month if date is from the same month
    df_main.loc[df_main.day < day_start, 'issue_date_var'] =\
        df_main.issue_date_var - pd.DateOffset(months = 1)
    df_main['issue_date_var'] = df_main.issue_date_var.astype('str')
    df_aggs = df_aggs[['site_id', 'issue_date_var'] + new_col_names]
    #Merge to df_main last available monthly columns' values before issue date
    df_main = pd.merge(df_main,
                       df_aggs,
                       how = 'left',
                       on = ['site_id', 'issue_date_var'])
    df_main.drop('issue_date_var', axis = 1, inplace = True)
    return df_main


def get_prev_daily(df_aggs: pd.DataFrame,
                   df_main: pd.DataFrame,
                   cols: list,
                   new_col_names: list,
                   date_col: str,
                   site_id_col: str,
                   issue_days_unique: np.array,
                   days_lag: int) -> pd.DataFrame:
    """
    Get previous available value before issue date from a monthly DataFrame.
    This function works appropriately only if lag day isn't greater than 6 (for
    February) and 8 (for 30-day long months).
    
    Args:
        df_aggs (pd.DataFrame): A DataFrame to get aggregates from
        df_main (pd.DataFrame): The main DataFrame to be merged with df_aggs
        cols (list): Columns to get previous available values from (from df_aggs)
        new_col_names (list): New names of columns to append
        date_col (str): df_aggs date column to aggregate on
        site_id_col (str): A column from df_aggs with site_id information
        issue_days_unique (list): A list of issue days from df_main. It should
            have values of 1, 8, 15, 22
        days_lag (int): How many days should be added to date from df_aggs to
            be constistent with issue dates from df_main
    Returns:
        df_main (pd.DataFrame): The main DataFrame with appended prev columns
    """
    df_aggs = df_aggs.copy()
    #Add suffix to columns to aggregate on
    for feat, new_name in zip(cols, new_col_names):
        df_aggs[new_name] = df_aggs[feat]
    #Make sure that site_id column from df_aggs is called site_id
    df_aggs.rename({site_id_col: 'site_id'}, axis = 1, inplace = True)
    #Create issue date one month and 5 days later to not look into future
    df_aggs[date_col] = pd.to_datetime(df_aggs[date_col])
    #Get year, month, day
    df_aggs['year'] = df_aggs[date_col].dt.year
    df_aggs['month'] = df_aggs[date_col].dt.month
    df_aggs['day'] = df_aggs[date_col].dt.day
    #Get the earliest date when values from df_aggs don't look into future
    df_aggs['issue_date_var'] = df_aggs[date_col] + pd.Timedelta(days = days_lag)
    #Add one month as a starting point only if it hasn't been yet appended
    df_aggs['issue_date_var'] = df_aggs.issue_date_var.astype('str')
    #Get year, month and day from issue_date_var
    date_issue_split = df_aggs.issue_date_var.str.split('-')
    df_aggs['year_issue'] = date_issue_split.str[0].astype('int')
    df_aggs['month_issue'] = date_issue_split.str[1].astype('int')
    df_aggs['day_issue'] = date_issue_split.str[2].astype('int')
    #Get earliest matching day from df_main's issue days for each row.
    #Days without looking into future were already selected and now the closest
    #day from df_main issue_dates will be determined to match with df_main 
    df_aggs['issue_date_day'] = np.nan
    for issue_day in issue_days_unique:
        df_aggs.loc[(df_aggs.issue_date_day.isna()) &
                    (df_aggs.day_issue <= issue_day),
                    'issue_date_day'] = issue_day
    #Add month and year for merging
    df_aggs['issue_date_month'] = df_aggs.month_issue
    df_aggs['issue_date_year'] = df_aggs.year_issue
    #If issue date should be from the next month (day_issue>22), set issue_day
    #to 1 and add one month
    df_aggs.loc[(df_aggs.issue_date_day.isna()), 'issue_date_month'] =\
        df_aggs.issue_date_month + 1
    df_aggs.loc[(df_aggs.issue_date_day.isna()), 'issue_date_day'] = 1
    #Amendment to issue_date_month if it was December
    df_aggs.loc[df_aggs.issue_date_month == 13, 'issue_date_year'] =\
        df_aggs.issue_date_year + 1
    df_aggs.loc[df_aggs.issue_date_month == 13, 'issue_date_month'] = 1
    #Get final issue_date to merge with df_main
    df_aggs['issue_date_var'] = pd.to_datetime(
        df_aggs.issue_date_year.astype('str') + '-' +\
        df_aggs.issue_date_month.astype('int').astype('str') + '-' +\
        df_aggs.issue_date_day.astype('int').astype('str')).astype('str')
    #Make sure that values are correctly sorted
    df_aggs = df_aggs.sort_values(['site_id', 'issue_date_var']).reset_index(drop = True)
    #Get last observation for given issue dates. This way it will be the latest
    #available observation for each issue data. Keep only indexes
    idx_to_keep = df_aggs[['site_id', 'issue_date_var']].\
        drop_duplicates(keep = 'last').index
    #Get rows to merge with train
    df_to_merge = df_aggs.loc[idx_to_keep]
    #Get columns to merge
    feats_to_keep = ['site_id', 'issue_date_var'] + new_col_names
    df_to_merge = df_to_merge[feats_to_keep]
    #Change issue_date name to be consistent with df_main convention
    df_to_merge.rename({'issue_date_var': 'issue_date'}, axis = 1, inplace = True)
    #Merge with df_main
    df_main = pd.merge(df_main,
                       df_to_merge,
                       how = 'left',
                       on = ['site_id', 'issue_date'])
    return df_main


def preprocess_monthly_naturalized_flow(train_monthly_naturalized_flow: pd.DataFrame) -> pd.DataFrame:
    """
    Append issue dates and shift by 1 month to be able to safely merge with
    other datasets without looking into future.
    General information on this dataset is in https://www.drivendata.org/competitions/254/reclamation-water-supply-forecast-dev/page/797/#antecedent-monthly-naturalized-flow.
    Data came from NRCS (https://www.nrcs.usda.gov/) and RFCs
    (https://water.weather.gov/ahps/rfc/rfc.php) sources.
    
    Args:
        train_monthly_naturalized_flow (pd.DataFrame): train monthly
            naturalized flow
        test_monthly_naturalized_flow (pd.DataFrame): test monthly
            naturalized flow
    Returns:
        monthly_naturalized_flow (pd.DataFrame): merged monthly naturalized
            flow with some auxiliary features
    """
    #Sort values
    monthly_naturalized_flow = train_monthly_naturalized_flow.\
        sort_values(['site_id', 'year', 'month']).reset_index(drop = True)
    #Get issue dates
    monthly_naturalized_flow['issue_date'] = pd.to_datetime\
        (monthly_naturalized_flow['year'].astype('str') + '-' +\
             monthly_naturalized_flow['month'].astype('str'))
    monthly_naturalized_flow['issue_date'] =\
        monthly_naturalized_flow['issue_date'].astype('str')
    #Shift by 1 month to be able to easily merge without looking into future
    monthly_naturalized_flow['issue_date'] =\
        (pd.to_datetime(monthly_naturalized_flow.issue_date) +\
         pd.DateOffset(months = 1)).astype('str')
    return monthly_naturalized_flow


def nat_flow_sum_cumul_since_apr(train_monthly_naturalized_flow: pd.DataFrame,
                                 df_main: pd.DataFrame,
                                 new_col_name: str,
                                 month_end: int) -> pd.DataFrame:
    """
    Get sum of naturalized flow for a given site_id between April and selected
    month. It won't be used as a feature. It will be used in postprocessing in
    model training pipeline to make sure that predictions made for Apr-Jul
    volumes aren't less than those from Apr/Apr-May/Apr-Jun. There is no need
    for filling missing values for df_main data from previous months, as
    the transformations will be done only for months without looking into future.
    
    Args:
        train_monthly_naturalized_flow (pd.DataFrame): Monthly naturalized flow
            after some initial preprocessing
        df_main (pd.DataFrame): The main DataFrame to be merged with a sum of
            naturalized flow from selected months
        new_col_name (str): A column name for the new column with a sum of
            naturalized flow from selected months
        month_end (int): Up to which month the sum of naturalized flow should
            be calculated
    Returns:
        df_main (pd.DataFrame): The main DataFrame with appended cumulative
            naturalized flow
    """
    #Get sum of monthly naturalized flow between April and selected month for
    #a given site_id-forecast_year combination. Don't calculate it for
    #pecos_r_nr_pecos site_id for now 
    nat_flow_sum = train_monthly_naturalized_flow[
        (train_monthly_naturalized_flow.month.between(4, month_end)) &
        (train_monthly_naturalized_flow.site_id != 'pecos_r_nr_pecos')].\
        groupby(['site_id', 'forecast_year'])['nat_flow'].sum().reset_index()
    #Calculate pecos_r_nr_pecos separately. For this site_id, naturalized flow
    #between March and July is calcluated in the competition, so use
    #nat_flow_sum since March for this site_id.
    #detroit_lake_inflow doesn't need similar change, Apr-Jun volume is
    #predicted for this site_id but it's safe to do it even for July as
    #train_monthly_naturalized_flow dataset doesn't include July for this site
    nat_flow_sum_pecos = train_monthly_naturalized_flow[
        (train_monthly_naturalized_flow.month.between(3, month_end)) &
        (train_monthly_naturalized_flow.site_id == 'pecos_r_nr_pecos')].\
        groupby(['site_id', 'forecast_year'])['nat_flow'].sum().reset_index()
    #Merge pecos with other site_ids
    nat_flow_sum = pd.concat([nat_flow_sum, nat_flow_sum_pecos]).reset_index(drop = True)
    #Change nat_flow 0 to nan
    nat_flow_sum.loc[nat_flow_sum.nat_flow == 0, 'nat_flow'] = np.nan
    #Set name for column to add to df_main
    nat_flow_sum = nat_flow_sum.rename({'nat_flow': new_col_name}, axis = 1)
    #Merge with df_main
    df_main = pd.merge(df_main,
             nat_flow_sum,
             how = 'left',
             left_on = ['site_id', 'year'],
             right_on = ['site_id', 'forecast_year'])
    df_main.drop('forecast_year', axis = 1, inplace = True)
    return df_main


def preprocess_snotel(snotel: pd.DataFrame,
                      sites_to_snotel_stations: pd.DataFrame) -> pd.DataFrame:
    """
    Add helper features to SNOTEL. Average values over stations and add date
    details.
    
    Args:
        snotel (pd.DataFrame): SNOTEL DataFrame
        sites_to_snotel_stations (pd.DataFrame): site_id-station mapping
    Returns:
        snotel (pd.DataFrame): SNOTEL DataFrame with additional information
    """
    #Add site_id to snotel
    sites_to_snotel_stations['stationTriplet'] =\
        sites_to_snotel_stations.stationTriplet.str.replace(':', '_')
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

    #Add year_forecast as year from given water year
    snotel['year_forecast'] = snotel.year
    snotel.loc[snotel['month'].astype(int).between(10, 12), 'year_forecast'] =\
        snotel.year_forecast + 1
    #Add issue date with a 1 day lag to be able to merge easily with train data
    snotel['issue_date'] = pd.to_datetime(snotel.date) + pd.DateOffset(days = 1)
    snotel['issue_date'] = snotel.issue_date.astype('str')
    return snotel
