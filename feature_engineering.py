import pandas as pd
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


def preprocess_monthly_naturalized_flow(train_monthly_naturalized_flow: pd.DataFrame,
                                        test_monthly_naturalized_flow: pd.DataFrame) -> pd.DataFrame:
    """
    Merge train and test monthly naturalized flow, append issue dates and
    shift by 1 month to be able to safely merge with other datasets without
    looking into future.
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
    #Merge train and test monthly naturalized flow
    monthly_naturalized_flow =\
        pd.concat([train_monthly_naturalized_flow, test_monthly_naturalized_flow])
    monthly_naturalized_flow = monthly_naturalized_flow.\
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
    return snotel
