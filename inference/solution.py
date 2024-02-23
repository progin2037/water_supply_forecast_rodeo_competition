from collections.abc import Hashable
from pathlib import Path
from typing import Any

import os
import pickle
from loguru import logger
import pandas as pd
import numpy as np

from wsfr_read.streamflow import read_test_monthly_naturalized_flow, read_usgs_streamflow_data
from wsfr_read.sites import read_metadata

#Suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None


def preprocess(src_dir: Path, data_dir: Path, preprocessed_dir: Path) -> dict[Hashable, Any]: 
    #Read models for all quantiles and months
    models = dict()
    for quantile in [10, 50, 90]:
        for month in [1, 2, 3, 4, 5, 6, 7]:
            model_path = src_dir / f'models/lgbm_{quantile}_{month}_month_2023_12_21.pkl'
            models[f'q_{quantile}_{month}_month'] = pd.read_pickle(model_path)
    
    #Read all distributions to use
    with open(src_dir / "all_distr_dict", "rb") as fp:
        all_distr_dict = pickle.load(fp)    
    #Read fitted distributions to train volume. Distribution was fitted per each site_id
    with open(src_dir / "distr_per_site_50_outliers_2_5_best", "rb") as fp:
        distr_results = pickle.load(fp)
    #Read amendments to distributions
    with open(src_dir / "distr_amendments", "rb") as fp:
        distr_to_change = pickle.load(fp)
    #Read issue date encoding
    with open(src_dir / "issue_date_encoded", "rb") as fp:
        issue_date_encoded = pickle.load(fp)
    #Read min and max volume values per site_id
    min_max_site_id = pd.read_pickle(src_dir / "min_max_site_id.pkl")

    return {"models": models,
            "distr_results": distr_results,
            "distr_to_change": distr_to_change,
            "issue_date_encoded": issue_date_encoded,
            "all_distr_dict": all_distr_dict,
            "min_max_site_id": min_max_site_id}


def predict(
    site_id: str,
    issue_date: str,
    assets: dict[Any, Any],
    src_dir: Path,
    data_dir: Path,
    preprocessed_dir: Path,
) -> tuple[float, float, float]:

    DISTR_PERC = 0.4 #Set how much importance should be given for distribution

    #Get longitude
    site_metadata_df = read_metadata()
    longitude = site_metadata_df.loc[site_metadata_df.index == site_id, 'longitude'].item()


    #Get monthly_naturalized_flow features
    try:
        #Read monthly_naturalized_flow
        monthly_naturalized_flow = read_test_monthly_naturalized_flow(site_id, issue_date)
        monthly_naturalized_flow.reset_index(inplace = True)
        #Get latest value 
        nat_flow_prev = monthly_naturalized_flow['volume'].iloc[-1]
        #Get average volume from data since April
        nat_flow_Apr_mean = monthly_naturalized_flow.loc[(monthly_naturalized_flow.month >= 4) &
                                                         (monthly_naturalized_flow.month < 10),
                                                         'volume'].mean()
        #Ratio between month 11 and 10
        nat_flow_10 = monthly_naturalized_flow.loc[monthly_naturalized_flow.month == 10, 'volume'].item()
        nat_flow_11 = monthly_naturalized_flow.loc[monthly_naturalized_flow.month == 11, 'volume'].item()
        nat_flow_11_to_10_ratio = nat_flow_11 / nat_flow_10
    except Exception as e:
        logger.info(f"Monthly naturalized flow wasn't properly processed on {issue_date} for {site_id}. Filling values with nans")
        logger.info(e)
        #Fill nans For sites without monthly_naturalized_flow 
        nat_flow_prev = np.nan
        nat_flow_Apr_mean = np.nan
        nat_flow_11_to_10_ratio = np.nan


    #Get SNOTEL features

    #Get all paths from directory
    def get_paths(path_dir: str,
                  extension: str) -> list:
        paths = []
        for root, dirs, files in os.walk(path_dir):
            for file in files:
                full_path = os.path.join(root, file)
                if full_path.endswith(extension):
                    #Append the file name to the list
                    paths.append(os.path.join(root, file))
        return paths
    try:
        year = issue_date[:4]
        #Get all paths from SNOTEL directory
        snotel_paths = get_paths(data_dir / 'snotel/', '.csv')
        #logger.info('snotel_paths done')
        #logger.info(snotel_paths[0])
        
        snotel_paths = pd.Series(snotel_paths)

        #Keep only SNOTEL daily data
        snotel_paths_meta = snotel_paths[snotel_paths.str.contains('station_metadata|sites_to_snotel_stations')]
        #logger.info('snotel_paths_meta done')
        #logger.info(snotel_paths_meta.iloc[0])
        snotel_paths_data = snotel_paths[~snotel_paths.index.isin(snotel_paths_meta.index)]
        #Split paths by '/'
        snotel_paths_data_split = snotel_paths_data.str.split('/')
        #logger.info('split slash done')
        #logger.info(snotel_paths_data_split.iloc[0])

        #Read sites_to_snotel_stations
        sites_to_snotel_stations = pd.read_csv(data_dir / 'snotel\sites_to_snotel_stations.csv')
        #Get stations connected to given site_id
        sites_to_snotel_stations = sites_to_snotel_stations[sites_to_snotel_stations.site_id == site_id]
        #Change stations format
        sites_to_snotel_stations['stationTriplet'] = sites_to_snotel_stations.stationTriplet.str.replace(':', '_')
        #logger.info('changed station format')
        #logger.info(sites_to_snotel_stations.iloc[0])
        
        #Get years and stations from paths
        snotel_paths_years = snotel_paths_data_split.str[-2].str[2:]
        #logger.info('added snotel_paths_years')
        #logger.info(snotel_paths_years.iloc[0])
        
        snotel_paths_stations = snotel_paths_data_split.str[-1].str.rstrip('.csv')
        #logger.info('added snotel_paths_stations')
        #logger.info(snotel_paths_stations.iloc[0])

        #Get idxs of stations and years that match current data point
        idx_station = snotel_paths_stations[snotel_paths_stations.isin(sites_to_snotel_stations.stationTriplet)].index
        idx_year = snotel_paths_years[snotel_paths_years == year].index

        #Keep only matching paths
        snotel_paths_final = snotel_paths_data[(snotel_paths_data.index.isin(idx_station)) &
                                               (snotel_paths_data.index.isin(idx_year))]

        #Create snotel_df
        snotel_df = pd.DataFrame({'date': pd.Series(dtype = 'str'),
                                  'WTEQ_DAILY': pd.Series(dtype = 'float'),
                                  'STATION': pd.Series(dtype = 'str')})
        #Iterate over different paths
        for snotel_path in snotel_paths_final:
            #Get station name
            station = snotel_path.split('/')[-1].strip('.csv')
            #Read a file
            try:
                df_to_add = pd.read_csv(snotel_path)[['date', 'WTEQ_DAILY']]
            except:
                #If one of the columns was missing, go to next dataframe
                logger.info(f'WTEQ_DAILY was missing from given station data {station} on {issue_date} for {site_id}. Go to the next station.')
                continue
            #Keep data before issue_date. SNOTEL seems to upload data every day
            df_to_add = df_to_add[df_to_add['date'] < issue_date]
            #Append station
            df_to_add['STATION'] = station
            #Append to dafarame
            snotel_df = pd.concat([snotel_df, df_to_add], axis = 0, ignore_index = True)
        #Get average WTEQ_DAILY value per date (avg  from different stations)
        snotel_df = snotel_df.groupby(['date'])['WTEQ_DAILY'].mean().reset_index().sort_values('date')

        #Get latest value
        WTEQ_DAILY_prev = snotel_df['WTEQ_DAILY'].iloc[-1]
        #Get month
        snotel_df['month'] = snotel_df.date.str[5:7].astype('int')
        #Get average value since April
        WTEQ_DAILY_Apr_mean = snotel_df.loc[(snotel_df.month >= 4) & (snotel_df.month < 10), 'WTEQ_DAILY'].mean()

    except Exception as e:
        logger.info(f"SNOTEL wasn't properly processed on {issue_date} for {site_id}. Filling values with nans")
        logger.info(e)
        WTEQ_DAILY_prev = np.nan
        WTEQ_DAILY_Apr_mean = np.nan


    #Get USGS streamflow data:
    try:
        streamflow = read_usgs_streamflow_data(site_id, issue_date)
        #Get std value
        discharge_cfs_mean_std = streamflow['discharge_cfs_mean'].std()
        #Get month
        streamflow['month'] = streamflow.datetime.dt.month

        #Get average value since April
        discharge_cfs_mean_Apr_mean =\
            streamflow.loc[(streamflow.month >= 4) & (streamflow.month < 10), 'discharge_cfs_mean'].mean()
    except Exception as e:
        logger.info(f"USGS streamflow wasn't properly processed on {issue_date} for {site_id}. Filling values with nans")
        logger.info(e)
        discharge_cfs_mean_std = np.nan
        discharge_cfs_mean_Apr_mean = np.nan


    #Get month_day from issue_date
    issue_month_day = issue_date[5:]
    #Encode for model
    issue_date_no_year = assets['issue_date_encoded'][issue_month_day]

    #Create DataFrame from calculated values
    test = pd.DataFrame({'site_id': [site_id],
                         'issue_date': [issue_date],
                         'longitude': [longitude],
                         'nat_flow_prev': [nat_flow_prev],
                         'nat_flow_Apr_mean': [nat_flow_Apr_mean],
                         'nat_flow_11_to_10_ratio': [nat_flow_11_to_10_ratio],
                         'WTEQ_DAILY_prev': [WTEQ_DAILY_prev],
                         'WTEQ_DAILY_Apr_mean': [WTEQ_DAILY_Apr_mean],
                         'discharge_cfs_mean_std': [discharge_cfs_mean_std],
                         'discharge_cfs_mean_Apr_mean': [discharge_cfs_mean_Apr_mean],
                         'issue_date_no_year': [issue_date_no_year]
                        })

    #Get features for different months
    train_feat_1 = ['site_id', 'nat_flow_prev', 'WTEQ_DAILY_prev', 'issue_date_no_year',
                    'nat_flow_11_to_10_ratio']

    train_feat_2 = ['site_id', 'nat_flow_prev', 'WTEQ_DAILY_prev', 'issue_date_no_year', 'discharge_cfs_mean_std',
                    'longitude']

    train_feat_3 = ['site_id', 'nat_flow_prev', 'WTEQ_DAILY_prev', 'issue_date_no_year', 'discharge_cfs_mean_std',
                    'longitude']

    train_feat_4 = ['site_id', 'nat_flow_prev', 'WTEQ_DAILY_prev', 'issue_date_no_year', 'discharge_cfs_mean_std',
                    'longitude']

    train_feat_5 = ['site_id', 'nat_flow_prev', 'WTEQ_DAILY_prev', 'issue_date_no_year', 'discharge_cfs_mean_std',
                    'longitude', 'WTEQ_DAILY_Apr_mean', 'discharge_cfs_mean_Apr_mean']

    train_feat_6 = ['site_id', 'nat_flow_prev', 'WTEQ_DAILY_prev', 'issue_date_no_year', 'discharge_cfs_mean_std',
                    'longitude', 'WTEQ_DAILY_Apr_mean', 'discharge_cfs_mean_Apr_mean', 'nat_flow_Apr_mean']

    train_feat_7 = ['site_id', 'nat_flow_prev', 'WTEQ_DAILY_prev', 'issue_date_no_year', 'discharge_cfs_mean_std',
                    'longitude', 'nat_flow_Apr_mean', 'WTEQ_DAILY_Apr_mean', 'discharge_cfs_mean_Apr_mean']

    train_feat_dict = {1: train_feat_1,
                       2: train_feat_2,
                       3: train_feat_3,
                       4: train_feat_4,
                       5: train_feat_5,
                       6: train_feat_6,
                       7: train_feat_7}

    #Change site_id to category for LightGBM model
    categorical = ['site_id']
    for cat in categorical:
        test[cat] = test[cat].astype('category')

    #Get month to choose appropriate model
    month = int(issue_date[5:7])
    
    test['volume_10'] = np.nan
    test['volume_50'] = np.nan
    test['volume_90'] = np.nan
    test['volume_10_lgbm'] = np.nan
    test['volume_90_lgbm'] = np.nan
    test['volume_10_distr'] = np.nan
    test['volume_90_distr'] = np.nan
    
    test['volume_50'] = assets['models'][f'q_50_{month}_month'].predict(test[train_feat_dict[month]])
    test['volume_10_lgbm'] = assets['models'][f'q_10_{month}_month'].predict(test[train_feat_dict[month]])
    test['volume_90_lgbm'] = assets['models'][f'q_90_{month}_month'].predict(test[train_feat_dict[month]])


    def get_quantiles_from_distr(data: pd.DataFrame,
                                 min_max_site_id: pd.DataFrame,
                                 all_distr_dict: dict,
                                 distr_per_site: list,
                                 distr_to_change: list,
                                 site_id: str):
        """
        1. Gex LightGBM prediction (x)
        2. Calculate in what point on CDF function x appears on (q0.5)
            * 3600 volume is on 0.64 quantile, then q(0.5) = 0.64)
        3. Calculate in what point on CDF function max possible historical data for site_id (max(site_id)) appears on (qmax)
            * 5000 volue (maximum per site_id from historical data) is on 0.98, quantile, then qmax = 0.98
        3. Set x to max possible value (max(site_id)) if x > max(site_id)
        4. Calculate a difference on CDF between qmax and q0.5
        5. This difference becomes a base for calculating quantile0.9 (q0.9). The quantile is calculated based on proportions. 
            qmax - q0.5 = value
            q0.9 = q0.5 + 4/5 * value (4/5 as thanks to that the distance is proportional; it would be 3/5 for q0.8)
        6. Do similarly for min value
            1. Calculate in what point on CDF function min possible historical data for site_id (min(site_id)) appears on (qmin)
            2. Set x to min possible value (min(site_id)) if x > min(site_id)
            3. Calculate a difference on CDF between q0.5 and qmin
            4. This difference becomes a base for calculating quantile0.9 (q0.9). The quantile is calculated based on proportions. 
                q0.5 - qmin = value  
                q0.1 = q0.5 - 4/5 * value

        Thanks to this approach, quantile values are continous (except LightGBM predictions that exceed min/max site_id value) and
        will never exceed min, max values range.
        """
        #Amendments to distributions
        #Get site_ids that require distribution change
        sites_distr_to_change = [x[0] for x in distr_to_change]
        #Get site_ids from distr_per_site
        site_ids_distr_per_site = [x[0] for x in distr_per_site]
        #Get indexes to change from site_ids_distr_per_site (distr_per_site)
        idxs_to_change = [site_ids_distr_per_site.index(x) for x in sites_distr_to_change]
        #Get indexes to change distr_per_site values. Those indexes are from distr_to_change
        idxs_change_with = list(range(len(sites_distr_to_change)))
        #Change values for specified indexes
        for idx_to_change, idx_change_with in zip(idxs_to_change, idxs_change_with):
            distr_per_site[idx_to_change] = distr_to_change[idx_change_with]

        
        #Get distribution from this site_id
        distr_per_site = [x for x in assets['distr_results'] if x[0] == site_id]
        
        for site_id, site_params in distr_per_site:
            #Get distribution name
            distr = next(iter(site_params.keys()))
            #Get distribution parameters
            distr_args = site_params[distr]
            #Center distribution on our volume_50 prediction
            site_cdf_50 = data[data.site_id == site_id]['volume_50'].apply(
                lambda x: all_distr_dict[distr].cdf(x, **distr_args))

            #Get site_cdf_10 and site_cdf_90 based on proportions
            #Get min and max values for site_id volume
            max_site = min_max_site_id.loc[min_max_site_id.index == site_id, 'max']
            min_site = min_max_site_id.loc[min_max_site_id.index == site_id, 'min']
            #Get CDFs of min and max volume for current site_id
            max_cdf = pd.Series(all_distr_dict[distr].cdf(max_site, **distr_args))
            min_cdf = pd.Series(all_distr_dict[distr].cdf(min_site, **distr_args))
            #Fill values greater than max_cdf with max_cdf
            site_cdf_50.loc[site_cdf_50 > max_cdf] = max_cdf
            #Fill values less than min_cdf with min_cdf
            site_cdf_50.loc[site_cdf_50 < min_cdf] = min_cdf
            #Calculate site_cdf_90 based on proportions
            site_cdf_90 = site_cdf_50 + (max_cdf - site_cdf_50) * 0.8
            #Calculate site_cdf_10 based on proportions
            site_cdf_10 = site_cdf_50 - (site_cdf_50 - min_cdf) * 0.8
            #Get 0.9 quantile values
            q_90 = site_cdf_90.apply(lambda x: all_distr_dict[distr].ppf(q = x, **distr_args))
            #Get 0.1 quantile values
            q_10 = site_cdf_10.apply(lambda x: all_distr_dict[distr].ppf(q = x, **distr_args))

            data.loc[data.site_id == site_id, 'volume_90_distr'] = q_90
            data.loc[data.site_id == site_id, 'volume_10_distr'] = q_10
        return data

    test = get_quantiles_from_distr(test,
                                    assets['min_max_site_id'],
                                    assets['all_distr_dict'],
                                    assets['distr_results'], 
                                    assets['distr_to_change'],
                                    site_id)
    
    #Make amendments to results
    
    #Get maximum and minimum values for site_id to make amendments to Q0.1 and Q0.9.
    #Using values greater than historical range is assumed prohibited.
    max_min_site = assets['min_max_site_id']
    #Add min and max for site_id as 'max' and 'min' columns
    test = pd.merge(test,
                    max_min_site,
                    how = 'left',
                    left_on = 'site_id',
                    right_index = True)

    #Change volume values values greater than min (max) for site_id to that min (max) value.
    #Do it also for distribution volume, though it shouldn't exceed maximum values, just to be certain.
    test.loc[test['volume_50'] < test['min'], 'volume_50'] = test['min']
    test.loc[test['volume_50'] > test['max'], 'volume_50'] = test['max']

    test.loc[test['volume_10_lgbm'] < test['min'], 'volume_10_lgbm'] = test['min']
    test.loc[test['volume_10_distr'] < test['min'], 'volume_10_distr'] = test['min']

    test.loc[test['volume_90_lgbm'] > test['max'], 'volume_90_lgbm'] = test['max']
    test.loc[test['volume_90_distr'] > test['max'], 'volume_90_distr'] = test['max']

    #Clipping, if volume_90 < volume_50 -> change volume_90 to volume_50
    #          if volume_50 < volume_10 -> change volume_10 to volume_50            
    #Do it also for distribution estimate to be certain that it's used indeed.
    test.loc[test.volume_90_lgbm < test.volume_50, 'volume_90_lgbm'] = test.volume_50
    test.loc[test.volume_50 < test.volume_10_lgbm, 'volume_10_lgbm'] = test.volume_50

    test.loc[test.volume_90_distr < test.volume_50, 'volume_90_distr'] = test.volume_50
    test.loc[test.volume_50 < test.volume_10_distr, 'volume_10_distr'] = test.volume_50

    #Get weighted average from distribution and models for Q0.1 and Q0.9
    test['volume_10'] = DISTR_PERC * test.volume_10_distr + (1 - DISTR_PERC) * test.volume_10_lgbm
    test['volume_90'] = DISTR_PERC * test.volume_90_distr + (1 - DISTR_PERC) * test.volume_90_lgbm
    
    return test.volume_10.item(), test.volume_50.item(), test.volume_90.item()