import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm
import pickle
from loguru import logger

from utils import ReadAllData, get_outliers, all_distr_dict, distr_param_values_to_dict

import time

start = time.time()

dfs = ReadAllData()

#Get train df from dfs list
df = dfs.train.copy()
#Get unique site ids
site_ids_unique = dfs.site_ids_unique.copy()
#Remove missing values
df = df[df.volume.notna()].reset_index(drop = True)

#Remove outliers
OUT_THRES = 2.5
zscores_outliers = get_outliers(df, OUT_THRES)
df = df.iloc[~df.index.isin(zscores_outliers.index)]

#Remove troublesome distributions
distr_trouble = ['kstwo', 'rv_continuous', 'rv_histogram', 'studentized_range',
                 'vonmises_fisher', 'vonmises', 'gausshyper', 'powerlaw',
                 'vonmises_line', 'ncx2', 'arcsine', 'kappa4']
for distr in distr_trouble:
    del all_distr_dict[distr]

#Find best distributions for each site_id. Best parameters are fitted for each
#distribution
distr_results = []
for site_id in tqdm(site_ids_unique):
    print(site_id)
    for distr_name, distr in all_distr_dict.items():
        #Keep best estimate of distribution parameters in distr_results and
        #best parameters in distr_params
        args = distr._fitstart(df[df.site_id == site_id].volume)
        x0, func, restore, args = distr._reduce_func(args, {})
        try:
            res = minimize(func, x0, args=(df[df.site_id == site_id].volume,),
                           method='Nelder-Mead')
            distr_results.append([site_id, distr_name, res.fun, tuple(res.x)])
        except:
            logger.info(f'There was an error with {distr_name} for {site_id}.\
                  Ignoring the distribution and trying the next one.')

#Save results
with open("data\distr_per_site_forecast_50_outliers_2_5", "wb") as fp:
    pickle.dump(distr_results, fp)

#Keep only best distribution fit for each site_id
distr_results_best = pd.DataFrame(distr_results).sort_values(2).drop(2, axis = 1).\
    groupby(0).head(1).values.tolist()
#Change to appropriate format
distr_results_best = [[x[0], {x[1]: distr_param_values_to_dict(x[1], x[2])}]
                      for x in distr_results_best]
#Save best results
with open("data\distr_per_site_forecast_50_outliers_2_5_best", "wb") as fp:
    pickle.dump(distr_results_best, fp)

end = time.time()
elapsed = end - start
