# water_supply_forecast_rodeo_competition

## Introduction
The repository contains a full Python solution for the Water Supply Forecast Rodeo (https://www.drivendata.org/competitions/group/reclamation-water-supply-forecast/) competition for its Forecast Stage.
The competition aims at predicting water supply for 26 western US hydrologic sites. Forecasts are made for different quantiles - 0.1, 0.5 and 0.9. Additionally, the predictions are made for different
issue dates. For most hydrologic sites, the predictions are made for the volume of water flow between April and July, but the forecasts are issued in different months (4 weeks from January, February, March, April,
May, June, July).

The solution makes predictions based on the approach of creating LightGBM models for different months and averaging their results with distribution estimates from historical data.

The aim of the first stage of the competition (Hindcast Stage, hidcast_stage branch, https://www.drivendata.org/competitions/257/reclamation-water-supply-forecast-hindcast/) was
to make predictions for historical data (2005, 2007, 2009, 2011, 2013, 2015, 2017, 2019, 2021, 2023).

The second one (Forecast Stage, forecast_stage branch, https://www.drivendata.org/competitions/259/reclamation-water-supply-forecast/) is a real-time stage. Predictions are made on the competition
server for water volume forecasts between April (March for one hydrologic site) and July (June for one hydrologic site) 2024. Inference code is available on this branch.

In the third stage (Final Prize Stage, main branch, https://www.drivendata.org/competitions/262/reclamation-water-supply-forecast-final/) the objective was to make predictions using LOOCV
(Leave-One-Out Cross Validation) with 20 folds where one of the years between 2004-2023 is a test set. It is different from the other stages in the way that using this fold CV year data for
creating aggregations is prohibited, so each fold's data has to be processed separately.

Water supply forecast combining LightGBM model with distribution estimation approach.pdf contains a 12-page summary of the solution.

Some scripts are only available for some stages (for example get_predictions.py was removed from the Final Stage, as predictions were run in the LOOCV pipeline, as well as inference/ directory,
as inference wasn't needed for the Final Stage). They were removed to avoid confusion that some scripts exist but aren't used. The removed scripts could still be found in other branches.

## Content
1. Scripts to run to create and test models
	1. cds_downloads.py - it downloads CDS data (CDS monthly data and seasonal forecasts). However, using notebooks/CDS downloads.ipynb is recommended to easier keep track of already downloaded data.
	2. data_processing.py - it processes the data and saves a DataFrame to be used in model_training.py.
	3. model_params.py - contains model hyperparameters and features for different months to be used in model_training.py and get_predictions.py. It saves these parameters as files for simplicity.
	4. distribution_estimates.py - it fits data for each hydrologic site to different distributions, optimizes parameters for the distribution and selects distributions with the best fit to data. 
	5. model_training.py - 	trains the models. There are different parameters to choose from for this script. By default, RUN_FINAL_SOLUTION is set to True, which ignores other parameters and
	trains all models required to obtain the final solution. Models' hyperparameters were already read from model_params.py but creating hyperparameters by yourself is also supported to check
	if hardcoded hyperparameters are indeed the output of running the function. Keep in mind that hyperparameters optimization takes long (20-50 hours).
2. Auxiliary scripts
	1. utils.py - general utilities, functions/list/dictionary that could be used for different tasks.
	2. feature_engineering.py - functions to facilitate feature engineering.
	3. train_utils.py - functions dedicated for model training.
	4. cv_and_hyperparams_opt.py - functions with main cv/hyperparameters tuning logic.
3. Distribution estimates (data/distr)
	1. Contains 4 types of output files from distribution_estimates.py:
		1. distr_per_site_[year] - all fitted distributions with site_id-distribution_name-distribution_fit-parameter_values combinations from a given year.
		2. distr_per_site_best_[year] - one best fit for each site_id from a given year (site_id-distribution_name-parameter_values combinations).
		3. distr_amend_[year] - amendments to make to distr_per_site_best_[year]. It contains (site_id-distribution_name-parameter_values combinations) of site_ids with distributions to change
		for a given year.
		4. distr_final_[year] - final distributions used in the model. It is the result of merging distr_per_site_best_[year] with distr_amend_[year] amendments for selected site_ids for a given year.
	2. Estimated parameters were saved to facilitate the process, as running distribution_estimates.py for all years takes about 4-6 hours.
4. Results from repo (results/)
	1. Contains submission_2024_03_28.pkl with final results submitted to the competition.
	2. Contains hyperparameters tuning results for the Forecast Stage of the competition, one per month (different quantiles were optimized together).
5. Notebooks (notebooks/)
	1. Additional analyses.
	2. CDS data download. It was provided in a notebook to facilitate keeping track of download progress.
6. Water supply forecast combining LightGBM model with distribution estimation approach.pdf - a 12-page summary of the solution.
## How to run
The solution was created using Python version 3.10.13.

*Keep in mind that results won't be exactly the same as those from models/ repo directory when downloading data again, as some of the datasets could be updated (it happened for example
with USGS streamflow. There was a data update in 2024 but data available on 2023-11-10 was used in the solution, to not take into account future update that wasn't available at a time
when the predictions would have been made if it was run real-time).*

1. Clone this repo.
2. Install packages from requirements (`pip install -r requirements.txt`).
	1. If you run into problems with using eccodes packages, try to install it with conda (`conda install -c conda-forge eccodes==2.33.0`)
	2. Follow the official guidelines to use CDS API https://cds.climate.copernicus.eu/api-how-to. It requires creating an account, saving API key and agreeing to the Terms of Use
	of every datasets that you intend to download. When running CDS download for the first time, a link to agreeing to the Terms should be provided.
3. Create data/ directory within the repo. All data should be downloaded to this directory.
4. Download data from the competition website https://www.drivendata.org/competitions/254/reclamation-water-supply-forecast-dev/data/. 
5. Download additional data from the competition repo https://github.com/drivendataorg/water-supply-forecast-rodeo-runtime.
	1. Clone the repo.
	2. Replace hindcast_test_config.yml (https://github.com/drivendataorg/water-supply-forecast-rodeo-runtime/blob/main/data_download/hindcast_test_config.yml)
	with hindcast_test_config.yml from this repo (https://github.com/progin2037/water_supply_forecast_rodeo_competition/blob/main/hindcast_test_config.yml). 
	3. Follow the instructions from the Data download section from the water-supply-forecast-rodeo-runtime repo (https://github.com/drivendataorg/water-supply-forecast-rodeo-runtime?tab=readme-ov-file#data-download)
	and download the data. It is, go to the cloned repo, install wsfr-download package: `pip install ./data_download/` and run `python -m wsfr_download bulk data_download/hindcast_test_config.yml`
	to start the download. Installing wsfr-download requirements (`pip install -r ./data_download/requirements.txt`) isn't necessary, as required libraries were already installed in step 2.
	4. If data was downloaded to water-supply-forecast-rodeo-runtime repo instead of the main repo water_supply_forecast_rodeo_competition, copy downloaded data to data/ directory from the main repo.
	5. Follow the instructions from the Requirements and installation section from Data reading, installing wsfr-read package from data_reading directory (`pip install ./data_reading/`)
	(https://github.com/drivendataorg/water-supply-forecast-rodeo-runtime/tree/main?tab=readme-ov-file#requirements-and-installation-1). Thanks to that, auxiliary library for
	reading data downloaded in the previous point could be used.
	6. Download CDS data. There are 2 options to achieve that:
		1. [Recommended] Use notebooks/CDS downloads.ipynb (https://github.com/progin2037/water_supply_forecast_rodeo_competition/blob/main/notebooks/CDS%20downloads.ipynb).
		2. Use cds_downloads.py (https://github.com/progin2037/water_supply_forecast_rodeo_competition/blob/main/cds_downloads.py). This way, it will be harder to keep track of already downloaded data.
6. Run data_processing.py.
7. Run model_params.py.
8. [OPTIONAL] Run distribution_estimates.py. Output from this script is already saved in this repository in data/distr, as running the script takes about 4-6 hours.
9. Run model_training.py.
