# water_supply_forecast_rodeo_competition

## Introduction
The repository contains a full Python solution for the Water Supply Forecast Rodeo (https://www.drivendata.org/competitions/group/reclamation-water-supply-forecast/) competition for its Forecast Stage.
The competition aims at predicting water supply for 26 western US hydrologic sites. Forecasts are made for different quantiles - 0.1, 0.5 and 0.9. Additionally, the predictions are made for different
issue dates. For most hydrologic sites, the predictions are made for the volume of water flow between April and July, but the forecasts are issued in different months (4 weeks from January, February, March, April,
May, June, July).

The solution makes predictions based on the approach of creating LightGBM models for different months and averaging their results with distribution estimates from historical data.

The aim of the first stage of the competition (Hindcast Stage, hidcast_stage branch, https://www.drivendata.org/competitions/257/reclamation-water-supply-forecast-hindcast/) was
to make predictions for historical data (2005, 2007, 2009, 2011, 2013, 2015, 2017, 2019, 2021, 2023).

Second one (Forecast Stage, forecast_stage branch, https://www.drivendata.org/competitions/259/reclamation-water-supply-forecast/) is a real-time stage. Predictions are made on the competition
server for water volume forecasts between April (March for one hydrologic site) and July (June for one hydrologic site) 2024.

In the third stage (Final Prize Stage, main branch, https://www.drivendata.org/competitions/262/reclamation-water-supply-forecast-final/) the objective was to make predictions using LOOCV
(Leave-One-Out Cross Validation) with 20 folds where one of the years between 2004-2023 is a test set. It is different from the other stages in the way that using this fold CV year data for
creating aggregations is prohibited, so each fold's data has to be processed separately.

## Content
1. Scripts to run to create and test models
	1. data_processing.py - it processes the data and saves a DataFrame to be used in model_training.py.
	2. model_params.py - contains model hyperparameters and features for different months to be used in model_training.py and get_predictions.py. It saves these parameters as files for simplicity.
	3. distribution_estimates.py - it fits data for each hydrologic site to different distributions, optimizes parameters for the distribution and selects distributions with the best fit to data. 
	4. model_training.py - 	train and saves models. There are different parameters to choose from for this script. By default, RUN_CV and RUN_TRAINING are set to True, whereas
	RUN_HYPERPARAMS_TUNING is set to False. Models' hyperparameters were already read from model_params.py but creating hyperparameters by yourself is also supported to check
	if hardcoded hyperparameters are indeed the output of running the function. Keep in mind that hyperparameters optimization takes long (20-50 hours).
	5. get_predictions.py - it runs trained models on the test data and saves the results. Forecast Stage doesn't use a static test set	but it is still helpful to have a working prediction pipeline.
2. Auxiliary scripts
	1. utils.py - general utilities, functions/list/dictionary that could be used for different tasks.
	2. feature_engineering.py - functions to facilitate feature engineering.
	3. train_utils.py - functions dedicated for model training.
	4. cv_and_hyperparams_opt.py - functions with main cv/hyperparameters tuning logic.
3. Models from repo (models/)
	1. Contains models trained for each month (1 (Jan), 2 (Feb), 3 (Mar), 4 (Apr), 5 (May), 6 (Jun), 7 (Jul)) and quantile (0.1/0.5/0.9) that were used in the Forecast Stage of the competition.
	Running this repo code should result in the same models.
4. Results from repo (results/)
	1. Contains hyperparameters tuning results for the Forecast Stage of the competition, one per month (different quantiles were optimized together). It is also a directory where train results
	will be appended after running prediction pipeline from get_predictions.py.
3. Inference (inference/)
	1. Inference code used for the Forecast Stage on the competition server. It is similar to the main code, adapting to the code submition format
	(https://www.drivendata.org/competitions/257/reclamation-water-supply-forecast-hindcast/page/809/).
4. Notebooks (notebooks/)
	1. Additional analyses.
## How to run
The solution was created using Python version 3.10.13.

*Keep in mind that results won't be exactly the same as those from models/ repo directory when downloading data again, as some of the datasets could be updated (it happened with USGS streamflow).*

1. Clone this repo.
2. Install packages from requirements (`pip install -r requirements.txt`).
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
6. Run data_processing.py.
7. Run model_params.py.
8. Run distribution_estimates.py.
9. Run model_training.py.
10. Run get_predictions.py.
