# water_supply_forecast_rodeo_competition

## Introduction
The repository contains a full Python solution for the Water Supply Forecast Rodeo (https://www.drivendata.org/competitions/group/reclamation-water-supply-forecast/) competition for its Hindcast Stage.
The competition aims at predicting water supply for 26 western US hydrologic sites. Forecasts are made for different quantiles - 0.1, 0.5 and 0.9. Additionally, the predictions are made for different
issue dates. For most hydrologic sites, the predictions are made for the volume of water flow between April and July, but the forecasts are issued in different months (4 weeks from January, February, March, April,
May, June, July).

The solution makes predictions based on the approach of creating LightGBM models for different months and averaging their results with distribution estimates from historical data.

## Content
1. Scripts to run to create and test models
	1. data_processing.py - it processes the data and saves a DataFrame to be used in model_training.py.
	2. model_params.py - contains model hyperparameters and features for different months to be used in model_training.py and get_predictions.py. It saves these parameters as files for simplicity.
	3. distribution_estimates.py - it fits data for each hydrologic site to different distributions, optimizes parameters for the distribution and selects distributions with the best fit to data. 
	4. model_training.py - 	train and saves models. There are different parameters to choose from for this script. By default, RUN_CV and RUN_TRAINING are set to True, whereas
	RUN_HYPERPARAMS_TUNING is set to False. Models' hyperparameters were already read from model_params.py but creating hyperparameters by yourself is also supported to check
	if hardcoded hyperparameters are indeed the output of running the function. Keep in mind that hyperparameters optimization takes long (20-50 hours).
	5. get_predictions.py - it runs trained models on the test data and saves the results.
2. Auxiliary scripts
	1. utils.py - general utilities, functions/list/dictionary that could be used for different tasks.
	2. feature_engineering.py - functions to facilitate feature engineering.
	3. train_utils.py - functions dedicated for model training.
	4. cv_and_hyperparams_opt.py - functions with main cv/hyperparameters tuning logic.
3. Models from repo (models/)
	1. Contains models trained for each month (1 (Jan), 2 (Feb), 3 (Mar), 4 (Apr), 5 (May), 6 (Jun), 7 (Jul)) and quantile (0.1/0.5/0.9) that were used in the Hindcast Stage of the competition.
	Running this repo code should result in the same models.
4. Results from repo (results/)
	1. Contains hyperparameters tuning results for the Hindcast Stage of the competition, one per month (different quantiles were optimized together). It is also a directory where train results
	will be appended after running prediction pipeline from get_predictions.py.
3. Inference (inference/)
	1. Inference code used for the Hindcast Stage on the competition server. It is similar to the main code, adapting to the code submition format
	(https://www.drivendata.org/competitions/257/reclamation-water-supply-forecast-hindcast/page/809/).
4. Notebooks (notebooks/)
	1. Additional analyses.
## How to run
The solution was created using Python version 3.10.13.

1. Clone this repo.
2. Create data/ directory within the repo. All data should be downloaded to this directory.
3. Download data from the competition website https://www.drivendata.org/competitions/254/reclamation-water-supply-forecast-dev/data/. 
4. Download additional data from the competition repo https://github.com/drivendataorg/water-supply-forecast-rodeo-runtime.
	1. Clone the repo.
	2. Replace hindcast_test_config.yml (https://github.com/drivendataorg/water-supply-forecast-rodeo-runtime/blob/main/data_download/hindcast_test_config.yml)
	with hindcast_test_config.yml from this repo (https://github.com/progin2037/water_supply_forecast_rodeo_competition/blob/main/hindcast_test_config.yml). 
	3. Follow the instructions from the Data download section from the water-supply-forecast-rodeo-runtime repo (https://github.com/drivendataorg/water-supply-forecast-rodeo-runtime?tab=readme-ov-file#data-download)
	and download the data.
		1. Data should be downloaded to the data/ directory from this repo.
5. Run data_processing.py.
6. Run model_params.py.
7. Run distribution_estimates.py.
8. Run model_training.py.
9. Run get_predictions.py.
