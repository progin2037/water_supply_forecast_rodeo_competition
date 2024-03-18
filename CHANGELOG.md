The changelog was created to keep track of most important changes between different stages of the competition (Hindcast Stage, Forecast Stage, Final Prize Stage).

# Changes between Forecast Stage and Final Prize Stage
* added additional clipping with nat_flow_sum since April
* added pdsi_prev, removed nat_flow_11_to_10_ratio features
* removed outliers removal from full data processing (kept for distribution estimates and min_max_site_id_dict_final)
* added min_max_site_id_dict_final - min/max values for site_id without a year from the given LOOCV fold (20 DataFrames in a dictionary, each for one LOOCV year)
* added fitted distributions to data/distr directory
* added distribution fitting separately for all LOOCV years, amendments added to distribution estimation pipeline
* corrected CV results (added less weight for July)

# Changes between Hindcast Stage and Forecast Stage
* added 20-fold LOOCV with one changing year
* changed settings for hyperparams optimization and model training
* added interval coverage
* updated features to train
* updated distribution estimates based on full available data (with outliers removal)
* added new competition data that include Hindcast test years with their observed values
* changed DISTR_PERC to distr_perc_dict to assign different distribution weights for different months
* changed YEAR_SINCE from 1930 to 1965
* added discharge_cfs_mean_since_Oct_std
