# temperature-data-forecast
Here we fit Seasonal ARIMA and Holt-Winters Seasonal Smoothing models to a dataset, downloaded from https://data.world/data-society/global-climate-change-data, to model global monthly average land and ocean temperatures from January 1916 to December 2015.

We use the data from the last 3 years (36 months) in the dataset as a test set to evaluate the performance of the models. The Seasonal ARIMA model acheives lower mean absolute percentage error, mean absolute error, mean squared error and root mean squared error scores than the Holt-Winters Seasonal Smoothing model. Despite this, when testing the autocorrelation of the residuals of both models we see that there is strong evidence to suggest the residuals are not independently distributed, suggesting both models could be improved further.
