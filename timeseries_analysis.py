#%% importing and preparing data
''' IMPORTING DATA '''
import pandas as pd
# imports pandas module for working with dataframes

gt_data = pd.read_csv("GlobalTemperatures.csv", index_col = 0)
# sets gt_data as the data in GlobalTemperatures csv file
# with the first column set as the index

''' PREPARING DATA '''
gt_data1 = gt_data[(len(gt_data) - 1200):len(gt_data)]
# extracts the last 1200 observation which is the data
# of interest

GT_data = pd.Series(gt_data1["LandAndOceanAverageTemperature"])
# creates a pandas series containing only data for
# average global land and ocean temperature

GT_data.index = pd.to_datetime(GT_data.index)
# sets the index of GT_data to a datetime type variable

#%% visualising data
''' VISUALISING TIME SERIES AS LINE PLOT '''
import matplotlib.pyplot as plt
# imports the matplotlib.pyplot module for plotting data

plt.figure(figsize = (15, 5))
# creates a larger than usual figure to display the data
GT_data.plot()
# plots the average land and ocean temperature as a lineplot
plt.xlabel("Date")
# sets the x label as Date
plt.ylabel("Temperature (deg. celcius)")
# sets the y label as Temperature (deg. celcius)
plt.title("Global Average Land and Sea Temperature")
# gives the plot an appropriate title
plt.show()
# displays the plot
'''
In this plot, seasonality is clearly visible. The
seasonal period here is likely yearly as temperatures
vary throughout the year
'''

''' VISUALISING DECOMPOSITION OF TIME SERIES '''
from statsmodels.tsa.seasonal import seasonal_decompose
# imports the seasonal_decompose class from the
# statsmodel.tsa.seasonal module for decomposing
# time series

snl_dcp = seasonal_decompose(GT_data, model = "additive",
                             period = 12)
# decompose the time series into an additive model
# timeseries = trend + seasonal + residual components
# seasonal component here is 12, as data is monthly
# and a yearly seasonal trend is expected

snl_dcp.plot()
# plots the time series decomposed into a trend, seasonal
# and residual component
plt.show()
# displays the plot

plt.figure(figsize = (15, 5))
# creates a larger than usual figure to display the data
snl_dcp.seasonal.plot()
# plots only the seasonal component of the timer series
# decomposition
plt.show()
# displays the plot

'''
The above is a quick way to visualise the potential
trend and seasonal components of a time series. Alot
of the variation in the data can potentially be
explained by the seasonal component meaning any 
forecasting method will have to account for seasonality
'''
#%% stationarity, ACF and PACF plots
from statsmodels.tsa.stattools import adfuller
# imports the adfuller functions from the
# statsmodels.tsa.stattools module for test whether
# a time series is stationary
import numpy as np
# imports the numpy module for working with arrays

test = adfuller(GT_data)
print('Augmneted Dickey_fuller Statistic: %f' % test[0])
print('p-value: %f' % test[1])
# tests the GT_data to check if the data is
# stationary and prints the output

data_diff1 = np.array([])
# creates an empty array for the data differenced by
# a lag of 1
for i in range(1, (len(GT_data) - 1)):
    diff_val = GT_data[i] - GT_data[i - 1]
    data_diff1 = np.append(data_diff1, diff_val)
# creates an array of the values in GT-data
# differenced by a lag of 1

test1 = adfuller(data_diff1)
print('Augmneted Dickey_fuller Statistic: %f' % test1[0])
print('p-value: %f' % test1[1])
# tests the differenced GT_data to check if the data is
# stationary and prints the output

'''
differencing the data with a lag of 1 is enough to
make the data stationary
'''

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# imports the plot_acf and plot_pacf functions
# to view acf and pacf plot for a time series

import matplotlib.pyplot as plt
# imports the matplotlib.pyplot module for plotting
plot_acf(data_diff1) 
# plots an acf plot for the difference time series
plt.title("ACF plot for the differenced time series")
# gives the ACF plot an appropriate title
plt.show()
# displays the plot

plot_pacf(data_diff1)
# plots the pacf plot for the difference timeseries
plt.title("PACF plot for the differenced time series")
# gives the PACF plot an appropriate title
plt.show()
# displays the plot
    

#%% time series forecasting with SARIMAX
''' PREPARING DATA '''
train_data = GT_data[0:(len(GT_data) - 36)]
# creates a training data set using all data in GT_data
# except for the last 3 years of data

test_data = GT_data[(len(GT_data) - 36):len(GT_data)]
# uses the last 3 years of data in GT_data as a test
# data set

''' TRAINING MODEL '''
from statsmodels.tsa.statespace.sarimax import SARIMAX
# imports the SARIMAX class from the
# statsmodels.tsa.statesapce.sarimax module for using
# ARIMA type time series forecasting

sarima_model = SARIMAX(train_data, order = (3, 1, 1), 
                       seasonal_order = (1, 1, 1, 12))
# creates a moment for the SARIMAX class to build the
# time series forecasting model
'''
in the above model, parameters were chosen by
minimising the AIC statistic
'''

model_fit = sarima_model.fit()
# fits the training data to the SARIMA model

''' GENERATING PREDICTIONS AND EVALUATING MODEL '''
test_prdct = model_fit.forecast(36)
# uses the SARIMA model to predict values for the next
# 36 months - the same 36 months as in the test data

from sklearn.metrics import mean_absolute_percentage_error
# imports the mean_absolute_percentage_error function
# from the sklearn.metrics module
from sklearn.metrics import mean_absolute_error, mean_squared_error
# imports the mean_absolute_error and mean_squared_error 
# functions from the sklearn.metrics module
from math import sqrt
# imports the sqrt functions from the math module to
# find the square root of a number

sarima_MAPE = mean_absolute_percentage_error(test_data, test_prdct)
# calculates the mean absolute percentage error
# for the sarima model predictions for the test data
sarima_MAE = mean_absolute_error(test_data, test_prdct)
# calculates the mean absolute error for the SARIMA
# model predictions for the test data
sarima_MSE = mean_squared_error(test_data, test_prdct)
# calculates the mean_squared_error for the SARIMA
# model predictions for the test data
sarima_RMSE = sqrt(sarima_MSE)
# calculates the root mean squared error for the SARIMA
# model predictions for the test data

''' TESTING RESIDUALS '''
from statsmodels.stats.diagnostic import acorr_ljungbox
# imports the acorr_ljungbox function from the
# statsmodels.stats.diagnostic module for testing
# if the residuals of the model are independently
# distributed

sarima_ljungbox = acorr_ljungbox(model_fit.resid)
# tests the residuals of the timeseries using the
#ljungbox test to see if their independently
# distributed

''' VISUALISING THE MODEL '''
import matplotlib.pyplot as plt
# imports the matplotlib.pyplot module for plotting

plt.figure(figsize = (15, 5))
# creates a larger than usual figure
train_data.plot()
# plots the training data as a line plot
model_fit.predict()[1:len(model_fit.predict())].plot()
# plots predictions for the test data using the
# SARIMA model to see how well the model fits the data
plt.xlabel("Date")
plt.ylabel("Temperature (deg. celcius)")
plt.title("Comparing SARIMA model to the training data")
# gives the plot appropriate x and y axis labels
# and a title
plt.show()
# displays the plot

test_data.plot()
# plots the test data as a line plot
test_prdct.plot()
# plots the predicted values for the test 
# data as a line plot
plt.xlabel("Date")
plt.ylabel("Temperature (deg. celcius)")
plt.title("Comparing SARIMA predicted values to the test data")
# gives the plot appropriate x and y axis labels
# and a title
plt.show()
# displays the plot

#%% time series forecasting with Holt-Winters Seasonal Smoothing
''' PREPARING DATA '''
train_data = GT_data[0:(len(GT_data) - 36)]
# creates a training data set using all data in GT_data
# except for the last 3 years of data

test_data = GT_data[(len(GT_data) - 36):len(GT_data)]
# uses the last 3 years of data in GT_data as a test
# data set

''' BUILDING MODEL '''
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# imports the ExponentialSmoothing class from the
# statsmodels.tsa.holtwinters module for time series
# forecasting

hw_model = ExponentialSmoothing(train_data, trend = "add",
                                seasonal = "add",
                                seasonal_periods = 12)
# creates a moment of the ExponentialSnoothing class
# with the training data, fitting an additive model
# with a seasonal component of size 12

hw_fit = hw_model.fit()
# fits the training data to the Holt-Winters Seasonal
# Smoothing module

''' TEST DATA PREDICTIONS AND EVALUATION '''
test_prdct = hw_fit.forecast(36)
# uses the Holt-Winters Seasonal Smoothing model to
# forecast global average land and ocean temperature
# for next 36 months - same 36 months as in test_data

from sklearn.metrics import mean_absolute_percentage_error
# imports the mean_absolute_percentage_error function
# from the sklearn.metrics module
from sklearn.metrics import mean_absolute_error, mean_squared_error
# imports the mean_absolute_error and mean_squared_error 
# functions from the sklearn.metrics module
from math import sqrt
# imports the sqrt functions from the math module to
# find the square root of a number

hw_MAPE = mean_absolute_percentage_error(test_data, test_prdct)
# calculates the mean absolute percentage error
# for the HWSS model predictions for the test data
hw_MAE = mean_absolute_error(test_data, test_prdct)
# calculates the mean absolute error for the HWSS
# model predictions for the test data
hw_MSE = mean_squared_error(test_data, test_prdct)
# calculates the mean_squared_error for the HWSS
# model predictions for the test data
hw_RMSE = sqrt(hw_MSE)
# calculates the root mean squared error for the HWSS
# model predictions for the test data

''' TESTING RESIDUALS '''
from statsmodels.stats.diagnostic import acorr_ljungbox
# imports the acorr_ljungbox function from the
# statsmodels.stats.diagnostic module for testing
# if the residuals of the model are independently
# distributed

hw_ljungbox = acorr_ljungbox(hw_fit.resid)
# tests the residuals of the timeseries using the
# ljungbox test to see if their independently
# distributed

''' VISUALISING MODEL '''
train_prdct = hw_fit.predict(start = train_data.index[0],
                             end = train_data.index[len(train_data) - 1])
# generates predictions for the training data using 
# the HWSS model

import matplotlib.pyplot as plt
# imports the matplotlib.pyplot module for plotting

plt.figure(figsize = (15, 5))
# creates a larger than usual figure
train_data.plot()
# plots the training data as a line plot
train_prdct.plot()
# plots predictions for the test data using the
# HWSS model to see how well the model fits the data
plt.xlabel("Date")
plt.ylabel("Temperature (deg. celcius)")
plt.title("Comparing Holt-Winters Seasonal Smoothing model to the training data")
# gives the plot appropriate x and y axis labels
# and a title
plt.show()
# displays the plot

test_data.plot()
# plots the test data as a line plot
test_prdct.plot()
# plots the predicted values for the test 
# data as a line plot
plt.xlabel("Date")
plt.ylabel("Temperature (deg. celcius)")
plt.title("Comparing Holt-Winter Seasonal Smoothing predicted values to the test data")
# gives the plot appropriate x and y axis labels
# and a title
plt.show()
# displays the plot










