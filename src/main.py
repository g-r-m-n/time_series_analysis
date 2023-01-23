# -*- coding: utf-8 -*-
# run the following to create/update the requirements:
# !pipreqs --encoding utf-8

# %% load libraries
import numpy as np
import pandas as pd
import sys, time, os
from datetime import date

# determine the path to the source folder 
pth_to_src = 'C:/DEV/time_series_analysis/src/'

# data:
today = date.today().strftime('%Y%m%d')
# output folders:
path_to_data  = pth_to_src + 'input/'
output_folder = pth_to_src+ 'output/'+today+'/'
output_folder_plots  = output_folder+'plots/'
# create output_folders if they do not exist:
os.makedirs(output_folder,exist_ok=True)
os.makedirs(output_folder_plots,exist_ok=True)
# load utility functions
sys.path.append(pth_to_src+'/utils/')
from utility import *
# reload functions from utility
from importlib import reload
reload(sys.modules['utility'])    

np.random.seed(888) # set random seed for reproduceability

# %% load data and check it
# Read the dataframe:
df = pd.read_csv(path_to_data+'/train.csv',  parse_dates=['datetime'], infer_datetime_format=True) #  parse_dates=True, squeeze=True, header=0, index_col=0,
#drop casual and registered columns
df.drop(['casual', 'registered'], axis=1, inplace=True)

df = df.rename (columns= {'count': 'y'})
                      
# %% preprocess the data :
    
#set datetime as index
df = df.set_index(df.datetime)

#drop datetime column
df.drop('datetime', axis=1, inplace=True)

#create hour, day and month variables from datetime index
if 0:
    df['day']   = df.index.day
    df['month'] = df.index.month
    
    # Add a variable to indicate the day of the week:
    df['weekday'] = df.index.dt.day_name() 
    
    # Add a variable to indicate the hour of the day:
    df['hour'] = df.index.dt.hour
    
# sort dataframe 
if 0:
    df = df.sort_values(['user_id','session_id','timestamp','page_id'])
    # reset the index:
    df = df.reset_index()

# %% check the data
# basic descriptive statistics of the data set:
df.info()
df.describe().T

# Show distribution plots of data (from kernel density estimation)
if 0:
    import seaborn as sns
    from matplotlib import pyplot as plt
    for i in df.columns:
        ax = plt.subplot()
        sns.kdeplot(df[i], label=i)
        plt.show()
 
# create an autocorrelation plot
if 0:
    from pandas.plotting import autocorrelation_plot
    from matplotlib import pyplot as plt
    autocorrelation_plot(df.y) # (df.value.tolist())
    plt.show()
        
	
# %% Decompose series 	

    
decompose_time_series(df, output_folder_plots = output_folder_plots)

# Calculate ACF and PACF up to 50 lags

# Draw Plot
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
plot_acf(df.y.tolist(), lags=50, ax=axes[0])
plot_pacf(df.y.tolist(), lags=50, ax=axes[1], method='ywm')
	
# %% Test for stationary:
if 0:
    test_for_stationary(df,  y_var = 'y')

    

	
# %% Granger causality	
# Granger causality test is used to determine if one time series will be useful to forecast another.	
# Idea: if X causes Y, then the forecast of Y based on previous values of Y AND the previous values of X should outperform the forecast of Y based on previous values of Y alone.
# The Null hypothesis: the series in the second column, does not Granger cause the series in the first. 
from statsmodels.tsa.stattools import grangercausalitytests
df['month'] = df.index.month
gct = grangercausalitytests(df[['y', 'month']], maxlag=2)	

# If p-values are zero for all tests: So the ‘month’ indeed can be used to forecast the Y.
	



# %% tune, train, predict and plot model results using LGB:
# wo lagged variable
train_time_series_with_folds(df, SAVE_OUTPUT=0)


# %% tune, train, predict and plot model results using LGB:
# with lagged variable
df['lagged_y'] = df['y'].shift(24*7)

#drop NaNs after feature engineering
df.dropna(how='any', axis=0, inplace=True)


train_time_series_with_folds(df, output_folder_plots = output_folder_plots)



