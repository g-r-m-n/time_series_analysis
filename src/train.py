# -*- coding: utf-8 -*-


# %% Setting

VERBOSE                     = 1 # 0 or 1. If 1. verbose mode is used and more inforation is shown, such as addtional descriptive plots, or correlations, statistical tests, etc.

USE_LOG_TRANSFORMED_RESPONSE = 1 # # 0 or 1. If 1, log-transform the response value to model values that are numerical better tractable. Add plus one to avoid a log of zero.


# %% load libraries
import numpy as np
import pandas as pd
import sys, time, os
from datetime import date
from datetime import datetime
# determine the path to the source folder 
pth_to_src = 'C:/DEV/time_series_analysis/src/'

# date of today:
today = date.today().strftime('%Y%m%d')

# input folder:
input_folder  = pth_to_src + 'input/'

# output folder:
output_folder = pth_to_src+ 'output/'+today+'/' # with date of today. This way a daily history of results is automatically stored.
output_folder_plots  = output_folder+'plots/'
output_folder_model  = output_folder+'model/'
# create output_folder if not existant:
os.makedirs(output_folder,exist_ok=True)
os.makedirs(output_folder_plots,exist_ok=True)
os.makedirs(output_folder_model,exist_ok=True)
# load utility functions
sys.path.append(pth_to_src+'/utils/')
from utility import *
# reload functions from utility
from importlib import reload
reload(sys.modules['utility'])    

np.random.seed(888) # set random seed for reproduceability

# %% load data and check it 2022-02-09
custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
# Read the dataframe:
df = pd.read_csv(input_folder+'/data.csv',  parse_dates=['day'], date_parser=custom_date_parser, index_col = 0) #  

# drop duplicate rows:
df = df.drop_duplicates()

df = df.rename (columns= {'revenue': 'y'})

# revenue is zero for zero quantity sales:
df.loc[df.sales_quantity==0,'y'] = 0
             
# check for missing values:
if VERBOSE:
    print('\nNumber and rate of missing values:')
    print(df.isna().sum(),df.isna().mean())       


# log-transform the response value to model values that are numerical better tractable.
# add plus one to avoid a log of zero:
if USE_LOG_TRANSFORMED_RESPONSE:
    df['y'] = np.log(df['y'] +1)


# %% preprocess the data :
    
#set datetime as index
df = df.set_index([df.day,df.item_name])

#drop day and item_number column
df.drop(['day','item_name'], axis=1, inplace=True) 

#drop item_number column
df.drop(['item_number'], axis=1, inplace=True)


#create  month  and week day variables from datetime index
if 1:
    df['month'] = df.index.get_level_values('day').month_name()
    df = pd.get_dummies(data=df, columns= ['month'])
    
    # Add a variable to indicate the day of the week:
    df['weekday'] = df.index.get_level_values('day').day_name() 
    df = pd.get_dummies(data=df, columns= ['weekday'])
    
                
# add discount rate variable
df['discount'] = (df.suggested_retail_price - df.purchase_price)/df.suggested_retail_price 

#drop suggested_retail_price column and use instead the discount variable
df.drop(['suggested_retail_price'], axis=1, inplace=True)


# add lagged variables
# y:
df['lagged_y']  = df['y'].groupby(level='item_name').shift(1)
df['lagged2_y'] = df['y'].groupby(level='item_name').shift(2)

# sales_quantity:
df['lagged_sales_quantity'] = df['sales_quantity'].groupby(level='item_name').shift(1)
#df['lagged_orders_quantity'] = df['orders_quantity'].groupby(level='item_name').shift(1)



#drop NaNs after feature engineering
df.dropna(how='any', axis=0, inplace=True)

#drop sales_quantity column as it is not observed at the day before.
df.drop(['sales_quantity'], axis=1, inplace=True)

# %% check the data
# basic descriptive statistics of the data set:
if VERBOSE:
    df.info()
    df.describe().T

# pearson correlations
if VERBOSE:
    c1 = np.round(df.corr(),3)
    print(c1)

# Show distribution plots of data (from kernel density estimation)
if VERBOSE and 0:
    import seaborn as sns
    from matplotlib import pyplot as plt
    from pandas.api.types import is_numeric_dtype
    for i in df.columns:
        if is_numeric_dtype(df[i]):
            ax = plt.subplot()
            sns.kdeplot(df[i], label=i)
            plt.show()
     
# create an autocorrelation plot
if VERBOSE and 0:
    from pandas.plotting import autocorrelation_plot
    from matplotlib import pyplot as plt
    autocorrelation_plot(df.y) # (df.value.tolist())
    plt.show()
        
	
# %% Decompose series 	

if VERBOSE:    
    decompose_time_series(df, output_folder_plots = output_folder_plots)


	
# %% Granger causality	
# Granger causality test is used to determine if one time series will be useful to forecast another.	
# Idea: if X causes Y, then the forecast of Y based on previous values of Y AND the previous values of X should outperform the forecast of Y based on previous values of Y alone.
# The Null hypothesis: the series in the second column, does not Granger cause the series in the first. 
if VERBOSE:
    from statsmodels.tsa.stattools import grangercausalitytests
    df['month'] = df.index.get_level_values('day').month
    gct = grangercausalitytests(df[['y', 'month']], maxlag=2)	
    
    # If p-values are zero for all tests: So the ‘month’ indeed can be used to forecast the Y.
    	


# %% tune, train, predict and plot model results using LGB:
trained_model = train_time_series_with_folds(df, output_folder_plots = output_folder_plots)


# %% save or load the trained model 
import pickle
if 1:
    pickle.dump(trained_model, open(output_folder_model+'trained_model.pkl', "wb"))
# load the trained model:
# trained_model = pickle.load(open(output_folder_model+'trained_model.pkl', "rb"))
