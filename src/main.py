# -*- coding: utf-8 -*-

# %% load libraries
import numpy as np
import pandas as pd
import sys, time

# determine the path to the source folder 
pth_to_src = 'C:/DEV/time_series_analysis/src/'


# %% load data and check it
# Read the dataframe:
df = pd.read_csv(pth_to_src+'data/train.csv',  parse_dates=['datetime'], infer_datetime_format=True) #  parse_dates=True, squeeze=True, header=0, index_col=0,
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
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot as plt

def decompose_time_series(df, share_type='y', samples=250, period=24, decomposition_model_type='additive'):
    if samples == 'all':
        #decomposing all time series timestamps
        result = seasonal_decompose(df[share_type].values, period=period, model=decomposition_model_type, extrapolate_trend='freq')
    else:
        #decomposing a sample of the time series
        result = seasonal_decompose(df[share_type].values[-samples:], period=period, model=decomposition_model_type, extrapolate_trend='freq')
        
        
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result.plot().suptitle('Seasonal Decomposition', fontsize=22)
    plt.show()	
    	

decompose_time_series(df)

# Calculate ACF and PACF up to 50 lags

# Draw Plot
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
plot_acf(df.y.tolist(), lags=50, ax=axes[0])
plot_pacf(df.y.tolist(), lags=50, ax=axes[1], method='ywm')
	
# %% Test for stationary:

# Augmented Dickey Fuller (ADF) Test
if 0:
    # null hypothesis:  time series possesses a unit root and is non-stationary. 
    from statsmodels.tsa.stattools import adfuller, kpss
    result = adfuller(df.y.values, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print('Critial Values:')
        print(f'   {key}, {value}')

# Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test
if 0:
    # null hypothesis: time series does not possess a unit root and is stationary.
    result = kpss(df.y.values, regression='c')
    print('\nKPSS Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    for key, value in result[3].items():
        print('Critial Values:')
        print(f'   {key}, {value}')	
	

# If the time series is not stationary and we should use an AR-I-MA model instead of an ARMA.

	
# %% Granger causality	
# Granger causality test is used to determine if one time series will be useful to forecast another.	
# Idea: if X causes Y, then the forecast of Y based on previous values of Y AND the previous values of X should outperform the forecast of Y based on previous values of Y alone.
# The Null hypothesis: the series in the second column, does not Granger cause the series in the first. 
from statsmodels.tsa.stattools import grangercausalitytests
df['month'] = df.index.month
gct = grangercausalitytests(df[['y', 'month']], maxlag=2)	

# If p-values are zero for all tests: So the ‘month’ indeed can be used to forecast the Y.
	
# %% function to train, predict and plot model results:
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import RandomizedSearchCV

def train_time_series_with_folds(df,  y_var = 'y', horizon=24*7, TUNE = True):
    X = df.drop(y_var, axis=1)
    y = df[y_var]
    
    #take last week of the dataset for validation
    X_train, X_test = X.iloc[:-horizon,:], X.iloc[-horizon:,:]
    y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]
    
    #create, train and do inference of the model
    if TUNE:
        # Tune hyperparameters and final model using 10-fold cross-validation with 10 parameter settings sampled from random search. Random search can cover a larger area of the paramter space with the same number of consider setting compared to e.g. grid search.
        rs = RandomizedSearchCV(LGBMRegressor(), {
               'learning_rate': [0.01, 0.1, 0.3, 0.5],
                                'max_depth': [1, 5, 15, 20, 30],
                                'num_leaves': [10, 20, 30, 100],
                                'subsample': [0.1, 0.2, 0.8, 1]
            }, 
            cv=10, 
            return_train_score=False, 
            n_iter=10
        )
        print("\nTuning hyperparameters ..")
        rs.fit(X_train, y_train)    
        print("Tuned hyperparameters :(best parameters) ",rs.best_params_)
        model = LGBMRegressor(random_state=42, **rs.best_params_)
    else:
        model = LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    #calculate MAE
    mae = np.round(mean_absolute_error(y_test, predictions), 3)    
    
    #plot Observed vs prediction for the horizon of the dataset
    fig = plt.figure(figsize=(16,8))
    plt.title(f'Observed vs Prediction - MAE {mae}', fontsize=20)
    plt.plot(y_test, color='red')
    plt.plot(pd.Series(predictions, index=y_test.index), color='green')
    plt.xlabel('Hour', fontsize=16)
    plt.ylabel(y_var, fontsize=16)
    plt.legend(labels=['Real', 'Prediction'], fontsize=16)
    plt.grid()
    plt.show()
    
    #create a dataframe with the variable importances of the model
    df_importances = pd.DataFrame({
        'feature': model.feature_name_,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    #plot variable importances of the model
    plt.title('Variable Importances', fontsize=16)
    sns.barplot(x=df_importances.importance, y=df_importances.feature, orient='h')
    plt.show()



# %%
# wo lagged variable
train_time_series_with_folds(df)


# %%
# with lagged variable
df['lagged_y'] = df['y'].shift(24*7)

#drop NaNs after feature engineering
df.dropna(how='any', axis=0, inplace=True)


train_time_series_with_folds(df)



