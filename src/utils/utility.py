# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from statsmodels.tsa.stattools import adfuller, kpss



def decompose_time_series(df, share_type='y', samples=250, period=24, decomposition_model_type='additive', output_folder_plots = '', title1='Decomposition', SAVE_OUTPUT = 1):
    if samples == 'all':
        #decomposing all time series timestamps
        result = seasonal_decompose(df[share_type].values, period=period, model=decomposition_model_type, extrapolate_trend='freq')
    else:
        #decomposing a sample of the time series
        result = seasonal_decompose(df[share_type].values[-samples:], period=period, model=decomposition_model_type, extrapolate_trend='freq')
        
        
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result.plot().suptitle('Decomposition', fontsize=22)
    # Saving plot to pdf and png file
    if SAVE_OUTPUT:
        plt.savefig(output_folder_plots  +title1+'.pdf', dpi=100,bbox_inches="tight")
        #plt.title(title1, fontsize=20)
        plt.savefig(output_folder_plots  +title1+ '.png', dpi=100,bbox_inches="tight")    
    plt.show()	
    


def train_time_series_with_folds(df,  y_var = 'y', horizon=24*7, TUNE = True, output_folder_plots = '', title1='Prediction', title2= 'importance', SAVE_OUTPUT = 1):
    """function to tune, train, predict and plot model results"""
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
    # Saving plot to pdf and png file
    if SAVE_OUTPUT:
        plt.savefig(output_folder_plots  +title1+'.pdf', dpi=100,bbox_inches="tight")
        #plt.title(title1, fontsize=20)
        plt.savefig(output_folder_plots  +title1+ '.png', dpi=100,bbox_inches="tight")
    plt.show()
    
    #create a dataframe with the variable importances of the model
    df_importances = pd.DataFrame({
        'feature': model.feature_name_,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    #plot variable importances of the model
    plt.title('Variable Importances', fontsize=16)
    sns.barplot(x=df_importances.importance, y=df_importances.feature, orient='h')
    # Saving plot to pdf and png file
    if SAVE_OUTPUT:
        plt.savefig(output_folder_plots  +title2+'.pdf', dpi=100,bbox_inches="tight")
        #plt.title(title1, fontsize=20)
        plt.savefig(output_folder_plots  +title2+ '.png', dpi=100,bbox_inches="tight")    
    plt.show()
    
    
    
def test_for_stationary(df,  y_var = 'y'):
    """ Test for stationary.
    If the time series is not stationary and we should use an AR-I-MA model instead of an ARMA."""
    
    # Augmented Dickey Fuller (ADF) Test
    # null hypothesis:  time series possesses a unit root and is non-stationary. 
    result = adfuller(df.y.values, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print('Critial Values:')
        print(f'   {key}, {value}')

    # Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test
    # null hypothesis: time series does not possess a unit root and is stationary.
    result = kpss(df.y.values, regression='c')
    print('\nKPSS Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    for key, value in result[3].items():
        print('Critial Values:')
        print(f'   {key}, {value}')	    
    
    
    

