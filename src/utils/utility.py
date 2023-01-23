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
import itertools



def decompose_time_series(df, y_var='y', samples='all', period=24, decomposition_model_type='additive', output_folder_plots = '', title1='Decomposition', SAVE_OUTPUT = 1):
    items = np.unique(df.index.get_level_values('item_name'))
    for i in items:
        inx_i = df[y_var].index.get_level_values('item_name')==i
        if samples == 'all':
            #decomposing all time series timestamps
            result = seasonal_decompose(df.loc[inx_i,y_var].values, period=period, model=decomposition_model_type, extrapolate_trend='freq')
        else:
            #decomposing a sample of the time series
            result = seasonal_decompose(df.loc[inx_i,y_var].values[-samples:], period=period, model=decomposition_model_type, extrapolate_trend='freq')
            
            
        # Plot
        plt.rcParams.update({'figure.figsize': (10,10)})
        result.plot().suptitle('Decomposition'+'_'+str(i), fontsize=22)
        # Saving plot to pdf and png file
        if SAVE_OUTPUT:
            plt.savefig(output_folder_plots  +title1+'_'+str(i)+'.pdf', dpi=100,bbox_inches="tight")
            #plt.title(title1, fontsize=20)
            plt.savefig(output_folder_plots  +title1+'_'+str(i)+ '.png', dpi=100,bbox_inches="tight")    
        plt.show()	
    


def train_time_series_with_folds(df,  y_var = 'y', horizon=14, TUNE = True, output_folder_plots = '', title1='Prediction', title2= 'importance', SAVE_OUTPUT = 1):
    """function to tune, train, predict and plot model results"""
    X = df.drop(y_var, axis=1)
    y = df[y_var]
    
    #take last week of the dataset for validation
    inx_day = np.unique(df.index.get_level_values('day'))[-horizon]
    
    X_train, X_test = X.iloc[df.index.get_level_values('day')<inx_day,:], X.iloc[df.index.get_level_values('day')>=inx_day,:]
    y_train, y_test = y.iloc[df.index.get_level_values('day')<inx_day], y.iloc[df.index.get_level_values('day')>=inx_day]
    
    #create, train and do inference of the model
    if TUNE:
        # Tune hyperparameters and final model using 10-fold cross-validation with 10 parameter settings sampled from random search. Random search can cover a larger area of the paramter space with the same number of consider setting compared to e.g. grid search.
        rs = RandomizedSearchCV(LGBMRegressor(random_state=42), { 
               'learning_rate': [0.01, 0.1, 0.3, 0.5],
                                'max_depth': [3, 5, 15, 20, 30],
                                'num_leaves': [5, 10, 20, 30],
                                'subsample': [0.1, 0.2, 0.8, 1]
            }, 
            cv=10, 
            return_train_score=False, 
            n_iter=30
        )
        print("\nTuning hyperparameters ..")
        rs.fit(X_train, y_train)    
        print("Tuned hyperparameters :(best parameters) ",rs.best_params_)
        model = LGBMRegressor(random_state=42, **rs.best_params_)
    else:
        model = LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
   
    
    #plot Observed vs prediction for the horizon of the dataset
    items = np.unique(df.index.get_level_values('item_name'))
    for i in items:
        fig = plt.figure(figsize=(16,8))
        inx_i = y_test.index.get_level_values('item_name')==i
        #calculate MAE
        mae = np.round(mean_absolute_error(y_test[inx_i], predictions[inx_i]), 3) 
        plt.title(f'Observed vs Prediction - MAE {mae}', fontsize=20)
        plt.plot(pd.Series(y_test[inx_i].values,index=y_test.index.get_level_values('day')[inx_i]), color='red')
        plt.plot(pd.Series(predictions[inx_i], index=y_test.index.get_level_values('day')[inx_i]), color='green')
        plt.xlabel('day', fontsize=16)
        plt.ylabel(y_var+':'+str(i), fontsize=16)
        plt.legend(labels=['Real', 'Prediction'], fontsize=16)
        plt.grid()
        # Saving plot to pdf and png file
        if SAVE_OUTPUT:
            plt.savefig(output_folder_plots  +title1+'_'+str(i)+'.pdf', dpi=100,bbox_inches="tight")
            #plt.title(title1, fontsize=20)
            plt.savefig(output_folder_plots  +title1+'_'+str(i)+ '.png', dpi=100,bbox_inches="tight")
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
    
    return model
    

    
    
    

