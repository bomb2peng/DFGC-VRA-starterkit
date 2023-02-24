# -*- coding: utf-8 -*-
from cmath import phase
import os
from tkinter.font import names
import warnings
import time
import pandas as pd
import scipy.stats
import scipy.io
from scipy.optimize import curve_fit
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
# ignore all warnings
warnings.filterwarnings("ignore")

def read_mat(mat_path):
  mat = scipy.io.loadmat(mat_path)
  mat = np.asarray(mat['feats_mat'], dtype=np.float)
  mat[np.isnan(mat)] = 0
  mat[np.isinf(mat)] = 0
  return mat


def compute_metrics(y_pred, y):

  SRCC = scipy.stats.spearmanr(y, y_pred)[0]
  PLCC = scipy.stats.pearsonr(y, y_pred)[0]
  RMSE = np.sqrt(mean_squared_error(y, y_pred))
  return [SRCC, PLCC, RMSE]

def formatted_print(snapshot, phase, params, duration):
  print('======================================================')
  
  if not params == None:
    print('params: ', params)
  
  print('SRCC-'+phase+':', snapshot[0])
  print('PLCC-'+phase+':', snapshot[1])
  print('RMSE-'+phase+':', snapshot[2])
  print('======================================================')
  
  if not duration == None:
    print(' -- ' + str(duration) + ' seconds elapsed...\n\n')

def train_single_epoch(feats, mos, param_grid, rnd_seed):
    t_start = time.time()
    
    # grid search for parameter
    grid = RandomizedSearchCV(SVR(), param_grid, cv=3, n_jobs=-1, random_state=rnd_seed)
    scaler = MinMaxScaler().fit(feats)
    feats = scaler.transform(feats)
    
    grid.fit(feats, mos)
    best_params = grid.best_params_

    # retrain model with the best parameters
    regressor = SVR(C=best_params['C'], gamma=best_params['gamma'])
    regressor.fit(feats, mos)
    
    # predict
    mos_pred = regressor.predict(feats)
    # evaluate train acc
    metrics = compute_metrics(mos_pred, mos)

    t_end = time.time()
    formatted_print(metrics, phase='train', params=best_params, duration=(t_end - t_start))

    return regressor

def test_single_epoch(feats_test, regressor, output_pred=True):

  mos_test_pred = regressor.predict(feats_test)
  
  if output_pred:
    return mos_test_pred


if __name__=='__main__':

    
    feature_path = r'./feats/selected' 
    label_path = r'./label'
    out_path = r'./pred'
    algo_name = 'DFGC1st_withstd_feats360'

    # load training sets
    feats_train = read_mat(os.path.join(feature_path, 'DFGC-train_'+algo_name+'.mat'))

    df_train = pd.read_csv(os.path.join(label_path, 'train_set.csv'), skiprows=[])
    names_train = list(df_train['file'])
    mos_train = np.array(list(df_train['mos']), dtype=np.float)

    # load test sets
    phase = ['1', '2', '3']
    feats_test = []
    names_test = []
    for p in phase:
      feats = read_mat(os.path.join(feature_path, 'DFGC-test'+p+'_'+algo_name+'.mat'))
      df_test = pd.read_csv(os.path.join(label_path, 'test_set'+p+'.txt'), sep=',', names=['file'])
      feats_test.append(feats)
      names_test.append(list(df_test['file']))

    # train SVR model
    # param_grid and rnd_seed are used when grid searching for the best hyper parameters of SVR
    # you may modify them for a better result
    param_grid =  {'C': np.logspace(1, 10, 10, base=2), 'gamma': np.logspace(-8, 1, 10, base=2)}
    rnd_seed = 42
    model = train_single_epoch(feats_train, mos_train, param_grid, rnd_seed)
      
    # predict and save
    for p in phase:
      # set output_pred=True to retun prediction results
      pred_test = test_single_epoch(feats_test[int(p)-1], regressor=model, output_pred=True)
      out_df = pd.DataFrame({'file':names_test[int(p)-1], 'pred_mos':pred_test})
      out_df.to_csv(os.path.join(out_path, 'DFGC-test'+p+'_'+algo_name+'_pred.txt'), index=None, header=None)

