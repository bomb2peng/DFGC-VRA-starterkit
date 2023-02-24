import numpy as np
import pandas as pd
import scipy.io
import os
from selector import *

# ================== select features with given dimention ==========================


if __name__=='__main__':

    feats_path = r'./feats/raw'
    label_path = r'./label/train_set.csv'
    out_path = r'./feats/selected'

    algo_name = 'DFGC1st_withstd'
    num_workers = 5
    num_iters = 100
    num_feats = 280
    
    # load features and ground truth mos
    feats_file = os.path.join(feats_path, 'DFGC-train_'+algo_name+'_feats.mat')
    feats = scipy.io.loadmat(feats_file)
    feats = np.asarray(feats['feats_mat'], dtype=np.float)
    mos_df = pd.read_csv(label_path)
    mos_gt = np.array(list(mos_df['mos']), dtype=np.float)

    print('loaded feature mat shape:', feats.shape)
    
    # select features and save selected index
    _, mask = feat_sel_eval(feats, mos_gt, out_path, algo_name, num_feats, num_iters, num_workers)

    # save selected feature mat for training set
    pick_out(feats, mask, algo_name, 'train', num_feats, out_path)

    # save selected feature mat for test sets
    phase = ['1', '2', '3']
    for c in phase:
        feats_file = os.path.join(feats_path, 'DFGC-test'+c+'_'+algo_name+'_feats.mat')
        feats_test = scipy.io.loadmat(feats_file)
        feats_test = np.asarray(feats_test['feats_mat'], dtype=np.float)
        pick_out(feats_test, mask, algo_name, 'test'+c, num_feats, out_path)
