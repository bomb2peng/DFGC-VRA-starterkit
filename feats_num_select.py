import numpy as np
import pandas as pd
import scipy.io
import os
from selector import *

# ================== select dimention of features ==========================


if __name__=='__main__':

    feats_path = r'./feats/raw'
    label_path = r'./label/train_set.csv'
    out_path = r'./eval'

    algo_name = 'DFGC1st_withstd'
    num_workers = 5
    num_iters = 10
    num_feats = list(range(20, 520, 20))

    # load features and ground truth mos
    feats_file = os.path.join(feats_path, 'DFGC-train_'+algo_name+'_feats.mat')
    feats = scipy.io.loadmat(feats_file)
    feats = np.asarray(feats['feats_mat'], dtype=np.float)
    mos_df = pd.read_csv(label_path)
    mos_gt = np.array(list(mos_df['mos']), dtype=np.float)

    print('loaded feature mat shape:', feats.shape)

    param_selection_fnum_svr(feats, mos_gt, algo_name, out_path, num_feats, num_iters, num_workers)
