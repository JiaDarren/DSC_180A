import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import linear_model
import warnings
from sklearn.exceptions import ConvergenceWarning

from multiprocessing import Pool
from datetime import datetime
import os
import time
import SNN_utils
import feature_eval_utils as fe_util

DATA_PATH = '../../data'

def eval_feature(X, i):
    # Randomize feature
    orig_feature = X[:,i]
    X[:,i] = [1] * len(X[:,i])
    print(f"worker {os.getpid()}: {sum(X[0,:])}")
    # time.sleep(np.random.randint(10))
    # Restore orig Dataset
    X[:,i] = orig_feature
    
    # Save evaluated scores
    return 0, 0

if __name__ == '__main__':
    print("Importing Data")
    start = datetime.now()
    feature_matrix = pd.read_csv(f'{DATA_PATH}/processed/feature_matrix.csv')
    
    print("Preparing Data for Feature Evaluation")  
    
    data = np.zeros([240, 240])
    input_args = [(data, i) for i in range(240)]

    print("Setting Pool")
    with Pool() as pool:
        outputs = pool.starmap(eval_feature, input_args) 
    end = datetime.now()

    # print(f"Total time: {end - start}")
    # with open("file.txt", "w") as out_file:
    #     out_file.write(str(outputs))