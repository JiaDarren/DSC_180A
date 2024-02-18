import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import linear_model
import warnings
from sklearn.exceptions import ConvergenceWarning

from multiprocessing import Pool
from datetime import datetime
from tqdm import tqdm
import SNN_utils
import feature_eval_utils as fe_util

DATA_PATH = '../../data'

def eval_feature(X, y, i, feature_names):
    print(f"Evaluating Feature {i+1}: {feature_names[i]}")
    # Randomize feature
    orig_feature = X[:,i]
    np.random.shuffle(X[:,i])

    # Create New Datasets
    X_train, X_valid, X_test, y_train, y_valid, y_test = fe_util.train_valid_test_split(X, y, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).reshape(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    # Define and Train Models
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        # Define Logistic Regression model
        lin_model = linear_model.LogisticRegression(max_iter=1000)
        # Train Logistic Regression model
        lin_model.fit(X_train, y_train)

    # Define Sequential NN model
    model = nn.Sequential(
        nn.Linear(X.shape[1], 12),
        nn.ReLU(),
        nn.Linear(12, 24),
        nn.ReLU(),
        nn.Linear(24, 24),
        nn.ReLU(),
        nn.Linear(24, 12),
        nn.ReLU(),
        nn.Linear(12, 6),
        nn.ReLU(),
        nn.Linear(6, 1),
        nn.Sigmoid()
    )
    # loss function and optimizer
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # Train Sequential NN model
    model, train_history, valid_history = SNN_utils.train_SNN(
        model, 
        loss_fn,
        optimizer,
        X_train_tensor, 
        y_train_tensor, 
        X_valid_tensor, 
        y_valid_tensor, 
        n_epochs = 200,
        batch_size = 128,
        display=False,
        file_name=f"data/model_{i}"
    )
    
    # Evaluate Logistic Rergression model
    test_pred = lin_model.predict(X_test)
    lin_reg_score = fe_util.get_metrics(test_pred, y_test, display=False)[-1]
    
    # Evaluate Sequential NN model    
    # Find optimal threshold
    valid_pred_prob = model(X_valid_tensor).detach().numpy()[:,0]
    threshold = SNN_utils.find_best_prob(valid_pred_prob, y_valid, display=False)
    # Find test roc_auc score
    test_pred_prob = model(X_test_tensor).detach().numpy()[:,0]
    test_pred = test_pred_prob > threshold
    SNN_score = fe_util.get_metrics(test_pred, y_test, display=False)[-1]

    # Restore orig Dataset
    X[:,i] = orig_feature

    # Save evaluated scores
    return lin_reg_score, SNN_score

if __name__ == '__main__':
    print("Importing Data")
    start = datetime.now()
    feature_matrix = pd.read_csv(f'{DATA_PATH}/processed/feature_matrix.csv')
    
    print("Preparing Data for Feature Evaluation")  
    X = feature_matrix.iloc[:,1:].to_numpy()
    y = feature_matrix.iloc[:,0].to_numpy()
    num_features = X.shape[1]
    feature_names = dict(zip(
        np.arange(0, num_features),
        feature_matrix.iloc[:,1:].columns
    ))

    input_args = []
    for i in tqdm(range(num_features)):
        input_args.append((X.copy(), y, i, feature_names))

    print("Setting Pool")
    with Pool() as pool:
        outputs = pool.starmap(eval_feature, input_args) 
    end = datetime.now()

    print(f"Total time: {end - start}")
    with open("file.txt", "w") as out_file:
        out_file.write(str(outputs))