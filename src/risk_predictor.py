import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DATA_PATH = '../data'
MODEL_PATH = 'models'

if __name__ == '__main__':
    feature_matrix_path = f'{DATA_PATH}/processed/HOLDOUT_SNN_feature_matrix.csv'
    prediction_path = f'holdout_predictions_group1.csv'
    
    # Get SNN Data
    print("Importing Data")
    feature_matrix = pd.read_csv(feature_matrix_path)
    feature_matrix_tensor = torch.tensor(
        feature_matrix.to_numpy(), 
        dtype=torch.float32
    )

    # Define the model
    print("Setting Model")
    model = nn.Sequential(
        nn.Linear(feature_matrix.shape[1], 12),
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
    model.load_state_dict(torch.load(f'{MODEL_PATH}/nn_base_params.pt'))
    
    # Make Prediction
    print("Predicting")
    pred_prob = model(feature_matrix_tensor).detach().numpy()[:,0]

    # Save predictions
    print("Save Predictions")
    pred_df = pd.DataFrame({'prediction':pred_prob})
    pred_df.to_csv(prediction_path)