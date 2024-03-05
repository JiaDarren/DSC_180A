import pandas as pd
import re
import numpy as np

DATA_PATH = '../data'
OUTPUTS_PATH = '../src/features'

def get_top_features(include_y, input_file_path, output_file_path):
    # Read scores from eval_feature.py
    # Split scores into model-specific scores
    scores = []
    for i in range(6):
        with open(f"{OUTPUTS_PATH}/feature_scores_{i}.txt", "r") as f:
            feature_scores = eval(f.readline())
            scores.append(feature_scores)
    # Get median score from all evaluations
    scores = np.median(scores, axis=0)

    # Get feature names
    feature_matrix = pd.read_csv(input_file_path)
    if include_y:
        feature_names = list(feature_matrix.iloc[:,1:].columns)
    else:
        feature_names = list(feature_matrix.columns)

    # Bind feature name with associated score
    scores = dict(zip(feature_names, scores))

    # Sort based on score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])

    # Save top 40 features
    top_scores = [name for name,_ in sorted_scores[:40]]

    # Get top 40 features
    feature_matrix = feature_matrix[top_scores]

    if include_y:
        # Append target to beginning of feature matrix
        feature_matrix.insert(
            0, 
            column='FPF_TARGET',
            value=feature_matrix['FPF_TARGET']
        )
    # Save feature matrix
    feature_matrix.to_csv(output_file_path, index=False)


if __name__ == '__main__':
    include_y = False
    input_file_path = f'{DATA_PATH}/processed/HOLDOUT_feature_matrix.csv'
    output_file_path = f'{DATA_PATH}/processed/HOLDOUT_SNN_feature_matrix.csv'

    get_top_features(include_y, input_file_path, output_file_path)