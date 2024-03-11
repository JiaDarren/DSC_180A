import pandas as pd
import re
import numpy as np

DATA_PATH = '../data'
OUTPUTS_PATH = '../src/features'

def get_top_features(include_y, feature_matrix):
    # Read scores from eval_feature.py
    # Split scores into model-specific scores
    scores = []
    for i in range(6):
        with open(f"{OUTPUTS_PATH}/feature_scores_{i}.txt", "r") as f:
            feature_scores = eval(f.readline())
            scores.append(feature_scores)
    # Get median score from all evaluations
    scores = np.median(scores, axis=0)

    # Bind feature name with associated score
    if include_y:
        feature_names = list(
            feature_matrix
            .drop(['FPF_TARGET', 'prism_consumer_id'], axis=1)
            .columns
        )
    else:
        feature_names = list(
            feature_matrix
            .drop('prism_consumer_id', axis=1)
            .columns
        )
    scores = dict(zip(feature_names, scores))

    # Sort based on score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])

    # Save top 40 features
    top_scores = [name for name,_ in sorted_scores[:40]]
    top_scores.insert(0, 'prism_consumer_id')
    if include_y: 
        top_scores.append('FPF_TARGET')
    top_score_matrix = feature_matrix[top_scores]

    # Save feature matrix
    return top_score_matrix


if __name__ == '__main__':
    include_y = False
    input_file_path = f'{DATA_PATH}/processed/HOLDOUT_feature_matrix.csv'
    output_file_path = f'{DATA_PATH}/processed/HOLDOUT_SNN_feature_matrix.csv'
    # Get feature names
    feature_matrix = pd.read_csv(input_file_path)

    top_score_matrix = get_top_features(include_y, feature_matrix)
    top_score_matrix.to_csv(output_file_path, index=False)