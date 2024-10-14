# This code is for demonstration purposes only, showcasing the process of recursive feature elimination.
# The feature elimination process was completed on a supercomputer.
# Original data and final results can be obtained from the author.

# Import necessary libraries
from Functions import *
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np
import sys

# Main execution starts here
if __name__ == "__main__":
    # Load and shuffle data
    # "data_MMX_s_ds.csv" is the feature-engineered dataset for model training
    file_path = "data_MMX_s_ds.csv"

    # Split data into training and testing sets
    train_data, test_data = load_and_split_data(file_path, train_ratio=0.8, seed=10)

    # Separate features and labels
    train_features, train_labels = separate_features_and_labels(train_data)
    test_features, test_labels = separate_features_and_labels(test_data)

    # Feature selection
    # columns_selected are the features selected from prior feature selection results
    columns_selected = ['Comp_L3Norm', 'mean_Number', 'maxdiff_Number', 'max_Number',
                        'min_Number', 'dev_MendeleevNumber', 'maxdiff_MeltingT', 'dev_MeltingT',
                        'most_MeltingT', 'maxdiff_Electronegativity', 'dev_Electronegativity',
                        'mean_NpValence', 'dev_NValance', 'max_NValance', 'maxdiff_NpUnfilled',
                        'mean_NdUnfilled', 'dev_NUnfilled', 'most_NUnfilled', 'dev_GSvolume_pa',
                        'mean_GSmagmom', 'dev_GSmagmom', 'maxdiff_SpaceGroupNumber',
                        'dev_SpaceGroupNumber', 'frac_pValence']

    # Train data with selected features
    train_features_selected = train_data[columns_selected]

    # Normalize training data
    train_stats = train_features_selected.describe().transpose()
    train_features_normalized = normalize_features(train_features_selected, train_stats)

    # Remove specific features
    # columns_to_remove are features removed by prior recursive feature elimination
    columns_to_remove = ["dev_NValance", "dev_GSmagmom", "most_NUnfilled", "most_MeltingT", "maxdiff_Number"]
    remaining_columns = [col for col in columns_selected if col not in columns_to_remove]

    # Select a feature to remove in the current iteration based on input argument
    j = int(sys.argv[1])
    feature_to_remove = remaining_columns[j]

    # Update training data without the selected feature
    updated_columns = [col for col in train_features_normalized.columns if col not in columns_to_remove and col != feature_to_remove]
    train_features_final = train_features_normalized[updated_columns]

    # Define hyperparameter grid for grid search
    max_depth = range(7, 16, 1)
    max_features = range(1, 5, 1)
    n_estimators = range(3600, 4600, 5)
    params_dict = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
    }

    # Initialize RandomForestRegressor
    random_forest_model = RandomForestRegressor(n_jobs=-1, random_state=0)

    # Perform grid search with cross-validation
    grid_search_cv = perform_grid_search(random_forest_model, params_dict, train_features_final, train_labels)

    # Get the best estimator
    best_rf_model = grid_search_cv.best_estimator_

    # Evaluate model with cross-validation
    r2_score_mean, mae_score_mean, rmse_score_mean, error_ratio = evaluate_model(best_rf_model, train_features_final, train_labels)

    # Compile scores
    de = 1
    feature_index = [j]
    removed_feature = [feature_to_remove]
    scores = np.c_[de, feature_index, removed_feature, r2_score_mean, mae_score_mean, rmse_score_mean, error_ratio]
    columns_scores = ["de", "feature_index", "removed_feature", "r2", "mae", "rmse", "error_ratio"]
    scores_df = pd.DataFrame(scores, columns=columns_scores)
    best_params_df = pd.DataFrame([grid_search_cv.best_params_])
    final_score_df = pd.concat([scores_df, best_params_df], axis=1)

    # Save the scores to a CSV file
    output_path = f"score-{j}.csv"
    final_score_df.to_csv(output_path, index=False)

    print("Computation complete. Results saved to:", output_path)