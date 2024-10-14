# Import necessary libraries
from Functions import *
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.model_selection import train_test_split

# Main processing function
def main(file_path, train_ratio=0.8, seed=10, variance_threshold=0.005, significance_level=0.05, correlation_threshold=0.95):
    # Load and split data
    train_data, test_data = load_and_split_data(file_path, train_ratio, seed)
    train_labels = train_data.iloc[:, -2]
    train_features = train_data.iloc[:, :-2]
    test_features = test_data.iloc[:, :-2]

    # Apply variance threshold
    train_features_filtered = apply_variance_threshold(train_features)

    # Remove duplicate features
    train_features_no_duplicates = remove_duplicate_features(train_features_filtered)

    # Normalize and apply second variance threshold
    train_features_normalized = normalize_and_apply_variance_filter(train_features_no_duplicates, threshold=variance_threshold)

    # Standardize data
    train_features_standardized = standardize_data(train_features_normalized)

    # Select significant features
    train_features_selected = select_k_best_features(train_features_standardized, train_labels, significance_level=significance_level)

    # Remove highly correlated features
    final_train_features = remove_high_correlation_features(train_features_selected, threshold=correlation_threshold)

    # Prepare final training and test data
    final_test_features = test_features[final_train_features.columns]
    standardized_test_features = standardize_data(final_test_features)
    
    print("Final Standardized Training Data Shape:", final_train_features.shape)
    print("Final Standardized Test Data Shape:", standardized_test_features.shape)
    print("Final selected features:", final_train_features.columns.tolist())


# Run the main process
# "data_MMX_s_tt.csv" and "data_MMX_s_ds.csv" are results from feature buliding and data augmentation

file_path = "data_MMX_s_ds.csv"
main(file_path, train_ratio=0.8, seed=10, variance_threshold=0.005, significance_level=0.05, correlation_threshold=0.95)

file_path = "data_MMX_s_tt.csv"
main(file_path, train_ratio=0.8, seed=20, variance_threshold=0.01, significance_level=0.05, correlation_threshold=0.95)
