
# Load and split data function
def load_and_split_data(file_path, train_ratio=0.8, seed=10):
    data = pd.read_csv(file_path).sample(frac=1, random_state=10)
    train_data = data.sample(frac=train_ratio, random_state=seed)
    test_data = data.drop(train_data.index)
    return train_data, test_data

# Variance threshold filter function
def apply_variance_threshold(df, threshold=0):
    vt = VarianceThreshold(threshold=threshold)
    df_filtered = pd.DataFrame(vt.fit_transform(df), columns=df.columns[vt.get_support()])
    print("After Variance Threshold:", df_filtered.shape[1], "features")
    return df_filtered

# Remove duplicate features function
def remove_duplicate_features(df):
    duplicates = []
    for i in range(df.shape[1]):
        for j in range(i + 1, df.shape[1]):
            if df.iloc[:, i].equals(df.iloc[:, j]):
                duplicates.append(df.columns[j])
    df_reduced = df.drop(columns=duplicates)
    print("After Removing Duplicates:", df_reduced.shape[1], "features")
    return df_reduced

# Normalization and variance filter combination function
def normalize_and_apply_variance_filter(df, threshold=0.005):
    df_normalized = df / df.mean()
    vt = VarianceThreshold(threshold=threshold)
    df_filtered = pd.DataFrame(vt.fit_transform(df_normalized), columns=df_normalized.columns[vt.get_support()])
    print("After Second Variance Threshold:", df_filtered.shape[1], "features")
    return df_filtered

# Standardization function
def standardize_data(df):
    desc = df.describe().transpose()
    df_standardized = (df - desc['mean']) / desc['std']
    print("After Standardization:", df_standardized.shape[1], "features")
    return df_standardized

# Feature selection function (SelectKBest)
def select_k_best_features(df, target, significance_level=0.05):
    F, pvalues_f = f_regression(df, target)
    k = F.shape[0] - (pvalues_f > significance_level).sum()
    skb = SelectKBest(f_regression, k=k)
    df_selected = pd.DataFrame(skb.fit_transform(df, target), columns=df.columns[skb.get_support()])
    print("After SelectKBest:", df_selected.shape[1], "features")
    return df_selected

# High/Low correlation removal function
def remove_high_correlation_features(df, threshold=0.95):
    corr_matrix = df.corr()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns 
               if any(upper_tri[column] > threshold) or any(upper_tri[column] < -threshold)]
    df_reduced = df.drop(columns=to_drop)
    print("After Correlation Filtering:", df_reduced.shape[1], "features")
    return df_reduced


# Separate features and labels
def separate_features_and_labels(data, label_index=-1, exclude_last_n=2):
    labels = data.iloc[:, label_index]
    features = data.iloc[:, :-exclude_last_n]
    return features, labels

# Normalize training data
def normalize_features(data, stats):
    return (data - stats['mean']) / stats['std']

# Remove specific features from the data
def remove_features(data, columns_to_remove):
    return data[[col for col in data.columns if col not in columns_to_remove]]

# Perform grid search with cross-validation
def perform_grid_search(model, param_grid, features, labels, cv=10, n_jobs=-1):
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        n_jobs=n_jobs,
        cv=cv
    )
    grid_search.fit(features, labels)
    return grid_search

# Evaluate model with cross-validation
def evaluate_model(model, features, labels, cv=10):
    r2_score_mean = cross_val_score(model, features, labels, cv=cv).mean()
    mae_score_mean = cross_val_score(model, features, labels, cv=cv, scoring="neg_mean_absolute_error").mean()
    rmse_score_mean = cross_val_score(model, features, labels, cv=cv, scoring="neg_root_mean_squared_error").mean()
    error_ratio = mae_score_mean / labels.mean()
    return r2_score_mean, mae_score_mean, rmse_score_mean, error_ratio