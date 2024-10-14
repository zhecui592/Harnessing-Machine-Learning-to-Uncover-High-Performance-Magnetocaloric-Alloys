# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Define columns and models for datasets
# Columns are derived from Recursive Feature Elimination results
columns = [
    ["dev_Number", "dev_Electronegativity", "mean_NpValence", "dev_MeltingT", "dev_CovalentRadius", "most_MendeleevNumber"],
    ["maxdiff_SpaceGroupNumber", "maxdiff_Electronegativity", "Comp_L3Norm", "max_Number", "dev_MeltingT", "maxdiff_NpUnfilled", "dev_MendeleevNumber"]
]

# RandomForest models are selected based on Model Optimization results
RF_models = [
    RandomForestRegressor(n_jobs=-1, random_state=0, max_depth=11, max_features=2, n_estimators=374),
    RandomForestRegressor(n_jobs=-1,random_state=0,max_depth=11,max_features=2,n_estimators=3600)
]

# Load datasets
# "data_MMX_s_tt.csv" and "data_MMX_s_ds.csv" are results from feature engineering and data augmentation
# "MMX_Pre_MnCoNiGeSi.csv" is derived from prediction space establishment and its feature engineering
# "MMX_Pre_MnCoNiGeSi.csv" is too large to upload to GitHub. Please contact the author if needed.

data_files = ["data_MMX_s_tt.csv", "data_MMX_s_ds.csv"]
data = [pd.read_csv(file) for file in data_files]
data_pre = pd.read_csv("MMX_Pre_MnCoNiGeSi.csv")

# Shuffle prediction data
data_pre = data_pre.sample(frac=1, random_state=10)
data_pre_no = list(data_pre.iloc[:, -1])

# Initialize lists for storing data
train_data_x = []
train_data_y = []
data_pre_x = []
data_pre_y = []

# Train models and make predictions
for i in range(2):
    # Shuffle training data
    data[i] = data[i].sample(frac=1, random_state=0)
    
    # Extract features and targets for training
    data_x = pd.DataFrame(data[i], columns=columns[i])
    data_y = data[i].iloc[:, -2]
    data_no = data[i].iloc[:, -1]
    
    # Prepare prediction data
    data_pre_x.append(pd.DataFrame(data_pre, columns=columns[i]))

    # Train the RandomForest model
    rf = RF_models[i]
    rf.fit(data_x, data_y)
    
    # Predict on the prediction dataset
    data_pre_y.append(rf.predict(data_pre_x[i]))

# Filter predictions based on conditions
y_T_tt_pre_new = []
y_T_ds_pre_new = []
y_T_pre_no = []

for i in range(len(data_pre_y[0])):
    if 260 < data_pre_y[0][i] < 290 and data_pre_y[1][i] > 40:
        y_T_tt_pre_new.append(data_pre_y[0][i])
        y_T_ds_pre_new.append(data_pre_y[1][i])
        y_T_pre_no.append(data_pre_no[i])

# Create DataFrame for filtered predictions
data_x_pp = pd.DataFrame({
    "No.": y_T_pre_no,
    "TT Prediction": y_T_tt_pre_new,
    "DS Prediction": y_T_ds_pre_new
})

# Save the filtered predictions to a CSV file
data_x_pp.to_csv("filtered_predictions.csv", index=False)