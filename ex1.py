from datasets import load_dataset
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import mlflow
import mlflow.xgboost # Using mlflow.xgboost for specific XGBoost model logging

# Set the tracking URI. MLflow will store tracking data in a local './mlruns' directory.
# If you have a dedicated MLflow tracking server, you would put its URI here (e.g., http://your-mlflow-server:5000).
mlflow.set_tracking_uri('http://localhost:5000/') # Using a local directory for simplicity

print(f"MLflow Version: {mlflow.__version__}")
print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

#########################################
########## SECOND PART ###################
#########################################

# Load the California Housing dataset from Hugging Face
try:
    # Using the gvlassis version which provides the 8 numeric features directly
    housing_dataset = load_dataset('gvlassis/california_housing', split='train')
    print("Successfully loaded California Housing dataset.")
except Exception as e:
    print(f"Failed to load dataset: {e}")
    print("Please ensure you have internet connectivity and the dataset name is correct.")
    # If loading fails, you might want to stop or use a very simple fallback for demonstration if absolutely necessary
    # For this dataset, failure is less likely than with very large or experimental datasets.
    raise e # Re-raise the exception to halt if dataset loading fails

# Convert to pandas DataFrame for easier manipulation
df = housing_dataset.to_pandas()

print("\nDataset Preview (First 5 rows):")
print(df.head())
print(f"\nDataset Shape: {df.shape}")
print("\nDataset Features and Types:")
print(df.info())

# Define features and target based on the dataset's known structure
# Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
# Target: MedHouseVal
feature_columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
target_column = 'MedHouseVal'

X = df[feature_columns]
y = df[target_column]

# The California Housing dataset (gvlassis version) is clean and doesn't typically have NaNs
# However, adding a check or a simple imputer is good practice for robustness if needed in other contexts.
print(f"\nMissing values in features (X): {X.isnull().sum().sum()}")
print(f"Missing values in target (y): {y.isnull().sum()}")

# If there were NaNs, a simple strategy would be:
# X = X.fillna(X.mean())
# y = y.fillna(y.mean())

print("\nFeatures (X) shape:", X.shape)
print("Target (y) shape:", y.shape)


#########################################
########## THIRD PART ###################
#########################################

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Validation set size: {X_val.shape[0]} samples")

# Initialize and train the XGBoost Regressor model
# We'll use a few common hyperparameters. More advanced tuning will be covered later.
params = {
    'objective': 'reg:squarederror', # Objective function for regression
    'n_estimators': 150,             # Number of boosting rounds (trees)
    'learning_rate': 0.1,            # Step size shrinkage
    'max_depth': 7,                  # Maximum depth of a tree (adjusted from 3 for potentially more complex data)
    'subsample': 0.8,                # Subsample ratio of the training instance
    'colsample_bytree': 0.8,         # Subsample ratio of columns when constructing each tree
    'random_state': 42               # For reproducibility
}

model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the model
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"\nModel Performance on Validation Set:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2 Score): {r2:.4f}")



#########################################
########## FOURTH PART ###################
#########################################



experiment_name = "XGBoost_Run_More_Estimators_Housing"
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name="XGBoost_Initial_Run_Housing") as run:
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id
    print(f"Starting MLflow Run: {run.info.run_name}")
    print(f"Run ID: {run_id}")
    print(f"Experiment ID: {experiment_id}")
    
    # Log parameters used for this run
    mlflow.log_params(params) # Log all parameters from the dict
    mlflow.log_param("train_test_split_random_state", 42)
    mlflow.log_param("dataset_name", "gvlassis/california_housing")
    mlflow.log_param("features_used_count", len(feature_columns))

    # Re-train the model and evaluate INSIDE the mlflow.start_run() context for proper logging
    model_in_run = xgb.XGBRegressor(**params)
    model_in_run.fit(X_train, y_train)
    y_pred_in_run = model_in_run.predict(X_val)
    
    mse_in_run = mean_squared_error(y_val, y_pred_in_run)
    mae_in_run = mean_absolute_error(y_val, y_pred_in_run)
    r2_in_run = r2_score(y_val, y_pred_in_run)
    
    # Log metrics
    metrics_to_log = {"mse": mse_in_run, "mae": mae_in_run, "r2_score": r2_in_run}
    mlflow.log_metrics(metrics_to_log)
    print(f"Logged Metrics: MSE={mse_in_run:.4f}, MAE={mae_in_run:.4f}, R2={r2_in_run:.4f}")

    # Log the trained model using MLflow's XGBoost flavor
    # This saves the model in a format MLflow understands, allowing for easy loading later.
    # The 'artifact_path' is a name for the model within this run's artifacts.
    mlflow.xgboost.log_model(model_in_run, artifact_path="xgboost-housing-model")
    print(f"Model logged under path: xgboost-housing-model")

    # You can also log arbitrary files as artifacts
    # For example, let's log the list of features used as a text file
    with open("features_california_housing.txt", "w") as f:
        for feature in feature_columns:
            f.write(f"{feature}\n")
    mlflow.log_artifact("features_california_housing.txt", artifact_path="feature_info")
    print(f"Logged 'features_california_housing.txt' artifact to 'feature_info' directory.")

    # Set a tag for this run for easier filtering/organization
    mlflow.set_tag("model_type", "XGBoost_Regressor")
    mlflow.set_tag("data_version", "1990_census")
    print("Set tags for the run.")

    print(f"\nMLflow Run {run.info.run_name} completed and logged.")

print("\nCheck the 'mlruns' directory in your file system. It should now contain experiment data.")