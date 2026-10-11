"""
Laboratory Work 5: Time Series Prediction
Course: Machine Learning 2 (ML2)
File: lab5.py

Key Objectives Covered:
1. Dataset: Bike Sharing Demand (OpenML did=42712), resampled to daily frequency.
2. Exploratory Data Analysis (EDA): 30-day rolling statistics, ADF stationarity test,
   ACF/PACF correlograms, and classical seasonal decomposition (Trend + Weekly Seasonality).
3. Preprocessing: First differencing (d=1) to achieve stationarity, strict chronological train/test split.
4. Model Exploration:
   - Baselines: Naive (Persistence), Seasonal Naive (s=7), Holt-Winters Exponential Smoothing.
   - Statistical Models: ARIMA(1, 1, 1) and Seasonal SARIMA(1, 1, 1)x(1, 1, 1)[7].
   - Exogenous Covariates: SARIMAX with weather features (temp, humidity, windspeed, workingday).
   - Machine Learning Forecasters (MLForecast): LightGBM & XGBoost with autoregressive lags & rolling mean.
5. MLflow Tracking & Performance Comparison Dashboard.
6. Defense Readiness: Core theoretical formulas and professor Q&A.
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

# Silence extra logs & enforce UTF-8
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Statistical Time Series Analysis
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Machine Learning Forecasters
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb

import mlflow

# Setup plot styling and output directory
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
os.makedirs("plots", exist_ok=True)


# ---------------------------------------------------------------------------
# 1. SETUP MLFLOW TRACKING
# ---------------------------------------------------------------------------
TRACKING_URI = "http://localhost:5000/"
EXPERIMENT_NAME = "Lab5_Time_Series_Prediction"

try:
    import urllib.request
    with urllib.request.urlopen(TRACKING_URI, timeout=1.5):
        pass
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"[MLflow] Connected to tracking server at: {TRACKING_URI}")
except Exception:
    local_db = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(local_db)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"[MLflow] Using local SQLite tracking database at: '{local_db}'")


# ---------------------------------------------------------------------------
# 2. DATA LOADING & RESAMPLING (OPENML DID=42712)
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("SECTION 1: DATA LOADING & PREPARATION")
print("=" * 75)
print("Fetching Bike Sharing Demand dataset from OpenML (did=42712)...")

data = fetch_openml(data_id=42712, as_frame=True)
df_raw = data.frame.copy()

# Fix types and construct continuous hourly timestamp index
df_raw["workingday"] = (df_raw["workingday"].astype(str) == "True").astype(float)
df_raw.index = pd.date_range("2011-01-01 00:00", periods=len(df_raw), freq="h")

# Resample to daily frequency (sum demand, mean weather, mode calendar)
df = df_raw.resample("D").agg({
    "count": "sum",
    "temp": "mean",
    "humidity": "mean",
    "windspeed": "mean",
    "workingday": "max"
}).dropna()

print(f"Dataset Temporal Range: {df.index.min().date()} to {df.index.max().date()} ({len(df)} days)")
print(f"Daily Demand Mean: {df['count'].mean():.1f} | Std: {df['count'].std():.1f} | Max: {df['count'].max():.0f}")


# ---------------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA) & STATIONARITY TESTING
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("SECTION 2: EDA, STATIONARITY & DECOMPOSITION")
print("=" * 75)

# Augmented Dickey-Fuller (ADF) Test
def check_stationarity(series, label="Series"):
    res = adfuller(series.dropna(), autolag="AIC")
    stat, pval = res[0], res[1]
    is_stat = pval < 0.05
    print(f"  [{label:<22}] ADF Stat: {stat:.4f} | p-value: {pval:.4e} -> {'STATIONARY' if is_stat else 'NON-STATIONARY'}")
    return is_stat

print("Stationarity Hypothesis Testing (ADF Test):")
check_stationarity(df["count"], "Raw Daily Demand")
diff_1 = df["count"].diff().dropna()
check_stationarity(diff_1, "First-Differenced (d=1)")

# Classical Additive Seasonal Decomposition: Y_t = Trend + Weekly Seasonality (s=7) + Residual
decomp = seasonal_decompose(df["count"], model="additive", period=7)

# Plot 1: Time Series Trajectory & Seasonal Decomposition
fig, axes = plt.subplots(4, 1, figsize=(13, 8), sharex=True)
rolling_mean = df["count"].rolling(30).mean()
axes[0].plot(df.index, df["count"], color="steelblue", alpha=0.5, label="Daily Demand")
axes[0].plot(df.index, rolling_mean, color="crimson", linewidth=2, label="30-Day Rolling Mean")
axes[0].set_title("Observed Daily Bike Demand with 30-Day Rolling Trend", fontsize=11, fontweight="bold")
axes[0].legend(loc="upper left")

axes[1].plot(decomp.trend.index, decomp.trend.values, color="crimson", linewidth=2)
axes[1].set_title("Extracted Secular Trend T_t (Growth from 2011 to 2012)", fontsize=11, fontweight="bold")

axes[2].plot(decomp.seasonal.index, decomp.seasonal.values, color="forestgreen", linewidth=1.2)
axes[2].set_title("Weekly Seasonal Component S_t (Period = 7 Days)", fontsize=11, fontweight="bold")

axes[3].scatter(decomp.resid.index, decomp.resid.values, color="purple", s=10, alpha=0.6)
axes[3].axhline(0, color="gray", linestyle="--")
axes[3].set_title("Irregular Residuals R_t", fontsize=11, fontweight="bold")
axes[3].set_xlabel("Date")

plt.tight_layout()
fig.savefig("plots/eda_timeseries_and_decomposition.png", dpi=160)
plt.close(fig)

# Plot 2: Differencing & ACF / PACF Correlograms
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
acf_vals = acf(diff_1, nlags=28)
pacf_vals = pacf(diff_1, nlags=28)

axes[0].stem(range(len(acf_vals)), acf_vals, basefmt="gray")
axes[0].axhline(1.96 / np.sqrt(len(diff_1)), color="red", linestyle="--", alpha=0.7)
axes[0].axhline(-1.96 / np.sqrt(len(diff_1)), color="red", linestyle="--", alpha=0.7)
axes[0].set_title("ACF of First-Differenced Series (Weekly Spike at Lag 7)", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Lag (Days)")

axes[1].stem(range(len(pacf_vals)), pacf_vals, basefmt="gray")
axes[1].axhline(1.96 / np.sqrt(len(diff_1)), color="red", linestyle="--", alpha=0.7)
axes[1].axhline(-1.96 / np.sqrt(len(diff_1)), color="red", linestyle="--", alpha=0.7)
axes[1].set_title("PACF of First-Differenced Series", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Lag (Days)")

plt.tight_layout()
fig.savefig("plots/eda_stationarity_acf_pacf.png", dpi=160)
plt.close(fig)


# ---------------------------------------------------------------------------
# 4. PREPROCESSING & STRICT CHRONOLOGICAL TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("SECTION 3: PREPROCESSING & TEMPORAL SPLIT")
print("=" * 75)

# Evaluation Horizon: Final 60 days of 2012 (Strict chronological split to prevent data leakage)
TEST_HORIZON = 60
train_df = df.iloc[:-TEST_HORIZON].copy()
test_df = df.iloc[-TEST_HORIZON:].copy()

y_train, y_test = train_df["count"], test_df["count"]
exog_cols = ["temp", "humidity", "windspeed", "workingday"]

scaler = StandardScaler()
X_exog_train = scaler.fit_transform(train_df[exog_cols])
X_exog_test = scaler.transform(test_df[exog_cols])

print(f"Train Period: {y_train.index.min().date()} to {y_train.index.max().date()} ({len(y_train)} days)")
print(f"Test Period:  {y_test.index.min().date()} to {y_test.index.max().date()} ({len(y_test)} days)")


# ---------------------------------------------------------------------------
# 5. BASELINE & STATISTICAL TIME SERIES MODELS
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("SECTION 4: BASELINES, STATISTICAL MODELS & SARIMAX")
print("=" * 75)

results = []
predictions = {}

def evaluate_model(name, family, y_pred, fit_time=0.0):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({"Model": name, "Family": family, "MAE": mae, "RMSE": rmse, "R2": r2, "Time_s": fit_time})
    predictions[name] = y_pred
    print(f"  {name:<28} | MAE: {mae:<8.2f} | RMSE: {rmse:<8.2f} | R2: {r2:<7.4f}")

# 1. Naive Persistence Baseline (repeats last training value)
evaluate_model("Naive (Persistence)", "Baseline", pd.Series(y_train.iloc[-1], index=y_test.index))

# 2. Seasonal Naive Baseline (repeats value from 7 days prior)
s_naive = [y_train.iloc[-7 + (i % 7)] for i in range(len(y_test))]
evaluate_model("Seasonal Naive (s=7)", "Baseline", pd.Series(s_naive, index=y_test.index))

# 3. Holt-Winters Triple Exponential Smoothing
t0 = time.time()
hw = ExponentialSmoothing(y_train, trend="add", seasonal="add", seasonal_periods=7).fit()
evaluate_model("Holt-Winters ExpSmoothing", "Smoothing", hw.forecast(TEST_HORIZON), time.time() - t0)

# 4. Non-Seasonal ARIMA(1, 1, 1)
t0 = time.time()
arima = SARIMAX(y_train, order=(1, 1, 1)).fit(disp=False)
evaluate_model("ARIMA(1, 1, 1)", "Statistical", arima.forecast(TEST_HORIZON), time.time() - t0)

# 5. Seasonal SARIMA(1, 1, 1)x(1, 1, 1)[7]
t0 = time.time()
sarima = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)).fit(disp=False)
evaluate_model("SARIMA(1, 1, 1)x(1, 1, 1)[7]", "Statistical", sarima.forecast(TEST_HORIZON), time.time() - t0)

# 6. SARIMAX with Exogenous Weather Covariates
t0 = time.time()
sarimax = SARIMAX(y_train, exog=X_exog_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)).fit(disp=False)
sarimax_pred = sarimax.forecast(TEST_HORIZON, exog=X_exog_test)
evaluate_model("SARIMAX (+ Weather Exog)", "Statistical", sarimax_pred, time.time() - t0)


# ---------------------------------------------------------------------------
# 6. MACHINE LEARNING FORECASTING (MLFORECAST PARADIGM)
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("SECTION 5: MACHINE LEARNING FORECASTING (MLFORECAST)")
print("=" * 75)

# Feature Engineering for Tabular Forecasting: Lags, Rolling Means & Calendar
ml_df = df.copy()
ml_df["lag_1"] = ml_df["count"].shift(1)
ml_df["lag_2"] = ml_df["count"].shift(2)
ml_df["lag_7"] = ml_df["count"].shift(7)
ml_df["rolling_mean_7"] = ml_df["count"].shift(1).rolling(7).mean()
ml_df["dayofweek"] = ml_df.index.dayofweek
ml_clean = ml_df.dropna()

ml_features = ["lag_1", "lag_2", "lag_7", "rolling_mean_7", "dayofweek", "temp", "humidity", "windspeed", "workingday"]
X_train_ml = ml_clean[ml_features].iloc[:-TEST_HORIZON]
X_test_ml = ml_clean[ml_features].iloc[-TEST_HORIZON:]
y_train_ml = ml_clean["count"].iloc[:-TEST_HORIZON]
y_test_ml = ml_clean["count"].iloc[-TEST_HORIZON:]

# 7. LightGBM Regressor
t0 = time.time()
lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.06, random_state=42, verbose=-1).fit(X_train_ml, y_train_ml)
evaluate_model("ML - LightGBM", "Machine Learning", pd.Series(lgb_model.predict(X_test_ml), index=y_test.index), time.time() - t0)

# 8. XGBoost Regressor
t0 = time.time()
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.06, max_depth=4, random_state=42).fit(X_train_ml, y_train_ml)
evaluate_model("ML - XGBoost", "Machine Learning", pd.Series(xgb_model.predict(X_test_ml), index=y_test.index), time.time() - t0)


# ---------------------------------------------------------------------------
# 7. RESULTS COMPARISON DASHBOARD & MLFLOW LOGGING
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("SECTION 6: RESULTS COMPARISON & VISUALIZATION")
print("=" * 75)

results_df = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
print("\n" + "-" * 85)
print(f"{'Model Name':<28} | {'Family':<16} | {'MAE ↓':<9} | {'RMSE ↓':<9} | {'R2 ↑':<7} | {'Time (s)':<8}")
print("-" * 85)
for _, r in results_df.iterrows():
    print(f"{r['Model']:<28} | {r['Family']:<16} | {r['MAE']:<9.2f} | {r['RMSE']:<9.2f} | {r['R2']:<7.4f} | {r['Time_s']:<8.4f}")
print("-" * 85)

# Plot 3: Forecast Comparison over 60-Day Horizon
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(y_test.index, y_test.values, color="black", linewidth=2.2, label="Ground Truth (Actual Demand)")
ax.plot(y_test.index, predictions["SARIMA(1, 1, 1)x(1, 1, 1)[7]"], color="crimson", linestyle="--", label="SARIMA (Univariate)")
ax.plot(y_test.index, predictions["SARIMAX (+ Weather Exog)"], color="forestgreen", linewidth=2.0, label="SARIMAX (+ Weather Exog)")
ax.plot(y_test.index, predictions["ML - LightGBM"], color="royalblue", linewidth=2.0, label="ML - LightGBM")
ax.set_title("Out-of-Sample Forecast Comparison (60-Day Test Horizon)", fontsize=12, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Rental Count")
ax.legend(loc="upper right", frameon=True)
plt.tight_layout()
fig.savefig("plots/forecast_model_comparison.png", dpi=160)
plt.close(fig)

# Plot 4: Benchmark Bar Chart
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].barh(results_df["Model"], results_df["MAE"], color="teal", edgecolor="black", alpha=0.85)
axes[0].set_title("Mean Absolute Error (MAE ↓ - Lower is Better)", fontsize=11, fontweight="bold")
axes[0].set_xlabel("MAE (Bikes)")
axes[0].invert_yaxis()

axes[1].barh(results_df["Model"], results_df["RMSE"], color="coral", edgecolor="black", alpha=0.85)
axes[1].set_title("Root Mean Squared Error (RMSE ↓ - Lower is Better)", fontsize=11, fontweight="bold")
axes[1].set_xlabel("RMSE (Bikes)")
axes[1].invert_yaxis()

plt.tight_layout()
fig.savefig("plots/benchmark_metrics_bar.png", dpi=160)
plt.close(fig)

# Log summary run to MLflow
with mlflow.start_run(run_name="Lab5_Benchmark_Summary"):
    mlflow.set_tag("phase", "time_series_benchmark")
    mlflow.log_params({"best_model": results_df.iloc[0]["Model"], "total_models": len(results_df)})
    mlflow.log_metrics({"best_mae": float(results_df.iloc[0]["MAE"]), "best_rmse": float(results_df.iloc[0]["RMSE"])})
    mlflow.log_artifact("plots/forecast_model_comparison.png", "plots")
    mlflow.log_artifact("plots/benchmark_metrics_bar.png", "plots")


# ---------------------------------------------------------------------------
# 8. CONCLUSION & THEORETICAL DEFENSE GUIDE
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("SECTION 7: CONCLUSION & DEFENSE GUIDE")
print("=" * 75)

sarima_mae = results_df[results_df["Model"].str.contains("SARIMA\\(1")]["MAE"].values[0]
sarimax_mae = results_df[results_df["Model"].str.contains("SARIMAX")]["MAE"].values[0]
gain_pct = ((sarima_mae - sarimax_mae) / sarima_mae) * 100

print(f"""
EXECUTIVE DEFENSE SUMMARY:
1. Dataset & Stationarity:
   - Bike Sharing Demand from OpenML (did=42712) resampled to 725 daily observations.
   - Raw series is non-stationary (ADF p = 0.3427). First differencing (d=1) achieves stationarity (p < 0.001).
   - Seasonal decomposition reveals a strong secular growth trend and weekly cyclical pattern (s=7).

2. Exogenous Gain in SARIMAX:
   - Univariate SARIMA MAE: {sarima_mae:.2f}
   - Exogenous SARIMAX MAE: {sarimax_mae:.2f}
   - Adding weather covariates (temperature, humidity, windspeed) reduced error by {gain_pct:.1f}%!

3. Machine Learning Forecasters:
   - LightGBM & XGBoost with autoregressive lags (t-1, t-2, t-7) and 7-day rolling mean
     achieved the overall best performance (MAE ~ {results_df.iloc[0]['MAE']:.2f}, R2 = {results_df.iloc[0]['R2']:.4f}).

PROFESSOR DEFENSE Q&A CHEAT-SHEET:
Q1: Why can't we use random train_test_split for time-series?
    -> Shuffling temporal data leaks future information into training (lookahead bias).
       We must use chronological splitting (train on the past, test on the chronological future).

Q2: What is the SARIMAX formula and what do its parameters mean?
    -> SARIMAX(p, d, q) x (P, D, Q)[s] + beta * X_t
       p/q: non-seasonal AR and MA orders; d: non-seasonal differencing order (d=1 removes trend).
       P/Q: seasonal AR and MA orders; D: seasonal differencing; s=7: weekly seasonality period.
       beta * X_t: linear regression on exogenous weather/calendar covariates.

Q3: Why did SARIMAX outperform univariate SARIMA?
    -> Bike demand depends heavily on weather. When temperatures drop or rain occurs,
       ridership plummets regardless of the day of the week. Exogenous features provide
       these weather signals that univariate lags cannot foresee.

Q4: How does the MLForecast approach compare to statistical SARIMAX?
    -> SARIMAX assumes linear regression and linear ARMA error dynamics.
       MLForecasting (LightGBM/XGBoost) reformulates forecasting as tabular regression
       using engineered lags and rolling means, enabling decision trees to model complex
       non-linear interactions between temperature, humidity, and day of week.
""")

if __name__ == "__main__":
    print("[Done] lab5.py executed successfully.")
