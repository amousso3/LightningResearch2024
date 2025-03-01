import xarray as xr
import cdsapi
import netCDF4 as nc
from goes2go import GOES
import requests
from datetime import datetime, timedelta
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from xgboost import XGBRegressor

def logit(target_da, predictors_da_list):
    # Initialize an array to store coefficients
    coefficients = np.full((len(predictors_da_list), 24), np.nan)

    # StandardScaler instance
    scaler = StandardScaler()

    # Loop through each hour of the day
    for hour in range(24):
        # Select data for the current hour
        target_data = target_da.sel(time=target_da.time.dt.hour == hour)
        predictors = []
        for predictor in predictors_da_list:
            predictor_data = predictor.sel(time=predictor.time.dt.hour == hour)
            predictor_data = predictor_data.values.flatten()
            predictors.append(predictor_data)

        # Flatten the dataarrays
        y_hour = target_data.values.flatten()

        # Combine predictors into a single feature matrix
        X_hour = np.vstack(predictors).T

        # Remove NaN values
        valid_mask = ~np.isnan(X_hour).any(axis=1) & ~np.isnan(y_hour)
        X_hour = X_hour[valid_mask]
        y_hour = y_hour[valid_mask]

        # Ensure binary target (presence/absence of lightning)
        y_hour = (y_hour > 0).astype(int)

        # Skip if there's insufficient data for regression
        if len(y_hour) < 10 or len(np.unique(y_hour)) < 2:
            coefficients[:, hour] = np.nan
            continue

        # Apply standard scaling to predictors
        X_hour_scaled = scaler.fit_transform(X_hour)

        # Perform logistic regression
        model = LogisticRegression()
        model.fit(X_hour_scaled, y_hour)

        # Store coefficients for this hour
        coefficients[:, hour] = model.coef_[0]

    return coefficients

def xgb_binary(target_da, predictors_da_list):
    # Initialize an array to store importance
    feature_importances = np.full((len(predictors_da_list), 24), np.nan)

    # Loop through each hour of the day
    for hour in range(24):
        # Select data for the current hour
        target_data = target_da.sel(time=target_da.time.dt.hour == hour)
        predictors = []
        for predictor in predictors_da_list:
            predictor_data = predictor.sel(time=predictor.time.dt.hour == hour)
            predictor_data = predictor_data.values.flatten()
            predictors.append(predictor_data)

        # Flatten the dataarrays
        y_hour = target_data.values.flatten()

        # Combine predictors into a single feature matrix
        X_hour = np.vstack(predictors).T

        # Remove NaN values
        valid_mask = ~np.isnan(X_hour).any(axis=1) & ~np.isnan(y_hour)
        X_hour = X_hour[valid_mask]
        y_hour = y_hour[valid_mask]

        # Ensure binary target (presence/absence of lightning)
        y_hour = (y_hour > 0).astype(int)

        # Skip if there's insufficient data for regression
        if len(y_hour) < 10 or len(np.unique(y_hour)) < 2:
            feature_importances[:, hour] = np.nan
            continue

        # Train Gradient Boosting Classifier
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        model.fit(X_hour, y_hour)

        # Store feature importances for this hour
        feature_importances[:, hour] = model.feature_importances_
    return feature_importances

def xgb_reg(target_da, predictors_da_list):
     # Initialize an array to store importance
    feature_importances = np.full((len(predictors_da_list), 24), np.nan)

    # StandardScaler instance
    scaler = StandardScaler()

    # Loop through each hour of the day
    for hour in range(24):
        # Select data for the current hour
        target_data = target_da.sel(time=target_da.time.dt.hour == hour)
        predictors = []
        for predictor in predictors_da_list:
            predictor_data = predictor.sel(time=predictor.time.dt.hour == hour)
            predictor_data = predictor_data.values.flatten()
            predictors.append(predictor_data)

        # Flatten the dataarrays
        y_hour = target_data.values.flatten()

        # Combine predictors into a single feature matrix
        X_hour = np.vstack(predictors).T

        # Remove NaN values
        valid_mask = ~np.isnan(X_hour).any(axis=1) & ~np.isnan(y_hour)
        X_hour = X_hour[valid_mask]
        y_hour = y_hour[valid_mask]

        # Skip if there's insufficient data for regression
        if len(y_hour) < 10:
            feature_importances[:, hour] = np.nan
            continue

        # Train Gradient Boosting Regressor
        model = XGBRegressor()
        model.fit(X_hour, y_hour)

        # Store feature importances for this hour
        feature_importances[:, hour] = model.feature_importances_
    return feature_importances

def plot_scores(time_range, score_type, scores, parameters, filename):
    # Plot the feature importances as time series
    hours = np.arange(time_range)
    plt.figure(figsize=(10, 6))
    for row in range(0, np.shape(scores)[0]):
        plt.plot(hours, scores[row, :], label=parameters[row])
    plt.xlabel('Hour of Day')
    plt.ylabel(score_type)
    plt.legend()
    plt.savefig(filename)
    plt.show()
    