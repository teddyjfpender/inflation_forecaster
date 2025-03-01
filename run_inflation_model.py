#!/usr/bin/env python3
# run_inflation_model.py - Replicates the original main.py functionality using the SDK

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load environment variables
load_dotenv()

# Import SDK components
from src.sdk import get_sdk
from src.utils import setup_logger, save_chart
from src.data_fetcher import build_daily_training_dataset
from src.models.simple_models import (
    train_random_forest, 
    train_xgboost, 
    train_multiple_linear_regression,
    evaluate_model
)

# Setup logger
logger = setup_logger(__name__)

def plot_true_vs_predicted_scatter(true_values, preds, model_label: str):
    """Plot true vs. predicted values as a scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(true_values, preds, alpha=0.5)
    # add x = y line
    ax.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], ls='--', color='gray')
    ax.set_title(f"{model_label} True vs Predicted Values", fontsize=16)
    ax.set_xlabel("True Values", fontsize=14)
    ax.set_ylabel("Predicted Values", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    return fig

def plot_with_confidence_interval(dates, true_values, preds, lower_bound, upper_bound, model_label: str):
    """Plot predictions along with a confidence interval."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dates, true_values, label='True Values', color='black', linewidth=2)
    ax.plot(dates, preds, label=f'{model_label} Predictions', color='orange', linestyle='--', linewidth=2)
    ax.fill_between(dates, lower_bound, upper_bound, color='orange', alpha=0.4, label='Confidence Interval')
    ax.set_title(f"{model_label} Prediction with Confidence Interval", fontsize=16)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Target CPI 12-month % Change", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    return fig

def extend_dataset_to_present(dataset):
    """
    Extend dataset to include entries up to the current date.
    This is critical for forecasting current inflation values.
    
    Args:
        dataset: Original dataset from build_daily_training_dataset()
        
    Returns:
        Extended dataset with additional rows up to current date
    """
    last_date = pd.to_datetime(dataset.index.max())
    today = pd.to_datetime(datetime.datetime.now().strftime("%Y-%m-%d"))
    
    # If the dataset is already up to date, return it as is
    if last_date >= today:
        logger.info("Dataset already includes the current date.")
        return dataset
        
    # Create a date range from the day after the last date to today
    extension_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end=today)
    
    if len(extension_dates) == 0:
        logger.info("No additional dates needed.")
        return dataset
        
    logger.info(f"Extending dataset by {len(extension_dates)} days to include data up to today.")
    
    # Get the last row's values to use as a starting point for the extension
    last_row = dataset.iloc[-1].copy()
    
    # Create extension dataframe
    extension_data = []
    
    # Copy forward the last known values and update dates
    for extension_date in extension_dates:
        new_row = last_row.copy()
        
        # Set appropriate days_until_print (increasing by one day for each extension day)
        # This assumes the target doesn't change for future dates
        
        # For actual extension implementation, you'd want to include latest market data here
        # This would require fetching the latest data for features like crude_oil, sp500, etc.
        
        # For now, just carry forward the last known values and adjust dates
        extension_data.append(new_row)
    
    extension_df = pd.DataFrame(extension_data, index=extension_dates.strftime("%Y-%m-%d"))
    
    # Combine original dataset with extension
    extended_dataset = pd.concat([dataset, extension_df])
    
    logger.info(f"Dataset extended successfully. New shape: {extended_dataset.shape}")
    
    return extended_dataset

def main():
    # Use the SDK for configuration and utilities
    sdk = get_sdk("config/model_config.yaml")
    logger.info("SDK initialized")
    
    # 1. Prepare data - use the direct function call like the original
    logger.info("Building daily training dataset based on monthly CPI prints...")
    dataset = build_daily_training_dataset()
    logger.info(f"Dataset prepared with shape: {dataset.shape}")
    
    # Extend dataset to include dates up to today
    dataset = extend_dataset_to_present(dataset)
    
    # Log dataset information like the original
    logger.info("Dataset head:")
    logger.info(dataset.head())
    logger.info("Dataset tail:")
    logger.info(dataset.tail())
    
    # Save dataset to CSV just like the original
    dataset.to_csv('dataset.csv')
    
    # Ensure the index is in datetime format (needed for plotting) - just like original
    dataset.index = pd.to_datetime(dataset.index)
    
    # Define feature and target columns from the config
    target_col = sdk.config['target_variable']
    feature_cols = sdk.config['feature_variables']
    
    # Verify required columns exist
    missing = [col for col in [target_col] + feature_cols if col not in dataset.columns]
    if missing:
        raise ValueError(f"Missing required columns in training dataset: {missing}")
    
    # --- Train-Test Split (matching original code) ---
    X = dataset[feature_cols]
    y = dataset[target_col]
    indices = dataset.index
    
    logger.info(f"Using features: {feature_cols}")
    logger.info(f"Target variable: {target_col}")
    
    # Random split with a fixed seed for reproducibility - same as original
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=0.2, random_state=42
    )
    
    # --- Normalize Features - same as original ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_all_scaled = scaler.transform(X)
    
    # --- Train Models - calling the same functions as the original ---
    # Multiple Linear Regression
    ml_model, ml_diag = train_multiple_linear_regression(X_train_scaled, y_train, feature_cols)
    rmse_ml, preds_ml_test = evaluate_model(ml_model, X_test_scaled, y_test, model_type='ml')
    preds_ml_all = ml_model.predict(X_all_scaled)
    logger.info(f"Multiple Linear Regression RMSE on test set: {rmse_ml}")

    # Random Forest
    rf_model = train_random_forest(X_train_scaled, y_train)
    rmse_rf, preds_rf_test = evaluate_model(rf_model, X_test_scaled, y_test, model_type='rf')
    preds_rf_all = rf_model.predict(X_all_scaled)
    logger.info(f"Random Forest RMSE on test set: {rmse_rf}")

    # XGBoost
    xgb_model = train_xgboost(X_train_scaled, y_train)
    rmse_xgb, preds_xgb_test = evaluate_model(xgb_model, X_test_scaled, y_test, model_type='xgb')
    preds_xgb_all = xgb_model.predict(X_all_scaled)
    logger.info(f"XGBoost RMSE on test set: {rmse_xgb}")
    
    # --- Ensemble Model (Average of Predictions) ---
    preds_ensemble_all = (preds_ml_all + preds_rf_all + preds_xgb_all) / 3.0
    ensemble_rmse = np.sqrt(np.mean((y - preds_ensemble_all) ** 2))
    logger.info(f"Ensemble Model RMSE on the entire dataset: {ensemble_rmse}")
    
    # Create ensemble test predictions
    preds_ensemble_test = (preds_ml_test + preds_rf_test + preds_xgb_test) / 3.0
    
    # --- Create and save visualizations ---
    dates_all = dataset.index
    
    # For Multiple Linear Regression:
    ci_factor_ml = 1.96 * rmse_ml  # 95% CI approximation using normal assumption
    lower_bound_ml_all = preds_ml_all - ci_factor_ml
    upper_bound_ml_all = preds_ml_all + ci_factor_ml
    fig_ml = plot_with_confidence_interval(
        dates_all, y, preds_ml_all,
        lower_bound_ml_all, upper_bound_ml_all,
        "Multiple Linear Regression"
    )
    fig_ml_test = plot_true_vs_predicted_scatter(
        y_test, preds_ml_test, "Multiple Linear Regression"
    )
    save_chart(fig_ml, "MultipleLinearRegression_vs_True_all.png")
    save_chart(fig_ml_test, "MultipleLinearRegression_vs_True_test.png")
    
    # For Random Forest:
    ci_factor_rf = 1.96 * rmse_rf
    lower_bound_rf_all = preds_rf_all - ci_factor_rf
    upper_bound_rf_all = preds_rf_all + ci_factor_rf
    fig_rf = plot_with_confidence_interval(
        dates_all, y, preds_rf_all,
        lower_bound_rf_all, upper_bound_rf_all,
        "Random Forest"
    )
    fig_rf_test = plot_true_vs_predicted_scatter(
        y_test, preds_rf_test, "Random Forest"
    )
    save_chart(fig_rf, "RandomForest_vs_True_all.png")
    save_chart(fig_rf_test, "RandomForest_vs_True_test.png")
    
    # For XGBoost:
    ci_factor_xgb = 1.96 * rmse_xgb
    lower_bound_xgb_all = preds_xgb_all - ci_factor_xgb
    upper_bound_xgb_all = preds_xgb_all + ci_factor_xgb
    fig_xgb = plot_with_confidence_interval(
        dates_all, y, preds_xgb_all,
        lower_bound_xgb_all, upper_bound_xgb_all,
        "XGBoost"
    )
    fig_xgb_test = plot_true_vs_predicted_scatter(
        y_test, preds_xgb_test, "XGBoost"
    )
    save_chart(fig_xgb, "XGBoost_vs_True_all.png")
    save_chart(fig_xgb_test, "XGBoost_vs_True_test.png")
    
    # For Ensemble Model:
    ci_factor_ensemble = 1.96 * ensemble_rmse
    lower_bound_ensemble_all = preds_ensemble_all - ci_factor_ensemble
    upper_bound_ensemble_all = preds_ensemble_all + ci_factor_ensemble
    fig_ensemble = plot_with_confidence_interval(
        dates_all, y, preds_ensemble_all,
        lower_bound_ensemble_all, upper_bound_ensemble_all,
        "Ensemble Model"
    )
    fig_ensemble_test = plot_true_vs_predicted_scatter(
        y_test, preds_ensemble_test, "Ensemble Model"
    )
    save_chart(fig_ensemble, "EnsembleModel_vs_True_all.png")
    save_chart(fig_ensemble_test, "EnsembleModel_vs_True_test.png")
    
    # Save the trained models using the SDK functionality
    logger.info("Saving models via SDK...")
    
    # Add models to SDK to use its saving functionality
    sdk.models = {
        'multiple_linear_regression': {'model': ml_model, 'features': feature_cols},
        'random_forest': {'model': rf_model, 'features': feature_cols},
        'xgboost': {'model': xgb_model, 'features': feature_cols}
    }
    
    # Also add predictions to SDK for historical store update
    all_predictions = pd.DataFrame({
        'actual': y,
        'multiple_linear_regression_prediction': preds_ml_all,
        'random_forest_prediction': preds_rf_all,
        'xgboost_prediction': preds_xgb_all,
        'ensemble_prediction': preds_ensemble_all
    }, index=dates_all)
    
    # Add confidence intervals
    all_predictions['multiple_linear_regression_lower'] = lower_bound_ml_all
    all_predictions['multiple_linear_regression_upper'] = upper_bound_ml_all
    all_predictions['random_forest_lower'] = lower_bound_rf_all
    all_predictions['random_forest_upper'] = upper_bound_rf_all
    all_predictions['xgboost_lower'] = lower_bound_xgb_all
    all_predictions['xgboost_upper'] = upper_bound_xgb_all
    
    sdk.predictions = all_predictions
    
    # Save models
    sdk.save_model()
    
    # Update historical predictions
    logger.info("Updating historical predictions...")
    sdk.update_historical_predictions("data/prediction_store.json")
    
    logger.info("Process completed successfully!")

if __name__ == '__main__':
    main() 