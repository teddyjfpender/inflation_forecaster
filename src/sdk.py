# src/sdk.py

import os
import logging
import yaml
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# Import internal modules
from src.data_fetcher import build_daily_training_dataset
from src.models.simple_models import (
    train_multiple_linear_regression,
    train_random_forest,
    train_xgboost,
    evaluate_model,
    predict_multiple_linear_regression,
    predict_with_confidence_interval
)
from src.store_manager import (
    load_prediction_store,
    save_prediction_store,
    update_store
)
from src.utils import (
    save_chart,
    setup_logger
)

# Setup logger
logger = setup_logger(__name__)

class FinancialForecastingSDK:
    """
    Main SDK class for financial indicator forecasting.
    Provides a unified interface for data fetching, model training,
    forecasting, evaluation, and historical prediction storage.
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize the SDK with configuration.
        
        Args:
            config_path: Path to the YAML configuration file.
        """
        self.config = self.load_config(config_path)
        self.models = {}
        self.data = None
        self.predictions = None
        
        # Create necessary directories
        Path("src/saved_models").mkdir(parents=True, exist_ok=True)
        Path("docs/images").mkdir(parents=True, exist_ok=True)
        
    def load_config(self, config_path: str) -> dict:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file.
            
        Returns:
            Dictionary containing the configuration.
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def prepare_data(self) -> pd.DataFrame:
        """
        Prepare the data for training and forecasting.
        
        Returns:
            DataFrame containing the prepared data.
        """
        try:
            logger.info("Preparing data for modeling...")
            data = build_daily_training_dataset()
            self.data = data
            logger.info(f"Data prepared successfully with shape {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def train_models(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Train multiple models as specified in the configuration.
        
        Args:
            data: Optional DataFrame to use for training. If None, uses self.data.
            
        Returns:
            Dictionary mapping model names to trained model objects.
        """
        if data is None:
            if self.data is None:
                self.data = self.prepare_data()
            data = self.data
        
        target = self.config['target_variable']
        features = self.config['feature_variables']
        
        # Prepare training data
        X = data[features]
        y = data[target]
        
        # Train-test split
        from sklearn.model_selection import train_test_split
        test_size = self.config['model_parameters']['train_test_split']['test_size']
        random_state = self.config['model_parameters']['train_test_split']['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scaling if specified
        if self.config['model_parameters']['scaling']:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            self.scaler = scaler
        
        models = {}
        evaluations = {}
        
        # Train models based on configuration
        if 'multiple_linear_regression' in self.config['models']:
            logger.info("Training Multiple Linear Regression model...")
            fit_intercept = self.config['models']['multiple_linear_regression'].get('fit_intercept', True)
            mlr_model, mlr_stats = train_multiple_linear_regression(
                X_train, y_train, features, fit_intercept=fit_intercept
            )
            models['multiple_linear_regression'] = {
                'model': mlr_model,
                'stats': mlr_stats,
                'features': features
            }
            mse, preds = evaluate_model(mlr_model, X_test, y_test, 'mlr')
            evaluations['multiple_linear_regression'] = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'predictions': preds
            }
            logger.info(f"Multiple Linear Regression RMSE: {np.sqrt(mse):.4f}")
        
        if 'random_forest' in self.config['models']:
            logger.info("Training Random Forest model...")
            n_estimators = self.config['models']['random_forest'].get('n_estimators', 100)
            random_state = self.config['models']['random_forest'].get('random_state', 42)
            
            from sklearn.ensemble import RandomForestRegressor
            rf_model = RandomForestRegressor(
                n_estimators=n_estimators, 
                random_state=random_state
            )
            rf_model.fit(X_train, y_train)
            
            models['random_forest'] = {
                'model': rf_model,
                'features': features
            }
            mse, preds = evaluate_model(rf_model, X_test, y_test, 'rf')
            evaluations['random_forest'] = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'predictions': preds,
                'feature_importance': dict(zip(features, rf_model.feature_importances_))
            }
            logger.info(f"Random Forest RMSE: {np.sqrt(mse):.4f}")
        
        if 'xgboost' in self.config['models']:
            logger.info("Training XGBoost model...")
            n_estimators = self.config['models']['xgboost'].get('n_estimators', 100)
            random_state = self.config['models']['xgboost'].get('random_state', 42)
            
            from xgboost import XGBRegressor
            xgb_model = XGBRegressor(
                n_estimators=n_estimators, 
                random_state=random_state
            )
            xgb_model.fit(X_train, y_train)
            
            models['xgboost'] = {
                'model': xgb_model,
                'features': features
            }
            mse, preds = evaluate_model(xgb_model, X_test, y_test, 'xgb')
            evaluations['xgboost'] = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'predictions': preds,
                'feature_importance': dict(zip(features, xgb_model.feature_importances_))
            }
            logger.info(f"XGBoost RMSE: {np.sqrt(mse):.4f}")
        
        self.models = models
        self.evaluations = evaluations
        return models
    
    def forecast(self, data: Optional[pd.DataFrame] = None, days_ahead: int = 30) -> pd.DataFrame:
        """
        Generate forecasts using trained models.
        
        Args:
            data: Optional DataFrame to use for forecasting. If None, uses self.data.
            days_ahead: Number of days to forecast ahead.
            
        Returns:
            DataFrame containing forecasts from all models.
        """
        if not self.models:
            logger.warning("No trained models available. Training models first.")
            self.train_models()
        
        if data is None:
            if self.data is None:
                self.data = self.prepare_data()
            data = self.data
        
        features = self.config['feature_variables']
        
        # Get the latest data for forecasting
        forecast_df = data.sort_index().tail(days_ahead + 1).copy()
        X_forecast = forecast_df[features]
        
        # Apply scaling if necessary
        if hasattr(self, 'scaler'):
            X_forecast_scaled = self.scaler.transform(X_forecast)
        else:
            X_forecast_scaled = X_forecast
        
        # Generate predictions for each model
        predictions = pd.DataFrame(index=forecast_df.index)
        predictions['actual'] = forecast_df[self.config['target_variable']]
        
        for model_name, model_data in self.models.items():
            model = model_data['model']
            
            if model_name == 'multiple_linear_regression':
                preds, lower, upper = predict_with_confidence_interval(model, X_forecast)
                predictions[f'{model_name}_prediction'] = preds
                predictions[f'{model_name}_lower'] = lower
                predictions[f'{model_name}_upper'] = upper
            else:
                preds = model.predict(X_forecast_scaled if hasattr(self, 'scaler') else X_forecast)
                predictions[f'{model_name}_prediction'] = preds
        
        self.predictions = predictions
        logger.info(f"Generated forecasts for {len(predictions)} dates")
        return predictions
    
    def save_model(self, model_name: Optional[str] = None) -> None:
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model to save. If None, saves all models.
        """
        if not self.models:
            logger.warning("No trained models to save.")
            return
        
        models_dir = Path("src/saved_models")
        models_dir.mkdir(parents=True, exist_ok=True)
        today_str = datetime.today().strftime('%Y-%m-%d')
        
        to_save = [model_name] if model_name else self.models.keys()
        
        for name in to_save:
            if name not in self.models:
                logger.warning(f"Model {name} not found.")
                continue
            
            model_data = self.models[name]
            save_path = models_dir / f"{today_str}-{name}.pkl"
            
            with open(save_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Saved {name} model to {save_path}")
    
    def update_historical_predictions(self, store_path: str = "data/prediction_store.json", target_var: str = "cpi_12_month_change") -> None:
        """
        Update the historical predictions store with new forecasts.
        
        Args:
            store_path: Base path for the prediction store (will be modified to use the structured format).
            target_var: Target variable being predicted, used for directory organization.
        """
        if self.predictions is None:
            logger.warning("No predictions available to update the store.")
            return
        
        # Create the directory structure if it doesn't exist
        prediction_dir = Path(f"data/predictions/{target_var}")
        prediction_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure we have predictions up to today
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Check if our predictions include today
        if today not in self.predictions.index:
            logger.warning(f"Predictions do not include current date ({today}). This might result in outdated forecasts.")
        
        # Process each model's predictions
        for model_name in self.models.keys():
            if f"{model_name}_prediction" not in self.predictions.columns:
                logger.warning(f"No predictions found for model {model_name}")
                continue
            
            # Define the specific store path for this model
            model_store_path = prediction_dir / f"{model_name}_predictions.json"
            
            # Load existing predictions for this model if they exist
            try:
                if model_store_path.exists():
                    with open(model_store_path, 'r') as f:
                        store = json.load(f)
                else:
                    store = {
                        "model_name": model_name,
                        "target_variable": target_var,
                        "last_updated": datetime.now().isoformat(),
                        "historical_predictions": []
                    }
            except Exception as e:
                logger.error(f"Error loading prediction store for {model_name}: {e}")
                store = {
                    "model_name": model_name,
                    "target_variable": target_var,
                    "last_updated": datetime.now().isoformat(),
                    "historical_predictions": []
                }
            
            # Extract predictions for this model
            model_preds = self.predictions[['actual', f'{model_name}_prediction']].copy()
            model_preds.columns = ['actual', 'prediction']
            
            # Add confidence intervals if available
            if f"{model_name}_lower" in self.predictions.columns:
                model_preds['lower_bound'] = self.predictions[f'{model_name}_lower']
                model_preds['upper_bound'] = self.predictions[f'{model_name}_upper']
            
            # Prepare the predictions in the required format for storage
            pred_records = []
            for date, row in model_preds.iterrows():
                # Convert date to string if it's a datetime object
                date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                
                record = {
                    "date": date_str,
                    "prediction": float(row['prediction'])  # Ensure it's a native Python float
                }
                
                if 'actual' in row and not pd.isna(row['actual']):
                    record["actual"] = float(row['actual'])
                
                if 'lower_bound' in model_preds.columns:
                    record["confidence_interval"] = [
                        float(row['lower_bound']), 
                        float(row['upper_bound'])
                    ]
                
                pred_records.append(record)
            
            # Update the store with new predictions
            existing_dates = {pred["date"] for pred in store["historical_predictions"]}
            
            # Only add new predictions
            new_records = [rec for rec in pred_records if rec["date"] not in existing_dates]
            
            if new_records:
                store["historical_predictions"].extend(new_records)
                store["last_updated"] = datetime.now().isoformat()
                logger.info(f"Added {len(new_records)} new predictions to {model_name} store")
            else:
                logger.info(f"No new predictions to add for {model_name}")
            
            # Save the updated store
            with open(model_store_path, 'w') as f:
                json.dump(store, f, indent=2)
            
            logger.info(f"Updated prediction store at {model_store_path}")
        
        # Also update the original store path for backward compatibility
        if store_path:
            try:
                store = load_prediction_store(store_path)
                
                for model_name in self.models.keys():
                    if f"{model_name}_prediction" not in self.predictions.columns:
                        continue
                    
                    # Extract predictions for this model
                    model_preds = self.predictions[['actual', f'{model_name}_prediction']].copy()
                    model_preds.columns = ['actual', 'prediction']
                    
                    # Add confidence intervals if available
                    if f"{model_name}_lower" in self.predictions.columns:
                        model_preds['lower_bound'] = self.predictions[f'{model_name}_lower']
                        model_preds['upper_bound'] = self.predictions[f'{model_name}_upper']
                    
                    # Prepare the predictions in the required format for storage
                    pred_records = []
                    for date, row in model_preds.iterrows():
                        # Convert date to string if it's a datetime object
                        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                        
                        record = {
                            "date": date_str,
                            "prediction": float(row['prediction'])  # Ensure it's a native Python float
                        }
                        
                        if 'lower_bound' in model_preds.columns:
                            record["confidence_interval"] = [
                                float(row['lower_bound']), 
                                float(row['upper_bound'])
                            ]
                        
                        pred_records.append(record)
                    
                    # Update the store for this model
                    if model_name not in store:
                        store[model_name] = {
                            "last_updated": datetime.now().isoformat(),
                            "historical_predictions": []
                        }
                    
                    updated_store = update_store(pred_records, store[model_name])
                    store[model_name] = updated_store
                
                # Save the updated store
                save_prediction_store(store, store_path)
                logger.info(f"Updated legacy prediction store at {store_path}")
            except Exception as e:
                logger.error(f"Error updating legacy prediction store: {e}")
    
    def plot_forecasts(self, save: bool = True) -> None:
        """
        Plot the forecasts from all models against the actual values.
        
        Args:
            save: Whether to save the plots to disk.
        """
        if self.predictions is None:
            logger.warning("No predictions available to plot.")
            return
        
        import matplotlib.pyplot as plt
        
        # Plot time series
        plt.figure(figsize=(12, 8))
        plt.plot(self.predictions.index, self.predictions['actual'], 'k-', label='Actual')
        
        for model_name in self.models.keys():
            col = f"{model_name}_prediction"
            if col in self.predictions.columns:
                plt.plot(self.predictions.index, self.predictions[col], '--', label=f'{model_name}')
                
                # Plot confidence intervals if available
                lower_col = f"{model_name}_lower"
                upper_col = f"{model_name}_upper"
                if lower_col in self.predictions.columns and upper_col in self.predictions.columns:
                    plt.fill_between(
                        self.predictions.index,
                        self.predictions[lower_col],
                        self.predictions[upper_col],
                        alpha=0.2
                    )
        
        plt.title('Financial Indicator Forecast')
        plt.xlabel('Date')
        plt.ylabel(self.config['target_variable'])
        plt.legend()
        plt.grid(True)
        
        if save:
            save_chart(plt, "forecast_comparison.png")
        else:
            plt.show()
        
        # Plot feature importance for applicable models
        for model_name in ['random_forest', 'xgboost']:
            if model_name in self.evaluations and 'feature_importance' in self.evaluations[model_name]:
                importance = self.evaluations[model_name]['feature_importance']
                
                # Sort features by importance
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                features = [item[0] for item in sorted_features]
                values = [item[1] for item in sorted_features]
                
                plt.figure(figsize=(10, 8))
                plt.barh(features, values)
                plt.title(f'{model_name} Feature Importance')
                plt.xlabel('Importance')
                plt.tight_layout()
                
                if save:
                    save_chart(plt, f"{model_name}_feature_importance.png")
                else:
                    plt.show()

# Function to get an SDK instance with configuration
def get_sdk(config_path: str = "config/model_config.yaml") -> FinancialForecastingSDK:
    """
    Get an instance of the FinancialForecastingSDK.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        An instance of FinancialForecastingSDK.
    """
    return FinancialForecastingSDK(config_path) 