# src/models/simple_models.py

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Union, Dict, Any
import warnings

def train_arima(
    train_series: pd.Series, 
    order: Tuple[int, int, int] = (1, 1, 1), 
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0)
) -> SARIMAX:
    """
    Train a SARIMAX model on the target series.
    """
    model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    return results

def train_random_forest(
    X_train: Union[pd.DataFrame, np.ndarray], 
    y_train: Union[pd.Series, np.ndarray]
) -> RandomForestRegressor:
    """
    Train a Random Forest regressor.
    """
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf

def train_xgboost(
    X_train: Union[pd.DataFrame, np.ndarray], 
    y_train: Union[pd.Series, np.ndarray]
) -> XGBRegressor:
    """
    Train an XGBoost regressor.
    """
    xgb = XGBRegressor(n_estimators=100, random_state=42)
    xgb.fit(X_train, y_train)
    return xgb

def evaluate_model(
    model: Any, 
    X_test: Union[pd.DataFrame, np.ndarray, pd.Series], 
    y_test: Union[pd.Series, np.ndarray], 
    model_type: str = 'rf'
) -> Tuple[float, np.ndarray]:
    """
    Evaluate a model and return the RMSE and predictions.
    
    For ARIMA, X_test is ignored in favor of forecasting.
    For other models, predictions are made on the provided features.
    """
    if isinstance(model, tuple):
        model = model[0]
    
    if model_type == 'arima':
        preds = model.forecast(steps=len(y_test))
    else:
        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test)
        preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse, preds

def train_multiple_linear_regression(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    feature_names: list = None,
    fit_intercept: bool = False
) -> Tuple[LinearRegression, Dict[str, Any]]:
    """
    Train a multiple linear regression model and return diagnostics.
    """
    if isinstance(X_train, np.ndarray):
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        X_train = pd.DataFrame(X_train, columns=feature_names)
    elif not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be either a pandas DataFrame or a numpy array.")
    
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same length.")
    
    if X_train.isnull().any().any() or (isinstance(y_train, pd.Series) and y_train.isnull().any()):
        warnings.warn("Missing values detected in the input data.")
    
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X_train, y_train)
    # Store training data and compute additional statistics for prediction intervals
    X_train_arr = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    model.X_train_ = X_train_arr
    model.y_train_ = y_train.values if isinstance(y_train, pd.Series) else y_train
    y_pred = model.predict(X_train)
    r2 = model.score(X_train, y_train)
    n = X_train_arr.shape[0]
    p = X_train_arr.shape[1]
    if fit_intercept:
        df = n - p - 1
        X_design = np.column_stack([np.ones(n), X_train_arr])
    else:
        df = n - p
        X_design = X_train_arr
    sigma_squared = np.sum((y_train - y_pred)**2) / df
    model.sigma_squared_ = sigma_squared
    model.fit_intercept_ = fit_intercept
    model.XtX_inv_ = np.linalg.inv(X_design.T.dot(X_design))
    
    coef_df = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': model.coef_
    })
    if fit_intercept and hasattr(model, 'intercept_'):
        intercept_df = pd.DataFrame({'feature': ['intercept'], 'coefficient': [model.intercept_]})
        coef_df = pd.concat([intercept_df, coef_df], ignore_index=True)
    
    X_scaled = StandardScaler().fit_transform(X_train)
    eigenvals = np.linalg.eigvals(X_scaled.T @ X_scaled)
    condition_number = np.sqrt(np.max(eigenvals) / np.min(eigenvals))
    
    if condition_number > 30:
        warnings.warn(f"High condition number ({condition_number:.2f}) indicates potential multicollinearity.")
    
    diagnostics = {
        'r2_score': r2,
        'condition_number': condition_number,
        'n_observations': n,
        'n_features': p
    }
    
    return model, diagnostics

def predict_multiple_linear_regression(
    model: LinearRegression,
    X_test: pd.DataFrame,
    return_std: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Make predictions using a trained multiple linear regression model.
    
    Optionally returns a (simplified) standard deviation for predictions.
    """
    if not isinstance(model, LinearRegression):
        raise TypeError("Model must be a LinearRegression instance.")
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    
    predictions = model.predict(X_test)
    if return_std:
        mse = np.mean((predictions - model.predict(X_test)) ** 2)
        std = np.sqrt(mse) * np.ones_like(predictions)
        return predictions, std
    return predictions

def predict_with_confidence_interval(
    model: LinearRegression,
    X_test: pd.DataFrame,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Make predictions with confidence intervals for a multiple linear regression model.
    This function uses the training data stored in the model to compute prediction variances.

    Returns:
        predictions: The predicted values.
        lower_bound: The lower bound of the confidence interval.
        upper_bound: The upper bound of the confidence interval.
    """
    from scipy.stats import t
    predictions = model.predict(X_test)
    if model.fit_intercept_:
        X_design_test = np.column_stack([np.ones(len(X_test)), X_test.values])
        df = model.X_train_.shape[0] - (model.X_train_.shape[1] + 1)
    else:
        X_design_test = X_test.values
        df = model.X_train_.shape[0] - model.X_train_.shape[1]
    # Compute the standard error for each prediction:
    # SE = sqrt( sigma^2 * (1 + x' (X'X)^{-1} x) )
    se_pred = np.sqrt(model.sigma_squared_ * (1 + np.sum((X_design_test.dot(model.XtX_inv_)) * X_design_test, axis=1)))
    # t critical value for a two-tailed test
    t_val = t.ppf(1 - alpha / 2, df)
    lower_bound = predictions - t_val * se_pred
    upper_bound = predictions + t_val * se_pred
    return predictions, lower_bound, upper_bound
