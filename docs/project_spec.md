# Project Specification for Financial Indicator Forecasting SDK

This document outlines the design for a standardized and extensible SDK that supports rapid development of forecasting models for financial indicators (e.g., CPI and other variables from BLS and FRED). The SDK unifies data fetching, model training, forecasting, evaluation, and historical prediction storage.

---

## Repository Structure


```yaml
financial-forecasting-sdk/ 
├── README.md 
├── .env # Environment variables (e.g., BLS_API_KEY, FRED_API_KEY) 
├── config/ 
│ └── model_config.yaml # Configuration file for targets, features, and model parameters 
├── docs/ 
│ ├── images/ # Directory for saving generated charts 
│ └── report.md # Project documentation/report 
├── src/ 
│ ├── init.py 
│ ├── data_fetcher.py # Module for data fetching from BLS, FRED, Yahoo Finance 
│ ├── models/ 
│ │ ├── init.py 
│ │ ├── simple_models.py # Implementation of simple models (e.g., Linear, RF, XGBoost) 
│ │ └── advanced_models.py # (Optional) More advanced forecasting models 
│ ├── sdk.py # Core SDK logic: configuration, training, forecasting, saving 
│ ├── store_manager.py # Functions to create/update a JSON store of historical predictions 
│ └── utils.py # Utility functions (logging, chart saving, file I/O, etc.) 
├── tests/ 
│ ├── test_data_fetcher.py 
│ ├── test_models.py 
│ └── test_store_manager.py 
└── main.py # Entry point for running experiments using the SDK
```


---

## 1. Generalizing the Code into an SDK

**Module: `src/sdk.py`**

- **Purpose:** Provide a unified interface to:
  - Load configuration from a YAML file.
  - Fetch and preprocess data.
  - Train and evaluate various models.
  - Generate forecasts and update historical predictions.
  - Save trained models and generated outputs.

- **Key Functions:**
  - `load_config(config_path: str) -> dict`: Reads the YAML configuration.
  - `prepare_data(config: dict) -> pd.DataFrame`: Uses `data_fetcher` to create the training dataset.
  - `train_models(data: pd.DataFrame, config: dict) -> dict`: Trains multiple models (e.g., multiple linear regression, random forest, XGBoost) with parameters specified in the configuration.
  - `forecast(models: dict, data: pd.DataFrame, config: dict) -> pd.DataFrame`: Generates predictions for both historical and future dates.
  - `save_model(model, model_type: str) -> None`: Saves models to a standardized directory with date-based filenames.
  - `update_historical_predictions(predictions: pd.DataFrame, store_path: str) -> None`: Merges new predictions with the stored JSON object of historical forecasts.

---

## 2. Configuration File (YAML)

**File: `config/model_config.yaml`**

This YAML file defines the target variable, feature variables, and model parameters.

```yaml
# config/model_config.yaml
target_variable: target_CPI_12_month_change
feature_variables:
  - days_until_print
  - crude_oil
  - natural_gas
  - sp500
  - vix
  - treasury_yield
  - unemployment_rate
  - m2
  - fed_funds
  - gold
  - silver
  - copper
  - dollar
model_parameters:
  train_test_split:
    test_size: 0.2
    random_state: 42
  scaling: true
models:
  multiple_linear_regression:
    fit_intercept: true
  random_forest:
    n_estimators: 100
    random_state: 42
  xgboost:
    n_estimators: 100
    random_state: 42
```

---

### 3. Standardized Historical Predictions Store
Module: src/store_manager.py

Purpose: Define a standardized JSON structure and utility functions to create and update a historical predictions store.

JSON Structure Example:
```json
{
  "model_name": "random_forest",
  "last_updated": "2025-03-01T00:00:00",
  "historical_predictions": [
    {"date": "2025-02-28", "prediction": 2.3, "confidence_interval": [2.1, 2.5]},
    {"date": "2025-03-01", "prediction": 2.5, "confidence_interval": [2.3, 2.7]}
  ]
}
```
Key Functions:
load_prediction_store(store_path: str) -> dict: Loads the JSON file if it exists; otherwise, returns a default structure.
save_prediction_store(store: dict, store_path: str) -> None: Saves the JSON object to disk.
update_store(new_predictions: pd.DataFrame, store: dict) -> dict: Merges new predictions with existing ones, only appending records with dates later than the last update.

---

4. Utility Functions to Update Historical Data Stores
Module: src/utils.py

Purpose: Provide helper functions to merge and update prediction records.

Example Function:
```py
def merge_predictions(existing: list, new: list) -> list:
    """
    Merge new predictions with existing ones.
    Only include new predictions with dates later than the last recorded date.
    """
    if existing:
        last_date = max(entry['date'] for entry in existing)
    else:
        last_date = None
    merged = existing.copy()
    for entry in new:
        if last_date is None or entry['date'] > last_date:
            merged.append(entry)
    return merged
```
Additional utility functions should cover:
Saving charts to docs/images/.
Standard logging setup.
Reading and writing JSON/CSV files.

--- 

5. Output Structure for Inference
During model inference, the SDK should produce the following outputs:

Forecast File:
A single file (CSV or JSON) containing:
Date
Actual values (if available)
Predictions from each model (e.g., multiple_linear_regression, random_forest, xgboost, ensemble)
Confidence intervals (if computed)
Saved Models:
Models are stored in src/saved_models/ with date-based filenames.
Charts:
Visualizations (e.g., time series plots, scatter plots with true vs. predicted) are saved to docs/images/ via the save_chart utility.
Additional Considerations
Environment Variables:
Use a .env file and the python-dotenv package to manage API keys (BLS_API_KEY, FRED_API_KEY).

Testing:
Include unit tests for each module in the tests/ directory to ensure proper functionality.

Documentation:
Update README.md with detailed setup instructions, usage examples, and guidelines for extending the SDK.

Error Handling & Logging:
Use consistent logging (with appropriate logging levels) and graceful error handling across modules, especially for API calls and data merging.

Extensibility:
Design functions so that new models or data sources can be added with minimal changes to the overall codebase.

Final Summary
This specification outlines the architecture for a unified forecasting SDK. It supports data fetching, model training, evaluation, forecasting, and historical prediction storage in a consistent and configurable manner. By adhering to this design, you can efficiently build, test, and deploy multiple forecasting models, ensuring that configuration, repository structure, and output formats remain standardized.