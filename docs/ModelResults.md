# Model Results and Testing Accuracy

This document summarizes model performance.

## Models Evaluated

1. **Simple Models:**
   - **ARIMA/SARIMAX**
   - **Random Forest Regressor**
   - **XGBoost Regressor**

2. **Deep Learning Model:**
   - **LSTM**

## Evaluation Metrics

- **RMSE:** Used to compare forecast accuracy.
- Models were trained/tested using an 80/20 split.

## Example Results

- **ARIMA RMSE:** ~0.45  
- **Random Forest RMSE:** ~0.40  
- **XGBoost RMSE:** ~0.38  
- **LSTM RMSE:** ~0.35  
- **Ensemble RMSE:** ~0.33  

*Note: These values are examples; actual performance may vary.*

## Ensemble Method

The final forecast is computed as a weighted average of the individual model predictions (with weights inversely proportional to their RMSE).
