import pandas as pd
import numpy as np

def generate_features(cpi_df, fred_series_dict=None, yahoo_data_dict=None):
    """
    Merge CPI data with optional FRED and Yahoo Finance features, and create lag features.

    :param cpi_df: DataFrame containing CPI data with either a 'date' column or date index.
    :param fred_series_dict: Optional dictionary {feature_name: pandas Series} from FRED.
    :param yahoo_data_dict: Optional dictionary {ticker: DataFrame} from Yahoo Finance.
    :return: DataFrame with merged features.
    """
    df = cpi_df.copy()
    
    # Only set index if 'date' is a column and not already the index
    if 'date' in df.columns and not df.index.name == 'date':
        df.set_index('date', inplace=True)
    
    print(f"Shape after initial load: {df.shape}")
    
    # Create lagged features first
    df['CPI_lag1'] = df['CPI'].shift(1)
    df['CPI_lag12'] = df['CPI'].shift(12)
    
    print(f"Shape after creating lags: {df.shape}")
    
    # Calculate year-over-year inflation rate
    df['inflation_yoy'] = (df['CPI'] / df['CPI'].shift(12) - 1) * 100
    
    print(f"Shape after calculating inflation: {df.shape}")
    
    # Add FRED series features (resampled to monthly)
    if fred_series_dict:
        # Convert all FRED series to DataFrames with consistent frequency
        fred_dfs = {}
        for key, series in fred_series_dict.items():
            # Convert series to DataFrame and resample
            series_df = pd.DataFrame(series)
            series_df.columns = [key]
            series_df = series_df.resample('ME').last()
            fred_dfs[key] = series_df
        
        # Merge all FRED DataFrames
        for key, fred_df in fred_dfs.items():
            df = df.join(fred_df, how='left')
            print(f"Shape after adding {key}: {df.shape}")
    
    # Add Yahoo Finance features (monthly averages)
    if yahoo_data_dict:
        for ticker, data in yahoo_data_dict.items():
            monthly_data = data['Close'].resample('ME').mean()
            df = df.join(pd.DataFrame(monthly_data, columns=[ticker]), how='left')
            print(f"Shape after adding {ticker}: {df.shape}")
    
    # Print info about NaN values before dropping
    print("\nNaN values before dropping:")
    print(df.isna().sum())
    
    # Drop NaN values but only for rows after we have enough data for lagged features
    start_date = df.index[12]  # Start after we have enough data for 12-month lag
    df = df[df.index >= start_date]
    
    # Forward fill any missing values in FRED data (common for financial data)
    fred_columns = list(fred_series_dict.keys()) if fred_series_dict else []
    if fred_columns:
        df[fred_columns] = df[fred_columns].fillna(method='ffill')
    
    # Drop any remaining NaN values
    df = df.dropna()
    
    print(f"\nFinal shape: {df.shape}")
    print("\nFirst few rows of final dataset:")
    print(df.head())
    
    return df

def feature_importance(df, target_column='target_CPI_12_month_change'):
    """
    Calculate feature importance using Random Forest
    
    Args:
        df: DataFrame containing features and target
        target_column: name of the target column (default: 'target_CPI_12_month_change')
    
    Returns:
        DataFrame with feature importance scores
    """
    from sklearn.ensemble import RandomForestRegressor
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    importance = model.feature_importances_
    features = X.columns
    fi_df = pd.DataFrame({'feature': features, 'importance': importance}).sort_values('importance', ascending=False)
    return fi_df
