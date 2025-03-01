# src/data_fetcher.py

import os
import datetime
import logging
from typing import Optional
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from fredapi import Fred
from pathlib import Path
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# BLS API configuration
BLS_API_KEY = os.getenv('BLS_API_KEY', None)
BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

def fetch_bls_cpi(series_id: str = 'CUUR0000SA0', start_year: str = '2000', end_year: str = '2025') -> pd.DataFrame:
    """
    Fetch CPI data from the Bureau of Labor Statistics (BLS) API.
    Processes data in 20‐year chunks.
    
    Returns:
        A DataFrame with a datetime index and a 'CPI' column.
    """
    headers = {'Content-type': 'application/json'}
    start_year_int = int(start_year)
    end_year_int = int(end_year)
    dfs = []

    # Get BLS API key from environment, log warning if missing
    api_key = os.getenv('BLS_API_KEY')
    if not api_key:
        logger.warning("BLS_API_KEY environment variable not set. Some API functionality may be limited.")
    
    # Create cache directory if it doesn't exist
    cache_dir = Path("data_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"bls_cpi_{series_id}_{start_year}_{end_year}.pkl"
    
    # Try to load from cache first
    if cache_file.exists():
        try:
            logger.info(f"Loading BLS CPI data from cache: {cache_file}")
            return pd.read_pickle(cache_file)
        except Exception as e:
            logger.warning(f"Failed to load cached BLS data: {e}")
    
    for chunk_start in range(start_year_int, end_year_int + 1, 20):
        chunk_end = min(chunk_start + 19, end_year_int)
        payload = {
            "seriesid": [series_id],
            "startyear": str(chunk_start),
            "endyear": str(chunk_end),
        }
        # Only add API key if it exists
        if api_key:
            payload["registrationkey"] = api_key
        
        try:
            response = requests.post(BLS_API_URL, json=payload, headers=headers)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"BLS API request failed for years {chunk_start}-{chunk_end}: {e}")
            continue

        json_data = response.json()
        if 'Results' not in json_data or 'series' not in json_data['Results']:
            logger.warning(f"No data returned for years {chunk_start}-{chunk_end}")
            continue

        results = json_data['Results']['series'][0].get('data', [])
        if not results:
            logger.warning(f"Empty results for years {chunk_start}-{chunk_end}")
            continue

        df = pd.DataFrame(results)
        # Remove the "M" prefix and convert to datetime
        df['period'] = df['period'].str.replace('M', '', regex=False)
        df['date'] = pd.to_datetime(df['year'] + '-' + df['period'] + '-01', errors='coerce')
        df = df.dropna(subset=['date'])
        df.sort_values('date', inplace=True)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df[['date', 'value']].rename(columns={'value': 'CPI'})
        df.set_index('date', inplace=True)
        dfs.append(df)

    if not dfs:
        raise ValueError("No data retrieved from BLS API for the specified period.")
    final_df = pd.concat(dfs)
    final_df = final_df[~final_df.index.duplicated(keep='first')]
    final_df.sort_index(inplace=True)
    
    # Save to cache
    try:
        final_df.to_pickle(cache_file)
        logger.info(f"Saved BLS CPI data to cache: {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to cache BLS data: {e}")
        
    return final_df

def fetch_fred_cpi(start_date: str = '2000-01-01', end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch CPI data from FRED using the "CPIAUCSL" series.
    
    Returns:
        A DataFrame with a datetime index and a 'CPI' column.
    """
    fred_api_key = os.getenv('FRED_API_KEY', 'YOUR_FRED_API_KEY')
    fred = Fred(api_key=fred_api_key)
    series = fred.get_series("CPIAUCSL", observation_start=start_date, observation_end=end_date)
    df = pd.DataFrame({'date': series.index, 'CPI': series.values})
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)
    return df

def fetch_fred_series(api_key: str, series_id: str, start_date: str = '2000-01-01', end_date: Optional[str] = None) -> pd.Series:
    """
    Fetch a data series from FRED.
    
    Returns:
        A pandas Series of the requested data.
    """
    # Create cache directory if it doesn't exist
    cache_dir = Path("data_cache")
    cache_dir.mkdir(exist_ok=True)
    
    # Generate cache filename based on parameters
    cache_key = f"{series_id}_{start_date}_{end_date or 'latest'}"
    cache_file = cache_dir / f"fred_{cache_key}.pkl"
    
    # Try to load from cache first
    if cache_file.exists():
        try:
            logger.info(f"Loading FRED data from cache: {cache_file}")
            return pd.read_pickle(cache_file)
        except Exception as e:
            logger.warning(f"Failed to load cached FRED data: {e}")
    
    # Validate API key
    if api_key == 'YOUR_FRED_API_KEY' or not api_key:
        logger.error("Valid FRED_API_KEY not provided. Set the FRED_API_KEY environment variable.")
        # Return dummy data if API key is invalid
        dummy_dates = pd.date_range(start=start_date, end=end_date or datetime.datetime.today(), freq='M')
        dummy_series = pd.Series(index=dummy_dates, data=np.zeros(len(dummy_dates)))
        logger.warning(f"Returning dummy data for {series_id} due to missing API key")
        return dummy_series
    
    # If we have a valid API key, proceed with the API call
    try:
        fred = Fred(api_key=api_key)
        series = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        
        # Save to cache
        try:
            series.to_pickle(cache_file)
            logger.info(f"Saved FRED data to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache FRED data: {e}")
            
        return series
    except ValueError as e:
        logger.error(f"FRED API error for series {series_id}: {e}")
        # Return dummy data on error
        dummy_dates = pd.date_range(start=start_date, end=end_date or datetime.datetime.today(), freq='M')
        dummy_series = pd.Series(index=dummy_dates, data=np.zeros(len(dummy_dates)))
        return dummy_series

def fetch_yahoo_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical financial data from Yahoo Finance for a given ticker.
    
    Returns:
        A DataFrame containing the ticker's historical data.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            logger.warning(f"No data returned for ticker {ticker}")
        return data
    except Exception as e:
        logger.error(f"Error downloading data for ticker {ticker}: {e}")
        return pd.DataFrame()

def build_daily_training_dataset() -> pd.DataFrame:
    """
    Constructs a training dataset for nowcasting the next CPI print.
    For every day between two consecutive monthly CPI prints (fetched from BLS),
    the target is the upcoming print's 12‐month percentage change. Daily features 
    are collected from Yahoo Finance and FRED.
    
    Returns:
        A DataFrame with one row per day including features, 'days_until_print',
        and the target 'target_CPI_12_month_change'.
    """
    # Load configuration 
    config_path = os.getenv('CONFIG_PATH', 'config/model_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 1. Fetch and process monthly CPI data from BLS using config values
    current_year = datetime.datetime.today().year
    series_id = config['data_sources']['bls'].get('series_id', 'CUUR0000SA0')
    start_year = config['data_sources']['bls'].get('start_year', '2000')
    end_year = config['data_sources']['bls'].get('end_year', str(current_year))
    
    cpi_df = fetch_bls_cpi(series_id=series_id, start_year=start_year, end_year=end_year)
    cpi_df.sort_index(inplace=True)
    cpi_df['CPI_12_month_change'] = cpi_df['CPI'].pct_change(12)
    cpi_df.dropna(subset=['CPI_12_month_change'], inplace=True)

    # New: Compute previous CPI 12-month change as an additional lag feature.
    cpi_df['prev_CPI_12_month_change'] = cpi_df['CPI_12_month_change'].shift(1)
    
    # Define date ranges from config or calculate from CPI data
    start_date = config['data_sources']['fred'].get('start_date', cpi_df.index.min().strftime('%Y-%m-%d'))
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')

    # 2. Fetch daily Yahoo Finance data.
    yahoo_tickers = {
        'crude_oil': 'CL=F',
        'natural_gas': 'NG=F',
        'sp500': '^GSPC',
        'gold': 'GC=F',
        'silver': 'SI=F',
        'copper': 'HG=F',
        'dollar': 'DX-Y.NYB',
        'vix': '^VIX'
    }
    # Create a DataFrame with a complete daily index
    daily_index = pd.date_range(start=start_date, end=end_date, freq='D')
    yahoo_data = pd.DataFrame(index=daily_index)
    for name, ticker in yahoo_tickers.items():
        df = fetch_yahoo_data(ticker, start_date, end_date)
        if not df.empty and 'Close' in df.columns:
            yahoo_data[name] = df['Close']
        else:
            logger.warning(f"Ticker {ticker} returned empty or missing 'Close' data.")
    if yahoo_data.empty:
        raise ValueError("Failed to fetch any Yahoo Finance data. Check your connection and ticker symbols.")
    yahoo_data = yahoo_data.ffill().bfill()

    # 3. Fetch daily FRED data.
    fred_series_ids = {
        'treasury_yield': 'DGS10',        # 10-Year Treasury Yield
        'unemployment_rate': 'UNRATE',    # Unemployment Rate
        'm2': 'M2SL',                     # Money Supply, M2
        'fed_funds': 'FEDFUNDS',           # Federal Funds Rate
    }
    fred_data = {}
    fred_api_key = os.getenv('FRED_API_KEY', 'YOUR_FRED_API_KEY')
    for name, series_id in fred_series_ids.items():
        series = fetch_fred_series(fred_api_key, series_id, start_date=start_date, end_date=end_date)
        series_df = pd.DataFrame(series, columns=[name])
        series_df.index = pd.to_datetime(series_df.index)
        series_df = series_df.reindex(daily_index).ffill()
        fred_data[name] = series_df[name]
    fred_df = pd.DataFrame(fred_data)

    # 4. Merge Yahoo and FRED data.
    daily_features = pd.concat([yahoo_data, fred_df], axis=1)
    daily_features.index.name = 'date'
    daily_features.sort_index(inplace=True)
    daily_features = daily_features.ffill().bfill()

    # New: Add lagged percentage change features for each daily feature.
    for col in daily_features.columns:
        daily_features[f"{col}_pct_change_14"] = daily_features[col].pct_change(14)
        daily_features[f"{col}_pct_change_30"] = daily_features[col].pct_change(30)
    # Forward/backward fill the newly created lagged change columns.
    daily_features = daily_features.ffill().bfill()

    # 5. Build the training dataset: for each monthly CPI print, assign each preceding day the target.
    training_rows = []
    monthly_prints = cpi_df.index.sort_values()
    previous_print = None
    for print_date in monthly_prints:
        target_value = cpi_df.loc[print_date, 'CPI_12_month_change']
        # Retrieve the previous CPI 12-month change as a lag feature.
        prev_inflation = cpi_df.loc[print_date, 'prev_CPI_12_month_change'] if 'prev_CPI_12_month_change' in cpi_df.columns else np.nan
        start_interval = daily_features.index.min() if previous_print is None else previous_print
        day_range = pd.date_range(start=start_interval, end=print_date - pd.Timedelta(days=1), freq='D')
        for current_day in day_range:
            if current_day in daily_features.index:
                features = daily_features.loc[current_day].to_dict()
                features['days_until_print'] = (print_date - current_day).days
                features['target_CPI_12_month_change'] = target_value
                features['prev_CPI_12_month_change'] = prev_inflation
                # Optionally, include print_date and current_day for reference.
                features['print_date'] = print_date.strftime('%Y-%m-%d')
                features['current_day'] = current_day.strftime('%Y-%m-%d')
                training_rows.append(features)
        previous_print = print_date

    training_df = pd.DataFrame(training_rows)
    training_df.set_index('current_day', inplace=True)
    return training_df

if __name__ == '__main__':
    # For testing: build and display the training dataset.
    dataset = build_daily_training_dataset()
    logger.info("Training dataset head:")
    logger.info(dataset.head())
    logger.info("Training dataset tail:")
    logger.info(dataset.tail())
