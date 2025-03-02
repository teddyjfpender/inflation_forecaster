# Financial Indicator Forecasting SDK

A standardized and extensible SDK for developing forecasting models for financial indicators (e.g., CPI and other variables from BLS and FRED). The SDK unifies data fetching, model training, forecasting, evaluation, and historical prediction storage.

## Features

- **Unified SDK Interface:** A standardized interface for configuration, data preparation, model training, and forecasting.
- **Data Integration:** Fetch CPI data from the Bureau of Labor Statistics (BLS) plus additional indicators from FRED and Yahoo Finance.
- **Configurable Models:** Train multiple models (e.g., Linear Regression, Random Forest, XGBoost) with parameters specified in a YAML configuration file.
- **Historical Prediction Storage:** Store and track forecasts over time to evaluate model performance.
- **Visualization Tools:** Generate and save charts for model comparisons and feature importance.
- **Improved Organization:** Structured prediction storage by target variable and model type.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/inflation_forecaster.git
   cd inflation_forecaster
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your API keys (BLS_API_KEY, FRED_API_KEY) to the `.env` file

## Usage

### Running the Inflation Model

The main entry point is `run_inflation_model.py`, which provides a streamlined way to train models and generate forecasts:

```bash
# Run the inflation model with all default settings
python run_inflation_model.py

# Use the SDK as a command-line tool (if installed with pip install -e .)
inflation_forecaster
```

### Scheduled Model Runs

The repository includes several options for setting up scheduled model runs:

- **GitHub Actions**: Automatically runs every day at 6 AM UTC (recommended)
- **Python Script**: Flexible script with logging and GitHub PR creation
- **Shell Script**: Simple bash script for cron setups
- **Docker Container**: Isolated environment with optional scheduling
- **Systemd Service**: For Linux servers

For detailed setup instructions, see [docs/ScheduledModelRuns.md](docs/ScheduledModelRuns.md).

### Prediction Storage

Predictions are stored in an organized directory structure:

```
data/predictions/{target_variable}/{model_name}_predictions.json
```

For example:
```
data/predictions/cpi_12_month_change/random_forest_predictions.json
```

### Project Structure

```
inflation_forecaster/
├── README.md
├── .env                          # Environment variables (API keys)
├── config/
│   └── model_config.yaml         # Configuration file
├── docs/
│   ├── images/                   # Directory for saved charts
│   ├── report.md                 # Technical documentation
│   └── project_spec.md           # Project specifications
├── src/
│   ├── data_fetcher.py           # Data fetching from BLS, FRED, Yahoo
│   ├── models/
│   │   └── simple_models.py      # Model implementations
│   ├── sdk.py                    # Core SDK logic
│   ├── store_manager.py          # Historical predictions store
│   ├── saved_models/             # Saved model files
│   └── utils.py                  # Utility functions
├── tests/
│   ├── test_data_fetcher.py
│   └── test_models.py
├── run_inflation_model.py        # Main script to run the model
└── setup.py                      # Package setup
```

## Configuration

The SDK uses a YAML configuration file (`config/model_config.yaml`) to specify:

- Target variable (e.g., CPI 12-month change)
- Feature variables (e.g., crude oil, S&P 500, treasury yield)
- Model parameters (e.g., train-test split, scaling)
- Specific model configurations (e.g., Random Forest n_estimators)
- Data source parameters

## Development

To contribute to this project:

1. Make sure you have the development dependencies installed:
   ```
   pip install -e .
   ```

2. Run tests:
   ```
   python -m unittest discover tests
   ```

3. Add new features or models to the appropriate modules

## License

MIT License
