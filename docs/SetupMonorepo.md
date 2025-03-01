# Setting Up the Python Monorepo

This guide explains how to set up and organize the inflation_forecaster monorepo.

## Directory Structure

inflation_forecaster/ 
├── docs/ │ ├── SUMMARY.md │ ├── GettingStarted.md │ ├── EDA.md │ ├── ModelResults.md │ └── SetupMonorepo.md ├── notebooks/ │ └── eda.ipynb ├── src/ │ ├── init.py │ ├── data_fetcher.py │ ├── feature_engineering.py │ ├── main.py │ ├── models/ │ │ ├── init.py │ │ ├── simple_models.py │ │ ├── deep_models.py │ │ └── ensemble.py │ └── utils/ │ ├── init.py │ ├── logger.py │ └── config.py ├── tests/ │ ├── test_data_fetcher.py │ └── test_models.py ├── README.md ├── requirements.txt └── setup.py


## Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/inflation_forecaster.git
   cd inflation_forecaster
```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Set Environment Variables:**
   ```bash
   export BLS_API_KEY='your_bls_api_key'
   export FRED_API_KEY='your_fred_api_key'
   ```

5. Launch EDA Notebook:
   ```bash
   jupyter notebook notebooks/eda.ipynb
   ```