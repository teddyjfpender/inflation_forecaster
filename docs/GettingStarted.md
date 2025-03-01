# Getting Started

Welcome to the Inflation Forecaster library.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/inflation_forecaster.git
   cd inflation_forecaster

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. Install dependencies:
   ```bash
   pip install --upgrade -r requirements.txt
   pip install -e .
   ```

4. Set environment variables:
   ```bash
   export BLS_API_KEY='your_bls_api_key'
   export FRED_API_KEY='your_fred_api_key'
   ```

5. Train & Save the Models:
   ```bash
   python src/main.py
   ```

6. Predict Inflation:
   ```bash
   python predict_inflation.py --current_date 02-20-2025 --print_date 03-15-2025
   ```

## Repository Structure

 - **src/**: Core modules for data fetching, feature engineering, models, and utilities.
 - **notebooks/**: Jupyter notebooks for exploratory data analysis.
 - **docs/**: GitBook-style documentation.
 - **tests/**: Unit tests.

---

### File: docs/EDA.md

```markdown
# Exploratory Data Analysis

This document details the EDA performed on the CPI data and supplementary features.

## Data Sources

- **BLS CPI Data:** Monthly Consumer Price Index from the Bureau of Labor Statistics.
- **FRED Data:** Supplementary economic indicators.
- **Yahoo Finance Data:** Market data (e.g. crude oil prices).

## EDA Process

1. **Visualization:**  
   - Plot CPI trends over time.
   - Examine seasonality and trends.
2. **Feature Engineering:**  
   - Creation of lag features (e.g. previous month, 12-month lag).
   - Calculation of year-over-year inflation.
3. **Feature Importance:**  
   - Use a Random Forest to assess which features are most predictive.

For full details and code, see the `notebooks/eda.ipynb` notebook.

```