import os

class Config:
    BLS_API_KEY = os.getenv('BLS_API_KEY', 'YOUR_BLS_API_KEY')
    FRED_API_KEY = os.getenv('FRED_API_KEY', 'YOUR_FRED_API_KEY')
    START_YEAR = '2000'
    END_YEAR = '2024'
    # Extend with additional configuration parameters as needed
