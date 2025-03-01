import unittest
import pandas as pd
import numpy as np
from src.models import simple_models, deep_models

class TestModels(unittest.TestCase):
    def setUp(self):
        # Create a synthetic dataset for testing
        dates = pd.date_range(start='2020-01-01', periods=50, freq='M')
        self.df = pd.DataFrame({
            'date': dates,
            'CPI': np.linspace(100, 150, 50)
        })
        self.df['CPI_lag1'] = self.df['CPI'].shift(1)
        self.df['CPI_lag12'] = self.df['CPI'].shift(12)
        self.df['inflation_yoy'] = self.df['CPI'].pct_change(12) * 100
        self.df.dropna(inplace=True)
        self.X = self.df[['CPI', 'CPI_lag1']]
        self.y = self.df['inflation_yoy']
    
    def test_arima(self):
        model = simple_models.train_arima(self.df['CPI'])
        self.assertIsNotNone(model)
    
    def test_random_forest(self):
        model = simple_models.train_random_forest(self.X, self.y)
        self.assertIsNotNone(model)
    
    def test_xgboost(self):
        model = simple_models.train_xgboost(self.X, self.y)
        self.assertIsNotNone(model)
    
    def test_lstm_preparation(self):
        X, y = deep_models.prepare_lstm_data(self.df.set_index('date'), target_column='inflation_yoy', timesteps=3)
        self.assertEqual(X.shape[0], len(y))
    
if __name__ == '__main__':
    unittest.main()
