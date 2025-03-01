import unittest
import pandas as pd
from src.data_fetcher import fetch_bls_cpi

class TestDataFetcher(unittest.TestCase):
    def test_fetch_bls_cpi(self):
        # This test assumes that a valid BLS_API_KEY is set in your environment.
        df = fetch_bls_cpi(start_year='2020', end_year='2021')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('date', df.columns)
        self.assertIn('CPI', df.columns)
        self.assertGreater(len(df), 0)

if __name__ == '__main__':
    unittest.main()