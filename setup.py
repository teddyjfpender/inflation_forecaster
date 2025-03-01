from setuptools import setup, find_packages

setup(
    name='inflation_forecaster',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A library to forecast US inflation using statistical and deep learning models.',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'statsmodels',
        'pmdarima',
        'xgboost',
        'tensorflow',
        'keras',
        'yfinance',
        'fredapi',
        'jupyter',
        'python-dotenv',
        'pyyaml',
        'requests'
    ],
    entry_points={
        'console_scripts': [
            'inflation_forecaster=run_inflation_model:main'
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
    ],
)
