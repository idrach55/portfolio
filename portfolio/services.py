"""
Author: Isaac Drachman
Date: 8/16/2021

Simple API clients for various financial data providers -- mostly AlphaVantage. 
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import requests

# CONSTANTS
KEY_DIR = os.environ.get('KEY_DIR', 'keys')

class AlphaVantage:
    def __init__(self, api_key: Optional[str] = None):
        """
        Create the API service object.
        """
        self.api_key = api_key
        if self.api_key is None:
            self.api_key = open(f'{KEY_DIR}/alphavantage.keys','r').read().split('\n')[0]
        elif self.api_key[-5:] == '.keys':
            self.api_key = open(api_key,'r').read().split('\n')[0]
        self.root_url = 'https://www.alphavantage.co/query?'

    def chart(self, symbol: str, adjusted: bool = True) -> pd.DataFrame:
        """
        Return historical data (OHLC, dividend, split, etc.)
        """
        suffix = '_ADJUSTED' if adjusted else ''
        function = f'TIME_SERIES_DAILY{suffix}'
        url = self.root_url+'function={}&symbol={}&outputsize=full&apikey={}'.format(function, symbol, self.api_key)
        #print(url)
        r = requests.get(url)

        # If time series key unavailable, data not retrieved - raise error.
        assert r.json().get('Time Series (Daily)') is not None, f'missing data for {symbol}'

        # Assume json is well-formed and includes data.
        data = pd.DataFrame(r.json()['Time Series (Daily)']).T[::-1].astype(np.float)
        data.index = pd.to_datetime(data.index)
        data.columns = [col[col.find(' ')+1:] for col in data.columns]
        return data

    def earnings(self, symbol: str):
        url = self.root_url+'function=EARNINGS&symbol={}&apikey={}'.format(symbol, self.api_key)
        r = requests.get(url)
        data = r.json()
        return data

    def quote(self, symbol: str) -> pd.DataFrame:
        url = self.root_url+'function=GLOBAL_QUOTE&symbol={}&apikey={}'.format(symbol, self.api_key)
        r = requests.get(url)
        return r.json()

    def overview(self, symbol: str) -> pd.DataFrame:
        url = self.root_url+'function=OVERVIEW&symbol={}&apikey={}'.format(symbol, self.api_key)
        r = requests.get(url)
        return r.json()

class Fred:
    def __init__(self):
        self.url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id={}'

    def load(self, symbol: str) -> pd.Series:
        """
        T10YIE : 10 year treasury inflation breakeven
        BAMLH0A0HYM2EY : BAML HY corp yield index
        BAMLC0A0CMEY   : BAML IG corp yield index
        CPILFESL       : Urban CPI ex-food & energy
        VIXCLS         : VIX index
        """
        df = pd.read_csv(self.url.format(symbol),index_col=0)
        df.index = pd.to_datetime(df.index)
        return df[symbol]