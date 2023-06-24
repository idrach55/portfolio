"""
Author: Isaac Drachman
Date: 8/16/2021

Simple API clients for various financial data providers -- mostly AlphaVantage. 
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import quandl
import requests

# CONSTANTS
KEY_DIR = 'keys'


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
        if r.json().get('Time Series (Daily)') is None:
            raise Exception(r.json()['Note'])

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


class Quandl:
    def __init__(self, token: Optional[str] = None):
        self.token = token
        if self.token is None:
            self.token = open(f'{KEY_DIR}/quandl.keys','r').read().split('\n')[0]
        elif self.token[-5:] == '.keys':
            self.token = open(token,'r').read().split('\n')[0]

    def get_yield_curve(self) -> pd.DataFrame:
        def rename_col(col):
            if 'MO' in col:
                return col[:col.find(' ')]+'m'
            elif 'YR' in col:
                return col[:col.find(' ')]+'y'
        data = quandl.get('USTREASURY/YIELD', authtoken=self.token)
        data.columns = [rename_col(col) for col in data.columns]
        return data

    def get_ndx_factor_idx(self, factors: List[str]) -> pd.DataFrame:
        data = pd.DataFrame()
        for factor in factors:
            data[factor] = quandl.get('NASDAQOMX/{}'.format(factor), authtoken=self.token)['Index Value']
        return data


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


# Unused services: Polygon, IEX
class Polygon:
    def __init__(self, api_key: Optional[str] = None):
        # Process api key as either given or filename
        self.api_key = api_key
        if self.api_key[-5:] == '.keys':
            self.api_key = open(api_key,'r').read().split('\n')[0]
        self.root_url = 'https://api.polygon.io/v2/'

    def aggs(self, ticker: str, from_date: str, to_date: str, timespan: str= 'day', unadjusted: bool = False) -> pd.DataFrame:
        supported_spans = ['minute','hour','day','week','month','quarter','year']
        assert timespan in supported_spans, f'timespan {timespan} not supported'

        # Construct url for this request and GET it
        url = self.root_url+'aggs/ticker/{}/range/1/{}/{}/{}?unadjusted={}&sort=asc&apiKey={}'
        r = requests.get(url.format(ticker,timespan,from_date,to_date,unadjusted,self.api_key))

        # Cast to pandas dataframe and clean
        data = pd.DataFrame(r.json()['results'])
        data.index = pd.to_datetime(data.t, unit='ms')
        data.index = pd.to_datetime(data.index.date)
        return data.drop('t',axis=1)

    def dividends(self, ticker: str, adjusted: bool = True) -> pd.DataFrame:
        url = self.root_url+'reference/dividends/{}?apiKey={}'
        r = requests.get(url.format(ticker,self.api_key))

        data = pd.DataFrame(r.json()['results'])
        data.exDate = pd.to_datetime(data.exDate)
        if not adjusted:
            return data

        factor = pd.Series([1.0]*len(data), index=data.exDate)
        stock_splits = self.splits(ticker)
        stock_splits = stock_splits.loc[stock_splits.exDate > factor.index[-1]]
        for idx,split in stock_splits.iterrows():
            factor[factor.index < split.exDate] *= split.ratio
        data_adj = data.copy()
        data_adj.index = data_adj.exDate
        data_adj.amount *= factor
        return data_adj

    def splits(self, ticker: str) -> pd.DataFrame:
        url = self.root_url+'reference/splits/{}?apiKey={}'
        r = requests.get(url.format(ticker,self.api_key))
        data = pd.DataFrame(r.json()['results'])
        data.exDate = pd.to_datetime(data.exDate)
        return data


class IEX:
    def __init__(self, token: Optional[str] = None):
        self.token = token
        if self.token[-5:] == '.keys':
            self.token = open(token,'r').read().split('\n')[0]
        self.root_url = 'https://cloud.iexapis.com/stable/'

    def chart(self, symbol: str, range: str, close_only: bool = True) -> pd.DataFrame:
        url = self.root_url+'stock/{}/chart/{}?chartCloseOnly={}&token={}'
        r = requests.get(url.format(symbol,range,close_only,self.token))
        data = pd.DataFrame(r.json())
        data.index = pd.to_datetime(data.date)
        data = data.drop('date',axis=1)
        return data

    # Return adjusted closes for several symbols over a range
    def charts(self, symbols: List[str], range: str) -> pd.DataFrame:
        data = pd.DataFrame()
        for symbol in symbols:
            price_data = self.chart(symbol, range, close_only=True)
            data[symbol] = price_data.close
        return data

    def dividends(self, symbol: str, range: str) -> pd.DataFrame:
        url = self.root_url+'stock/{}/dividends/{}?token={}'
        r = requests.get(url.format(symbol,range,self.token))
        data = pd.DataFrame(r.json())
        for column in ['exDate','paymentDate','recordDate','declaredDate']:
            data[column] = pd.to_datetime(data[column])
        return data