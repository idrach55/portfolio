"""
Author: Isaac Drachman
Date: 8/16/2021

Code for downloading stock and other financial data, processing/transforming, and computing derived statistics.
"""

from .services import AlphaVantage as av
from .services import Quandl as ql

import pandas as pd
import numpy as np
import time
import tqdm
import os
import scipy.optimize as opt
import requests

from bs4 import BeautifulSoup as Soup
from datetime import datetime, date
from sklearn.linear_model import LinearRegression as OLS
from urllib.request import urlopen
from zipfile import ZipFile
from bs4 import BeautifulSoup
from io import StringIO
from typing import Dict, List, Tuple

# CONSTANTS
DATA_DIR = 'data'


def download_rental_data(key='All'):
    """
    Download median asking rent and inventory from StreetEasy.

    key: All, OneBd
    """
    categories = ['medianAskingRent_{}'.format(key), 'rentalInventory_{}'.format(key)]
    for cat in categories:
        if '{}.csv'.format(cat) in os.listdir(DATA_DIR):
            os.remove('{}/{}.csv'.format(DATA_DIR, cat))

        url = 'https://streeteasy-market-data-download.s3.amazonaws.com/rentals/{}/{}.zip'.format(key, cat)
        with open('data.zip','wb') as file:
            file.write(urlopen(url).read())
        ZipFile('data.zip').extractall(f'{DATA_DIR}/')
        os.remove('data.zip')


def has_options(symbol):
    r = requests.get('https://finance.yahoo.com/quote/{}/options?p={}'.format(symbol, symbol))
    soup = BeautifulSoup(r.text, features='lxml')
    for span in soup.find_all('span'):
        if span.get_text() == 'Options data is not available':
            return False
    return True


def get_ishares_data(etf: str) -> pd.DataFrame:
    """
    Get iShares ETF composiiton.
    """
    etfs = {'IVV': '/us/products/239726/ishares-core-sp-500-etf',
            'HYG': '/us/products/239565/ishares-iboxx-high-yield-corporate-bond-etf'}

    # Find corresponding link for ETF and read whole page
    base_url = 'https://www.ishares.com'
    r = requests.get('{}{}'.format(base_url, etfs[etf]))
    # Locate link for holdings csv
    soup = BeautifulSoup(r.text, features='lxml')
    links = soup.find_all('a', href=True)
    a = [link for link in links if link.get_text() == 'Detailed Holdings and Analytics']
    if len(a) == 0:
        raise Exception('error: no holdings link found')
    csv_url = a[0].get('href')
    # Request holdings csv, put into temp file, and read into pandas dataframe
    r = requests.get('{}{}'.format(base_url, csv_url))
    data = r.text[r.text.find('\nName'):]
    tmpfile = StringIO(data)
    df = pd.read_csv(tmpfile, index_col=0)
    return df.iloc[:-1]


def get_etf_category(symbol: str) -> str:
    """
    Get the category from etfdb.com. These are also cached.
    """

    cache = pd.DataFrame()
    if 'etfs.csv' in os.listdir(DATA_DIR):
        cache = pd.read_csv(f'{DATA_DIR}/etfs.csv',index_col=0)
        if symbol in cache.index:
            return cache.loc[symbol].category

    url = 'https://etfdb.com/etf/{}/#etf-ticker-profile'
    r = requests.get(url.format(symbol))
    soup = Soup(r.text, features='lxml')
    spans = soup.find_all('span', class_='stock-quote-data')
    category = spans[2].get_text().strip() if len(spans) >= 3 else 'none'
    cache.append(pd.Series({'category': category}, name=symbol)).to_csv(f'{DATA_DIR}/etfs.csv')
    return category


def date_offset(timespan: str) -> pd.Timedelta:
    """
    Turn a date code, ex '5y', '3m', '1w' into a cutoff date.
    """

    multiplier = 1
    if timespan[-1] == 'y':
        multiplier = 52
    elif timespan[-1] == 'm':
        multiplier = 12
    return pd.Timedelta('{:d}w'.format(multiplier*int(timespan[:-1])))


def get_max_drawdown(ts, index=False) -> float:
    """
    Compute max drawdown on price history.

    ts:     pd.Series or list of prices
    index:  (optional) flag whether to return the location of the drawdown
    returns maximum drawdown amount, and location in series if index=True
    """

    maxdraw = 0
    peak    = -99999
    point   = -1
    for i in range(1,len(ts)):
        if ts[i] > peak:
            peak = ts[i]
        if (peak - ts[i])/peak > maxdraw:
            point = i
            maxdraw = (peak - ts[i])/peak
    if index:
        return maxdraw, point
    return maxdraw


def get_crash_moves(portfolio: pd.Series, prices: pd.DataFrame, days=30) -> pd.Series:
    """
    Compute the returns of a portfolio's constituents during the portfolio's max drawdown.

    portfolio:  pd.Series of portfolio prices
    prices:     pd.DataFrame of constituent prices
    days:       # of days to lookback prior to max drawdown (default 30)
    returns     underlying moves over drawdown period
    """
    
    draw,idx = get_max_drawdown(portfolio, index=True)
    # If lookback preceeds start of data, use start of data instead.
    if days > idx:
        days = idx
    drawdowns = (prices.iloc[idx] - prices.iloc[idx-days])/prices.iloc[idx-days]
    return drawdowns.sort_values()


def years_of_data(series) -> float:
    """
    Return difference btwn start and end index of series/dataframe in years.
    """
    return (series.index[-1] - series.index[0]).days / 365


def get_treasury_data() -> pd.DataFrame:
    url = 'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value=all'
    data = []
    dates = []
    page_idx = 1
    while True:
        page_url = url+f'&page={page_idx}'
        r = requests.get(page_url)
        soup = Soup(r.text, 'lxml')
        entries = soup.find_all('entry')
        if len(entries) == 0:
            break
        for entry in entries:
            dates.append(pd.to_datetime(entry.find('d:new_date').text))
            value = {}
            for month in [1,3,6]:
                key = f'd:bc_{month}month'
                if entry.find(key) is None:
                    continue
                value[f'{month}m'] = float(entry.find(key).text)/100.0
            for year in [1,2,3,5,7,10,20,30]:
                key = f'd:bc_{year}year'
                if entry.find(key) is None:
                    continue
                value[f'{year}y'] = float(entry.find(key).text)/100.0
            data.append(value)
        page_idx += 1
    df = pd.DataFrame(data)
    df.index = dates
    return df


def get_treasury(match_idx=None, data_age_limit=10) -> pd.DataFrame:
    """
    Get treasury data from Quandl. Same method as for stocks -- will check for existing data
    downloaded within past 10 days and will refresh otherwise.

    match_idx:  index of timeseries along which to take subset of treasury data
    returns     dataframe of treasury yields by tenor
    """

    # Load cached data if available and within certain age (days).
    existing_data = os.listdir(f'{DATA_DIR}/')
    existing_syms = {fname[:fname.find('_')]: fname for fname in existing_data}

    reload = True
    if 'treasury' in existing_syms.keys():
        fname = existing_syms['treasury']
        data_date = pd.to_datetime(fname[fname.find('_')+1:fname.find('.')])
        if (datetime.today() - data_date).days <= data_age_limit:
            treasury = pd.read_csv('{}/{}'.format(DATA_DIR,fname),index_col=0)
            treasury.index = pd.to_datetime(treasury.index)
            reload = False
        else:
            os.remove('{}/{}'.format(DATA_DIR,fname))
    if reload:
        treasury = get_treasury_data()
        treasury.to_csv('{}/treasury_{}.csv'.format(DATA_DIR,datetime.today().strftime('%d%b%y')))
    if match_idx is None:
        return treasury
    subset_idx = treasury.index[treasury.index.isin(match_idx)]
    return treasury.loc[subset_idx]


def get_risk_free(match_idx) -> float:
    """
    Get annualized risk-free rate as compounded 1m t-bills over specified timeframe.

    match_idx:  index of timeseries along which to take subset of treasury data
    returns     float for annualized compound yield
    """
    treasury = get_treasury(match_idx=match_idx)
    years = years_of_data(treasury)
    return np.log(np.exp(np.sum(treasury['1m']/360)))/years


def is_symbol_cached(symbol: str):
    """
    Check if data for symbol is cached, and return age/filename if found.

    returns -1, None if symbol not cached, otherwise data age, filename
    """

    existing_data = os.listdir(DATA_DIR)
    existing_syms = {fname[:fname.find('_')]: fname for fname in existing_data}
    if symbol in existing_syms.keys():
        fname = existing_syms[symbol]
        data_date = pd.to_datetime(fname[fname.find('_')+1:fname.find('.')])
        return (datetime.today() - data_date).days, fname
    else:
        return -1, None


def load_with_stall(symbols: List[str], data_age_limit=10) -> Dict[str, pd.DataFrame]:
    counter = 0
    for symbol in tqdm.tqdm(symbols):
        is_cached = is_symbol_cached(symbol)[0] 
        if is_cached == -1 or is_cached > data_age_limit:
            get_data([symbol], data_age_limit=data_age_limit)
            counter += 1
        if counter == 5:
            time.sleep(60.0)
            counter = 0
    return get_data(symbols, data_age_limit=data_age_limit)


def get_data(symbols: List[str], data_age_limit=10) -> Dict[str, pd.DataFrame]:
    """
    Load historical price data either from cache or API.

    symbols:        list of symbols to load
    data_age_limit: load cached data if available and within age (days)
    returns         dictionary of (symbol, dataframe)
    """

    data = {}
    for symbol in symbols:
        # Store downloaded data into its own dataframe to avoid truncating index to prior symbols.
        data_single = pd.DataFrame()
        data_age, fname = is_symbol_cached(symbol)

        # If data is found in data directory and within age limit.
        if data_age != -1 and data_age <= data_age_limit:
            data_single = pd.read_csv('{}/{}'.format(DATA_DIR,fname),index_col=0)
            data_single.index = pd.to_datetime(data_single.index)
            # Skip download and go to next symbol.
            data[symbol] = data_single
            continue
        # If data is found but too old.
        elif data_age > data_age_limit:
            os.remove('{}/{}'.format(DATA_DIR,fname))

        # If program flow gets here, data was either unavailable or too old.
        try:
            # Set up alphavantage API - uses default keyfile.
            service = av()
            data_single = service.chart(symbol, adjusted=True)
        except:
            # Likely from limit on 5 requests/minute.
            # Drop the symbol and the user may re-run if desired (others will be cached then).
            print('error: failed to load {}'.format(symbol))
            continue
        data_single.to_csv('{}/{}_{}.csv'.format(DATA_DIR,symbol,datetime.today().strftime('%d%b%y')))
        data_single.index = pd.to_datetime(data_single.index)
        data[symbol] = data_single

    # 'Close' prices are actual close, need to be adjusted for splits to have any use.
    for symbol in symbols:
        r_factor = data[symbol]['split coefficient'][::-1].cumprod().shift(1)
        r_factor.iloc[0] = 1.0
        data[symbol]['close'] = data[symbol]['close'] / r_factor
    return data


def get_prices(data: Dict[str, pd.DataFrame], field='adjusted close') -> pd.DataFrame:
    """
    Aggregate a field from each symbol into one dataframe.

    data:   risk data object, ie dict of pd.DataFrame
    field:  'adjusted close' adjusted for splits & dividends, 'close' is only adjusted for splits
    returns pd.DataFrame
    """
    prices = pd.concat([df[field] for df in data.values()],axis=1,sort=True)
    prices.columns = data.keys()
    return prices


def get_divs(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Get dividends and derived data.

    data:   risk data object, ie dict of dataframes
    returns dictionary of (symbol, dataframe)
    """

    div_data = {}
    for symbol in data.keys():
        divs = data[symbol].loc[data[symbol]['dividend amount'] > 0]
        df = pd.DataFrame()
        df['amount'] = divs['dividend amount']
        df['yield'] = divs['dividend amount'] / divs['close']
        df['period'] = divs.index.to_series().diff().map(lambda dt: dt.days/365)
        # Drop dividends > 5% (assume these are special dividends)
        div_data[symbol] = df.drop(df.loc[df['yield'] >= 0.05].index, axis=0)
    return div_data


def get_div_vol(data: pd.DataFrame) -> (pd.Series, pd.Series):
    """
    Return stddev of annual income / average annual income.
    """

    divs   = get_divs(data)
    divvol  = {}
    divdraw = {}
    for key, value in divs.items():
        # If no dividends, return 0%.
        if len(divs[key]) == 0:
            divvol[key] = 0.0
            continue
        # Sum annual income, include years with at least 1/4 typical divs (ex: 1 qtr if quarterly)
        # and normalize by typical number of divs per year.
        ann_divs = agg_to_period(divs[key].amount, freq='A', norm=1.0/divs[key].period.mean(), cutoff=0.25/divs[key].period.mean())
        divvol[key]  = ann_divs.std() / ann_divs.mean()
        divdraw[key] = get_max_drawdown(ann_divs)
    return pd.Series(divvol), pd.Series(divdraw)


def get_indic_yields(data: Dict[str, pd.DataFrame], last=False, timespan='1y') -> pd.Series:
    """
    Get indic yields using dividends paid over timespan.

    data:       risk data object, ie dict of dataframes
    last:       if True, returns last div / avg. period (using all divs), otherwise return sum(divs)
    timespan:   window from which to pull historical dividends
    returns     pd.Series of yields indexed by symbol
    """

    div_data = get_divs(data)
    prices   = get_prices(data, field='close')

    yields = []
    for symbol, divs in div_data.items():
        window = divs[datetime.today() - date_offset(timespan):].amount
        ann_factor = date_offset(timespan).days / 365
        if len(window) == 0:
            yields.append(0.0)
            continue
        if last:
            window = divs.iloc[-1].amount
            ann_factor = divs.period.mean()
        yields.append(window.sum() / prices[symbol].dropna().iloc[-1] / ann_factor)
    return pd.Series(yields, index=div_data.keys()).fillna(0.0)


def is_earnings_cached(symbol: str):
    """
    Check if earnings data for symbol is cached, and return age/filename if found.

    returns -1, None if earnings not cached, otherwise data age, filename
    """

    existing_data = os.listdir(f'{DATA_DIR}/')
    existing_data = [fname for fname in existing_data if 'earnings-' in fname]
    existing_syms = {fname[:fname.find('_')][9:]: fname for fname in existing_data}
    if symbol in existing_syms.keys():
        fname = existing_syms[symbol]
        data_date = pd.to_datetime(fname[fname.find('_')+1:fname.find('.')])
        return (datetime.today() - data_date).days, fname
    else:
        return -1, None


def get_earnings(symbols: List[str], data_age_limit=30) -> Dict[str, pd.DataFrame]:
    """
    Get raw earnings data per symbol, structured as:
    fiscal_quarter_end, report_date, report_eps, estimate_eps  
    """
    
    service = av()
    earns = {}
    for symbol in symbols:
        data_age, fname = is_earnings_cached(symbol)
        # If data is found in data directory and within age limit.
        if data_age != -1 and data_age <= data_age_limit:
            earnings = pd.read_csv('{}/{}'.format(DATA_DIR,fname),index_col=0)
            # Skip download and go to next symbol.
            earns[symbol] = earnings
            continue
        # If data is found but too old.
        elif data_age > data_age_limit:
            os.remove('{}/{}'.format(DATA_DIR,fname))

        # If program flow gets here, data was either unavailable or too old.
        try:
            earnings = pd.DataFrame(service.earnings(symbol)['quarterlyEarnings'])
            earnings = earnings[::-1]
            earnings.to_csv('{}/earnings-{}_{}.csv'.format(DATA_DIR, symbol, datetime.today().strftime('%d%b%y')))
        except:
            # Likely from limit on 5 requests/minute.
            # Drop the symbol and the user may re-run if desired (others will be cached then).
            print('error: failed to load {}'.format(symbol))
            continue
        earns[symbol] = earnings
    return earns


def get_reported(earns: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build dataframe of reportedEPS by quarter for multiple names.
    """
    
    earns_df = pd.DataFrame()
    for symbol in earns.keys():
        earns[symbol].fiscalDateEnding = pd.to_datetime(earns[symbol].fiscalDateEnding)
        quarter = pd.PeriodIndex(earns[symbol].fiscalDateEnding, freq='Q')
        earns_df[symbol] = pd.Series(earns[symbol].groupby(quarter).sum().reportedEPS, name=symbol)
    return earns_df


def quickload(symbols: List[str], field='adjusted close') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate get_data, get_prices, get_indic_yields with default optional parameters.

    returns pd.DataFrames for prices and yields
    """
    data = get_data(symbols)
    prices = get_prices(data, field=field)
    yields = get_indic_yields(data)
    return prices, yields


def get_risk_return(prices: pd.DataFrame, ann_factor: int = 252) -> (pd.Series, pd.Series, pd.Series, pd.DataFrame):
    """
    Compute following metrics:
    total return (annualized)
    annualized volatility over whole period
    annualized downvol (volatility | -return) over whole period
    covariance matrix over whole period

    prices: pd.DataFrame of price data per symbol, may include NAs
    returns pd.Series for ann. total return, vol, downvol (indexed by symbol), and pd.DataFrame object for covar
    """

    results = []
    # Do each symbol individually
    for symbol in prices.columns:
        px      = prices[symbol].dropna()
        returns = px.pct_change()
        rf      = get_risk_free(px.index)
        years   = years_of_data(px)

        series = pd.Series(dtype=np.float)
        series['tr']      = np.log(px.iloc[-1]/px.iloc[0]) / years
        series['vol']     = np.sqrt( (np.log(1.0 + returns)**2).mean() * ann_factor )
        series['downvol'] = np.sqrt( (np.log(1.0 + returns[returns < 0])**2).mean(axis=0) * ann_factor )
        series['sharpe']  = (series.tr - rf) / series.vol
        series['sortino'] = (series.tr - rf) / series.downvol
        series.name = symbol
        results.append(series)
    results = pd.DataFrame(results)
    variance = pd.DataFrame(results.vol).dot(pd.DataFrame(results.vol).T)
    covar = variance*prices.dropna().pct_change().corr()
    return results, covar


def get_metrics(prices: pd.DataFrame, data: Dict[str,pd.DataFrame] = None, ann_factor: int = 252) -> (pd.DataFrame, pd.DataFrame):
    """
    Compute risk metrics + div yield per symbol.

    prices: pd.DataFrame of price timeseries per symbol
    data:   risk-data object, will compute yield/yield stability
    returns metrics pd.DataFrame and covar pd.DataFrame
    """
    results, covar = get_risk_return(prices, ann_factor=ann_factor)

    metrics = pd.DataFrame()
    metrics = metrics.assign(tr=results.tr, vol=results.vol, sharpe=results.sharpe, sortino=results.sortino)
    metrics = metrics.assign(maxdraw=[get_max_drawdown(prices[ticker].dropna()) for ticker in metrics.index])
    if data is not None:
        yields = get_indic_yields(data)
        divstd = get_div_vol(data)
        metrics = metrics.assign(divs=yields, divstd=divstd[0])
    return metrics, covar


def match_indices(series_a, series_b):
    """
    Return subsets of A & B with shared indices.
    """
    subset_idx = series_a.index[series_a.index.isin(series_b.index)]
    return series_a.loc[subset_idx], series_b.loc[subset_idx]


def agg_to_period(series: pd.Series, freq='A', norm=252, cutoff=60) -> pd.Series:
    """
    Group series into periods and return periods with substantial data, optionally normalized by length.

    freq:   'A' or 'Q' for annual or quarterly
    norm:   normalize assuming each period 'should' have this many data points
    cutoff: include periods with at least this many data points
    """
    grouped = series.groupby(series.index.to_period(freq=freq))
    summed  = grouped.sum()[grouped.count() >= cutoff]
    normed  = (summed / grouped.count() * norm) if type(norm) == int else summed
    return normed.dropna()


def snapshot(symbols: List[str], from_date=None):
    data = get_data(symbols)
    prices = get_prices(data).dropna()
    if from_date is not None and pd.to_datetime(from_date) > prices.index[0]:
        prices = prices.loc[from_date:]
    first_index = prices.index[0]
    print(f'since {first_index:%Y-%m-%d}'.format())
    metrics = 100.0 * get_metrics(prices, data)[0]
    metrics['sharpe']  /= 100.0
    metrics['sortino'] /= 100.0
    return metrics.style.format('{:.1f}%').format({'sharpe': '{:.2f}', 'sortino': '{:.2f}'})


# Bond Math
def bond_dates(maturity):
    """
    Build semi-annual schedule ending at maturity and starting at least today.
    """
    dates = pd.period_range(end=maturity, freq='6M', periods=100).to_timestamp()
    return dates[dates >= datetime.today()] + (pd.to_datetime(maturity) - dates[-1])

def bond_pv(coupon, rate, maturity):
    """
    Solve for bond PV assuming semi-annual coupon.
    """
    dates = bond_dates(maturity)
    years = (dates - datetime.today()).days/360
    return 100.0 * coupon/2 * np.sum(1.0 / (1.0 + rate)**years) + 100.0 / (1.0 + rate)**years[-1]

def bond_rate(pv, coupon, maturity):
    """
    Solve for bond yield.
    """
    return opt.brentq(lambda rate: bond_pv(coupon, rate, maturity) - pv, -0.05, 0.50)

def bond_call(pv, coupon, rate, maturity):
    """
    Solve for call date of bond, given coupon, YTC, and max maturity. Assumes all coupon dates are call dates.
    """
    dates = bond_dates(maturity)
    idx   = 0
    diff  = np.abs(bond_pv(coupon, rate, dates[0]) - pv)
    for idx_ in range(1, len(dates)):
        if np.abs(bond_pv(coupon, rate, dates[idx_]) - pv) < diff:
            diff = np.abs(bond_pv(coupon, rate, dates[idx_]) - pv)
            idx = idx_ 
    return dates[idx], np.round(bond_pv(coupon, rate, dates[idx]), 2)