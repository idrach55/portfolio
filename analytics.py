"""
Author: Isaac Drachman
Date: 8/16/2021

Taxable portfolio analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import risk
from .risk import FundCategory, Utils


def process_jpm(fname: str, save: bool = True) -> Optional[pd.DataFrame]:
    # Load file (export csv from JP Morgan online.)
    df = pd.read_csv('jpm/{}'.format(fname))
    
    # Drop content at & below FOOTNOTES row. 
    df = df.iloc[:df.loc[df['Asset Class'] == 'FOOTNOTES'].index[0]]

    # Municipal bonds show NaN tickers under Tax-Exempt Core, replace with IG muni ETF.
    # Replace cash with T-Bill ETF and preferred stocks with ETF.
    new_tickers = []
    for idx, row in df.iterrows():
        if not pd.isna(row.Ticker) and ' ' in row.Ticker:
            new_tickers.append('PFF')
        elif not pd.isna(row.Ticker):
            new_tickers.append(row.Ticker)
        elif row['Asset Strategy Detail'] in ['Tax-Exempt Core']:
            new_tickers.append('MUB')
        elif row['Asset Strategy Detail'] == 'Cash':
            new_tickers.append('BIL')
        else:
            new_tickers.append(row.Ticker)
    df.Ticker = new_tickers
    old_value = df.Value.sum()
    df = df.drop(df.loc[df.Ticker.isna()].index)

    muni = df.loc[df.Ticker == 'MUB']
    pff  = df.loc[df.Ticker == 'PFF']
    if len(muni) > 0:
        df = df.drop(muni.index, axis=0)
    if len(pff) > 0:
        df = df.drop(pff.index, axis=0)

    df = df[['Ticker','Value']]
    df.index = df.Ticker
    df = df.Value
    if len(muni) > 0:
        df = pd.concat([df, pd.Series({'MUB': muni.Value.sum()})])
    if len(pff) > 0:
        df = pd.concat([df, pd.Series({'PFF': pff.Value.sum()})])
    df.name = 'value'
    df.index.name = 'symbol'

    if save:
        df.to_csv('portfolios/{}'.format(fname))
    else:
        return df


def agg_folio_csv(folios: List[str], saveas: str) -> None:
    folios = ['portfolios/{}'.format(folio) for folio in folios]
    saveas = 'portfolios/{}'.format(saveas)
    dfs = [pd.read_csv(folio, index_col=0) for folio in folios]
    symbols = dfs[0].index
    for df in dfs[1:]:
        symbols = symbols.append(df.index)
    symbols = symbols.unique()
    total = dfs[0].reindex(symbols).fillna(0.0)
    for df in dfs[1:]:
        total += df.reindex(symbols).fillna(0.0)
    total.to_csv(saveas)


@dataclass
class TaxBrackets:
    federal_income: float
    state: float
    federal_lt: float

    @staticmethod
    def default() -> TaxBrackets:
        return TaxBrackets(0.388, 0.068, 0.238)

    def getLTCapGains(self) -> float:
        return self.federal_lt + self.state
    
    def getIncome(self) -> float:
        return self.federal_income + self.state
    
    def getTaxesByAsset(self, assets: List[str]) -> pd.Series:
        taxes = pd.Series({symbol: self.getLTCapGains() for symbol in assets})
        for symbol in assets:
            # Assume mutual fund if symbol > 4 chars.
            category = FundCategory.createFromSymbol(symbol)
            if category == FundCategory.MUNI:
                taxes[symbol] = self.state
            elif category.getIsTaxableBond():
                taxes[symbol] = self.getIncome()
        return taxes

class TaxablePortfolio:
    """
    To analyze performance/income of taxable account.
    Assume investor is taking out all dividends/interest in a year.
    """
    def __init__(self, 
                 basket: pd.Series, 
                 brackets: Optional[TaxBrackets] = None, 
                 from_date: Optional[str] = None, 
                 reinvest: float = 0.0, 
                 categorize: bool = True):
        self.basket   = basket
        self.reinvest = reinvest
        self.brackets = brackets if brackets else TaxBrackets.default()

        # First, assume all income taxed as qualified dividends.
        self.taxes = pd.Series({symbol: brackets.getLTCapGains() for symbol in basket.index})
        # If requested, lookup each security on ETFDB to categorize as bonds/municipals.
        if categorize:
            self.taxes = brackets.getTaxesByAsset(basket.index)

        self.data   = risk.get_data(basket.index)

        # Quote yields as post-tax
        self.yields = risk.get_indic_yields(self.data, last=True)
        self.yields *= (1.0 - self.taxes)

        # Leave these with NAs for full data per symbol (to compute risks)
        self.prices_tr = risk.get_prices(self.data, field='adjusted close')
        self.prices_pr = risk.get_prices(self.data, field='close')

        # Drop NAs for use in building portfolio (want all symbols live)
        self.prices    = self.prices_pr.dropna()

        if from_date is not None:
            self.prices    = self.prices[from_date:]
            self.prices_tr = self.prices_tr[from_date:]
            self.prices_pr = self.prices_pr[from_date:]

        # Build dividends dataframe
        div_data  = risk.get_divs(self.data)
        self.divs = pd.concat([div_data[symbol]['amount'] for symbol in self.basket.index], axis=1)
        self.divs.columns = self.basket.index
        self.divs = self.divs.reindex(self.prices.index).fillna(0.0)

        prices_pr_ = self.prices_pr.dropna()
        prices_tr_ = self.prices_tr.dropna()

        # Compute portfolios with 0% reinvestment and 100% tax-free investment.
        self.value_pr = (1.0 + (prices_pr_.pct_change() * self.basket).sum(axis=1)).cumprod()
        self.value_tr = (1.0 + (prices_tr_.pct_change() * self.basket).sum(axis=1)).cumprod()

        # Closed-form computation for share quantity / reinvestment.
        self.taxed_excess = (prices_tr_ / prices_tr_.iloc[0] - prices_pr_ / prices_pr_.iloc[0]) * (1.0 - self.taxes)
        self.value = self.reinvest * (self.taxed_excess * self.basket).sum(axis=1) + self.value_pr
        self.shares = pd.concat([self.value] * len(self.basket), axis=1)
        self.shares.columns = self.basket.index
        self.shares *= self.basket / self.prices

        # Compute post-tax dividends/interest
        # total_inco = total post-tax income paid by assets whether or not it's reinvested
        # total_divs = cashflow net of reinvestment (if any)
        self.total_inco = (self.shares * self.divs * (1.0 - self.taxes)).sum(axis=1)
        self.total_divs = self.total_inco * (1.0 - self.reinvest)

    def get_cashflow(self):
        """
        Sum post-tax income in each period and return as step function with index as self.value.
        """
        yearly = self.total_divs.groupby(self.total_divs.index.year).sum()
        cashflow = self.value.copy()
        for year in yearly.index:
            cashflow.loc[cashflow.index.year == year] = yearly[year]
        return cashflow

    def get_trailing_yield(self, days=63, smooth=True):
        """
        Get post-tax yield as annualized sum of quarter's income / value on last day of quarter.

        smooth: use the rolling quarterly mean of the income time series
        """
        trailing_inco = 252.0 / days * self.total_inco.rolling(days).sum() 
        if smooth:
            trailing_inco = trailing_inco.rolling(63).mean()
        return (trailing_inco / self.value).dropna()

    def get_metrics(self, align=False):
        """
        Get metrics for underlyings and portfolio. Include taxable yield (by divs) in sharpe, but show yields as taxable (by income) in dataframe.
        """
        # Compute volatility of post-tax dividend yield
        divstd, divdraw = risk.get_div_vol(self.data)

        # Compute metrics for basket, but rename as price-return.
        # Sharpe/sortino based on total return.
        prices_pr_ = self.prices_pr.dropna() if align else self.prices_pr
        prices_tr_ = self.prices_tr.dropna() if align else self.prices_tr

        results_pr, covar = risk.get_risk_return(prices_pr_)
        results_tr, covar = risk.get_risk_return(prices_tr_)

        live_since = pd.Series({symbol: self.prices_pr[symbol].dropna().index[0] for symbol in self.prices.columns})

        metrics = results_tr.copy()
        metrics = metrics.assign(pr=results_pr.tr, maxdraw=[Utils.getMaxDrawdown(prices_pr_[ticker]) for ticker in metrics.index])
        metrics = metrics.assign(divs=self.yields, divstd=divstd, divdraw=divdraw, live=live_since)

        returns = self.value.pct_change()[1:]
        vol   = np.sqrt( (np.log(1.0 + returns)**2).mean()*252 )
        dnvol = np.sqrt( (np.log(1.0 + returns[returns < 0.0])**2).mean()*252 )
        rf    = risk.get_risk_free(self.value.index)

        # Requote PR/TR using weights & full history per security, rather than only full portfolio's history.
        sharpe   = (np.dot(self.basket, metrics.tr) - rf)/vol
        sortino  = (np.dot(self.basket, metrics.tr) - rf)/dnvol
        ann_divs = Utils.aggregateToPeriod(self.total_inco, freq='A')
        divstd   = ann_divs.std() / ann_divs.mean()
        divdraw  = Utils.getMaxDrawdown(ann_divs)
        pmetrics = pd.Series({'pr': np.dot(self.basket, metrics.pr), 'tr': np.dot(self.basket, metrics.tr), 'vol': vol, 'sharpe': sharpe, 
                              'sortino': sortino, 'maxdraw': Utils.getMaxDrawdown(self.value), 'divs': np.dot(self.yields, self.basket),
                              'live': self.prices.index[0], 'divstd': divstd, 'divdraw': divdraw})
        metrics.loc['portfolio'] = pmetrics
        return metrics[['pr','tr','vol','sharpe','sortino','maxdraw','divs','divstd','divdraw','live']], covar