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

from .risk import CloseMethod, Driver, FundCategory, Utils


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
                 reinvest: float = 0.0, 
                 categorize: bool = True):
        self.basket   = basket
        self.reinvest = reinvest
        self.brackets = brackets if brackets else TaxBrackets.default()

        # First, assume all income taxed as qualified dividends.
        self.taxes = pd.Series({symbol: self.brackets.getLTCapGains() for symbol in basket.index})
        # If requested, lookup each security on ETFDB to categorize as bonds/municipals.
        if categorize:
            self.taxes = self.brackets.getTaxesByAsset(basket.index)

        self.data   = Driver.getData(basket.index)

        # Quote yields as post-tax
        self.yields = Utils.getIndicYields(self.data, last=True)
        self.yields *= (1.0 - self.taxes)

        # Leave these with NAs for full data per symbol (to compute risks)
        self.prices_tr = Utils.getPrices(self.data, method=CloseMethod.ADJUSTED)
        self.prices_pr = Utils.getPrices(self.data, method=CloseMethod.RAW)

        # Drop NAs for use in building portfolio (want all symbols live)
        self.prices    = self.prices_pr.dropna()

        # Build dividends dataframe
        div_data  = Utils.getDivs(self.data)
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
        # Sum post-tax income in each period and return as step function with index as self.value.
        yearly = self.total_divs.groupby(self.total_divs.index.year).sum()
        cashflow = self.value.copy()
        for year in yearly.index:
            cashflow.loc[cashflow.index.year == year] = yearly[year]
        return cashflow

    def get_trailing_yield(self, days=63, smooth=True):
        # Get post-tax yield as annualized sum of quarter's income / value on last day of quarter.
        # smooth: use the rolling quarterly mean of the income time series
        trailing_inco = 252.0 / days * self.total_inco.rolling(days).sum() 
        if smooth:
            trailing_inco = trailing_inco.rolling(63).mean()
        return (trailing_inco / self.value).dropna()

    def get_metrics(self, align=False):
        # Get metrics for underlyings and portfolio. Include taxable yield (by divs) in sharpe, but show yields as taxable (by income) in dataframe.
        # Compute volatility of post-tax dividend yield
        divstd, divdraw = Utils.getDividendVol(self.data)

        # Compute metrics for basket, but rename as price-return.
        # Sharpe/sortino based on total return.
        prices_pr_ = self.prices_pr.dropna() if align else self.prices_pr
        prices_tr_ = self.prices_tr.dropna() if align else self.prices_tr

        results_pr, covar = Utils.getRiskReturn(prices_pr_)
        results_tr, covar = Utils.getRiskReturn(prices_tr_)

        live_since = pd.Series({symbol: self.prices_pr[symbol].dropna().index[0] for symbol in self.prices.columns})

        metrics = results_tr.copy()
        metrics = metrics.assign(pr=results_pr.tr, maxdraw=[Utils.getMaxDrawdown(prices_pr_[ticker]) for ticker in metrics.index])
        metrics = metrics.assign(divs=self.yields, divstd=divstd, divdraw=divdraw, live=live_since)

        returns = self.value.pct_change()[1:]
        vol   = np.sqrt( (np.log(1.0 + returns)**2).mean()*252 )
        dnvol = np.sqrt( (np.log(1.0 + returns[returns < 0.0])**2).mean()*252 )
        rf    = Utils.getRiskFree(self.value.index)

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