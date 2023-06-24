"""
Author: Isaac Drachman
Date: 6/21/2023

Factor decomposition.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.optimize as opt
from sklearn.linear_model import LinearRegression as OLS
from sklearn.metrics import r2_score

from .risk import Utils, load_with_stall


class FactorUniverse(Enum):
    STYLE = 1
    SECTOR = 2
    ASSET = 3

    @staticmethod
    def create(universe: str) -> FactorUniverse:
        assert universe in ['style', 'sector', 'asset']
        if universe == 'style':
            return FactorUniverse.STYLE
        elif universe == 'sector':
            return FactorUniverse.SECTOR
        elif universe == 'asset':
            return FactorUniverse.ASSET

    @staticmethod
    def makeFactorsStyle(returns: pd.DataFrame) -> pd.DataFrame:
        # Treasury bonds: SHY = 1-3y, IEI = 3-7y, IEF = 7-10y, TLH = 10-20y, TLT = 20y+
        factors = pd.DataFrame()
        factors['equities'] = returns.VT
        factors['rates'] = returns.IEI
        factors['credit'] = returns.HYG - returns.IEI
        factors['commods'] = returns.GSG

        # Secondary macro
        factors['inflation'] = regress_factor(returns['TIP'], returns[['IEF']])
        factors['emerging'] = 0.5*(returns.EEM - returns.VTI + returns.EMB - returns.IEF)
        factors['usequity'] = returns.VTI - returns.EFA
        factors['usdollar'] = returns.UUP

        # Macro style
        factors['shortvol'] = regress_factor(returns['PUTW'], returns[['VTI']])
        factors['municipal'] = returns.MUB - returns.IEF
        factors['realestate'] = returns.IYR - returns.VTI

        # Equity style factors
        factors['smallcap'] = returns.IWM - returns.VTI
        factors['lowrisk']  = returns.USMV - returns.VTI
        factors['momentum'] = returns.MTUM - returns.VTI
        factors['quality']  = returns.QUAL - returns.VTI
        factors['value']    = returns.IWD - returns.IWF
        return factors
    
    @staticmethod
    def makeFactorsSector(returns: pd.DataFrame, components: pd.Series) -> pd.DataFrame:
        factors = pd.DataFrame()
        for sector, etf in components.items():
            if sector != 'market':
                factors[sector] = returns[etf]
                #factors[sector] = regress_factor(returns[etf], returns[['SPY']])
        return factors
    
    @staticmethod
    def makeFactorsAsset(returns: pd.DataFrame, components: pd.Series) -> pd.DataFrame:
        factors = pd.DataFrame()
        for asset, etf in components.items():
            factors[asset] = returns[etf]
        return factors

    def getComponents(self) -> pd.Series:
        factors = pd.read_csv('factors.csv',index_col=1)
        return factors.loc[factors.universe == self.name].etf
    
    def getFactors(self) -> pd.DataFrame:
        """
        Generate factors from different universes:
        exposure = style/size/geo factors + bonds
        sector   = SPDR sectors
        """
        # Components from which we'll build our factors
        components = self.getComponents()

        # Get data and build prices dataframe
        data    = load_with_stall(list(components.values))
        prices  = Utils.getPrices(data).dropna()
        returns = prices.pct_change().iloc[1:]

        # Build factors dataframe by transforming components.
        # todo: make factors better / add more
        if self == FactorUniverse.STYLE:
            # Core macro factors
            factors = FactorUniverse.makeFactorsStyle(returns)
        elif self == FactorUniverse.SECTOR:
            factors = FactorUniverse.makeFactorsSector(returns, components)
        elif self == FactorUniverse.ASSET:
            factors = FactorUniverse.makeFactorsAsset(returns, components)
        # Take cumulative product of returns to generate price series
        factors  = (1.0 + factors).cumprod(axis=0)
        # Prepend 1.0 for each factor's initial price: just a niceity
        return pd.concat([pd.DataFrame({factor: 1.0 for factor in factors.columns}, index=[prices.index[0]]), factors])
    

def regress_factor(A: pd.Series, B: pd.DataFrame) -> pd.Series:
    """
    Regress s.t. A = beta x B + epsilon
    """
    fitted = OLS().fit(B, A)
    return A - (fitted.coef_ * B).sum(axis=1)


def decompose(prices: pd.Series, factors: pd.DataFrame) -> Tuple[float, pd.Series, pd.DataFrame]:
    """
    Perform factor decomp given "market" factors.

    prices: pd.Series of asset prices
    factors: pd.DataFrame of factor prices
    returns r-squared score, pd.Series factor-predicted price, pd.DataFrame of factor coefs
    """

    returns = prices.pct_change()[1:]
    factors = factors.dropna().pct_change()[1:]

    returns_s, factors_s = Utils.matchIndices(returns, factors)

    fitted = OLS().fit(factors_s, returns_s)
    r_squared = fitted.score(factors_s, returns_s)

    df = pd.DataFrame(fitted.coef_,index=factors.columns,columns=['coef'])
    df = df.assign(weight=100*df.coef.abs()/df.coef.abs().sum())

    # Compute the cumulative performance portfolio of factors
    pred = (1.0 + (factors_s*df.coef).sum(axis=1)).cumprod()
    # Prepend 1.0 to the series: just a niceity
    # pred = pd.Series({prices.index[0]: 1.0}).append(pred)
    return r_squared, pred, df


def decompose_const(prices: pd.Series, factors: pd.DataFrame) -> Tuple[float, pd.Series]:
    returns = prices.pct_change()[1:]
    factors = factors.dropna().pct_change()[1:]

    returns_s, factors_s = Utils.matchIndices(returns, factors)
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    def err(w):
        return (((w * factors_s).sum(axis=1) - returns_s)**2).sum()
    res = opt.minimize(err, [1.0/len(factors.columns)]*len(factors.columns), constraints=cons, bounds=[(0.0,1.0)]*len(factors.columns))
    weights = pd.Series(res['x'], index=factors.columns)
    # weights = weights.loc[weights > 0.0]
    weights[weights.abs() < 0.01] = 0.0
    weights /= weights.sum()
    rsq = r2_score(returns_s, (weights * factors_s).sum(axis=1))
    return rsq, weights


def decompose_multi(prices: pd.DataFrame, factors: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Perform factor decomp on multiple names.
    """
    rsq  = pd.Series()
    pred = pd.DataFrame()
    df   = {}
    for symbol in prices.columns:
        rsq[symbol], pred[symbol], df[symbol] = decompose(prices[symbol], factors)
    return rsq, pred, df