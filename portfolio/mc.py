"""
Author: Isaac Drachman
Date: 4/22/2023

MC portfolio projections.
"""

import sysconfig
from typing import Tuple

import numpy as np
import pandas as pd
from arch.univariate import GARCH, Normal, ZeroMean

# x86 = (sysconfig.get_platform().split("-")[-1].lower() == 'x86_64')
# import portfolio.mcpp as mcpp
from portfolio.analytics import TaxablePortfolio, TaxBrackets
from portfolio.factors import FactorUniverse, decompose_const
from portfolio.risk import Driver, Utils


class Garch:
    @staticmethod
    def process(series: pd.Series, window: int = 5) -> pd.Series:
        # Take log-returns along series, normalize and winsorize.
        ret = np.log(series/series.shift(1))[1:]
        ret = ret - ret.mean()
        ret = ret[(ret > -window*ret.std()) & (ret < window*ret.std())]
        return ret

    @staticmethod
    def filter(ret: pd.Series, omega: float, alpha: float, beta: float) -> pd.Series:
        sigma_2 = np.zeros(len(ret))
        sigma_2[0] = omega / (1.0 - alpha - beta)
        for t in range(1, len(ret)):
            sigma_2[t] = omega + alpha * ret[t-1]**2 + beta * sigma_2[t-1]
        return sigma_2

    @staticmethod
    def mle(params: Tuple[float, float, float], ret: pd.Series) -> float:
        sigma_2 = Garch.filter(ret, *params)
        # Return the negative as to minimize in optimization.
        return -np.sum(-np.log(sigma_2) - ret**2/sigma_2)

    @staticmethod
    def getLongRunVol(fit, scale=100):
        return np.sqrt(fit.params['omega'] / (1.0 - fit.params['alpha[1]'] - fit.params['beta[1]']) * 252) / scale

    def fit(ret: pd.Series):
        # For numerical stability, multiply returns by 100x. 
        # omega scales quadratically, other params are constant
        am = ZeroMean(100 * ret, rescale=False)
        am.volatility = GARCH(p=1, q=1)
        am.distribution = Normal() # GeneralizedError()
        return am.fit(disp='off')

    def fitMulti(symbols):
        # Normalize & winsorize returns, then fit GARCH(1,1).
        returns = {}
        fits    = {}
        epsilon = {}
        for symbol in symbols:
            prices = Driver.getPrices([symbol])
            returns[symbol] = Garch.process(prices[symbol])
            fits[symbol] = Garch.fit(returns[symbol])
            epsilon[symbol] = 100.0 * returns[symbol] / fits[symbol].conditional_volatility

        corr = np.zeros(shape=(len(symbols), len(symbols)))
        shared_idx = pd.DataFrame(returns).dropna().index
        for idx_i in range(len(symbols)):
            for idx_j in range(len(symbols)):
                corr[idx_i][idx_j] = epsilon[symbols[idx_i]][shared_idx].corr(epsilon[symbols[idx_j]][shared_idx])

        params = pd.DataFrame()
        params['omega'] = [fit.params.omega for fit in fits.values()]
        params['alpha'] = [fit.params['alpha[1]'] for fit in fits.values()]
        params['beta']  = [fit.params['beta[1]'] for fit in fits.values()]
        params.index = symbols
        return returns, fits, params, pd.DataFrame(corr, columns=symbols, index=symbols)

    def getMCPaths(drifts: np.array, corr: pd.DataFrame, sigma_last: np.array, params: pd.DataFrame, num_paths=1000, num_steps=252):
        paths   = np.zeros(shape=(num_paths, num_steps, len(drifts)))   
        sigma_2 = np.ones(shape=(num_paths, num_steps, len(drifts))) + sigma_last**2
            
        noise = np.random.multivariate_normal([0.0]*len(drifts), corr, size=(num_paths, num_steps))
        for symbol_idx in range(len(drifts)):
            for t in range(num_steps):
                if t > 0:
                    sigma_2[:,t,symbol_idx] = params.omega[symbol_idx] + params.alpha[symbol_idx] * paths[:,t-1,symbol_idx]**2 + params.beta[symbol_idx] * sigma_2[:,t-1,symbol_idx]
                paths[:,t,symbol_idx] = np.sqrt(sigma_2[:,t,symbol_idx]) * noise[:,t,symbol_idx]
        return np.exp(drifts/252.0 + paths/100).cumprod(axis=1)
    
    def getMCPathsCpp(drifts: np.array, corr: pd.DataFrame, sigma_last: np.array, params: pd.DataFrame, num_paths=1000, num_steps=252):
        paths   = np.zeros(shape=(num_paths, num_steps, len(drifts)))   
        sigma_2 = np.ones(shape=(num_paths, num_steps, len(drifts))) + sigma_last**2
            
        noise = np.random.multivariate_normal([0.0]*len(drifts), corr, size=(num_paths, num_steps))
        mcpp.genPaths(
            noise,
            params.omega.values,
            params.alpha.values,
            params.beta.values,
            sigma_2,
            paths
        )
        return np.exp(drifts/252.0 + paths/100).cumprod(axis=1)

# Asset replication for Monte Carlo
def do_basket(basket, cutoff=0.01):
    removes = []
    for remove in removes:
        remove_idx = np.where(basket.index == remove)[0][0]
        basket = basket[basket.index[0:remove_idx].append(basket.index[remove_idx+1:])]

    basket /= basket.sum()

    basket_ = basket[basket > cutoff]
    basket_ /= basket_.sum()

    dropped = 100.0 - 100.0*basket[basket_.index].sum()
    print(f'dropped {len(basket) - len(basket_)} name(s) totaling {dropped:.2f}%')

    # Just load for later.
    _data = Driver.getData(basket_)
    return basket_

def getReplication(basket: pd.Series, ltcma: pd.DataFrame):
    basket_ = do_basket(basket)
    folio = TaxablePortfolio(basket_)
    factors = FactorUniverse.ASSET.getFactors()
    _rsq, weights = decompose_const(folio.value, factors)
    return pd.Series(weights.values, index=ltcma.loc[weights.index].etf)

class MCPortfolio:
    def __init__(self, basket: pd.Series, brackets: TaxBrackets):
        self.basket = basket
        self.brackets = brackets
        self.taxes = brackets.getTaxesByAsset(basket.index)
        self.data = Driver.getData(basket.index)
        self.prices = Utils.getPrices(self.data, field='close')
        self.yields = Utils.getIndicYields(self.data, last=True)

    def generate_paths(self, years=10, drifts=None, vols=None, N=1000):
        # Use GARCH(1,1) with Multivariate Normal innovations.
        _, fits, params, corr = Garch.fitMulti(self.basket.index)
        sigma_last = np.array([fit.conditional_volatility[-1] for fit in fits.values()])

        # Drift is either given or uses historical average.
        log_r = np.log(self.prices.dropna()/self.prices.dropna().shift(1))
        self.drifts = drifts if drifts is not None else 252.0 * log_r.mean()
        paths = Garch.getMCPaths(self.drifts.values, corr, sigma_last, params, num_paths=N, num_steps=years*252)
        self.paths = np.concatenate([np.ones(shape=(paths.shape[0], 1, paths.shape[2])), paths], axis=1)

    def build(self, init_value, withdraw_fixed, withdraw_pct, fee=0.00, withdraw_explicit=None):
        N = self.paths.shape[0]
        qtr_index   = np.arange(0, (self.paths.shape[1] - 1) // 63 + 1) * 63
        self.shares = np.zeros(shape=(N, qtr_index.shape[0], self.paths.shape[2])) + (self.basket.values * init_value)
        self.costs  = np.zeros(shape=(N, qtr_index.shape[0], self.paths.shape[2])) + 1.0
        self.gains  = np.zeros(shape=(N, qtr_index.shape[0]))
        self.value  = np.zeros(shape=(N, qtr_index.shape[0])) + init_value
        self.income = np.zeros(shape=(N, qtr_index.shape[0]))
        self.resid  = np.zeros(shape=(N, qtr_index.shape[0]))
        self.taken  = np.zeros(shape=(N, qtr_index.shape[0]))

        if type(withdraw_fixed) == float:
            withdraw_fixed = np.zeros(shape=(qtr_index.shape[0] // 4)) + withdraw_fixed
        if type(withdraw_pct) == float:
            withdraw_pct = np.zeros(shape=(qtr_index.shape[0] // 4)) + withdraw_pct

        # For each quarter, qtr_index[qtr] = the daily index in paths for that quarter.
        for qtr in range(1, len(qtr_index)):
            # Grow portfolio by performance.
            self.value[:,qtr]  = np.sum(self.shares[:,qtr-1] * self.paths[:,qtr_index[qtr]], axis=1)
            self.income[:,qtr] = np.sum((self.basket * self.yields/4)*(1.0 - self.taxes))*self.value[:,qtr-1]
            self.taken[:,qtr]  = withdraw_fixed[(qtr-1)//4]/4 + self.value[:,qtr-1]*withdraw_pct[(qtr-1)//4]/4
            if withdraw_explicit is not None:
                # Per-path withdrawal (done for combining accounts)
                self.taken[:,qtr] += withdraw_explicit[:,qtr]
            self.resid[:,qtr]  = self.income[:,qtr] -  self.value[:,qtr-1]*fee/4 - self.taken[:,qtr]
            self.value[:,qtr] += self.resid[:,qtr]
            self.shares[:,qtr] = np.outer(self.value[:,qtr], self.basket.values) / self.paths[:,qtr_index[qtr]]

            # Rebalance and charge capital gains taxes (assuming all long-term.)
            shares_to_trade    = self.shares[:,qtr] - self.shares[:,qtr-1]
            amended_basis      = (self.costs[:,qtr-1]*self.shares[:,qtr-1] + shares_to_trade*self.paths[:,qtr_index[qtr]]) / np.maximum(self.shares[:,qtr], 1.0)
            self.costs[:,qtr]  = (shares_to_trade > 0) * amended_basis + (shares_to_trade < 0) * self.costs[:,qtr-1]

            self.gains[:,qtr]  = ((shares_to_trade < 0) * -shares_to_trade * (self.paths[:,qtr_index[qtr]] - self.costs[:,qtr-1])).sum(axis=1)
            self.value[:,qtr]  -= np.maximum(self.gains[:,qtr], 0.0) * self.brackets['div']
        
        # Snap negative portfolios to zero.
        for path_idx, step_idx in zip(np.where(self.value <= 0)[0], np.where(self.value <= 0)[1]):
            self.value[path_idx, step_idx:] = 0.0

    def get_undl_paths(self):
        # Return performance of underlying portfolio (assuming no tax/withdrawals) for each path.
        return np.cumprod(1.0 + (self.basket.values * (self.paths[:,1:,:] / self.paths[:,:-1,:] - 1.0)).sum(axis=2), axis=1)

    def get_max_drawdowns(self):
        return [Utils.getMaxDrawdown(path) for path in self.value if path[-1] > 0]