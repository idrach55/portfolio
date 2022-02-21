"""
Author: Isaac Drachman
Date: 8/16/2021

Factor decomposition, taxable portfolio analysis, and MC portfolio projections.
"""

from . import risk

import pandas as pd
import numpy as np
import time
import scipy.stats as stat

from sklearn.linear_model import LinearRegression as OLS
from sklearn.metrics import r2_score
from scipy.optimize import minimize, differential_evolution
from statsmodels.distributions.empirical_distribution import ECDF
from arch.univariate import ConstantMean, ZeroMean, GARCH, GeneralizedError, Normal
from typing import Dict, List, Tuple


def process_returns(series: pd.Series, window=5):
    # Take log-returns along series, normalize and winsorize.
    ret = np.log(series/series.shift(1))[1:]
    ret = ret - ret.mean()
    ret = ret[(ret > -window*ret.std()) & (ret < window*ret.std())]
    return ret


def garch_filter(ret: pd.Series, omega: float, alpha: float, beta: float):
    sigma_2 = np.zeros(len(ret))
    sigma_2[0] = omega / (1.0 - alpha - beta)
    for t in range(1, len(ret)):
        sigma_2[t] = omega + alpha * ret[t-1]**2 + beta * sigma_2[t-1]
    return sigma_2


def garch_mle(params, ret: pd.Series):
    sigma_2 = garch_filter(ret, *params)
    # Return the negative as to minimize in optimization.
    return -np.sum(-np.log(sigma_2) - ret**2/sigma_2)


def get_lr_vol(fit, scale=100):
    return np.sqrt(fit.params['omega'] / (1.0 - fit.params['alpha[1]'] - fit.params['beta[1]']) * 252) / scale


def garch_fit(ret: pd.Series):
    # For numerical stability, multiply returns by 100x. 
    # omega scales quadratically, other params are constant
    am = ZeroMean(100 * ret, rescale=False)
    am.volatility = GARCH(p=1, q=1)
    am.distribution = Normal() # GeneralizedError()
    return am.fit(disp='off')


def garch_fit_multi(symbols):
    # Normalize & winsorize returns, then fit GARCH(1,1).
    returns = {}
    fits    = {}
    epsilon = {}
    for symbol in symbols:
        prices = risk.get_prices(risk.get_data([symbol]))
        returns[symbol] = process_returns(prices[symbol])
        fits[symbol] = garch_fit(returns[symbol])
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


def garch_mc(drifts, corr, sigma_last, params, num_paths=1000, num_steps=252):
    paths   = np.zeros(shape=(num_paths, num_steps + 1, len(drifts)))   
    sigma_2 = np.ones(shape=(num_paths, num_steps + 1, len(drifts))) + sigma_last**2
        
    noise = np.random.multivariate_normal([0.0]*len(drifts), corr, size=(num_paths, num_steps))
    # noise = stat.gennorm(params.nu).rvs(size=(num_paths, num_steps))
    for symbol_idx in range(len(drifts)):
        for t in range(1, num_steps + 1):   
            sigma_2[:,t,symbol_idx] = params.omega[symbol_idx] + params.alpha[symbol_idx] * paths[:,t-1,symbol_idx]**2 + params.beta[symbol_idx] * sigma_2[:,t-1,symbol_idx]
            paths[:,t,symbol_idx] = np.sqrt(sigma_2[:,t,symbol_idx]) * noise[:,t-1,symbol_idx]
    return np.exp(drifts/252.0 + paths/100).cumprod(axis=1)


def process_jpm(fname: str, save=True):
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
        df = df.append(pd.Series({'MUB': muni.Value.sum()}))
    if len(pff) > 0:
        df = df.append(pd.Series({'PFF': pff.Value.sum()}))
    df.name = 'value'
    df.index.name = 'symbol'

    if save:
        df.to_csv('portfolios/{}'.format(fname))
    else:
        return df


def agg_folio_csv(folios: List[str], saveas: str):
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


def regress_factor(A: pd.Series, B: pd.DataFrame) -> pd.Series:
    """
    Regress s.t. A = beta x B + epsilon
    """
    fitted = OLS().fit(B, A)
    return A - (fitted.coef_ * B).sum(axis=1)


def get_components(universe: str = 'style') -> List[str]:
    """
    Read list of ETFs for given universe of factors.
    """
    factors = pd.read_csv('factors.csv',index_col=1)
    return factors.loc[factors.universe == universe].etf


def get_factors(universe: str = 'style') -> pd.DataFrame:
    """
    Generate factors from different universes:
    exposure = style/size/geo factors + bonds
    sector   = SPDR sectors

    returns pd.DataFrame of price per factor
    """

    # Treasury bonds: SHY = 1-3y, IEI = 3-7y, IEF = 7-10y, TLH = 10-20y, TLT = 20y+
    # Components from which we'll build our factors
    components = get_components(universe)

    # Get data and build prices dataframe
    loaded = 0
    for comp in components:
        if risk.is_symbol_cached(comp)[0] == -1 or risk.is_symbol_cached(comp)[0] > 10:
            risk.get_data([comp])
            loaded += 1
        if loaded == 5:
            print('hit 5 request limit, waiting 1 min...')
            loaded = 0
            time.sleep(60.0)
    data    = risk.get_data(list(components.values))
    prices  = risk.get_prices(data).dropna()
    returns = prices.pct_change().iloc[1:]

    # Build factors dataframe by transforming components.
    # todo: make factors better / add more
    factors = pd.DataFrame()

    if universe == 'style':
        # Core macro factors
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

    elif universe == 'sector':
        for sector, etf in components.items():
            if sector != 'market':
                factors[sector] = returns[etf]
                #factors[sector] = regress_factor(returns[etf], returns[['SPY']])

    elif universe == 'asset':
        for asset, etf in components.items():
            factors[asset] = returns[etf]

    # Take cumulative product of returns to generate price series
    factors  = (1.0 + factors).cumprod(axis=0)
    # Prepend 1.0 for each factor's initial price: just a niceity
    return pd.DataFrame({factor: 1.0 for factor in factors.columns}, index=[prices.index[0]]).append(factors)


def decompose(prices: pd.Series, factors: pd.DataFrame) -> Tuple[float, pd.Series, pd.DataFrame]:
    """
    Perform factor decomp given "market" factors.

    prices: pd.Series of asset prices
    factors: pd.DataFrame of factor prices
    returns r-squared score, pd.Series factor-predicted price, pd.DataFrame of factor coefs
    """

    returns = prices.pct_change()[1:]
    factors = factors.dropna().pct_change()[1:]

    returns_s, factors_s = risk.match_indices(returns, factors)

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

    returns_s, factors_s = risk.match_indices(returns, factors)
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    def err(w):
        return (((w * factors_s).sum(axis=1) - returns_s)**2).sum()
    res = minimize(err, [1.0/len(factors.columns)]*len(factors.columns), constraints=cons, bounds=[(0.0,1.0)]*len(factors.columns))
    weights = pd.Series(res['x'], index=factors.columns)
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


"""
Hardcoded (unfortunately) asset classes for some mutual funds so the right tax treatment can be applied to distributions.
"""
fund_categories = {'VWALX': 'National Munis', 'VWIUX': 'National Munis', 'VWITX': 'National Munis', 'VMMXX': 'MM Bond', 'SDSCX': 'Equity', 'JTISX': 'National Munis', 'TXRIX': 'National Munis', 
                   'VWEAX': 'HY Bond', 'VFSUX':'Bond', 'MAYHX': 'National Munis', 'VIPSX': 'Inflation Bond', 'SAMHX': 'HY Bond', 'ARTKX': 'Equity', 'JMSIX': 'Bond', 'KGGAX': 'Equity', 'LLPFX': 'Equity', 
                   'MPEMX': 'Equity', 'OAKEX': 'Equity', 'VWNAX': 'Equity', 'VMNVX': 'Equity', 'JPHSX': 'Bond', 'CSHIX': 'Bond', 'VFIDX': 'Bond', 'JHEQX': 'Equity', 'HLIEX': 'Equity'}
default_brackets = {'fed':0.388, 'state':0.068, 'div':0.306}

def taxes_by_asset(assets: List[str], brackets: Dict[str, float]):
    taxes = pd.Series({symbol: brackets['div'] for symbol in assets})
    for symbol in assets:
        # Assume mutual fund if symbol > 4 chars.
        category = risk.get_etf_category(symbol) if len(symbol) <= 4 or '-' in symbol else fund_categories[symbol]
        if category == 'New York Munis' or symbol in ['BNY','ENX']:
            taxes[symbol] = 0.0
        elif 'Munis' in category or symbol in ['JMST']:
            taxes[symbol] = brackets['state']
        elif 'Bond' in category and 'Preferred Stock' not in category:
            taxes[symbol] = brackets['fed'] + brackets['state']
    return taxes


class TaxablePortfolio:
    """
    To analyze performance/income of taxable account.
    Assume investor is taking out all dividends/interest in a year.
    """

    def __init__(self, basket: pd.Series, brackets: Dict[str,float] = default_brackets, from_date=None, reinvest=0.0, categorize=True):
        self.basket   = basket
        self.reinvest = reinvest
        self.brackets = brackets

        # First, assume all income taxed as qualified dividends.
        self.taxes = pd.Series({symbol: brackets['div'] for symbol in basket.index})
        # If requested, lookup each security on ETFDB to categorize as bonds/municipals.
        if categorize:
            self.taxes = taxes_by_asset(basket.index, brackets)

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
        metrics = metrics.assign(pr=results_pr.tr, maxdraw=[risk.get_max_drawdown(prices_pr_[ticker]) for ticker in metrics.index])
        metrics = metrics.assign(divs=self.yields, divstd=divstd, divdraw=divdraw, live=live_since)

        returns = self.value.pct_change()[1:]
        vol   = np.sqrt( (np.log(1.0 + returns)**2).mean()*252 )
        dnvol = np.sqrt( (np.log(1.0 + returns[returns < 0.0])**2).mean()*252 )
        rf    = risk.get_risk_free(self.value.index)

        # Requote PR/TR using weights & full history per security, rather than only full portfolio's history.
        sharpe   = (np.dot(self.basket, metrics.tr) - rf)/vol
        sortino  = (np.dot(self.basket, metrics.tr) - rf)/dnvol
        ann_divs = risk.agg_to_period(self.total_inco, freq='A')
        divstd   = ann_divs.std() / ann_divs.mean()
        divdraw  = risk.get_max_drawdown(ann_divs)
        pmetrics = pd.Series({'pr': np.dot(self.basket, metrics.pr), 'tr': np.dot(self.basket, metrics.tr), 'vol': vol, 'sharpe': sharpe, 
                              'sortino': sortino, 'maxdraw': risk.get_max_drawdown(self.value), 'divs': np.dot(self.yields, self.basket),
                              'live': self.prices.index[0], 'divstd': divstd, 'divdraw': divdraw})
        pmetrics.name = 'portfolio'
        return metrics.append(pmetrics)[['pr','tr','vol','sharpe','sortino','maxdraw','divs','divstd','divdraw','live']], covar


class MCPortfolio:
    def __init__(self, basket: pd.Series, brackets: Dict[str,float] = default_brackets):
        self.basket = basket
        self.brackets = brackets
        self.taxes = taxes_by_asset(basket.index, brackets)
        self.data = risk.get_data(basket.index)
        self.prices = risk.get_prices(self.data, field='close')
        self.yields = risk.get_indic_yields(self.data, last=True)

    def generate_paths(self, years=10, drifts=None, vols=None, N=1000):
        # Use GARCH(1,1) with Multivariate Normal innovations.
        returns, fits, params, corr = garch_fit_multi(self.basket.index)
        sigma_last = np.array([fit.conditional_volatility[-1] for fit in fits.values()])

        # Drift is either given or uses historical average.
        log_r = np.log(self.prices.dropna()/self.prices.dropna().shift(1))
        self.drifts = drifts if drifts is not None else 252.0 * log_r.mean()
        self.paths = garch_mc(self.drifts.values, corr, sigma_last, params, num_paths=N, num_steps=years*252)

    def build(self, init_value, withdraw, fee=0.00):
        N = self.paths.shape[0]
        qtr_index   = np.arange(0, (self.paths.shape[1] - 1) // 63 + 1) * 63
        self.shares = np.zeros(shape=(N, qtr_index.shape[0], self.paths.shape[2])) + (self.basket.values * init_value)
        self.costs  = np.zeros(shape=(N, qtr_index.shape[0], self.paths.shape[2])) + 1.0
        self.gains  = np.zeros(shape=(N, qtr_index.shape[0]))
        self.value  = np.zeros(shape=(N, qtr_index.shape[0])) + init_value
        self.income = np.zeros(shape=(N, qtr_index.shape[0]))
        self.resid  = np.zeros(shape=(N, qtr_index.shape[0]))

        if type(withdraw) == float:
            withdraw = np.zeros(shape=(qtr_index.shape[0] // 4)) + withdraw

        # For each quarter, qtr_index[qtr] = the daily index in paths for that quarter.
        for qtr in range(1, len(qtr_index)):
            # Grow portfolio by performance.
            self.value[:,qtr]  = np.sum(self.shares[:,qtr-1] * self.paths[:,qtr_index[qtr]], axis=1)
            self.income[:,qtr] = np.sum((self.basket * self.yields/4)*(1.0 - self.taxes))*self.value[:,qtr-1]
            self.resid[:,qtr]  = self.income[:,qtr] - withdraw[(qtr-1)//4]/4 - self.value[:,qtr-1]*fee/4
            self.value[:,qtr] += self.resid[:,qtr]
            self.shares[:,qtr] = np.outer(self.value[:,qtr], self.basket.values) / self.paths[:,qtr_index[qtr]]

            # Rebalance and charge capital gains taxes (assuming all long-term.)
            shares_to_trade    = self.shares[:,qtr] - self.shares[:,qtr-1]
            amended_basis      = (self.costs[:,qtr-1]*self.shares[:,qtr-1] + shares_to_trade*self.paths[:,qtr_index[qtr]])/self.shares[:,qtr]
            self.costs[:,qtr]  = (shares_to_trade > 0) * amended_basis + (shares_to_trade < 0)*self.costs[:,qtr-1]

            self.gains[:,qtr]  = ((shares_to_trade < 0) * -shares_to_trade * (self.paths[:,qtr_index[qtr]] - self.costs[:,qtr-1])).sum(axis=1)
            self.value[:,qtr]  -= (self.gains[:,qtr] > 0) * self.gains[:,qtr] * self.brackets['div']
        
        # Snap negative portfolios to zero.
        for path_idx, step_idx in zip(np.where(self.value <= 0)[0], np.where(self.value <= 0)[1]):
            self.value[path_idx, step_idx:] = 0.0

    def get_vols(self):
        # Return vol using asset log-returns and basket per path. Rather than taking vol using actual value.
        survivors = np.all(self.value > 0, axis=1)
        return  np.sqrt(252*((np.log(self.paths[survivors, 1:] / self.paths[survivors, :-1]) * self.basket.values).sum(axis=2)**2).mean(axis=1))

    def get_prs(self):
        # Return average quarterly log-return in annualized terms per path.
        survivors = np.all(self.value > 0, axis=1)
        return np.log(self.value[survivors,1:]/self.value[survivors,:-1]).mean(axis=1) * 4

    def get_undl_paths(self):
        # Return performance of underlying portfolio (assuming no tax/withdrawals) for each path.
        return np.cumprod(1.0 + (self.basket.values * (self.paths[:,1:,:] / self.paths[:,:-1,:] - 1.0)).sum(axis=2), axis=1)

    def get_max_drawdowns(self):
        return [risk.get_max_drawdown(path) for path in self.value if path[-1] > 0]

    def get_loss_probs(self, losses=[0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]):
        loss_probs = pd.DataFrame()
        loss_probs['Within 3y'] = np.array([np.sum(np.any([self.value[:,:4*3+1] < (1 - loss) * self.value[0][0]], axis=2))/len(self.value) for loss in losses])
        loss_probs['Ever'] = np.array([np.sum(np.any([self.value < (1 - loss) * self.value[0][0]], axis=2))/len(self.value) for loss in losses])
        loss_probs['At End'] = np.array([len(self.value[:,-1][self.value[:,-1] < (1 - loss) * self.value[0][0]])/len(self.value) for loss in losses])
        loss_probs.index = ['>{:.0f}%'.format(100*loss) for loss in losses]
        return loss_probs