# portfolio
Quantitative portfolio analytics I built for the investments I manage. Recently revamped!

## Components
<code>service.py</code> - lightweight API client for AlphaVantage which offers up to 20 years of prices, dividends, and earnings

<code>risk.py</code> - retrieves, processes, and runs some metrics on data

<code>analytics.py</code> - taxable portfolio analysis

<code>factors.py</code> - factor decomposition

<code>mc.py</code> - MonteCarlo portfolio projections

This library requires two folders -- <code>data</code> and <code>keys</code>. The keys folder should hold a plaintext file for each API with ending ".keys" containing the authentication token(s). The data folder is used to cache downloaded prices in csv files. A csv is re-downloaded when its data is requested and the age of the file exceeds a specified threshold (default is 10 days.) These can be in the same directory from where the code is run or overriden using enviroment varaibles.

## Example
### Example 1: basic risk/return data
```
from portfolio.risk import Driver, Utils

# Retrieve data for some macro ETFs
data = Driver.getData(['SPY','QQQ','AGG','HYG'])
prices = Utils.getPrices(data).dropna()

# table is dataframe with historic risk/return metrics and current yields
# covar is the covariance matrix of returns
table, covar = Utils.getMetrics(prices, data=data)
```
Year-to-date performance for multiple ETFs can be incorporated into a chart like the below (labelled by asset class.)<br>
<img src="https://github.com/idrach55/portfolio/blob/main/plots/ytd-assets.png?raw=true" width=600>

Historic risk/return metrics given a portfolio's composition.<br>
<img src="https://github.com/idrach55/portfolio/blob/main/plots/historic-metrics.png?raw=true" width=800>

### Example 2: single security factor decomposition
```
from portfolio.factors import FactorUniverse, decompose
prices = Utils.getPrices(Driver.getData(['AAPL']))

# get price series for some factors I call 'style'
factors = FactorUniverse.STYLE.getFactors()

# r_squared for the factor decomp
# pred is the predicted price series using the decomp
# df outlines the coefficients
r_squared, pred, df = decompose(prices['AAPL'], factors)
```

## In-Depth Analytics
### Factors
I set up 3 factor sets: <code>style</code>, <code>asset</code>, and <code>sector</code>. Each of these are built out of the total returns on ETFs because I couldn't find a free/cheap factor index data source. I generally aimed to use ETFs with 5y+ history. One issue with ETFs is that they bleed by their management fees. Further, less liquid ETFs may appear more volatile than their underlying asset(s). 

There are 2 functions to perform decomposition:<br>
<code>factors.decompose</code> -- Ordinary least-squares linear regression.<br>
<code>factors.decompose_const</code> -- Uses <code>scipy.minimize</code> on the sum-of-squares constraining the coefficients to be long-only and sum to unity. This can be useful for finding an approximate portfolio without shorting/leverage.
#### <code>style</code>
Loosely based on the usual sort of quant factors. I did some testing/design with the intent for them to be mostly orthogonal. These formulae refer to the tickers of each ETF.
|factor name|formula|description|
|-----------|-------|-----|
|<b>core macro</b>|
|equities|VT|vanguard total world market|
|rates|IEI|ishares 3-7 year bond etf|
|credit|HYG - IEI|ishares high yield vs 3-7y treasuries|
|commods|GSG|ishares gs commodity index|
|<b>secondary macro</b>|
|inflation|TIP - beta x IEF|ishares tips statically hedged with 7-10y treasuries|
|emerging|0.5 x (EEM - VTI + EMB - IEF)|average of EM vs US equity & bond spreads|
|usequity|VTI - EFA|US vs non-US developed mkt equities|
|usdollar|UUP|invesco dollar index|
|<b>macro style</b>|
|shortvol|PUTW - beta x VTI|statically hedged put-write strategy|
|municipal|MUB - IEF|ishares investment grade munis vs 7-10y treasuries|
|realestate|IYR - VTI|spread btwn REITs and equities|
|<b>equity style</b>|
|smallcap|IWM - VTI|russell 2k vs VTI (mostly S&P 500 like)|
|lowrisk|USMV - VTI|large cap low vol vs VTI|
|momentum|MTUM - VTI|large cap momentum vs VTI|
|quality|QUAL - VTI|large cap quality vs VTI|
|value|IWD - IWF|russell 1k value vs russel 1k growth| 

#### <code>asset</code>
These are single ETF factors meant to represent basic portfolio building blocks.

#### <code>sector</code>
Consists of SPY as a broad market factor, then a beta-hedged spread (vs SPY) for each XL-sector ETF. Easily outlines under/over-weight sector exposures compared to the S&P 500.  

### Portfolio Monte Carlo
The class <code>mc.MCPortfolio</code> generates simulated portfolio paths using a multi-asset GARCH model. It takes as inputs a basket, tax brackets, optional withdrawal amount and fees, number of paths, and time horizon. 

Performance and max drawdown at different percentiles for a given simulation.<br>
<img src="https://github.com/idrach55/portfolio/blob/main/plots/portfolio-mc.png?raw=true" width=800>

#### Underlying Dynamics
A GARCH(1,1) model with Normal innovations is fit to each security. A correlation matrix is computed among the residuals <code>u_t / sigma_t</code> to parameterize a multivariate Normal which is sampled as the noise for the simulation.

#### Portfolio Building
The projections output quarterly values along each path. The portfolio maintains constant weights, rebalancing quarterly and paying capital gains taxes if necessary. It also pays roughly the appropriate tax rate on each security. Withdrawals and fees are also taken out quarterly if specified. Using <code>np.quantile</code>, one can examine different percentile values over time. The function <code>get_loss_probs</code> computes what percent of portfolios take a loss at or greater than each threshold within certain timeframes.   
