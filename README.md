# portfolio
Quantitative portfolio analytics I built for the investments I manage.

## Components
<code>service.py</code><br>
Lightweight API clients for various financial data providers. Most heavily used is AlphaVantage, through which up to 20 years of prices, splits, dividends, and earnings can be retrieved for stocks, ETFs, and mutual funds. Quandl is supported for a few datasets as well. Also includes Polygon and IEX which I briefly tried out. Auth keys are retrieved from plaintext files with ending ".keys" from a "keys" folder in the directory the code is being run from.

<code>risk.py</code><br>
Retrieves, processes, and runs some metrics on data. 

<code>analytics.py</code><br>
Factor decomposition, taxable portfolio analysis, and Monte Carlo portfolio projections.

## Example
### Example 1: basic risk/return data
```
# If running from a different folder, can use sys.path hack:
import sys; sys.path.append('path_containing_portfolio_folder')
from portfolio import risk, analytics

# Retrieve data for some macro ETFs
data = risk.get_data(['SPY','QQQ','AGG','HYG'])
prices = risk.get_prices(data).dropna()

# table is dataframe with historic risk/return metrics and current yields
# covar is the covariance matrix of returns
table, covar = risk.get_metrics(prices, data=data)
```
### Example 2: single security factor decomposition
```
prices = risk.get_prices(risk.get_data(['AAPL']))

# get price series for some factors I call 'style'
factors = analytics.get_factors('style')

# r_squared for the factor decomp
# pred is the predicted price series using the decomp
# df outlines the coefficients
r_squared, pred, df = analytics.decompose(prices['AAPL'], factors)
```

## In-Depth Analytics
### Factors
I set up 3 factor sets: <code>style</code>, <code>asset</code>, and <code>sector</code>. Each of these are built out of the total returns on ETFs because I couldn't find a free/cheap factor index data source. I generally aimed to use ETFs with 5y+ history. One issue with ETFs is that they bleed by their management fees. Further, less liquid ETFs may appear more volatile than their underlying asset(s). 
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
|factor name|ETF|
|-----------|---|
|<b>fixed income and cash<b>|
|US Cash|BIL|
|US Mid Treasuries|IEF|
|US Long Treasuries|TLT|
|US Inflation-Linked|TIP|
|US IG Corp Bonds|LQD|
|US Agg Bonds|AGG|
|US HY Bonds|HYG|
|EM Sovereign Debt|EMB|
|US Muni 1-15 Yr Blend|MUB|
|US Muni High Yield|HYD|
|<b>equities</b>|
|US Large Cap|VV|
|US Mid Cap|VO|
|US Small Cap|VB|
|EAFE Equity|VEA|
|EM Equity|VWO|
|US Equity Value|VTV|
|US Equity Dividend|VYM|
|<b>alternatives</b>|
|US REITs|IYR|
|Commodities|GSG|

#### <code>sector</code>
Consists of SPY as a broad market factor, then a beta-hedged spread (vs SPY) for each XL-sector ETF. Easily outlines under/over-weight sector exposures compared to the S&P 500.  
