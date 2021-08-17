# portfolio
Portfolio analytics tools I built for the family office investments I manage.

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
I set up 3 factor sets: <code>style</code>, <code>asset</code>, and <code>sector</code>. Each of these are built out of the total returns on ETFs because I couldn't find a free/cheap factor index data source. Naturally one issue is that all the components have a management fee bleed. 
#### <code>style</code>
Loosely based on the usual sort of quant factors. I did some testing/design with the intent for them to be generally orthogonal. These formulae refer to the tickers of each ETF.
|factor name|formula|description|
|-----------|-------|-----|
|equities|=VT|vanguard total world market|
|rates|=IEI|ishares 3-7 year bond etf|
|credit|=HYG-IEI|spread between ishares high yield and treasury|
|commods|=GSG|ishares gs commodity index|
