"""
Author: Isaac Drachman
Date: 8/16/2021

Code for downloading stock and other financial data, processing/transforming, and computing derived statistics.
"""
from __future__ import annotations

import os
from datetime import datetime
from enum import Enum
from io import StringIO
from typing import Dict, List, Optional, Tuple, Union
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
import scipy.optimize as opt
from bs4 import BeautifulSoup

from .services import AlphaVantage as av

# CONSTANTS
DATA_DIR = os.environ.get("DATA_DIR", "data")


class UnitType(Enum):
    All = 1
    OneBd = 2

    def __str__(self):
        return self.name


class CloseMethod(Enum):
    RAW = 1
    ADJUSTED = 2

    def __str__(self) -> str:
        if self == CloseMethod.ADJUSTED:
            return "adjusted close"
        elif self == CloseMethod.RAW:
            return "close"


# All things that hit external API. These are here and not services as they refer to DATA_DIR for caching.
class Driver:
    @staticmethod
    def getRentalData(unit_type: UnitType) -> None:
        categories = [f"medianAskingRent_{unit_type}", f"rentalInventory_{unit_type}"]
        for cat in categories:
            if f"{cat}.csv" in os.listdir(DATA_DIR):
                os.remove(f"{DATA_DIR}/{cat}.csv")

            url = f"https://streeteasy-market-data-download.s3.amazonaws.com/rentals/{unit_type}/{cat}.zip"
            with open("data.zip", "wb") as file:
                file.write(urlopen(url).read())
            ZipFile("data.zip").extractall(f"{DATA_DIR}/")
            os.remove("data.zip")

    @staticmethod
    def getBlackRockData(etf: str) -> pd.DataFrame:
        # Get iShares ETF composiiton.
        etfs = {
            "IVV": "/us/products/239726/ishares-core-sp-500-etf",
            "HYG": "/us/products/239565/ishares-iboxx-high-yield-corporate-bond-etf",
        }

        # Find corresponding link for ETF and read whole page
        base_url = "https://www.ishares.com"
        r = requests.get("{}{}".format(base_url, etfs[etf]))
        # Locate link for holdings csv
        soup = BeautifulSoup(r.text, features="lxml")
        links = soup.find_all("a", href=True)
        a = [
            link
            for link in links
            if link.get_text() == "Detailed Holdings and Analytics"
        ]
        assert len(a) > 0, "no holdings link found"
        csv_url = a[0].get("href")
        # Request holdings csv, put into temp file, and read into pandas dataframe
        r = requests.get("{}{}".format(base_url, csv_url))
        data = r.text[r.text.find("\nName") :]
        tmpfile = StringIO(data)
        df = pd.read_csv(tmpfile, index_col=0)
        return df.iloc[:-1]

    @staticmethod
    def getETFCategory(symbol: str) -> str:
        cache = pd.DataFrame()
        if "etfs.csv" in os.listdir(DATA_DIR):
            cache = pd.read_csv(f"{DATA_DIR}/etfs.csv", index_col=0)
            if symbol in cache.index:
                return cache.loc[symbol].category

        url = "https://etfdb.com/etf/{}/#etf-ticker-profile"
        r = requests.get(url.format(symbol))
        soup = BeautifulSoup(r.text, features="lxml")
        spans = soup.find_all("span", class_="stock-quote-data")
        category = spans[2].get_text().strip() if len(spans) >= 3 else "none"
        pd.concat(
            [cache, pd.DataFrame({"category": [category]}, index=[symbol])], axis=0
        ).to_csv(f"{DATA_DIR}/etfs.csv")
        return category

    @staticmethod
    def getIsCached(symbol: str):
        existing_data = os.listdir(DATA_DIR)
        existing_syms = {fname[: fname.find("_")]: fname for fname in existing_data}
        if symbol in existing_syms.keys():
            fname = existing_syms[symbol]
            data_date = pd.to_datetime(fname[fname.find("_") + 1 : fname.find(".")])
            return (datetime.today() - data_date).days, fname
        else:
            return -1, None

    @staticmethod
    def getTreasuryData(match_idx=None, data_age_limit: int = 10) -> pd.DataFrame:
        data_age, fname = Driver.getIsCached("treasury")
        if data_age != -1 and data_age <= data_age_limit:
            df = pd.read_csv("{}/{}".format(DATA_DIR, fname), index_col=0)
            df.index = pd.to_datetime(df.index)
        else:
            url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value=all"
            data = []
            dates = []
            page_idx = 1
            while True:
                page_url = url + f"&page={page_idx}"
                r = requests.get(page_url)
                soup = BeautifulSoup(r.text, "lxml")
                entries = soup.find_all("entry")
                if len(entries) <= 1:
                    break
                for entry in entries:
                    dates.append(pd.to_datetime(entry.find("d:new_date").text))
                    value = {}
                    for month in [1, 3, 6]:
                        key = f"d:bc_{month}month"
                        if entry.find(key) is None:
                            continue
                        value[f"{month}m"] = float(entry.find(key).text) / 100.0
                    for year in [1, 2, 3, 5, 7, 10, 20, 30]:
                        key = f"d:bc_{year}year"
                        if entry.find(key) is None:
                            continue
                        value[f"{year}y"] = float(entry.find(key).text) / 100.0
                    data.append(value)
                page_idx += 1
            df = pd.DataFrame(data)
            df.index = dates
        if match_idx is None:
            return df
        subset_idx = df.index.intersection(match_idx)
        return df.loc[subset_idx]

    @staticmethod
    def getData(
        symbols: List[str], data_age_limit: int = 10
    ) -> Dict[str, pd.DataFrame]:
        # Load historical price data either from cache or API.
        data = {}
        for symbol in symbols:
            # Store downloaded data into its own dataframe to avoid truncating index to prior symbols.
            data_single = pd.DataFrame()
            data_age, fname = Driver.getIsCached(symbol)

            # If data is found in data directory and within age limit.
            if data_age != -1 and data_age <= data_age_limit:
                data_single = pd.read_csv("{}/{}".format(DATA_DIR, fname), index_col=0)
                data_single.index = pd.to_datetime(data_single.index)
                # Skip download and go to next symbol.
                data[symbol] = data_single
                continue
            # If data is found but too old.
            elif data_age > data_age_limit:
                os.remove("{}/{}".format(DATA_DIR, fname))

            # If program flow gets here, data was either unavailable or too old.
            service = av()
            data_single = service.chart(symbol, adjusted=True)
            data_single.to_csv(
                "{}/{}_{}.csv".format(
                    DATA_DIR, symbol, datetime.today().strftime("%d%b%y")
                )
            )
            data_single.index = pd.to_datetime(data_single.index)
            data[symbol] = data_single

        # 'Close' prices are actual close, need to be adjusted for splits to have any use.
        for symbol in symbols:
            r_factor = data[symbol]["split coefficient"][::-1].cumprod().shift(1)
            r_factor.iloc[0] = 1.0
            data[symbol]["close"] = data[symbol]["close"] / r_factor
        return data

    @staticmethod
    def getPrices(
        symbols: List[str],
        method: CloseMethod = CloseMethod.ADJUSTED,
        data_age_limit: int = 10,
    ) -> pd.DataFrame:
        data = Driver.getData(symbols, data_age_limit=data_age_limit)
        prices = pd.concat([df[str(method)] for df in data.values()], axis=1, sort=True)
        prices.columns = data.keys()
        return prices


# Hardcoded (unfortunately) asset classes for some mutual funds so the right tax treatment can be applied to distributions.
class FundCategory(Enum):
    MUNI = 1
    BOND = 2
    HY_BOND = 3
    MONEY_MKT = 5
    INFLATION_LINKED = 6
    EQUITY = 7
    PREFERRED = 8
    UNKNOWN = 9

    @staticmethod
    def createFromString(category: str) -> FundCategory:
        if "High Yield Bonds" in category:
            return FundCategory.HY_BOND
        elif "Equities" in category:
            return FundCategory.EQUITY
        elif "Preferred Stock" in category:
            return FundCategory.PREFERRED
        elif "Inflation-Protected" in category:
            return FundCategory.INFLATION_LINKED
        elif "Munis" in category:
            return FundCategory.MUNI
        elif "Bonds" in category:
            return FundCategory.BOND
        return FundCategory.UNKNOWN

    @staticmethod
    def createFromMutualFund(symbol: str) -> FundCategory:
        mapping = {
            "ARTKX": FundCategory.EQUITY,
            "SDSCX": FundCategory.EQUITY,
            "KGGAX": FundCategory.EQUITY,
            "LLPFX": FundCategory.EQUITY,
            "MPEMX": FundCategory.EQUITY,
            "OAKEX": FundCategory.EQUITY,
            "VWNAX": FundCategory.EQUITY,
            "VMNVX": FundCategory.EQUITY,
            "JHEQX": FundCategory.EQUITY,
            "HLIEX": FundCategory.EQUITY,
            "JITIX": FundCategory.MUNI,
            "JTISX": FundCategory.MUNI,
            "MAYHX": FundCategory.MUNI,
            "TXRIX": FundCategory.MUNI,
            "VWALX": FundCategory.MUNI,
            "VWITX": FundCategory.MUNI,
            "VWIUX": FundCategory.MUNI,
            "VWSUX": FundCategory.MUNI,
            "VMMXX": FundCategory.MONEY_MKT,
            "SAMHX": FundCategory.HY_BOND,
            "VWEAX": FundCategory.HY_BOND,
            "JMSIX": FundCategory.BOND,
            "VFSUX": FundCategory.BOND,
            "JPHSX": FundCategory.BOND,
            "CSHIX": FundCategory.BOND,
            "VFIDX": FundCategory.BOND,
            "VIPSX": FundCategory.INFLATION_LINKED,
            "VAIPX": FundCategory.INFLATION_LINKED,
        }
        return mapping.get(symbol, FundCategory.UNKNOWN)

    @staticmethod
    def createFromSymbol(symbol: str) -> FundCategory:
        if len(symbol) <= 4 or "-" in symbol:
            category = FundCategory.createFromString(Driver.getETFCategory(symbol))
        else:
            category = FundCategory.createFromMutualFund(symbol)
        if category == FundCategory.UNKNOWN:
            if symbol in ["BNY", "ENX", "BFK", "NVG", "JMST"]:
                category = FundCategory.MUNI
        return category

    def getIsTaxableBond(self) -> bool:
        return self in [
            FundCategory.BOND,
            FundCategory.INFLATION_LINKED,
            FundCategory.HY_BOND,
            FundCategory.MONEY_MKT,
        ]


# Compute relevant quantities.
# These functions are given dataframes and do not access remote data themselves.
class Utils:
    @staticmethod
    def getDateOffset(timespan: str) -> pd.Timedelta:
        # Turn a date code, ex '5y', '3m', '1w' into a cutoff date.
        multiplier = 1
        if timespan[-1] == "y":
            multiplier = 52
        elif timespan[-1] == "m":
            multiplier = 12
        return pd.Timedelta("{:d}w".format(multiplier * int(timespan[:-1])))

    @staticmethod
    def getMaxDrawdown(ts: Union[pd.Series, List[float]], index=False) -> float:
        # Location of drawdown in series if index=True
        maxdraw = 0
        peak = -99999
        point = -1
        for i in range(1, len(ts)):
            if ts[i] > peak:
                peak = ts[i]
            if (peak - ts[i]) / peak > maxdraw:
                point = i
                maxdraw = (peak - ts[i]) / peak
        if index:
            return maxdraw, point
        return maxdraw

    @staticmethod
    def getCrashMoves(
        portfolio: pd.Series, prices: pd.DataFrame, days: int = 30
    ) -> pd.Series:
        # Compute the returns of a portfolio's constituents during the portfolio's max drawdown.
        draw, idx = Utils.getMaxDrawdown(portfolio, index=True)
        # If lookback preceeds start of data, use start of data instead.
        if days > idx:
            days = idx
        drawdowns = (prices.iloc[idx] - prices.iloc[idx - days]) / prices.iloc[
            idx - days
        ]
        return drawdowns.sort_values()

    @staticmethod
    def getYears(series: Union[pd.Series, pd.DataFrame]) -> float:
        return (series.index[-1] - series.index[0]).days / 365

    @staticmethod
    def aggregateToPeriod(
        series: pd.Series, freq: str = "A", norm: int = 252, cutoff: int = 60
    ) -> pd.Series:
        """
        Group series into periods and return periods with substantial data, optionally normalized by length.

        norm:   normalize assuming each period 'should' have this many data points
        cutoff: include periods with at least this many data points
        """
        assert freq in ["A", "Q"]
        grouped = series.groupby(series.index.to_period(freq=freq))
        summed = grouped.sum()[grouped.count() >= cutoff]
        normed = (summed / grouped.count() * norm) if type(norm) == int else summed
        return normed.dropna()

    @staticmethod
    def getDividendVol(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        # Return stddev of annual income / average annual income.
        divs = Utils.getDivs(data)
        divvol = {}
        divdraw = {}
        for key, value in divs.items():
            # If no dividends, return 0%.
            if len(divs[key]) == 0:
                divvol[key] = 0.0
                continue
            # Sum annual income, include years with at least 1/4 typical divs (ex: 1 qtr if quarterly)
            # and normalize by typical number of divs per year.
            ann_divs = Utils.aggregateToPeriod(
                divs[key].amount,
                freq="A",
                norm=1.0 / divs[key].period.mean(),
                cutoff=0.25 / divs[key].period.mean(),
            )
            divvol[key] = ann_divs.std() / ann_divs.mean()
            divdraw[key] = Utils.getMaxDrawdown(ann_divs)
        return pd.Series(divvol), pd.Series(divdraw)

    @staticmethod
    def getPrices(
        data: Dict[str, pd.DataFrame], method: CloseMethod = CloseMethod.ADJUSTED
    ) -> pd.DataFrame:
        prices = pd.concat([df[str(method)] for df in data.values()], axis=1, sort=True)
        prices.columns = data.keys()
        return prices

    @staticmethod
    def getRiskReturn(
        prices: pd.DataFrame, ann_factor: int = 252
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
        # Compute metrics: total return, volatility, downvol (volatility | -return), covariance matrix
        results = []
        # Do each symbol individually
        for symbol in prices.columns:
            px = prices[symbol].dropna()
            returns = px.pct_change()
            rf = 0.0  # get_risk_free(px.index)
            years = Utils.getYears(px)

            series = pd.Series(float)
            series["tr"] = np.log(px.iloc[-1] / px.iloc[0]) / years
            series["vol"] = np.sqrt((np.log(1.0 + returns) ** 2).mean() * ann_factor)
            series["downvol"] = np.sqrt(
                (np.log(1.0 + returns[returns < 0]) ** 2).mean(axis=0) * ann_factor
            )
            series["sharpe"] = (series.tr - rf) / series.vol
            series["sortino"] = (series.tr - rf) / series.downvol
            series.name = symbol
            results.append(series)
        results = pd.DataFrame(results)
        variance = pd.DataFrame(results.vol).dot(pd.DataFrame(results.vol).T)
        covar = variance * prices.dropna().pct_change().corr()
        return results, covar

    @staticmethod
    def getDivs(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        div_data = {}
        for symbol in data.keys():
            divs = data[symbol].loc[data[symbol]["dividend amount"] > 0]
            df = pd.DataFrame()
            df["amount"] = divs["dividend amount"]
            df["yield"] = divs["dividend amount"] / divs["close"]
            df["period"] = divs.index.to_series().diff().map(lambda dt: dt.days / 365)
            # Drop dividends > 5% (assume these are special dividends)
            div_data[symbol] = df.drop(df.loc[df["yield"] >= 0.05].index, axis=0)
        return div_data

    @staticmethod
    def getIndicYields(
        data: Dict[str, pd.DataFrame], last: bool = False, timespan: str = "1y"
    ) -> pd.Series:
        # If last=True, returns last div / avg. period (using all divs), otherwise return sum(divs)
        div_data = Utils.getDivs(data)
        prices = Utils.getPrices(data, method=CloseMethod.RAW)

        yields = []
        for symbol, divs in div_data.items():
            window = divs[datetime.today() - Utils.getDateOffset(timespan) :].amount
            ann_factor = Utils.getDateOffset(timespan).days / 365
            if len(window) == 0:
                yields.append(0.0)
                continue
            if last:
                window = divs.iloc[-1].amount
                ann_factor = divs.period.mean()
            yields.append(window.sum() / prices[symbol].dropna().iloc[-1] / ann_factor)
        return pd.Series(yields, index=div_data.keys()).fillna(0.0)

    @staticmethod
    def getMetrics(
        prices: pd.DataFrame,
        data: Optional[Dict[str, pd.DataFrame]] = None,
        ann_factor: int = 252,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        results, covar = Utils.getRiskReturn(prices, ann_factor=ann_factor)
        metrics = pd.DataFrame()
        metrics = metrics.assign(
            tr=results.tr,
            vol=results.vol,
            sharpe=results.sharpe,
            sortino=results.sortino,
        )
        metrics = metrics.assign(
            maxdraw=[
                Utils.getMaxDrawdown(prices[ticker].dropna())
                for ticker in metrics.index
            ]
        )
        if data is not None:
            yields = Utils.getIndicYields(data)
            divstd = Utils.getDividendVol(data)
            metrics = metrics.assign(divs=yields, divstd=divstd[0])
        return metrics, covar

    @staticmethod
    def getRiskFree(match_idx: pd.Series) -> float:
        # Get annualized risk-free rate as compounded 1m t-bills over specified timeframe.
        treasury = Driver.getTreasuryData(match_idx=match_idx)
        years = Utils.getYears(treasury)
        return np.log(np.exp(np.sum(treasury["1m"] / 360))) / years

    @staticmethod
    def doBasket(basket: pd.Series, cutoff=0.01) -> pd.Series:
        removes = []
        for remove in removes:
            remove_idx = np.where(basket.index == remove)[0][0]
            basket = basket[
                basket.index[0:remove_idx].append(basket.index[remove_idx + 1 :])
            ]
        basket /= basket.sum()

        fixed = basket[basket > cutoff]
        fixed /= fixed.sum()

        dropped = 100.0 - 100.0 * basket[fixed.index].sum()
        print(f"dropped {len(basket) - len(fixed)} name(s) totaling {dropped:.2f}%")

        # Just load for later.
        _data = Driver.getData(fixed.index)
        return fixed


class Earnings:
    @staticmethod
    def getIsCached(symbol: str) -> Tuple[int, Optional[str]]:
        existing_data = os.listdir(f"{DATA_DIR}/")
        existing_data = [fname for fname in existing_data if "earnings-" in fname]
        existing_syms = {fname[: fname.find("_")][9:]: fname for fname in existing_data}
        if symbol in existing_syms.keys():
            fname = existing_syms[symbol]
            data_date = pd.to_datetime(fname[fname.find("_") + 1 : fname.find(".")])
            return (datetime.today() - data_date).days, fname
        else:
            return -1, None

    @staticmethod
    def getEarnings(
        symbols: List[str], data_age_limit: int = 30
    ) -> Dict[str, pd.DataFrame]:
        """
        Get raw earnings data per symbol, structured as:
        fiscal_quarter_end, report_date, report_eps, estimate_eps
        """
        service = av()
        earns = {}
        for symbol in symbols:
            data_age, fname = Earnings.getIsCached(symbol)
            # If data is found in data directory and within age limit.
            if data_age != -1 and data_age <= data_age_limit:
                earnings = pd.read_csv("{}/{}".format(DATA_DIR, fname), index_col=0)
                # Skip download and go to next symbol.
                earns[symbol] = earnings
                continue
            # If data is found but too old.
            elif data_age > data_age_limit:
                os.remove("{}/{}".format(DATA_DIR, fname))

            # If program flow gets here, data was either unavailable or too old.
            try:
                earnings = pd.DataFrame(service.earnings(symbol)["quarterlyEarnings"])
                earnings = earnings[::-1]
                earnings.to_csv(
                    "{}/earnings-{}_{}.csv".format(
                        DATA_DIR, symbol, datetime.today().strftime("%d%b%y")
                    )
                )
            except:
                # Likely from limit on 5 requests/minute.
                # Drop the symbol and the user may re-run if desired (others will be cached then).
                print("error: failed to load {}".format(symbol))
                continue
            earns[symbol] = earnings
        return earns

    @staticmethod
    def getReported(earns: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        earns_df = pd.DataFrame()
        for symbol in earns.keys():
            earns[symbol].fiscalDateEnding = pd.to_datetime(
                earns[symbol].fiscalDateEnding
            )
            quarter = pd.PeriodIndex(earns[symbol].fiscalDateEnding, freq="Q")
            earns_df[symbol] = pd.Series(
                earns[symbol].groupby(quarter).sum().reportedEPS, name=symbol
            )
        return earns_df
