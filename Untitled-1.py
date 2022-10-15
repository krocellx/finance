# %%
from multiprocessing import AuthenticationError
from multiprocessing.sharedctypes import Value
import os

import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from flask import Flask, request, jsonify, make_response  # noqa: E402
from datetime import datetime  # noqa: E402

load_dotenv(dotenv_path="./.env.local")

FMP_ROOT_URL = os.environ.get("FMP_ROOT_URL", "https://financialmodelingprep.com/api/")
FMP_KEY = os.environ.get("FMP_KEY", "")


def get_treasury_rate(start_date, end_date=None):
    """get 10 year treasury as risk free rate"""
    if not (end_date):
        end_date = start_date

    url = FMP_ROOT_URL + "v4/treasury"
    params = {"apikey": FMP_KEY, "from": start_date, "to": end_date}
    response = requests.get(url=url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        data = response.json()["error"]
        raise ValueError(data)


def get_stock_price(ticker, start_date=None, end_date=None):
    url = FMP_ROOT_URL + "v3/historical-price-full/" + ticker.upper()
    params = {"apikey": FMP_KEY}
    error_msg = ""

    if start_date:
        params["from"] = start_date
        params["to"] = end_date if end_date else start_date
        error_msg = f" from {start_date} to {end_date}"

    response = requests.get(url=url, params=params)

    if response.status_code == 200:
        data = response.json()
        if not (data):
            raise ValueError("No result found for ticker " + str(ticker) + error_msg)

        # sort data
        hist_price = data["historical"]
        sorted_hist_price = sorted(
            hist_price, key=lambda t: datetime.strptime(t["date"], "%Y-%m-%d")
        )
        data["historical"] = sorted_hist_price
        return data
    else:
        data = response.json()["error"]
        raise ValueError(data)


def get_live_quote(ls_tickers: list, short: bool = True) -> dict:
    """get live quote for list of"""

    quote_type = "quote-short" if short else "quote"
    error_msg = ""
    tickers = ",".join(ls_tickers).upper()
    url = FMP_ROOT_URL + f"v3/{quote_type}/{tickers}"
    params = {"apikey": FMP_KEY}

    current_time = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    response = requests.get(url=url, params=params)
    print(datetime.now())
    if response.status_code == 200:
        data = response.json()
        if not (data):
            raise ValueError("No result found for tickers " + str(tickers) + error_msg)
        # add time stemp in the list
        data = {"time": current_time, "quotes": data}
        return data
    elif response.status_code == 403:
        raise AuthenticationError("Incorrect Key")
    else:
        data = response.json()["error"]
        raise ValueError(data)


def get_live_quote(ls_tickers: list, short: bool = True) -> dict:
    """get live quote for list of"""

    quote_type = "quote-short" if short else "quote"
    error_msg = ""
    tickers = ",".join(ls_tickers).upper()
    url = FMP_ROOT_URL + f"v3/{quote_type}/{tickers}"
    params = {"apikey": FMP_KEY}

    current_time = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    response = requests.get(url=url, params=params)
    print(datetime.now())
    if response.status_code == 200:
        data = response.json()
        if not (data):
            raise ValueError("No result found for tickers " + str(tickers) + error_msg)
        # add time stemp in the list
        data = {"time": current_time, "quotes": data}
        return data
    elif response.status_code == 403:
        raise AuthenticationError("Incorrect Key")
    else:
        data = response.json()["error"]
        raise ValueError(data)


def get_intra_day_historical_price(ticker: str, interval: str):
    """get intraday qutote"""

    # validate interval
    if not (interval.lower() in ["1min", "5min", "15min", "30min", "1hour", "4hour"]):
        raise ValueError("Invalid interval: " + interval)
    error_msg = ""
    url = FMP_ROOT_URL + f"v3/historical-chart/{interval}/{ticker}"
    params = {"apikey": FMP_KEY}
    response = requests.get(url=url, params=params)

    if response.status_code == 200:
        data = response.json()
        if not (data):
            raise ValueError("No result found for tickers " + str(ticker) + error_msg)
        return data
    elif response.status_code == 403:
        raise AuthenticationError("Incorrect Key")
    else:
        data = response.json()["error"]
        raise ValueError(data)
    pass


if __name__ == "__main__":
    # print(get_treasury_rate("2022-07-29"))
    # print(get_stock_price("fff"))
    print(get_live_quote(["aapl", "spy"]))
    pass

# %%
data = get_stock_price("SPY","2001-01-01", "2017-12-31")

# %%
import pandas as pd
a = pd.DataFrame.from_dict(data["historical"])
a

# %%
a.to_excel("SPY", ignore_index=True)

# %%
a.to_excel("SPY", exclude_index=True)

# %%
a.to_excel("SPY", index=False)

# %%
a

# %%
type(a)

# %%
a.to_excel("SPY", index=False)

# %%
pip install openpyxl

# %%
a.to_excel("SPY", index=False)

# %%
a.to_excel("SPY.xlsx", index=False)

# %%

Ticker = "EM"
Sheet_name = "EM"
data = get_stock_price(Ticker,"2001-01-01", "2017-12-31")
df = pd.DataFrame.from_dict(data)
df.to_excel("Data.xlsx", sheet_name=Sheet_name)

# %%

Ticker = "EEM"
Sheet_name = "EM"
data = get_stock_price(Ticker,"2001-01-01", "2017-12-31")
df = pd.DataFrame.from_dict(data)
df.to_excel("Data.xlsx", sheet_name=Sheet_name)

# %%

Ticker = "USIG"
Sheet_name = "US.Corp"
data = get_stock_price(Ticker,"2001-01-01", "2017-12-31")
df = pd.DataFrame.from_dict(data)
df.to_excel("Data.xlsx", sheet_name=Sheet_name)

# %%

Ticker = "USHY"
Sheet_name = "US.HY"
data = get_stock_price(Ticker,"2001-01-01", "2017-12-31")
df = pd.DataFrame.from_dict(data)
df.to_excel("Data.xlsx", sheet_name=Sheet_name)

# %%

Ticker = "GOVT"
Sheet_name = "UST"
data = get_stock_price(Ticker,"2001-01-01", "2017-12-31")
df = pd.DataFrame.from_dict(data)
df.to_excel("Data.xlsx", sheet_name=Sheet_name)

# %%
Ticker = "GOVT"
Sheet_name = "UST"
data = get_stock_price(Ticker,"2001-01-01", "2017-12-31")
df = pd.DataFrame.from_dict(data)
df.to_excel("Data.xlsx", sheet_name=Sheet_name)

# %%
Ticker = "TIPS"
Sheet_name = "INF.L"
data = get_stock_price(Ticker,"2001-01-01", "2017-12-31")
df = pd.DataFrame.from_dict(data)
df.to_excel("Data.xlsx", sheet_name=Sheet_name)

# %%
Ticker = "IWM"
Sheet_name = "Russell2000"
data = get_stock_price(Ticker,"2001-01-01", "2017-12-31")
df = pd.DataFrame.from_dict(data)
df.to_excel("Data.xlsx", sheet_name=Sheet_name)

# %%
Ticker = "XGD"
Sheet_name = "Golds"
data = get_stock_price(Ticker,"2001-01-01", "2017-12-31")
df = pd.DataFrame.from_dict(data)
df.to_excel("Data.xlsx", sheet_name=Sheet_name)

# %%
Ticker = "GLD"
Sheet_name = "Golds"
data = get_stock_price(Ticker,"2001-01-01", "2017-12-31")
df = pd.DataFrame.from_dict(data)
df.to_excel("Data.xlsx", sheet_name=Sheet_name)

# %%
Ticker = "PSP"
Sheet_name = "PE"
data = get_stock_price(Ticker,"2001-01-01", "2017-12-31")
df = pd.DataFrame.from_dict(data)
df.to_excel("Data.xlsx", sheet_name=Sheet_name)

# %%
Ticker = "GSG"
Sheet_name = "Commodity"
data = get_stock_price(Ticker,"2001-01-01", "2017-12-31")
df = pd.DataFrame.from_dict(data)
df.to_excel("Data.xlsx", sheet_name=Sheet_name)

# %%
Ticker = "FEZ"
Sheet_name = "EStoxx50"
data = get_stock_price(Ticker,"2001-01-01", "2017-12-31")
df = pd.DataFrame.from_dict(data)
df.to_excel("Data.xlsx", sheet_name=Sheet_name)

# %%
Ticker = "EWJ"
Sheet_name = "Nikkei225"
data = get_stock_price(Ticker,"2001-01-01", "2017-12-31")
df = pd.DataFrame.from_dict(data)
df.to_excel("Data.xlsx", sheet_name=Sheet_name)

