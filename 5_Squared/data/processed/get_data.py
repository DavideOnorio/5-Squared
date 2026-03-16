import yfinance as yf
import pandas as pd
import requests

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {"User-Agent": "Mozilla/5.0"}
html = requests.get(url, headers=headers).text

sp500 = pd.read_html(html)[0]
tickers = sp500["Symbol"].tolist()
tickers = [t.replace(".", "-") for t in tickers]


data = yf.download(
    tickers,
    period="18mo",
    interval="1d",
    threads=True
)

data = data[["Close", "Volume"]]

data.to_csv("sp500_close_volume_18months.csv")

print("CSV saved!")