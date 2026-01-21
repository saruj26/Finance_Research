# -*- coding: utf-8 -*-
"""
S&P 500 (^GSPC) Download + Preprocessing + Save + Plots + OLS
Fixed for Google Colab + Google Drive
"""

# Imports
# -------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import yfinance as yf


# Path
PROJECT_PATH = "C:\Research\Finance_Research"
DATA_DIR = os.path.join(PROJECT_PATH, "Data\collected_data")
os.makedirs(DATA_DIR, exist_ok=True)

# Download Stock Price

ticker = "^GSPC"
start_date = "2016-01-01"
end_date = "2022-12-31"

# IMPORTANT: auto_adjust=False -> ensures "Adj Close" exists
gspc_prices = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

print(f"✅ Successfully retrieved S&P 500 data from {start_date} to {end_date}.")


# Preprocessing
gspc_prices = gspc_prices.reset_index()  # Date becomes a column
gspc_prices["formatted_date"] = pd.to_datetime(gspc_prices["Date"])

# create adjclose column (matches your old code style)
gspc_prices["adjclose"] = gspc_prices["Adj Close"]

# sort by date
gspc_prices = gspc_prices.sort_values("formatted_date").reset_index(drop=True)

# time trend
gspc_prices["time_trend"] = np.arange(len(gspc_prices))

# lag + returns
gspc_prices["adjclose_lag1"] = gspc_prices["adjclose"].shift(1)
gspc_prices["returns"] = (gspc_prices["adjclose"] - gspc_prices["adjclose_lag1"]) / gspc_prices["adjclose_lag1"]

# drop NA from lag
gspc_prices = gspc_prices.dropna().reset_index(drop=True)


# Save CSV
# -------------------------
output_file = os.path.join(DATA_DIR, "sp500_index.csv")
gspc_prices.to_csv(output_file, index=False)
print(f"✅ Saved CSV to: {output_file}")


# Visualization
# -------------------------
dates_list = gspc_prices["formatted_date"]
adjclose_list = gspc_prices["adjclose"]
returns_list = gspc_prices["returns"]

plt.figure(figsize=(8, 5))
plt.plot(dates_list, adjclose_list)
plt.axvline(dt.datetime(2019, 12, 31), color="black", linewidth=0.6, ls=":")
plt.axvline(dt.datetime(2020, 1, 30), color="red", linewidth=0.6, ls="-.")
plt.axvline(dt.datetime(2022, 2, 24), color="red", linewidth=0.6, ls="-.")
plt.xlabel("Date")
plt.ylabel("Adj Close")
plt.title("S&P 500 Adjusted Close")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(dates_list, returns_list)
plt.xlabel("Date")
plt.ylabel("Returns")
plt.title("S&P 500 Daily Returns")
plt.show()

# Histogram of returns
plt.figure(figsize=(8, 5))
plt.hist(gspc_prices["returns"], bins=50, density=True)
plt.xlabel("Returns")
plt.ylabel("Density")
plt.title("Returns Distribution")
plt.show()

# -------------------------
# OLS Regression
# -------------------------
# Formula method
model = smf.ols("adjclose ~ time_trend + adjclose_lag1", data=gspc_prices).fit()
print(model.summary())

# Numpy method (same result style)
X = gspc_prices[["time_trend", "adjclose_lag1"]].to_numpy()
X = sm.add_constant(X)
y = gspc_prices["adjclose"].to_numpy()

model2 = sm.OLS(y, X).fit()
print(model2.summary())
