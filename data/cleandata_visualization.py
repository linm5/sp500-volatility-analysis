import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = "../BDA_project/s&p_500_10_year_raw.csv"
raw_data = pd.read_csv(file_path, header=None)

print(raw_data.head(20))

# Extract relevant rows (assuming data starts at row 18 based on earlier analysis)
data_cleaned = raw_data.iloc[17:, :].copy()

# Assign meaningful column names
data_cleaned.columns = ["Exchange_Date", "Close", "High", "Low", "Open", "Column6", "Column7", "Column8"]

# Keep only relevant columns
data_cleaned = data_cleaned[["Exchange_Date", "Close", "High", "Low", "Open"]]

# Convert columns to appropriate data types
data_cleaned["Exchange_Date"] = pd.to_datetime(data_cleaned["Exchange_Date"], errors="coerce")
numeric_cols = ["Close", "High", "Low", "Open"]
data_cleaned[numeric_cols] = data_cleaned[numeric_cols].apply(pd.to_numeric, errors="coerce")

# Drop rows with missing values
data_cleaned = data_cleaned.dropna()

# Calculate daily log returns for ARCH modeling
data_cleaned = data_cleaned.sort_values(by="Exchange_Date")
data_cleaned["Log_Returns"] = np.log(data_cleaned["Close"] / data_cleaned["Close"].shift(1))

# Drop the first row with NA Daily_Returns
data_cleaned = data_cleaned.dropna()

# Save the cleaned data to a new CSV file
data_cleaned.to_csv("cleaned_s&p_500_data.csv", index=False)

# Visualize the Close prices and save the plot
plt.figure(figsize=(10, 6))
plt.plot(data_cleaned["Exchange_Date"], data_cleaned["Close"], label="Close Prices")
plt.title("S&P 500 Close Prices")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.savefig("../graphics/s&p500_close_prices.png")  
plt.show()

# Visualize the Daily Returns and save the plot
plt.figure(figsize=(10, 6))
plt.plot(data_cleaned["Exchange_Date"], data_cleaned["Log_Returns"], label="Log Returns", color="red")
plt.title("S&P 500 Daily Returns")
plt.xlabel("Date")
plt.ylabel("Daily Log Returns")
plt.legend()
plt.savefig("../graphics/s&p500_daily_log_returns.png")  
plt.show()

print(data_cleaned.head())
