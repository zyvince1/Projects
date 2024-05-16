# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:38:11 2024

@author: Vince
"""

import os
from xbbg import blp
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.x13 import X13Error
from datetime import datetime
from openpyxl import load_workbook





# Set the working directory
directory = 'G:/My Drive/new career/Python'  # Update with the path to your folder
os.chdir(directory)

# User inputs
excel_file = 'ticker.xlsx'  # Path to your Excel file
sheet_name = 'chn_tsf'      # Sheet name in the Excel file
name_column = 'name'        # Column name for long company names
ticker_column = 'ticker'    # Column name for tickers
start_date = '2010-01-31'
end_date = datetime.today().strftime('%Y-%m-%d')
field = 'PX_LAST'

# Read the Excel file and the specified sheet
df_sheet = pd.read_excel(excel_file, sheet_name=sheet_name)

# Extract names and tickers
names = df_sheet[name_column].tolist()
tickers = df_sheet[ticker_column].tolist()

# Fetch historical data for tickers
df_bbg = blp.bdh(tickers=tickers, flds=field, start_date=start_date, end_date=end_date)

# Ensure the index is in datetime format
df_bbg.index = pd.to_datetime(df_bbg.index)

# Adjust the DataFrame to have the long company names as column titles
df_bbg.columns = [names[tickers.index(ticker)] for ticker in df_bbg.columns.levels[0]]


# Create a full date range from the earliest to the latest date in the data
full_date_range = pd.date_range(start=df_bbg.index.min(), end=df_bbg.index.max(), freq='M')

def preprocess_series(series):
    longest_segment = pd.Series(dtype=series.dtype)
    current_segment = pd.Series(dtype=series.dtype)

    for i in range(len(series)):
        if not pd.isna(series.iloc[i]):
            current_segment = pd.concat([current_segment, pd.Series(series.iloc[i], index=[series.index[i]])])
        else:
            if len(current_segment) > len(longest_segment):
                longest_segment = current_segment
            current_segment = pd.Series(dtype=series.dtype)

    # Check if the last segment is the longest
    if len(current_segment) > len(longest_segment):
        longest_segment = current_segment

    # Enforce the frequency to monthly
    longest_segment = longest_segment.asfreq('M')

    # Calculate the shift required to make all values positive
    shift = -longest_segment.min() + 1e-6 if longest_segment.min() <= 0 else 0
    series_shifted = longest_segment + shift
    return series_shifted, shift



# Apply X-13ARIMA-SEATS to each series and store the results in a dictionary
adjusted_series = {}
shifts = {}

for column in df_bbg.columns:
    # Preprocess the series
    series, shift = preprocess_series(df_bbg[column])
    shifts[column] = shift
    
    try:
        # Apply X-13ARIMA-SEATS with log transformation
        result = sm.tsa.x13_arima_analysis(series, freq='M', log=True)
        # Reverse the shift after seasonal adjustment
        adjusted_series[column] = result.seasadj - shift
    except X13Error as e:
        print(f"Error processing {column}: {e}")

# Combine adjusted series into a single DataFrame, ensuring the full date range is used
df_adjusted = pd.DataFrame(index=full_date_range)

for column, series in adjusted_series.items():
    # Reindex each adjusted series to the full date range
    df_adjusted[column] = series.reindex(full_date_range)

df_adjusted = df_adjusted.sort_index(ascending=False)


def write_excel(filename, sheetname, dataframe):
    with pd.ExcelWriter(filename, engine='openpyxl', mode='a') as writer:
        workBook = writer.book
        try:
            workBook.remove(workBook[sheetname])
        except KeyError:
            print("Worksheet does not exist")
        finally:
            dataframe.to_excel(writer, sheet_name=sheetname, index=True)

write_excel('china.xlsx', 'chn_tsf', df_adjusted)