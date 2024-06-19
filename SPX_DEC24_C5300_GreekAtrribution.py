
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:24:01 2024

@author: Vince
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set the working directory
directory = 'G:/My Drive/new career/Python'  # Update with the path to your folder
os.chdir(directory)

# User inputs
excel_file = 'SPX.xlsx'  # Path to your Excel file
sheet_name = 'data'      # Sheet name in the Excel file

# Read the Excel file and the specified sheet
df = pd.read_excel(excel_file, sheet_name=sheet_name)
df['Dates'] = pd.to_datetime(df['Dates'])
df = df.set_index('Dates')

# Resample the data to weekly frequency
df_weekly = df.resample('W').last()

# Display the first few rows of the weekly dataframe to understand its structure
print(df_weekly.head())

# Helper functions
def calculate_delta_fd(option_prices, spx_prices):
    return (option_prices.shift(-1) - option_prices) / (spx_prices.shift(-1) - spx_prices)

def calculate_gamma_fd(delta_values, spx_prices):
    return (delta_values.shift(-1) - delta_values) / (spx_prices.shift(-1) - spx_prices)

def calculate_theta_fd(option_prices):
    return (option_prices.shift(-1) - option_prices) / (1 / 52)  # Weekly decay

def calculate_d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def calculate_d2(d1, sigma, T):
    return d1 - sigma * np.sqrt(T)

def calculate_vega(S, d1, T):
    return S * norm.pdf(d1) * np.sqrt(T)

def calculate_theta(S, d1, d2, K, T, r, sigma):
    return - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)

def plot_series(df, columns, title, ylabel):
    plt.figure(figsize=(12, 8))
    for col in columns:
        plt.plot(df.index, df[col], label=col)
    plt.xlabel('Dates')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

#%%
# Calculate Delta for Dec24 5300 Call Option using the Finite Difference Method
df_weekly['Delta_Dec24_5300'] = calculate_delta_fd(df_weekly['SPX_DEC24_C5300'], df_weekly['SPX'])

# Plot the Delta values
plot_series(df_weekly, ['Delta_Dec24_5300'], 'Delta Time Series for SPX Dec24 5300 Call Option', 'Delta')

# Calculate Gamma for Dec24 5300 Call Option using the Finite Difference Method
df_weekly['Gamma_Dec24_5300'] = calculate_gamma_fd(df_weekly['Delta_Dec24_5300'], df_weekly['SPX'])

# Plot the Gamma values
plot_series(df_weekly, ['Gamma_Dec24_5300'], 'Gamma Time Series for SPX Dec24 5300 Call Option', 'Gamma')

# Calculate Theta for Dec24 5300 Call Option using the Finite Difference Method
df_weekly['Theta_Dec24_5300'] = calculate_theta_fd(df_weekly['SPX_DEC24_C5300'])

# Plot the Theta values
plot_series(df_weekly, ['Theta_Dec24_5300'], 'Theta Time Series for SPX Dec24 5300 Call Option', 'Theta')

#%%
# Calculate the historical volatility (annualized)
df_weekly['log_return'] = np.log(df_weekly['SPX'] / df_weekly['SPX'].shift(1))
historical_volatility = df_weekly['log_return'].std() * np.sqrt(52)  # Adjust for weekly data

# Parameters for the Black-Scholes model
r = 0.045  # risk-free interest rate
sigma = historical_volatility  # assumed volatility
T_Dec24 = (pd.to_datetime('2024-12-20') - df_weekly.index).days / 365.0
K_5300 = 5300

# Calculate d1 and d2 for Dec24 5300 Call Option
df_weekly['d1_Dec24_5300'] = calculate_d1(df_weekly['SPX'], K_5300, T_Dec24, r, sigma)
df_weekly['d2_Dec24_5300'] = calculate_d2(df_weekly['d1_Dec24_5300'], sigma, T_Dec24)

# Calculate Delta using the analytical formula (N(d1))
df_weekly['Delta_Dec24_5300_analytical'] = norm.cdf(df_weekly['d1_Dec24_5300'])

# Calculate Gamma using the analytical formula
df_weekly['Gamma_Dec24_5300_analytical'] = norm.pdf(df_weekly['d1_Dec24_5300']) / (df_weekly['SPX'] * sigma * np.sqrt(T_Dec24))

# Plot the Delta values calculated using the analytical formula
plot_series(df_weekly, ['Delta_Dec24_5300_analytical'], 'Analytical Delta Time Series for SPX Dec24 5300 Call Option', 'Delta')

# Plot the Gamma values calculated using the analytical formula
plot_series(df_weekly, ['Gamma_Dec24_5300_analytical'], 'Analytical Gamma Time Series for SPX Dec24 5300 Call Option', 'Gamma')

# Plot comparison of Delta from the two methods
plot_series(df_weekly, ['Delta_Dec24_5300', 'Delta_Dec24_5300_analytical'], 'Comparison of Delta Methods for SPX Dec24 5300 Call Option', 'Delta')

# Plot comparison of Gamma from the two methods
plot_series(df_weekly, ['Gamma_Dec24_5300', 'Gamma_Dec24_5300_analytical'], 'Comparison of Gamma Methods for SPX Dec24 5300 Call Option', 'Gamma')

# Calculate Vega and Theta using the analytical formula
df_weekly['Vega_Dec24_5300_analytical'] = calculate_vega(df_weekly['SPX'], df_weekly['d1_Dec24_5300'], T_Dec24)
df_weekly['Theta_Dec24_5300_analytical'] = calculate_theta(df_weekly['SPX'], df_weekly['d1_Dec24_5300'], df_weekly['d2_Dec24_5300'], K_5300, T_Dec24, r, sigma)

# Plot the Vega values calculated using the analytical formula
plot_series(df_weekly, ['Vega_Dec24_5300_analytical'], 'Analytical Vega Time Series for SPX Dec24 5300 Call Option', 'Vega')

# Plot the Theta values calculated using the analytical formula
plot_series(df_weekly, ['Theta_Dec24_5300_analytical'], 'Analytical Theta Time Series for SPX Dec24 5300 Call Option', 'Theta')

#%%
# Calculate weekly returns and differences
df_weekly['SPX_return'] = df_weekly['SPX'].pct_change()
df_weekly['Delta_S'] = df_weekly['SPX'].diff()

# Calculate Delta return using analytical Greeks
df_weekly['Delta_Return_Dec24_5300'] = df_weekly['Delta_Dec24_5300_analytical'] * df_weekly['SPX_return']

# Calculate Gamma return using analytical Greeks
df_weekly['Gamma_Return_Dec24_5300'] = 0.5 * df_weekly['Gamma_Dec24_5300_analytical'] * (df_weekly['SPX_return'] ** 2)

# Calculate Theta return using analytical Greeks (weekly decay)
df_weekly['Theta_Return_Dec24_5300'] = df_weekly['Theta_Dec24_5300_analytical'] * (1/52) / df_weekly['SPX']

# Sum return contributions using analytical Greeks
df_weekly['Total_Return_Dec24_5300'] = df_weekly['Delta_Return_Dec24_5300'] + df_weekly['Gamma_Return_Dec24_5300'] + df_weekly['Theta_Return_Dec24_5300']

# Display the P/L attribution dataframe using analytical Greeks
pl_attribution_Dec24_5300 = df_weekly[['Total_Return_Dec24_5300', 'Delta_Return_Dec24_5300', 'Gamma_Return_Dec24_5300', 'Theta_Return_Dec24_5300']].dropna()
print("P/L Attribution for Dec24 5300 Call Option using Analytical Method:")
print(pl_attribution_Dec24_5300)

# Calculate cumulative returns using analytical Greeks
df_weekly['Cumulative_Total_Return_Dec24_5300'] = (1 + df_weekly['Total_Return_Dec24_5300']).cumprod() - 1
df_weekly['Cumulative_Delta_Return_Dec24_5300'] = (1 + df_weekly['Delta_Return_Dec24_5300']).cumprod() - 1
df_weekly['Cumulative_Gamma_Return_Dec24_5300'] = (1 + df_weekly['Gamma_Return_Dec24_5300']).cumprod() - 1
df_weekly['Cumulative_Theta_Return_Dec24_5300'] = (1 + df_weekly['Theta_Return_Dec24_5300']).cumprod() - 1

# Plot cumulative returns for Dec24 5300 Call Option using analytical Greeks
plt.figure(figsize=(12, 8))
plt.plot(df_weekly.index, df_weekly['Cumulative_Total_Return_Dec24_5300'], label='Total P/L')
plt.plot(df_weekly.index, df_weekly['Cumulative_Delta_Return_Dec24_5300'], label='Delta P/L')
plt.plot(df_weekly.index, df_weekly['Cumulative_Gamma_Return_Dec24_5300'], label='Gamma P/L')
plt.plot(df_weekly.index, df_weekly['Cumulative_Theta_Return_Dec24_5300'], label='Theta P/L')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Cumulative Return for Dec24 5300 Call Option (Analytical)')
plt.legend()
plt.grid(True)
plt.show()

