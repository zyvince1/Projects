import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import yfinance as yf

class StockMarketPCA:
    def __init__(self, tickers, start_date='2010-01-01', end_date='2020-01-01'):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    def fetch_data(self):
        self.data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Close']
        print(self.data.head())

    def calculate_returns(self):
        # Calculate daily percentage return and drop NA values
        return self.data.pct_change().dropna()

    def perform_pca(self, n_components=3):
        # Calculate returns
        returns = self.calculate_returns()

        # Standardize the data
        scaler = StandardScaler()
        returns_scaled = scaler.fit_transform(returns)

        # Perform PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(returns_scaled)
        explained_variance = pca.explained_variance_ratio_

        print(f'Explained Variance: {explained_variance}')
        return principal_components, explained_variance

    def plot_pca_results(self, principal_components):
        plt.figure(figsize=(12, 6))
        for i in range(principal_components.shape[1]):
            plt.plot(principal_components[:, i], label=f'PC{i+1}')
        plt.title('Principal Components of Stock Returns')
        plt.xlabel('Time')
        plt.ylabel('Principal Component Values')
        plt.legend()
        plt.show()

# Usage
if __name__ == "__main__":
    # Define a list of tickers
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA']
    pca_analysis = StockMarketPCA(tickers)
    pca_analysis.fetch_data()
    principal_components, explained_variance = pca_analysis.perform_pca(n_components=3)
    pca_analysis.plot_pca_results(principal_components)
