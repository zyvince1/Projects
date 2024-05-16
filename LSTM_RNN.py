# -*- coding: utf-8 -*-
"""
Created on Mon May 13 09:41:38 2024

@author: Vince
"""


### LSTM neural network model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
import yfinance as yf
from pandas_datareader import data as pdr

# Define a class for the Stock Price Predictor
class StockPricePredictor:
    def __init__(self, ticker, start_date, train_ratio=0.8, look_back=60):
        self.ticker = ticker
        self.start_date = start_date
        self.train_ratio = train_ratio
        self.look_back = look_back
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.train_data = None
        self.test_data = None

    def fetch_data(self):
        yf.pdr_override()
        self.data = pdr.get_data_yahoo(self.ticker, start=self.start_date)
        print(self.data.tail(), self.data.shape)

    def plot_close_price_history(self):
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(16,8))
        plt.title('Close Price History')
        plt.plot(self.data['Close'])
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.show()

    def preprocess_data(self):
        close_data = self.data['Close'].values.reshape(-1,1)
        scaled_data = self.scaler.fit_transform(close_data)
        train_size = int(len(scaled_data) * self.train_ratio)
        self.train_data = scaled_data[:train_size]
        self.test_data = scaled_data[train_size - self.look_back:]

    def build_model(self):
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.look_back, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        self.model.compile(optimizer='adamax', loss='mean_squared_error')

    def train_model(self):
        x_train = [self.train_data[i-self.look_back:i] for i in range(self.look_back, len(self.train_data))]
        y_train = self.train_data[self.look_back:, 0]
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = x_train.reshape(x_train.shape[0], self.look_back, 1)
        self.model.fit(x_train, y_train, epochs=1, batch_size=1)

    def make_predictions(self):
        x_test = []
        for i in range(self.look_back, len(self.test_data)):
            x_test.append(self.test_data[i - self.look_back:i, 0])
        
        x_test = np.array(x_test)
        x_test = x_test.reshape(x_test.shape[0], self.look_back, 1)
        predictions = self.model.predict(x_test)
        return self.scaler.inverse_transform(predictions)

    def evaluate_model(self, predictions):
        actual_prices = self.data['Close'].values[len(self.data) - len(predictions):]
        rmse = np.sqrt(mean_squared_error(predictions, actual_prices))
        print(f"Root Mean Squared Error: {rmse:.2f}")
        return actual_prices, predictions

    def plot_predictions(self, actual, predictions):
        plt.figure(figsize=(16,8))
        plt.title(f'Model Performance for {self.ticker}')
        plt.xlabel('Observation', fontsize=16)
        plt.ylabel('Close Price', fontsize=16)
        plt.plot(actual)
        plt.plot(predictions)
        plt.legend(['Actual', 'Predicted'], loc='lower right')
        plt.show()

# Usage
if __name__ == "__main__":
    predictor = StockPricePredictor('NVDA', '2000-01-01')
    predictor.fetch_data()
    predictor.plot_close_price_history()
    predictor.preprocess_data()
    predictor.build_model()
    predictor.train_model()
    predictions = predictor.make_predictions()
    actual, predictions = predictor.evaluate_model(predictions)
    predictor.plot_predictions(actual, predictions)
