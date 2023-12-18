import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler


class RandomForestPredictor:
    def __init__(self, ticker, start_date, end_date, forecast_out):
        # Initialize the predictor with the stock ticker, date range, and forecast out days
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.forecast_out = forecast_out
        self.data = None
        self.X_scaled = None
        self.y_scaled = None
        self.scaler = MinMaxScaler()
        self.rf = RandomForestRegressor(n_estimators=100, random_state=0)

    def fetch_data(self):
        # Download the stock data using yfinance
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        # Calculate high-low percentage and percentage change
        self.data['High_Low_Pct'] = (self.data['High'] - self.data['Low']) / self.data['Close']
        self.data['Pct_Change'] = (self.data['Close'] - self.data['Open']) / self.data['Open']
        # Select relevant data columns
        self.data = self.data[['High_Low_Pct', 'Pct_Change', 'Volume', 'Close']]
        # Shift the data for forecasting
        self.data['Prediction'] = self.data[['Close']].shift(-self.forecast_out)

    def prepare_data(self):
        # Prepare the data for training and testing
        X = self.data.drop('Prediction', axis=1)[:-self.forecast_out]
        y = self.data['Prediction'][:-self.forecast_out]
        # Scale the data
        self.X_scaled = self.scaler.fit_transform(X)
        self.y_scaled = self.scaler.fit_transform(y.values.reshape(-1,1))

    def train_model(self):
        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(self.X_scaled, self.y_scaled, test_size=0.2)
        # Train the model
        self.rf.fit(x_train, y_train)
        # Print the confidence score
        rf_confidence = self.rf.score(x_test, y_test)
        print("rf confidence: ", rf_confidence)
        return 

    def predict(self):
        # Predict the stock prices for the forecast out days
        x_forecast = self.X_scaled[-self.forecast_out:]
        return self.rf.predict(x_forecast)

    def plot_predictions(self, train_data, test_data, predictions):

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, figsize=(18, 12))
        
        # Set the window title
        fig.canvas.manager.set_window_title("Random Forest Regressor")

        # Create a continuous timeline for x-axis
        timeline = np.arange(len(train_data) + len(test_data))

        # Plot data on first subplot
        ax1.plot(timeline[:len(train_data)], self.scaler.inverse_transform(train_data.reshape(-1, 1)), color='orange', label="Training Data")
        ax1.plot(timeline[len(train_data):], self.scaler.inverse_transform(test_data.reshape(-1, 1)), color='red', label='Real Stock Price')
        ax1.plot(timeline[len(train_data):], self.scaler.inverse_transform(predictions.reshape(-1, 1)), color='blue', label='Predicted Stock Price')
        ax1.set_title(f'{self.ticker} Stock Price Prediction')
        ax1.set_xlabel('Time')
        ax1.set_ylabel(f'{self.ticker} Stock Price')
        ax1.legend()

        # Add a blank line (space) between the plots
        plt.subplots_adjust(hspace=0.5)

        # Plot data on second subplot
        ax2.plot(timeline[len(train_data):], self.scaler.inverse_transform(test_data.reshape(-1, 1)), color='red', label='Real Stock Price')
        ax2.plot(timeline[len(train_data):], self.scaler.inverse_transform(predictions.reshape(-1, 1)), color='blue', label='Predicted Stock Price')
        ax2.set_title(f'{self.ticker} Stock Price Prediction')
        ax2.set_xlabel('Time')
        ax2.set_ylabel(f'{self.ticker} Stock Price')
        ax2.legend()

        plt.show()


