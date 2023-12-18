import pandas as pd
import yfinance as yf

import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler

# Defining a class for ARIMA prediction
class ARIMAPredictor:
    # Initializing the class with ticker symbol, start and end dates
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.model = None

    # Method to load data from Yahoo Finance
    def load_data(self):
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        # Data processing
        data = data.copy()
        data = data.reset_index()
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        self.data = data['Close'].to_frame()

    # Method to prepare data by splitting it into training and testing sets
    def prepare_data(self):
        X = self.data.values
        size = int(len(X) * 0.8)
        train, test = X[0:size], X[size:len(X)]
        history = [x for x in train]
        return history, train, test, size

    # Method to create an ARIMA model
    def arima_model(self, history):
        self.model = ARIMA(history, order=(2,1,2))
        model_fit = self.model.fit()
        predictions = model_fit.forecast()
        return predictions[0]

    # Method to predict prices using the ARIMA model
    def predict_prices(self, history, test):
        predictions = list()
        for t in range(len(test)):
            yhat = self.arima_model(history)
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
        return predictions

    # Method to plot the real and predicted prices
    def plot_predictions(self, train, test, predictions, size):

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, figsize=(18, 12))

        # Set the window title
        fig.canvas.manager.set_window_title("ARIMA Model")

        # Plot data on first subplot
        ax1.plot(self.data.iloc[:size,:].index, train, color='orange', label="Training Data")
        ax1.plot(self.data.iloc[size:,:].index, test, color='red', label='Real Stock Price')
        ax1.plot(self.data.iloc[size:,:].index, predictions, color='blue', label='Predicted Stock Price')
        ax1.set_title(f'{self.ticker} Stock Price Prediction')
        ax1.set_xlabel('Time')
        ax1.set_ylabel(f'{self.ticker} Stock Price')
        ax1.legend()

        # Add a blank line (space) between the plots
        plt.subplots_adjust(hspace=0.5)

        # Plot data on second subplot
        ax2.plot(self.data.iloc[size:,:].index, test, color='red', label='Real Stock Price')
        ax2.plot(self.data.iloc[size:,:].index, predictions, color='blue', label='Predicted Stock Price')
        ax2.set_title(f'{self.ticker} Stock Price Prediction (Only Test Data)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel(f'{self.ticker} Stock Price')
        ax2.legend()

        plt.show()

