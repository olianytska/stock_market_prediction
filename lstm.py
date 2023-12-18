import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import matplotlib.pyplot as plt

class LSTMPredictor:
    def __init__(self, ticker, start_date, end_date):
        # Initialize the predictor with the stock ticker and date range
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        # Initialize a Sequential model
        self.model = Sequential()

    def load_data(self):
        # Download the stock data using yfinance
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        # Reshape the closing prices data
        self.data = data['Close'].values.reshape(-1, 1)

    def scale_data(self):
        # Scale the data to be between 0 and 1
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.data = self.scaler.fit_transform(self.data)

    def create_dataset(self, data, step):
        # Create a dataset where Xs is the number of passengers for a range of dates and
        # ys is the number of passengers for the next date
        Xs, ys = [], []
        for i in range(len(data) - step):
            Xs.append(data[i:(i+step), 0])
            ys.append(data[i + step, 0])
        return np.array(Xs), np.array(ys)

    def prepare_data(self, step):
        # Split the data into training and testing datasets
        train_data = self.data[:int(len(self.data)*0.8)]
        test_data = self.data[int(len(self.data)*0.8):]
        # Create the training and testing datasets
        X_train, y_train = self.create_dataset(train_data, step)
        X_test, y_test = self.create_dataset(test_data, step)
        # Reshape the datasets to fit the LSTM model
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        return X_train, y_train, X_test, y_test

    def build_model(self, input_shape):
        # Build the LSTM model
        self.model.add(LSTM(96, return_sequences=True, input_shape=input_shape))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(96, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization())
        self.model.add(Dense(1))

    def train_model(self, X_train, y_train, epochs=50, batch_size=32):
        # Compile and train the model
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict_prices(self, X_test):
        # Predict the stock prices for the test dataset
        predictions = self.model.predict(X_test)
        # Transform the predictions back to the original scale
        return self.scaler.inverse_transform(predictions)

    def plot_predictions(self, y_train, y_test, predictions):
    
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, figsize=(18, 12))
        
        # Set the window title
        fig.canvas.manager.set_window_title("LSTM Model")

        # Create a continuous timeline for x-axis
        timeline = np.arange(len(y_train) + len(y_test))

        # Plot data on first subplot
        ax1.plot(timeline[:len(y_train)], self.scaler.inverse_transform(y_train.reshape(-1,1)), color='orange', label="Training Data")
        ax1.plot(timeline[len(y_train):], self.scaler.inverse_transform(y_test.reshape(-1,1)), color='red', label='Real Stock Price')
        ax1.plot(timeline[len(y_train):], predictions, color='blue', label='Predicted Stock Price')
        ax1.set_title(f'{self.ticker} Stock Price Prediction')
        ax1.set_xlabel('Time')
        ax1.set_ylabel(f'{self.ticker} Stock Price')
        ax1.legend()

        # Add a blank line (space) between the plots
        plt.subplots_adjust(hspace=0.5)

        # Plot data on second subplot
        ax2.plot(self.scaler.inverse_transform(y_test.reshape(-1,1)), color='red', label='Real Stock Price')
        ax2.plot(predictions, color='blue', label='Predicted Stock Price')
        ax2.set_title(f'{self.ticker} Stock Price Prediction (Only Test Data)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel(f'{self.ticker} Stock Price')
        ax2.legend()

        plt.show()
