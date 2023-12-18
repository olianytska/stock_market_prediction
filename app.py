import tkinter as tk
from tkinter import ttk

from lstm import LSTMPredictor
from random_forest import RandomForestPredictor
from arima import ARIMAPredictor

# Defining the Application class
class Application(tk.Tk):
    def __init__(self, *args, **kwargs):
        # Initializing the tkinter window
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("Stock Predictor")  # Setting the title of the window
        self.geometry("800x700")  # Setting the size of the window

        # Creating a label and dropdown for ticker selection
        self.ticker_label = ttk.Label(self, text="Select Ticker:")
        self.ticker_label.pack()
        self.ticker_var = tk.StringVar()
        self.ticker_choices = tk.Entry(self, textvariable=self.ticker_var)
        self.ticker_choices.pack()

        # Creating a label and dropdown for model selection
        self.model_label = ttk.Label(self, text="Select Model:")
        self.model_label.pack()
        self.model_var = tk.StringVar()
        self.model_choices = ttk.Combobox(self, textvariable=self.model_var)
        self.model_choices['values'] = ('LSTM', 'ARIMA', 'Random Forest')
        self.model_choices.pack()

        # Creating a button to call the predict function
        self.button = ttk.Button(self, text="Predict", command=self.predict)
        self.button.pack()

    # Defining the predict function
    def predict(self):
        # Getting the selected ticker and model
        ticker_choice = self.ticker_var.get()
        model_choice = self.model_var.get()

        # If LSTM model is selected
        if model_choice == 'LSTM':
            # Initialize the LSTM model
            model = LSTMPredictor(ticker_choice, '2020-01-01', '2023-12-31')
            model.load_data()  # Load the data
            model.scale_data()  # Scale the data
            X_train, y_train, X_test, y_test = model.prepare_data(step=60)  # Prepare the data
            model.build_model(input_shape=(X_train.shape[1], 1))  # Build the model
            model.train_model(X_train, y_train)  # Train the model
            predictions = model.predict_prices(X_test)  # Make predictions
            model.plot_predictions(y_train, y_test, predictions)  # Plot the predictions

        # If ARIMA model is selected
        elif model_choice == 'ARIMA':
            # Initialize the ARIMA model
            model = ARIMAPredictor(ticker_choice, '2020-01-01', '2023-12-31')
            model.load_data()  # Load the data
            history, train, test, size = model.prepare_data()  # Prepare the data
            predictions = model.predict_prices(history, test)  # Make predictions
            model.plot_predictions(train, test, predictions, size)  # Plot the predictions

        # If Random Forest model is selected
        elif model_choice == 'Random Forest':
            # Initialize the Random Forest model
            model = RandomForestPredictor(ticker_choice, '2020-01-01', '2022-12-31', 30)
            model.fetch_data()  # Fetch the data
            model.prepare_data()  # Prepare the data
            model.train_model()  # Train the model
            predictions = model.predict()  # Make predictions
            model.plot_predictions(model.y_scaled[:-model.forecast_out+1], model.y_scaled[-model.forecast_out:], predictions)  # Plot the predictions

        else:
            print("Please select a model.")  # Print error message if no model is selected

# Running the application
if __name__ == "__main__":
    app = Application()
    app.mainloop()

