# Forecasting Stock Prices Using Long Short-Term Memory Networks (LSTMs)

![Prediction Example]([images\predicted_next_day5.png] "Prediction Example")

## Stack:
* Python
* PyTorch
* Alphavantage.co
* Numpy
* Matplotlib
---
## Introspection:
#### What did I learn?
* Long short-term memory (LSTM)
* Data preparation: acquiring data from APIs
* Data preparation: Normalising raw data
* Data preparation: Generating training and validation datasets
* Long short-term memory (LSTM) model
* Model training
* Model evaluation
* Predicting future stock prices

## Introduction:

This project dives into building a Long Short-Term Memory (LSTM) network to forecast stock prices. It starts by emphasising the importance of historical data for training the model. Whilst using the Alphavantage API, I was able to collect extensive price data, using a 20-year daily closing history for a variety of stock options. The below example is from the stock of IBM.
### Daily Close Price
![Daily Close Price]([images\Figure_1_Daily_close_price.png] "Daily Close Price")

![Daily Close Price](https://github.com/AlphaVantageSupport/time-series-forecasting-pytorch/raw/main/static/figure01-history-price.png "Daily Close Price")

Before feeding the data into the LSTM, it undergoes preprocessing. Normalization is crucial, transforming the stock prices into a standard range. This ensures all features contribute equally during training and avoids the model getting skewed by values with vastly different scales.

Another step involves splitting the data into training and validation sets. Typically, 80% of the data is used to train the model, and the remaining 20% is used to assess its generalisation on unseen data.

### Training and Validation data
![Training and Validation data]([images\Figure_2_Training_Data.png] "Training and Validation data")

The heart of the model is the LSTM network built using the PyTorch library. LSTMs excel at learning patterns in sequential data, making them ideal for tasks like stock price prediction where past prices can influence future trends. This specific model has a three-layer architecture:
* The first layer acts as a transformation layer, preparing the input data for the LSTM.
* The LSTM layer is the core of the network. It analyses the sequences of past closing prices (usually a defined window like 20 days) and extracts temporal dependencies within the data.
* The final layer leverages the learned patterns from the LSTM layer to generate predictions for the next day's closing price.

To prevent the model from simply memorising specific data points and hindering its ability to adapt to unseen data (overfitting), a technique called Dropout is employed. Dropout randomly excludes a certain percentage of neurons during training, forcing the model to rely on a broader set of features and improve its generalisation.

### Training the model
![Training the model]([images\training.png] "Training the model")

Training the model involves feeding it historical data and comparing its predicted prices with the actual closing prices. The model continuously adjusts its internal parameters to minimize the difference between its predictions and the actual values. This optimisation process is guided by an optimiser, such as “Adam”, which controls how the model updates its weights, and a learning rate that dictates the pace of learning. A slower learning rate allows for more gradual adjustments and can help prevent the model from overshooting the optimal solution.

### Evaluation on the validation set
![Evaluation on the validation set]([images\Prediction1B.png] "Evaluation on the validation set")

Once trained, the model's performance is evaluated on the validation set. Ideally, the predicted prices on both the training and validation sets should closely resemble the actual historical closing prices. This indicates that the model has effectively learned the underlying patterns from the training data and can generalise these patterns to make predictions on unseen data points.

### Prediction example:
![Prediction Example]([images\Zoomed-prediction5a.png] "Prediction Example")

![Prediction Example]([images\Zoomed-prediction5b.png] "Prediction Example")

## Closing...
Finally, with a well-trained model, you can leverage its forecasting capabilities. By feeding it the most recent 20 days of closing prices, you can obtain predictions for the next day's closing price. However, it's important to remember that stock markets are intricate systems influenced by a multitude of factors beyond historical price data. This model should be viewed as a tool to identify potential trends, and its predictions should not be used for financial decisions without careful consideration of other relevant market information and expert advice.

### Initial model - Next day prediction:
![Prediction Example]([images\predicted_next_day3.png] "Prediction Example")

### Altered model - Next day prediction:
![Prediction Example]([images\predicted_next_day5.png] "Prediction Example")

## Links:
Connect with me on [Linkedin](https://www.linkedin.com/in/dan-grout-430543167/)

Send me a [mail](mailto:dan.grout.architect@gmail.com)