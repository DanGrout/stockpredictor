"""
Train an LSTM model to predict stock price movements.
LSTMs stand for Long Short-Term Memory networks. 
They are a special kind of Recurrent Neural Network (RNN) architecture designed to address a limitation in traditional RNNs.
"""
#  ================ IMPORTS =====================
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from alpha_vantage.timeseries import TimeSeries

#Import python files:
from config import *
#Classes
from normalizer import Normalizer
from timeseries import TimeSeriesDataset
from lstmmodel import LSTMModel
#Functions
from data import *
from epoch import run_epoch

# ========================== MAIN =================================
def main():
    data_date, data_close_price, num_data_points, display_date_range = download_data(config)
    
    #============ PLOT 1 - Daily close price for Specified Stock ====================
    if logical(input("Plot 1 - Daily close price for Specified Stock? (y/n)?: ")):

        # Plot - Daily close price for Specified Stock.
        fig = figure(figsize=(25, 5), dpi=80)
        fig.patch.set_facecolor((1.0, 1.0, 1.0))
        plt.plot(data_date, data_close_price, color=config["plots"]["color_actual"])
        xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
        x = np.arange(0,len(xticks))
        plt.xticks(x, xticks, rotation='vertical')
        plt.title("Daily close price for " + config["alpha_vantage"]["symbol"] + ", " + display_date_range)
        plt.grid(visible=None, which='major', axis='y', linestyle='--')
        #plt.grid(b=None, which='major', axis='y', linestyle='--')
        plt.show()

    # Normalize the data
    scaler = Normalizer()
    normalized_data_close_price = scaler.fit_transform(data_close_price)

    # Split the dataset into two parts, for training and validation. Split the data into 80:20 - 80% of the data is used for training, with the remaining 20% for validating our model's performance in predicting future prices.
    data_x, data_x_unseen = prepare_data_x(normalized_data_close_price, window_size=config["data"]["window_size"])
    data_y = prepare_data_y(normalized_data_close_price, window_size=config["data"]["window_size"])

    # split dataset

    split_index = int(data_y.shape[0]*config["data"]["train_split_size"])
    data_x_train = data_x[:split_index]
    data_x_val = data_x[split_index:]
    data_y_train = data_y[:split_index]
    data_y_val = data_y[split_index:]
    
    #============ PLOT 2 - Daily close price for Specified Stock - Training and Validation Data. ====================
    if logical(input("Plot 2 - Daily close price for Specified Stock - Training and Validation Data? (y/n)?: ")):

        # Plot - Daily close price for Specified Stock - Training and Validation Data.
        # prepare data for plotting

        to_plot_data_y_train = np.zeros(num_data_points)
        to_plot_data_y_val = np.zeros(num_data_points)

        to_plot_data_y_train[config["data"]["window_size"]:split_index+config["data"]["window_size"]] = scaler.inverse_transform(data_y_train)
        to_plot_data_y_val[split_index+config["data"]["window_size"]:] = scaler.inverse_transform(data_y_val)

        to_plot_data_y_train = np.where(to_plot_data_y_train == 0, None, to_plot_data_y_train)
        to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)

        ## plots

        fig = figure(figsize=(25, 5), dpi=80)
        fig.patch.set_facecolor((1.0, 1.0, 1.0))
        plt.plot(data_date, to_plot_data_y_train, label="Prices (train)", color=config["plots"]["color_train"])
        plt.plot(data_date, to_plot_data_y_val, label="Prices (validation)", color=config["plots"]["color_val"])
        xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
        x = np.arange(0,len(xticks))
        plt.xticks(x, xticks, rotation='vertical')
        plt.title("Daily close prices for " + config["alpha_vantage"]["symbol"] + " - showing training and validation data")
        plt.grid(visible=None, which='major', axis='y', linestyle='--')
        plt.legend()
        plt.show()

    dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
    dataset_val = TimeSeriesDataset(data_x_val, data_y_val)
    print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
    print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)

    train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=True)

    model = LSTMModel(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]["lstm_size"], num_layers=config["model"]["num_lstm_layers"], output_size=1, dropout=config["model"]["dropout"])
    model = model.to(config["training"]["device"])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)

    print("Training commencement, please be patient...")
    for epoch in range(config["training"]["num_epoch"]):
        loss_train, lr_train = run_epoch(train_dataloader, model, optimizer, criterion, scheduler, is_training=True)
        loss_val, lr_val = run_epoch(val_dataloader, model, optimizer, criterion, scheduler)
        scheduler.step()
        
        print('Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}'
            .format(epoch+1, config["training"]["num_epoch"], loss_train, loss_val, lr_train))

    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
    train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=False)
    val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=False)

    model.eval()

    # predict on the training data, to see how well the model managed to learn and memorize
    predicted_train = np.array([])

    for idx, (x, y) in enumerate(train_dataloader):
        x = x.to(config["training"]["device"])
        out = model(x)
        out = out.cpu().detach().numpy()
        predicted_train = np.concatenate((predicted_train, out))
    
    # predict on the validation data, to see how the model does:
    predicted_val = np.array([])
    for idx, (x, y) in enumerate(val_dataloader):
        x = x.to(config["training"]["device"])
        out = model(x)
        out = out.cpu().detach().numpy()
        predicted_val = np.concatenate((predicted_val, out))
    
   
    #============ PLOT 3 - Compare predicted prices to actual prices ====================
    if logical(input("Plot 3 - Compare predicted prices to actual prices? (y/n)?: ")):
        # prepare data for plotting

        to_plot_data_y_train_pred = np.zeros(num_data_points)
        to_plot_data_y_val_pred = np.zeros(num_data_points)

        to_plot_data_y_train_pred[config["data"]["window_size"]:split_index+config["data"]["window_size"]] = scaler.inverse_transform(predicted_train)
        to_plot_data_y_val_pred[split_index+config["data"]["window_size"]:] = scaler.inverse_transform(predicted_val)

        to_plot_data_y_train_pred = np.where(to_plot_data_y_train_pred == 0, None, to_plot_data_y_train_pred)
        to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)

        # plots - Compare predicted prices to actual prices

        fig = figure(figsize=(25, 5), dpi=80)
        fig.patch.set_facecolor((1.0, 1.0, 1.0))
        plt.plot(data_date, data_close_price, label="Actual prices", color=config["plots"]["color_actual"])
        plt.plot(data_date, to_plot_data_y_train_pred, label="Predicted prices (train)", color=config["plots"]["color_pred_train"])
        plt.plot(data_date, to_plot_data_y_val_pred, label="Predicted prices (validation)", color=config["plots"]["color_pred_val"])
        plt.title("Compare predicted prices to actual prices")
        xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
        x = np.arange(0,len(xticks))
        plt.xticks(x, xticks, rotation='vertical')
        plt.grid(visible=None, which='major', axis='y', linestyle='--')
        plt.legend()
        plt.show()

    #============ PLOT 4 - Zoom in to examine predicted price on validation data portion ====================
    if logical(input("Plot 4 - Zoom in to examine predicted price on validation data portion? (y/n)?: ")):
        to_plot_data_y_val_subset = scaler.inverse_transform(data_y_val)
        to_plot_predicted_val = scaler.inverse_transform(predicted_val)
        to_plot_data_date = data_date[split_index+config["data"]["window_size"]:]

        # plots - Zoom in to examine predicted price on validation data portion

        fig = figure(figsize=(25, 5), dpi=80)
        fig.patch.set_facecolor((1.0, 1.0, 1.0))
        plt.plot(to_plot_data_date, to_plot_data_y_val_subset, label="Actual prices", color=config["plots"]["color_actual"])
        plt.plot(to_plot_data_date, to_plot_predicted_val, label="Predicted prices (validation)", color=config["plots"]["color_pred_val"])
        plt.title("Zoom in to examine predicted price on validation data portion")
        xticks = [to_plot_data_date[i] if ((i%int(config["plots"]["xticks_interval"]/5)==0 and (len(to_plot_data_date)-i) > config["plots"]["xticks_interval"]/6) or i==len(to_plot_data_date)-1) else None for i in range(len(to_plot_data_date))] # make x ticks nice
        xs = np.arange(0,len(xticks))
        plt.xticks(xs, xticks, rotation='vertical')
        plt.grid(visible=None, which='major', axis='y', linestyle='--')
        plt.legend()
        plt.show()
  

    #============= Predicting future stock prices ================:
    model.eval()

    x = torch.tensor(data_x_unseen).float().to(config["training"]["device"]).unsqueeze(0).unsqueeze(2) # this is the data type and shape required, [batch, sequence, feature]
    prediction = model(x)
    prediction = prediction.cpu().detach().numpy()

    #============ PLOT 5 - Predicted close price of the next trading day ====================
    if logical(input("Plot 5 - Predicted close price of the next trading day? (y/n)?: ")):
        # prepare plots
        plot_range = 10
        to_plot_data_y_val = np.zeros(plot_range)
        to_plot_data_y_val_pred = np.zeros(plot_range)
        to_plot_data_y_test_pred = np.zeros(plot_range)

        to_plot_data_y_val[:plot_range-1] = scaler.inverse_transform(data_y_val)[-plot_range+1:]
        to_plot_data_y_val_pred[:plot_range-1] = scaler.inverse_transform(predicted_val)[-plot_range+1:]

        to_plot_data_y_test_pred[plot_range-1] = scaler.inverse_transform(prediction)

        to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)
        to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)
        to_plot_data_y_test_pred = np.where(to_plot_data_y_test_pred == 0, None, to_plot_data_y_test_pred)

        # Plot - Predicted close price of the next trading day

        plot_date_test = data_date[-plot_range+1:]
        plot_date_test.append("tomorrow")

        fig = figure(figsize=(25, 5), dpi=80)
        fig.patch.set_facecolor((1.0, 1.0, 1.0))
        plt.plot(plot_date_test, to_plot_data_y_val, label="Actual prices", marker=".", markersize=10, color=config["plots"]["color_actual"])
        plt.plot(plot_date_test, to_plot_data_y_val_pred, label="Past predicted prices", marker=".", markersize=10, color=config["plots"]["color_pred_val"])
        plt.plot(plot_date_test, to_plot_data_y_test_pred, label="Predicted price for next day", marker=".", markersize=20, color=config["plots"]["color_pred_test"])
        plt.title("Predicted close price of the next trading day")
        plt.grid(visible=None, which='major', axis='y', linestyle='--')
        plt.legend()
        plt.show()

        print("Predicted close price of the next trading day:", round(to_plot_data_y_test_pred[plot_range-1], 2))
    
    print("END.")
# ========================== ^^^ MAIN ^^^ =================================

if __name__ == "__main__":
    main()