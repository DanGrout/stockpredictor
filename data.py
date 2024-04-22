"""
This file handles all the data related functions.
"""

from alpha_vantage.timeseries import TimeSeries
from timeseries import *
from config import *

def download_data(config):
    """
    Fetch data - gets daily adjusted time series in json object as data
    Gets Close price from data

    Args:
        Config Json setup.

    Returns:
        data_date, 
        data_close_price, 
        num_data_points, 
        display_date_range
    """
    ts = TimeSeries(key=gl_key)
    data, meta_data = ts.get_daily_adjusted(config["alpha_vantage"]["symbol"], outputsize=config["alpha_vantage"]["outputsize"])

    #Date
    data_date = [date for date in data.keys()]
    data_date.reverse()
    
    #Close price
    data_close_price = [float(data[date][config["alpha_vantage"]["key_adjusted_close"]]) for date in data.keys()]
    data_close_price.reverse()
    data_close_price = np.array(data_close_price)
    
    num_data_points = len(data_date)
    display_date_range = "from " + data_date[0] + " to " + data_date[num_data_points - 1]
    print("Number data points", num_data_points, display_date_range)

    return data_date, data_close_price, num_data_points, display_date_range

def prepare_data_x(x, window_size):
    # perform windowing
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0]))
    return output[:-1], output[-1]

def prepare_data_y(x, window_size):
    # # perform simple moving average
    # output = np.convolve(x, np.ones(window_size), 'valid') / window_size

    # use the next day as label
    output = x[window_size:]
    return output