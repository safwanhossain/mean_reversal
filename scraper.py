import numpy as np
import pandas as pd
import yfinance as yf

import time
from datetime import datetime
from matplotlib import pyplot as plt

# define the ticker symbol - wish there was a better way, but
# for now, this will do
VIX = "^VIX"

# Define all random helper functions here - should move to a 
# seperate file if/when this gets big enough
def _is_valid_interval(interval):
    interval_vals = [
        '1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo'
    ]
    assert interval in interval_vals

def _is_valid_field(field):
    field_vals = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    assert field in field_vals

def _is_valid_period(period):
    period_vals = ['1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max']
    assert period in period_vals

def _timestamp_to_utc(timestamp):
    ts_og = time.mktime(timestamp.timetuple())
    dt = datetime.fromtimestamp(ts_og)
    timestamp = dt.replace(tzinfo=timestamp.utcnow().tz).timestamp()
    return timestamp

def get_all_tickers():
    # Kinda sketchy getting this from wikipedia, but ... whatever
    payload=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    main_table = payload[0]
    # this dataset also includes nice things like sector and stuff
    # but for now, lets forget it
    all_tickers = main_table['Symbol'].values.tolist()
    # Add some indices to it
    indices = ['^VIX', 'SPY', '^DJIA']
    all_tickers.extend(indices)
    return all_tickers

def plot(ticker_name, data, period='5d', interval='1h'):
    # Data should be a 2d numpy array with the first dim
    # being utc time stamp
    assert len(data.shape) == 2 and data.shape[1] == 2
    y_vals = data[:,1]
    x_vals = [i for i in range(len(y_vals))]
    plt.plot(x_vals, y_vals, linestyle='--', marker='o')
    plt.title(f"{ticker_name} over {period} in {interval} increments")
    plt.xlabel("Normalized time")
    plt.ylabel("Quote value")
    plt.show()

########################### End Helper function #########################

def get_ticker_data(ticker, period='5d', interval='1h', field='Open'):
    """ Get data for that ticker, and return the field value at each 
    at each interval. Note we only care about the last 5 days
    
    YFinance gives us access to open, high, low, close on every
    interval. These are the field values

    Return type should be a (n x 2) array. Column 1 represents
    the time (in UTC format)
    """
    # Validate the input types
    _is_valid_field(field)
    _is_valid_interval(interval)
    
    # Download the data
    vix_data = yf.download(tickers=ticker, period='5d', interval=interval)
    
    # Field dict has pd timestamps, which we ideally convert to UTC 
    field_dict = vix_data[field].to_dict()
    np_out = np.ones((len(field_dict), 2), dtype=np.float32)
    for i, key in enumerate(field_dict.keys()):
        utc_time = _timestamp_to_utc(key)
        np_out[i,:] = [utc_time, field_dict[key]]
    return np_out

if __name__ == "__main__":
    all_tickers = get_all_tickers()
    curr_ticker = '^VIX'
    assert curr_ticker in all_tickers
    data = get_ticker_data(curr_ticker)
    print(data)
    plot(curr_ticker, data)
