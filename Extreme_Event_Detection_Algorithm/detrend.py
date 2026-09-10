import numpy as np
from scipy import signal
from statsmodels.tsa.seasonal import STL
import pandas as pd
'''
longterm detrend
'''
def series_linear_detrend(data_series):
    x = np.arange(len(data_series))
    mask = ~np.isnan(data_series)

    if np.sum(mask) < 2:
        return data_series

    coef = np.polyfit(x[mask], data_series[mask], 1)
    trend = np.polyval(coef, x)

    detrended = data_series - trend
    detrended[~mask] = np.nan
    return detrended

'''
seasonal detrend
'''
def mean_month_detrend(data_series):
    data_series = data_series.copy()
    length = len(data_series) // 12
    data_season_mean = np.reshape(data_series, (length, 12)).mean(axis=0)
    data_series -= np.tile(data_season_mean, length)
    return data_series

def z_score(data_series, start=0, end=30):
    '''
:param data_series: Time series.
:return: If the time series is shorter than 30 years, all available years
         are used to calculate the z-scores. If the time series is longer
         than 30 years, the monthly means and standard deviations used for
         z-score calculation are derived from the data of the specified
         30-year baseline period.
    '''
    length = len(data_series) // 12
    temp = np.reshape(data_series, (length, 12))
    if length <= 30:
        mon_mean = np.nanmean(temp, axis=0)
        mon_std = np.nanstd(temp, axis=0)
    else:
        mon_mean = np.nanmean(temp[start:end, :], axis=0)
        mon_std = np.nanstd(temp[start:end, :], axis=0)
    mon_std[mon_std == 0] = np.nan
    result = (temp - mon_mean) / mon_std
    result = result.reshape(-1)
    return result


'''
std
'''
def standard_process(data_series):
    if np.all(data_series==0):
        return data_series
    else:
        return data_series/np.std(data_series)

def sum_std_process(data_series):
    total = np.sum(data_series)
    if total == 0:
        return data_series
    return data_series / total


def process_data(data_cube, mask, detrend_type, allowed_type, process_name):
    if detrend_type is None:
        return data_cube

    process_func = allowed_type.get(detrend_type)
    if process_func is None:
        raise ValueError(f"Invalid {process_name}: {detrend_type}. Choose from {list(allowed_type.keys())}")

    mask_bool = mask != 0

    for i, j in zip(*np.where(mask_bool)):
        data_cube[i, j] = process_func(data_cube[i, j])

    return data_cube


# Determine whether to remove the long-term trend
def data_long_term_detrend(data_cube, mask, detrend_type=None):
    return process_data(data_cube, mask, detrend_type,
                        {"linear": series_linear_detrend
                         },
                        "long_term_detrend_type")


# Determine whether to remove the seasonal trend
def data_season_detrend(data_cube, mask, detrend_type=None):
    return process_data(data_cube, mask, detrend_type,
                        {"mean_month": mean_month_detrend,
                         "z_score":z_score,
                         },
                        "seasonal_detrend_type")


# Determine whether to standardize the data
def data_std(data_cube, mask, detrend_type=None):
    return process_data(data_cube, mask, detrend_type,
                        {"std": standard_process, "sum": sum_std_process},
                        "standard_type")