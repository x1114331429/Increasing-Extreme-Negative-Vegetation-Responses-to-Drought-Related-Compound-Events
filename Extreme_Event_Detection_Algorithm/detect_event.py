import numpy as np
import matplotlib.pyplot as plt
from Extreme_Event_Detection_Algorithm.detrend import data_season_detrend, data_long_term_detrend, data_std
import scipy.ndimage as ndimage
import pickle
from tools import cell_area_latlon_grid


# step 1 preprocess
def data_linear_seasonal_detrend(data_cube, mask,
                                 if_long_trend=True, long_term_detrend='linear',
                                 if_season_trend=False, season_detrend="mean_month",
                                 if_standard=False, std_type=None):
    '''
:param data_cube: Raw input data with dimensions (lat, lon, time).
                  For example, 20 years of global data at 0.5° resolution
                  would have dimensions of (360, 720, 20 * 12).
:param mask: Spatial mask for the region to be processed. Pixels to be
             processed are assigned a value of 1, while pixels to be
             excluded are assigned a value of 0.
:param if_long_trend: Whether to remove the long-term trend.
                      True indicates that the long-term trend will be removed.
:param long_term_detrend: Method used to remove the long-term trend.
                          By default, linear regression is used.
:param if_season_trend: Whether to remove the seasonal trend.
:param season_detrend: Method used to remove the seasonal component.
                       Options include subtracting the monthly mean or
                       calculating z-scores. See the comments in the
                       detrend file for details.
:param if_standard: Whether to standardize the data.
:param std_type: Standardization method. Options include dividing by the
                 standard deviation or dividing by the total sum.
:return: Processed data.
    '''
    if if_long_trend:
        data_long_term_detrend(data_cube, mask, long_term_detrend)
    if if_season_trend:
        data_season_detrend(data_cube, mask, season_detrend)
    if if_standard:
        data_std(data_cube, mask, std_type)
    return data_cube


# step 2 detect event
def mask_event_cube(data_cube, mask, method='z_score', z_score_threshold=1.5, growing_season_matrix=None,
                    flag='negative'):
    '''
:param data_cube: Detrended data, i.e., the output of
                  `data_linear_seasonal_detrend`.
:param mask: Spatial mask, as defined in the previous function.
:param method: Method used for anomaly detection. Here, `z_score` is used.
:param growing_season_matrix: Whether to restrict anomaly detection to the
                              growing season. The growing season is defined
                              as months with a mean temperature > 273.15 K.
                              If True, pass a temperature matrix with the
                              same dimensions as `data_cube`.
:return: An anomaly detection matrix with the same dimensions as
         `data_cube`. Values are 0 or 1, representing non-anomalous and
         anomalous points, respectively.
    '''
    if method == 'z_score':
        height, width, length = data_cube.shape
        event_cube = np.zeros((height, width, length)).astype(np.bool_)
        if growing_season_matrix is not None:
            valid_temp_mask = growing_season_matrix
        else:
            valid_temp_mask = np.ones((height, width, length))  # 不设置生长季
        if flag == 'negative':
            for i in range(height):
                for j in range(width):
                    if mask[i, j] != 0:
                        for k in range(length):
                            if data_cube[i, j, k] < 0 and valid_temp_mask[i, j, k] == 1:  # 如果满足 z_score <0 同时 也在生长季内
                                if data_cube[i, j, k] < (-1 * abs(z_score_threshold)):
                                    event_cube[i, j, k] = 1
        else:  # 正异常
            for i in range(height):
                for j in range(width):
                    if mask[i, j] != 0:
                        for k in range(length):
                            if data_cube[i, j, k] > 0 and valid_temp_mask[i, j, k] == 1:
                                if data_cube[i, j, k] > abs(z_score_threshold):
                                    event_cube[i, j, k] = 1
        return event_cube


# step 3 connect event
def connect_event(event_cube, save_path=None, delete_one_point=True, delete_small_event=True, structure_type=6,
                  min_duration=2, min_area=10000, data_res=0.5):
    '''
:param event_cube: The 0/1 binary matrix generated in the previous step.
:param save_path: Path where the final results will be saved.
:param delete_one_point: Whether to remove isolated points that are not
                         connected to any other points.
:param delete_small_event: Whether to impose a minimum size threshold
                           for events.
:param structure_type: Connectivity rule used for event identification.
                       Options are 6-, 18-, or 26-connectivity.
:return: The results are saved as key-value pairs in the form
         {key1: [[point1], [point2], ...], key2: [], ...}.
         Each key represents the ID of an individual event. The ID is used
         only as an identifier and has no specific meaning. Each point
         represents a location within an event in the form [lat, lon, time].
         Here, `time` denotes the time index. For example, a value of 23
         represents December of the second year.
    '''
    if structure_type == 6:
        structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    elif structure_type == 18:
        structure = np.zeros((3, 3, 3), dtype=int)
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    if abs(x - 1) + abs(y - 1) + abs(z - 1) <= 2:
                        structure[x, y, z] = 1
    elif structure_type == 26:
        structure = ndimage.generate_binary_structure(rank=3, connectivity=3)
    else:
        raise ValueError("structure_type only 6, 18, 26")
    
    labeled_array, num_features = ndimage.label(event_cube, structure=structure)
    connected_components = {region_id: [] for region_id in range(1, num_features + 1)}
    for i in range(labeled_array.shape[0]):
        for j in range(labeled_array.shape[1]):
            for k in range(labeled_array.shape[2]):
                if labeled_array[i, j, k] != 0:
                    connected_components[labeled_array[i, j, k]].append([i, j, k])
    region_area = cell_area_latlon_grid(data_res, 90, unit='km2')
    result = {}
    if delete_one_point:
        if delete_small_event:
            for key, value in connected_components.items():
                temp_space_list = set()
                temp_time_list = set()
                for i, j, k in value:
                    temp_space_list.add((i, j))
                    temp_time_list.add(k)
                area = 0
                for i,j in temp_space_list:
                    area += region_area[i,j]
                if len(temp_time_list) >= min_duration and area >= min_area:  # set minimam event
                    result[key] = value
            if save_path:
                with open(save_path, "wb") as f:
                    pickle.dump(result, f)
        else:
            for key, value in connected_components.items():
                if len(value) > 2:
                    result[key] = value
            if save_path:
                with open(save_path, "wb") as f:
                    pickle.dump(result, f)
    else:
        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump(connected_components, f)
    return result
