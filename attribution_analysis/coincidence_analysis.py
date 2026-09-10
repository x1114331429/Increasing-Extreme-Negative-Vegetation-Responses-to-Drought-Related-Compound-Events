import pickle
import numpy as np
import os
from sklearn.metrics import jaccard_score


def z_score_attribution(events_dir, var_data, time_leg, shuffle_result, alpha=0.05):
    result = {}
    nlat, nlon, nmon = var_data.shape
    for key, value in events_dir.items():
        value = np.array(value)
        valid_points = []
        for point in value:
            i, j, t = point
            lag = time_leg[i, j]
            new_t = t - lag
            if (new_t >= 0) and (new_t < nmon):
                valid_points.append([i, j, new_t])
        if len(valid_points) == 0:
            result[key] = np.nan
            continue
        valid_points = np.array(valid_points).astype(int)
        selected_values = var_data[
            valid_points[:, 0],
            valid_points[:, 1],
            valid_points[:, 2]
        ]
        points_len = np.count_nonzero(selected_values)
        obs_overlap = points_len / len(value)
        shuffle = np.array(shuffle_result[key])
        p = (np.sum(shuffle >= obs_overlap) + 1) / (len(shuffle) + 1)
        if p < alpha:
            result[key] = 1
    return result

def z_score_attribution_fire(events_dir, var_data, time_leg, shuffle_result, alpha=0.05):
    result = {}
    nlat, nlon, nmon = var_data.shape
    for key, value in events_dir.items():
        value = np.array(value)
        valid_points = []
        for point in value:
            i, j, t = point
            lag = time_leg[i, j]
            new_t = t - lag
            if (new_t >= 0) and (new_t < nmon):
                valid_points.append([i, j, new_t])
        if len(valid_points) == 0:
            result[key] = np.nan
            continue
        valid_points = np.array(valid_points).astype(int)
        selected_values = var_data[
            valid_points[:, 0],
            valid_points[:, 1],
            valid_points[:, 2]
        ]
        points_len = np.count_nonzero(selected_values)
        obs_overlap = points_len / len(value)
        shuffle = np.array(shuffle_result[key])
        p = (np.sum(shuffle <= obs_overlap) + 1) / (len(shuffle) + 1)
        if p < alpha:
            result[key] = 1
    return result

def land_use_similarity_event_result(events_dir, land_use_cube, shuffle_result, alpha=0.05):
    result = {}
    for key, value in events_dir.items():
        value = np.array(value)
        selected_matrix_t0 = land_use_cube[value[:, 0], value[:, 1], value[:, 2]]
        temp = value[:, 2] - 12
        if np.any(temp < 0):
            continue
        selected_matrix_t = land_use_cube[value[:, 0], value[:, 1], temp]
        J_obs = jaccard_score(selected_matrix_t0, selected_matrix_t,
                              average='macro', zero_division=0)
        J_shuffle = np.array(shuffle_result[key])
        p_value = (np.sum(J_shuffle <= J_obs) + 1) / (len(J_shuffle) + 1)
        significant = p_value < alpha
        if significant:
            result[key] = 1
    return result


def run_attribution(veg, scenario):
    # ===== 1. path =====
    event_path = f'../get_NEG_events/vegetation_events/{veg}/{veg}_T5_P19_z-1.5_d2_a10000.pkl'
    shuffle_base = f'./{veg}_shuffle/T5_P19/z-1.5_d2_a10000'
    output_base = f'./{veg}_attribution'
    os.makedirs(output_base, exist_ok=True)
    alpha = 0.01
    # ===== 2. load_data =====
    with open(event_path, 'rb') as f:
        event_dir = pickle.load(f)
    mask = np.load('../data/land_mask/veg_mask.npy')
    var_list = [
        'Tmp_high','Tmp_low',
        'Pre_high','Pre_low',
        'SM_high','SM_low',
        'VPD_high','VPD_low',
        'BA','Land_use'
    ]

    # ===== 3. main =====
    for var in var_list:
        print(f'{veg} | {scenario} | {var}')
        shuffle_path = f'{shuffle_base}/{var}.pkl'
        with open(shuffle_path, 'rb') as f:
            shuffle_result = pickle.load(f)
        # ===== 3.1 climate =====
        if var in ['Tmp_high', 'Tmp_low', 'Pre_high', 'Pre_low',
                   'SM_high', 'SM_low', 'VPD_high', 'VPD_low']:
            cli_matrix = np.load(f'../get_NEG_events/climate_events/{var}_event.npy')
            cli_name = var.split('_')[0]
            time_leg = np.load(f'../data/time_leg/{cli_name}_{veg}.npy')
            result = z_score_attribution(event_dir, cli_matrix, time_leg, shuffle_result, alpha=alpha)
        # ===== 3.2 Land use =====
        elif var == 'Land_use':
            cli_matrix = np.load('../data/cli_data/Land_use_01_24.npy')

            result = land_use_similarity_event_result(
                event_dir,
                cli_matrix,
                shuffle_result,
                alpha=alpha)
        # ===== 3.3 BA =====
        elif var == 'BA':
            cli_matrix = np.load('../get_NEG_events/climate_events/BA_event.npy')

            result = z_score_attribution_fire(
                event_dir,
                cli_matrix,
                np.zeros((360, 720)),
                shuffle_result,
                alpha=alpha
            )
        # ===== 4. save =====
        output_path = f'{output_base}/{var}.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(result, f)


# ===== parallel =====
vegetation_list = ['LAI', 'NDVI']
scenario_list = ['T5_P19']
for veg in vegetation_list:
    for scenario in scenario_list:
        run_attribution(veg, scenario)