import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
import os
from multiprocessing import Process, Pool

def z_score_attribution(events_dir, var_data, time_leg):
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
        result[key] = points_len / len(valid_points)
    return result


def land_use_similarity_event_result(events_dir, land_use_cube, time_leg):
    result = {}
    for key, value in events_dir.items():
        value = np.array(value)
        selected_matrix_t0 = land_use_cube[value[:, 0], value[:, 1], value[:, 2]]
        temp = value[:, 2] - 12
        if np.any(temp < 0):
            continue
        selected_matrix_t = land_use_cube[value[:, 0], value[:, 1], temp]
        similarity = jaccard_score(selected_matrix_t0, selected_matrix_t,average='macro', zero_division=0)
        result[key] = similarity
    return result


def shuffle_and_score(var, event_dir, cli_matrix, time_leg, output_path, attribution_func):
    event_scores = {eid: np.empty(1000) for eid in event_dir.keys()}
    if var != 'Land_use':
        for k in range(1000):
            print(f"{var} - Shuffle {k + 1}/1000")
            shuffled_data = np.empty_like(cli_matrix)
            for i in range(cli_matrix.shape[0]):
                for j in range(cli_matrix.shape[1]):
                    shuffled_data[i, j] = np.random.permutation(cli_matrix[i, j])
            result = attribution_func(event_dir, shuffled_data, time_leg)
            for eid, val in result.items():
                event_scores[eid][k] = val
    else:
        origin_land_use = cli_matrix[:,:,::12]
        for k in range(1000):
            print(f"{var} - Shuffle {k + 1}/1000")
            shuffled_data = np.empty_like(origin_land_use)
            for i in range(cli_matrix.shape[0]):
                for j in range(cli_matrix.shape[1]):
                    shuffled_data[i, j] = np.random.permutation(origin_land_use[i, j])
            shuffled_data = np.repeat(shuffled_data, 12, axis=2)
            result = attribution_func(event_dir, shuffled_data, time_leg)
            for eid, val in result.items():
                event_scores[eid][k] = val
    with open(output_path, 'wb') as f:
        pickle.dump(event_scores, f)

def make_event_name(veg, scenario, z, d, a):
    return f'../get_NEG_events/vegetation_events/{veg}/{veg}_{scenario}_z{z}_d{d}_a{a}.pkl'


def make_output_name(veg, scenario, z, d, a, var):
    return f'./{veg}_shuffle/{scenario}/z{z}_d{d}_a{a}/{var}.pkl'

def run_variable(var, veg, scenario, z, d, a):

    event_path = make_event_name(veg, scenario, z, d, a)

    with open(event_path, 'rb') as f:
        event_dir = pickle.load(f)

    print(f"attribution: {veg} | {scenario} | z={z}, d={d}, a={a} | {var}")

    # path
    output_dir = f'./{veg}_shuffle/{scenario}/z{z}_d{d}_a{a}'
    os.makedirs(output_dir, exist_ok=True)

    if var in ['Tmp_high','Tmp_low','Pre_high','Pre_low','SM_high','SM_low','VPD_high','VPD_low']:
        cli_matrix = np.load(f'../get_NEG_events/climate_events/{var}_event.npy')
        cli_name = var.split('_')[0]
        time_leg = np.load(f'../data/time_leg/{cli_name}_{veg}.npy')

        output_path = f'{output_dir}/{var}.pkl'

        shuffle_and_score(var, event_dir, cli_matrix, time_leg, output_path, z_score_attribution)

    elif var == 'Land_use':
        cli_matrix = np.load('../data/cli_data/Land_use_01_24.npy')

        output_path = f'{output_dir}/Land_use.pkl'

        shuffle_and_score(var, event_dir, cli_matrix, np.zeros((360, 720)), output_path,
                          land_use_similarity_event_result)

    elif var == 'BA':
        cli_matrix = np.load('../get_NEG_events/climate_events/BA_event.npy')

        output_path = f'{output_dir}/BA.pkl'

        shuffle_and_score(var, event_dir, cli_matrix, np.zeros((360, 720)), output_path,
                          z_score_attribution)


def wrapper(args):
    return run_variable(*args)

def run_one_group(veg, scenario, z, d, a, var_list):
    print(f"\n>>> RUN: {veg} | {scenario} | z={z}, d={d}, a={a}")

    tasks = [(var, veg, scenario, z, d, a) for var in var_list]

    with Pool(processes=min(len(var_list), os.cpu_count() - 1)) as pool:
        pool.map(wrapper, tasks)


if __name__ == '__main__':
    veg_list = ['LAI', 'NDVI']
    var_list = ['Tmp_high', 'Tmp_low', 'Pre_high', 'Pre_low',
                 'SM_high', 'SM_low', 'VPD_high', 'VPD_low', 'BA', 'Land_use']

    # baseline
    baseline = (-1.5, 2, 10000)

    print("=== baseline ===")
    for veg in veg_list:
        run_one_group(veg,
                      scenario='T5_P19',
                      z=baseline[0],
                      d=baseline[1],
                      a=baseline[2],
                      var_list=var_list,
                      )
