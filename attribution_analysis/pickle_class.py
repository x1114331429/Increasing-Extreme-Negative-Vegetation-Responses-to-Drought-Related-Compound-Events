import pickle
import numpy as np
import matplotlib.pyplot as plt
from tools import list_to_num, num_to_list

var_list = [
    'Tmp_high', 'Tmp_low',
    'Pre_high', 'Pre_low',
    'SM_high', 'SM_low',
    'VPD_high', 'VPD_low',
    'BA', 'Land_use'
]
veg_list = ['LAI','NDVI']

scenario_list = ['T5_P19']
for veg in veg_list:
    result = {}
    with open(f'../get_NEG_events/vegetation_events/{veg}/{veg}_T5_P19_z-1.5_d2_a10000.pkl', 'rb') as f:
        event_dir = pickle.load(f)
    for key,value in event_dir.items():
        temp_list = [0]*len(var_list)
        for index,var in enumerate(var_list):
            with open(f'./{veg}_attribution/{var}.pkl', 'rb') as f:
                cli_dir = pickle.load(f)
            # print(cli_dir.get(key))
            if cli_dir.get(key) == None:
                temp_list[index] = 0
            else:
                temp_list[index] = 1
        result[key] = list_to_num(temp_list)

    with open(f'./{veg}_attribution/result_shuffle_class.pkl', 'wb') as f:
        pickle.dump(result, f)

