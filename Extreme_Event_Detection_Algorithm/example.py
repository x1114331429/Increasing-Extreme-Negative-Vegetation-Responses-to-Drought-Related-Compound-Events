import os
from Extreme_Event_Detection_Algorithm.detect_event import (
    data_linear_seasonal_detrend,
    mask_event_cube,
    connect_event
)
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import linregress


def get_veg_events():
    mask = np.load('../data/land_mask/veg_mask.npy')  # Spatial mask

    vegetation_list = ['LAI', 'NDVI']
    scenario_list = ['T0_P19', 'T5_P19', 'T5_P28']

    # Baseline parameters
    baseline_z = -1.5
    baseline_duration = 2
    baseline_area = 10000

    # Sensitivity test parameters
    z_test = [-1, -2]
    duration_test = [3]
    area_test = [5000, 20000]

    for var in vegetation_list:
        print(f"====== Variable: {var} ======")
        os.makedirs(f'./vegetation_events/{var}', exist_ok=True)

        # Load vegetation data
        data = np.load(f'../data/veg_data/{var}_01_24.npy')

        # Remove the seasonal component and standardize the data using
        # the z-score method
        data_cube = data_linear_seasonal_detrend(
            data,
            mask,
            if_season_trend=True,
            season_detrend='z_score'
        )

        # print("  Save baseline anomaly data")
        # np.save(f'./vegetation_events/{var}_ano.npy', data_cube)

        # for scenario in scenario_list:
        #     print(f"--- Growing season: {scenario} ---")
        #     growing_season = np.load(
        #         f'../data/land_mask/growing_season_{scenario}.npy'
        #     )

        #     # ===== Baseline =====
        #     print(
        #         f"  [Baseline] z={baseline_z}, "
        #         f"d={baseline_duration}, a={baseline_area}"
        #     )

        #     event_cube = mask_event_cube(
        #         data_cube,
        #         mask,
        #         method='z_score',
        #         z_score_threshold=abs(baseline_z),
        #         growing_season_matrix=growing_season,
        #         flag='negative'
        #     )

        #     result = connect_event(
        #         event_cube,
        #         delete_small_event=True,
        #         save_path=(
        #             f'./vegetation_events/{var}/'
        #             f'{var}_{scenario}_z{baseline_z}_'
        #             f'd{baseline_duration}_a{baseline_area}.pkl'
        #         ),
        #         min_duration=baseline_duration,
        #         min_area=baseline_area
        #     )
        #     print(f"  Number of baseline events: {len(result)}")

        #     # ===== Sensitivity test: z-score threshold =====
        #     for z in z_test:
        #         print(f"  [Z test] z={z}")

        #         event_cube = mask_event_cube(
        #             data_cube,
        #             mask,
        #             method='z_score',
        #             z_score_threshold=abs(z),
        #             growing_season_matrix=growing_season,
        #             flag='negative'
        #         )

        #         result = connect_event(
        #             event_cube,
        #             delete_small_event=True,
        #             save_path=(
        #                 f'./vegetation_events/{var}/'
        #                 f'{var}_{scenario}_z{z}_'
        #                 f'd{baseline_duration}_a{baseline_area}.pkl'
        #             ),
        #             min_duration=baseline_duration,
        #             min_area=baseline_area
        #         )
        #         print(f"    Number of events: {len(result)}")

        #     # ===== Sensitivity test: minimum duration =====
        #     for d in duration_test:
        #         print(f"  [Duration test] duration={d}")

        #         event_cube = mask_event_cube(
        #             data_cube,
        #             mask,
        #             method='z_score',
        #             z_score_threshold=abs(baseline_z),
        #             growing_season_matrix=growing_season,
        #             flag='negative'
        #         )

        #         result = connect_event(
        #             event_cube,
        #             delete_small_event=True,
        #             save_path=(
        #                 f'./vegetation_events/{var}/'
        #                 f'{var}_{scenario}_z{baseline_z}_'
        #                 f'd{d}_a{baseline_area}.pkl'
        #             ),
        #             min_duration=d,
        #             min_area=baseline_area
        #         )
        #         print(f"    Number of events: {len(result)}")

        #     # ===== Sensitivity test: minimum area =====
        #     for a in area_test:
        #         print(f"  [Area test] area={a} km²")

        #         event_cube = mask_event_cube(
        #             data_cube,
        #             mask,
        #             method='z_score',
        #             z_score_threshold=abs(baseline_z),
        #             growing_season_matrix=growing_season,
        #             flag='negative'
        #         )

        #         result = connect_event(
        #             event_cube,
        #             delete_small_event=True,
        #             save_path=(
        #                 f'./vegetation_events/{var}/'
        #                 f'{var}_{scenario}_z{baseline_z}_'
        #                 f'd{baseline_duration}_a{a}.pkl'
        #             ),
        #             min_duration=baseline_duration,
        #             min_area=a
        #         )
        #         print(f"    Number of events: {len(result)}")


get_veg_events()


def get_cli_events():

    var_list = ['Tmp', 'Pre', 'SM', 'VPD']
    mask = np.load('../data/land_mask/veg_mask.npy')  # Spatial mask

    for var in var_list:
        print(var)
        # Load climate data
        data = np.load(
            f'../data/cli_data/{var}_01_24.npy'
        )

        # Remove the long-term and seasonal components and
        # standardize the data using the z-score method
        data_cube = data_linear_seasonal_detrend(
            data,
            mask,
            if_season_trend=True,
            season_detrend='z_score'
        )

        # Identify negative climate anomalies
        event_cube = mask_event_cube(
            data_cube,
            mask,
            method='z_score',
            growing_season_matrix=None,
            flag='negative'
        )

        np.save(
            f'./climate_events/{var}_low_event.npy',
            event_cube
        )

        # Identify positive climate anomalies
        event_cube = mask_event_cube(
            data_cube,
            mask,
            method='z_score',
            growing_season_matrix=None,
            flag='positive'
        )

        np.save(
            f'./climate_events/{var}_high_event.npy',
            event_cube
        )
    # burned_area = np.load('../data/cli_data/BA_01_24.npy')
    # burned_area = np.where(burned_area>0.1,1,0)
    # np.save('./climate_events/BA_event.npy',burned_area)
get_cli_events()