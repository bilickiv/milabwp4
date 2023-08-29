import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import os
import plotly.graph_objects as go
from enum import Enum, auto
from collections import defaultdict
from pathlib import Path
from scipy.signal import savgol_filter

dir_path_5 = "daily_avarages/5min/csvs"
image_dir_path_5 = "savgol/sgf_daily_avarages/5min/Pictures"
data_dir_path_5 = "savgol/sgf_daily_avarages/5min/csvs"

def calculate_end_values_differences(s_g_list, list_to_write, name):

    ceiling_sum_values = 0
    floor_sum_values = 0

    ceiling_values = []
    floor_values = []

    _e_v_values = []
    e_v__values_changes = []

    for i in range(1, len(s_g_list) - 1):
        if s_g_list[i - 1] < s_g_list[i] and s_g_list[i + 1] < s_g_list[i]:
            ceiling_sum_values += s_g_list[i]

            ceiling_values.append(s_g_list[i])

            _e_v_values.append(s_g_list[i])

        elif s_g_list[i - 1] > s_g_list[i] and s_g_list[i + 1] > s_g_list[i]:
            floor_sum_values += s_g_list[i]

            floor_values.append(s_g_list[i])

            _e_v_values.append(s_g_list[i])

    for i in range(1, len(_e_v_values) - 1):
        e_v__values_changes.append(_e_v_values[i] - _e_v_values[i - 1])


    e_v_avg_value = sum(_e_v_values)/len(_e_v_values)
    avg_ceiling_value = sum(ceiling_values)/len(ceiling_values)
    avg_floor_value = sum(floor_values)/len(floor_values)

    std_upper = np.std(ceiling_values)
    std_lower = np.std(floor_values)
    std_avg = np.std(_e_v_values)
    avg_ceiling_floor_diff = avg_ceiling_value - avg_floor_value

    len_ceiling_value = len(ceiling_values)
    len_floor_value = len(floor_values)
    len_e_v_values = len(_e_v_values)

    dict_to_write = {}
    dict_to_write["id"] = name
    dict_to_write["ceiling_sum_values"] = ceiling_sum_values
    dict_to_write["floor_sum_values"] = floor_sum_values
    dict_to_write["e_v_avg_value"] = e_v_avg_value
    dict_to_write["avg_ceiling_value"] = avg_ceiling_value
    dict_to_write["avg_floor_value"] = avg_floor_value
    dict_to_write["avg_ceiling_floor_diff"] = avg_ceiling_floor_diff
    dict_to_write["std_avg"] = std_avg
    dict_to_write["std_upper"] = std_upper
    dict_to_write["std_lower"] = std_lower
    dict_to_write["len_ceiling_value"] = len_ceiling_value
    dict_to_write["len_floor_value"] = len_floor_value
    dict_to_write["len_e_v_values"] = len_e_v_values
    dict_to_write["category"] = name[-1]

    list_to_write.append(dict_to_write)



def make_pictures_and_csvs(dir_path, image_dir_path, data_dir_path, x_tick_density=24):

    list_to_write = []

    limit = 0
    for _file in os.listdir(dir_path):
        if _file.endswith(".csv"):
            limit += 1
            if limit < 3000: #how many csvs to read
                filename = _file
                print(f"{filename}'s making has started!")
                df = pd.read_csv(f"{dir_path}/{filename}")

                name = filename.split(".")[0]
                _name = name.split("_")
                name = _name[0] + "_" + _name[1]

                x_names = []
                for time in df.iloc[:, 0]:
                    time = time.split(" ")[2:]
                    time = ' '.join(t for t in time)
                    x_names.append(time)

                win_size = 50
                poly_order = 7

                second_window_size = 30
                second_poly_order = 7
                _s_g_filtered_list = savgol_filter(df.iloc[:, 1], win_size, poly_order)
                s_g_filtered_list = savgol_filter(_s_g_filtered_list, second_window_size, second_poly_order)

                Path(f"{data_dir_path}/{win_size}_{poly_order}_{second_window_size}_{second_poly_order}").mkdir(parents=True, exist_ok=True)
                Path(f"{image_dir_path}/{win_size}_{poly_order}_{second_window_size}_{second_poly_order}").mkdir(parents=True, exist_ok=True)

                f_df = pd.DataFrame(zip(x_names, s_g_filtered_list), columns=['Date', 'Value'])
                f_df.to_csv(f"{data_dir_path}/{win_size}_{poly_order}_{second_window_size}_{second_poly_order}/{name}_Average_Activity_{win_size}_{poly_order}_{second_window_size}_{second_poly_order}.csv")

                plt.figure(num=None, figsize=(20, 6), dpi=200, facecolor='lightgrey', edgecolor='k')
                plt.plot(x_names, df.iloc[:,1], linewidth=0.5, label=f"Base")
                plt.plot(x_names, s_g_filtered_list, linewidth=0.5, label=f"Filtered")
                plt.title(f"{name}_Average_Activity\nwin_size {win_size}, polyorder {poly_order}\n 2nd_win_size {second_window_size}, 2nd_polyorder {second_poly_order}")
                plt.xticks(np.arange(0, len(x_names) + 1, x_tick_density))
                plt.xlabel("Time")
                plt.ylabel("Activity")
                plt.grid(False)
                plt.legend()
                plt.savefig(f"{image_dir_path}/{win_size}_{poly_order}_{second_window_size}_{second_poly_order}/{name}_Average_Activity_{win_size}_{poly_order}_{second_window_size}_{second_poly_order}.png")
                plt.close()

                calculate_end_values_differences(s_g_filtered_list, list_to_write, name)

    header_list = []

    for _dict in list_to_write:
        for key, value in _dict.items():
            header_list.append(key)
        break
    Path(f"{data_dir_path}").mkdir(parents=True,exist_ok=True)
    with open(f"{data_dir_path}/processed_e_v_values.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header_list)
        writer.writeheader()
        writer.writerows(list_to_write)



make_pictures_and_csvs(dir_path_5, image_dir_path_5, data_dir_path_5)