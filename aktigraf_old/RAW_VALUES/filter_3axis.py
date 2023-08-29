import seaborn as sns
import matplotlib.pyplot as plt
import csv
import os
from pyActigraphy.io import BaseRaw
import plotly.graph_objects as go
from enum import Enum, auto
from collections import defaultdict
from pathlib import Path
import datetime
import CK.cole_kripke_calculator as ckc
import pandas as pd
import numpy as np
from RAW_VALUES.plot_raw import get_butter_bandpass
import sys


pd.set_option('display.width', 300)
np.set_printoptions(linewidth=300)
pd.set_option('display.max_columns',300)

"""10 measurement / sec == 10Hz == 864.000 / day"""
periods_number = 864000
dir_path = "raw_values_path"
dir_path2 = "aggregalt_values_path"
dir_path3 = "ZCM/Values3"
frequency = "100ms"
_start_date = "01/01/2020" # dummy data
_uuid = "IdontHaveOne"  #dummy data
_format = "Pandas"


def make_bandpassed_raws():
    """
    This function fills the raw 3-axis datas' beginning and end with 0-s to make a full day, based on the already existing aggregated 1-axis datas.
    These csvs are used to calculate the ZCM values.
    :return: None
    """
    for file_agg in os.listdir(dir_path2):

        name = file_agg.split(".")[0]

        for _file_raw in os.listdir(dir_path):

            if _file_raw.startswith(name):
                with open(f"{dir_path2}/{name}.csv másolata.csv", 'r') as file:

                    print(f"{name}'s making has started!")

                    value_list = []  # concatenate each day's values
                    reader = csv.reader(file)

                    for index, row in enumerate(reader):
                        for value in row:
                            value_list.append(float(value))


                count_zeros_front = 0
                count_zeros_back = 0

                for v in value_list:
                    if v == 0:
                        count_zeros_front += 1
                    else:
                        break

                for v in reversed(value_list):
                    if v == 0:
                        count_zeros_back += 1
                    else:
                        break

                list_zeros_front = [0] * count_zeros_front
                list_zeros_back = [0] * count_zeros_back

                __temp_df = pd.read_csv(f"{dir_path}/{_file_raw}")
                _df_len = len(__temp_df["measurement_data_x"].tolist())

                measurement_data_x = list_zeros_front + __temp_df["measurement_data_x"].tolist() + list_zeros_back
                measurement_data_y = list_zeros_front + __temp_df["measurement_data_y"].tolist() + list_zeros_back
                measurement_data_z = list_zeros_front + __temp_df["measurement_data_z"].tolist() + list_zeros_back

                if len(measurement_data_x) == len(value_list):

                    measurement_lists = [measurement_data_x, measurement_data_y, measurement_data_z]

                    measurement_lists.append(get_butter_bandpass(measurement_data_x, 3))
                    measurement_lists.append(get_butter_bandpass(measurement_data_y, 3))
                    measurement_lists.append(get_butter_bandpass(measurement_data_z, 3))
                    print(len(value_list))
                    print(len(measurement_lists[3]))


                    with open(f"RAWS_FILTERED_PLUS_BANDPASS/{name}_raw_bandpass.csv", 'w', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile,
                                                fieldnames=['x', 'y', 'z', 'x_filtered', 'y_filtered', 'z_filtered'])
                        writer.writeheader()


                    ch_len = len(measurement_data_x)//100

                    for idx in range(0, 200):
                        list_to_write = []
                        if idx * ch_len > len(measurement_data_x):
                            break

                        elif len(measurement_data_x) - (ch_len * idx) > ch_len:
                            for i in range(0, ch_len):
                                r_idx = idx * ch_len + i
                                dict_to_write = {"x": measurement_lists[0][r_idx], "y": measurement_lists[1][r_idx], "z": measurement_lists[2][r_idx],
                                                 "x_filtered": measurement_lists[3][r_idx], "y_filtered": measurement_lists[4][r_idx],
                                                 "z_filtered": measurement_lists[5][r_idx]}
                                list_to_write.append(dict_to_write)

                        elif len(measurement_data_x) - (ch_len * idx) <= ch_len:
                            for i in range(ch_len*idx, len(measurement_data_x)):
                                dict_to_write = {"x": measurement_lists[0][i], "y": measurement_lists[1][i],
                                                 "z": measurement_lists[2][i],
                                                 "x_filtered": measurement_lists[3][i],
                                                 "y_filtered": measurement_lists[4][i],
                                                 "z_filtered": measurement_lists[5][i]}
                                list_to_write.append(dict_to_write)

                        Path(f"RAWS_FILTERED_PLUS_BANDPASS").mkdir(parents=True, exist_ok=True)

                        with open(f"RAWS_FILTERED_PLUS_BANDPASS/{name}_raw_bandpass.csv", 'a', newline='') as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=['x', 'y', 'z', 'x_filtered', 'y_filtered', 'z_filtered'])
                            writer.writerows(list_to_write)
                else:
                    print(f"{name} measurements doesn't have the same length!\n{len(measurement_data_x)=}\n{len(value_list)=}")

make_bandpassed_raws()



