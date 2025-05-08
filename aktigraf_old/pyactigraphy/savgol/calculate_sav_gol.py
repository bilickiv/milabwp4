import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import os
from pyActigraphy.io import BaseRaw
import plotly.graph_objects as go
from enum import Enum, auto
from collections import defaultdict
import pyActigraphy
from pathlib import Path

from scipy.signal import savgol_filter


dir_path_1 = "daily_avarages/1min/csvs"
dir_path_5 = "daily_avarages/5min/csvs"
image_dir_path_1 = "savgol/daily_avarages/1min/Pictures"
image_dir_path_5 = "savgol/daily_avarages/5min/Pictures"
data_dir_path_1 = "savgol/daily_avarages/1min/csv"
data_dir_path_5 = "savgol/daily_avarages/5min/csv"


def make_pictures_and_csvs(dir_path, image_dir_path, data_dir_path, x_tick_density = 24):

    limit = 0
    for _file in os.listdir(dir_path):
        if _file.endswith(".csv"):
            limit += 1
            if limit < 3000:
                filename = _file
                print(f"{filename}'s making has started!")
                df = pd.read_csv(f"{dir_path}/{filename}")

                name = filename.split(".")[0]

                x_names = []
                for time in df.iloc[:, 0]:
                    time = time.split(" ")[2:]
                    time = ' '.join(t for t in time)
                    x_names.append(time)

                for i in range(2,7):

                    for j in range(1, i):
                        filtered_list = savgol_filter(df.iloc[:, 1], i, j)

                        Path(f"{image_dir_path}/{name}").mkdir(parents=True, exist_ok=True)
                        Path(f"{data_dir_path}/{name}").mkdir(parents=True, exist_ok=True)

                        f_df = pd.DataFrame(zip(x_names ,filtered_list), columns=['Date', 'Value'])
                        f_df.to_csv(f"{data_dir_path}/{name}/{name}_Average_Activity_{i}_{j}.csv")

                        plt.figure(num=None, figsize=(20, 6), dpi=200, facecolor='lightgrey', edgecolor='k')
                        plt.plot(x_names, df.iloc[:,1], linewidth=0.5, label=f"Base")
                        plt.plot(x_names, filtered_list, linewidth=0.5, label=f"Filtered")
                        plt.title(f"{name}_Average_Activity\nwin_size {i}, polyorder {j}")
                        plt.xticks(np.arange(0, len(x_names) + 1, x_tick_density))
                        plt.xlabel("Time")
                        plt.ylabel("Activity")
                        plt.grid(False)
                        plt.legend()
                        plt.savefig(f"{image_dir_path}/{name}/{name}_Average_Activity_{i}_{j}.png")
                        plt.close()






make_pictures_and_csvs(dir_path_5, image_dir_path_5, data_dir_path_5)
print()
print()
print()
print()
print()
print()
make_pictures_and_csvs(dir_path_1, image_dir_path_1, data_dir_path_1, 120)
