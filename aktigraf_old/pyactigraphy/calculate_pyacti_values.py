import pandas as pd
import numpy as np
from numpy import isnan, nan, isin
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import os
from pyActigraphy.io import BaseRaw
import plotly.graph_objects as go
from enum import Enum, auto
from collections import defaultdict
import copy

periods_number = 864000
dir_path = "path_to_the_aggregated_1axis_files"
frequency = "100ms"
_start_date = "01/01/2020" # dummy data
_uuid = "IdontHaveOne"  #dummy data
_format = "Pandas"


filtered_days_path = "RAW_VALUES\RAW\proper_raw_days.csv"

def make_baseraws_days_consecutively(dir_path: str, periods_number: int, frequency: str, _start_date: str, _uuid: str, _format: str):
    list_to_write = []
    limit = 0
    dict_with_baseRaws = {}
    filtered_days_df = pd.read_csv(filtered_days_path)
    for _file in os.listdir(dir_path):
        limit += 1
        if limit < 3000:
            numbers_of_days = 0  # counting the number of days for the time_indexes, a row == 1 day
            with open(f"{dir_path}/{_file}", 'r') as file:
                print(f"{_file}'s making has started!")
                name = _file.split(".")[0]
                f_d_list = filtered_days_df[name].values.tolist()
                f_d_list = [int(d) for d in f_d_list if isnan(d) == False]

                if len(f_d_list) == 0:
                    continue

                print(f_d_list)

                reader = csv.reader(file)
                value_list = [] # concatenate each day's values

                for index, row in enumerate(reader):
                    if index + 1 in f_d_list:
                        for value in row:
                            value_list.append(float(value))
                        numbers_of_days += 1

            time_indexes = pd.date_range(start=_start_date, freq=frequency,  # dummy start_date since I Don't have one
                                         periods=periods_number * numbers_of_days)

            time_indexes2 = pd.date_range(start=_start_date, freq="1s",  # dummy start_date since I Don't have one
                                         periods=(periods_number * numbers_of_days)/10)

            dict_csv = {0: value_list} # 0 is just temporary
            df = pd.DataFrame(index=time_indexes, data=dict_csv)

            _chunks = [value_list[x:x + 10] for x in range(0, len(value_list), 1)]
            value_list2 = [np.mean(c) for c in _chunks]
            dict_csv2 = {0: value_list2}
            df2 = pd.DataFrame(index=time_indexes2, data=dict_csv2)


            column = df.columns[0]
            raw = BaseRaw(
                name=name,
                uuid=_uuid,
                format=_format,
                axial_mode=None,
                start_time=df.index[0],
                frequency=df.index.freq,
                period=(df.index[-1] - df.index[0]),
                data=df[column],
                light=None,
            )

            raw2 = BaseRaw(
                name=name,
                uuid=_uuid,
                format=_format,
                axial_mode=None,
                start_time=df2.index[0],
                frequency=df2.index.freq,
                period=(df2.index[-1] - df2.index[0]),
                data=df2[column],
                light=None,
            )

            _list = [] # returning a dict with a list value, because the other function also returns a list, so the return type is the same
            _list.append(raw)
            dict_with_baseRaws[raw.name] = _list # commented out to clear memory
            dict_to_add_to_list = defaultdict(None)
            dict_to_add_to_list["id"] = name
            dict_to_add_to_list['ADAT'] = raw.ADAT(binarize=False)
            dict_to_add_to_list['IV'] = raw.IV(binarize=False)
            dict_to_add_to_list['IS'] = raw.IS(binarize=False)
            dict_to_add_to_list['M10'] = raw2.M10(binarize=False)
            dict_to_add_to_list['L5'] = raw2.L5(binarize=False)
            dict_to_add_to_list['RA'] = raw2.RA(binarize=False)
            list_to_write.append(dict_to_add_to_list)
    field_names = ['id','M10','RA', 'IS', 'ADAT', 'L5', 'IV']
    with open(f'pyactivalues.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(list_to_write)

baseRaws_dict = make_baseraws_days_consecutively(dir_path, periods_number, frequency, _start_date, _uuid, _format)