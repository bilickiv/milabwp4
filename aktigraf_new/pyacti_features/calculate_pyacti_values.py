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


filtered_days_path = r"I:\Munka\Elso\project\RAW_VALUES\RAW\proper_raw_days.csv"

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

#baseRaws_dict = make_baseraws_days_consecutively(dir_path, periods_number, frequency, _start_date, _uuid, _format)

def calculate_psykose_pyacti():

    wrong_data_df = pd.read_csv(r"I:\Munka\Elso\tovabbi_csvk\wrong_data.csv")
    wrong_data_df['timestamp_beginning'] = pd.to_datetime(wrong_data_df['timestamp_beginning'])
    wrong_data_df['timestamp_end'] = pd.to_datetime(wrong_data_df['timestamp_end'])
    LIST_TO_WRITE = []

    dirpaths = [r"I:\Munka\Elso\tovabbi_csvk\psykose\control",
                r"I:\Munka\Elso\tovabbi_csvk\psykose\patient",
                r"I:\Munka\Elso\tovabbi_csvk\depression\data\condition"]

    for d in dirpaths:
        for file in os.listdir(d):
            print(f"{file} is in the making!")
            df = pd.read_csv(f"{d}\\{file}")

            _name = "psykose" if "control" or "patient" in file else "depresjon"
            wd_df = wrong_data_df[wrong_data_df["name"] == f"{_name}_{file.split('.')[0]}"]

            wrong_dates = []
            for _, row in wd_df.iterrows():
                begin =  pd.to_datetime(row['timestamp_beginning'])
                end =  pd.to_datetime(row['timestamp_end'])
                begin_strft = begin.strftime("%Y-%m-%d")
                end_strft = end.strftime("%Y-%m-%d")
                if end.hour > 3:
                    date_range = pd.date_range(begin_strft, end_strft, freq='D')
                else:
                    date_range = pd.date_range(begin_strft, end_strft, freq='D', closed="left")
                wrong_dates.extend(date_range.strftime('%Y-%m-%d').tolist())
            df = df[~df["date"].str[:10].isin(wrong_dates)]

            df["activity"] = [int(d) for d in df["activity"].values]
            consecutive_zeros = df['activity'].rolling(1440).sum() == 0
            timestamps_to_delete = df.loc[consecutive_zeros, 'date'].unique()
            df = df[~df["date"].isin(timestamps_to_delete)]

            if "control_3.csv" in file:
                act = df["activity"].values
                plt.plot(np.arange(0, len(act), 1), act)
                plt.show()


            time_indexes = pd.date_range(start=pd.to_datetime(df.timestamp.values[0]), freq="1min",  # dummy start_date since I Don't have one
                                         periods=df.shape[0])
            df["timestamp"] = time_indexes
            df.set_index("timestamp", inplace=True)




            dict_csv = {0: df["activity"]}  # 0 is just temporary
            df = pd.DataFrame(index=time_indexes, data=dict_csv)

            column = df.columns[0]

            raw = BaseRaw(
                name="control_1",
                uuid=_uuid,
                format=_format,
                axial_mode="mono-axial",
                start_time=df.index[0],
                frequency=df.index.freq,
                period=(df.index[-1] - df.index[0]),
                data=df[column],
                light=None,
            )

            dict_to_write = {}
            dict_to_write["name"] = file.split(".")[0]
            dict_to_write['ADAT'] = raw.ADATp(binarize=False, period="1D")
            dict_to_write ["M10"] = raw.M10p(binarize=False, period="1D")
            dict_to_write ["L5"] = raw.L5p(binarize=False, period="1D")
            dict_to_write ["RA"] = raw.RAp(binarize=False, period="1D")
            dict_to_write["IV"] = raw.IV(binarize=False)
            dict_to_write["IS"] = raw.IS(binarize=False)
            LIST_TO_WRITE.append(dict_to_write)

    field_names = LIST_TO_WRITE[0].keys()
    with open(f'pyactivalues_psykose.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(LIST_TO_WRITE)
#calculate_psykose_pyacti()

def calculate_szte_pyacti():

    LIST_TO_WRITE = []

    dirpaths = [r"I:\Munka\Elso\project\ZCM\Values_RAW_600\ZCM\0.05\timestampelt2"]

    filtered_days_df = pd.read_csv(filtered_days_path)


    for d in dirpaths:
        for file in os.listdir(d):

            names = file.split("_")

            f_d_list = filtered_days_df[names[0] + "_" + names[1]].values.tolist()
            f_d_list = [int(d) for d in f_d_list if isnan(d) == False]
            if len(f_d_list) < 2:
                continue
            print(f"{file} is in the making!")
            df = pd.read_csv(f"{d}\\{file}")

            time_indexes = pd.date_range(start=pd.to_datetime(df.timestamp[0]), freq="1min",  # dummy start_date since I Don't have one
                                         periods=df.shape[0])
            df["timestamp"] = time_indexes
            df["zcm_value"] = [int(d) for d in df["zcm_value"].values]
            df.set_index("timestamp", inplace=True)

            df2 = pd.DataFrame()

            for f_d in f_d_list:
                slice = df.iloc[(f_d-1)*1440: f_d*1440]
                df2 = pd.concat([df2, slice], axis=0)

            dict_csv = {0: df2["zcm_value"].values.tolist()}  # 0 is just temporary

            print(df2["zcm_value"].values.tolist())
            print(len(df2["zcm_value"].values.tolist())/1440)
            time_indexes2 = pd.date_range(start=pd.to_datetime(df2.index[0]), freq="1min",
                                         # dummy start_date since I Don't have one
                                         periods=df2.shape[0])
            df2.index = time_indexes2
            df3 = pd.DataFrame(index=df2.index, data=dict_csv)

            print(df3)

            column = df3.columns[0]

            raw = BaseRaw(
                name="control_1",
                uuid=_uuid,
                format=_format,
                axial_mode="mono-axial",
                start_time=df3.index[0],
                frequency=df3.index.freq,
                period=(df3.index[-1] - df3.index[0]),
                data=df3[column],
                light=None,
            )

            dict_to_write = {}
            dict_to_write["name"] = file.split("_")[0]
            dict_to_write ["ADAT"] = raw.ADATp(binarize=False, period="1D")
            dict_to_write ["M10"] = raw.M10p(binarize=False, period="1D")
            dict_to_write ["L5"] = raw.L5p(binarize=False, period="1D")
            dict_to_write ["RA"] = raw.RAp(binarize=False, period="1D")
            dict_to_write["IV"] = raw.IV(binarize=False)
            dict_to_write["IS"] = raw.IS(binarize=False)
            print(dict_to_write)
            LIST_TO_WRITE.append(dict_to_write)

    field_names = LIST_TO_WRITE[0].keys()
    with open(f'pyactivalues_szte_filtered.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(LIST_TO_WRITE)
calculate_szte_pyacti()