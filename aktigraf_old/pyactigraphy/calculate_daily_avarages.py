import pandas as pd
import csv
import os
from pyActigraphy.io import BaseRaw
import plotly.graph_objects as go
from numpy import nan, isnan, isin
from pathlib import Path

"""10 measurement / sec == 10Hz == 864.000 / day"""
periods_number = 864000
dir_path = "1axis_aggregated_path"
frequency = "100ms"
_start_date = "01/01/2020" # dummy data
_uuid = "IdontHaveOne"  #dummy data
_format = "Pandas"

filtered_days_path = "RAW_VALUES\RAW\proper_raw_days.csv"

def make_baseraws_days_consecutively(dir_path: str, periods_number: int, frequency: str, _start_date: str, _uuid: str, _format: str):
    """Calculates Daily Profiles for the savitzky-golay algorithm"""
    limit = 0

    freq_list = ["1min", "5min", "15min", "30min"]

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
                print(f_d_list)
                if len(f_d_list) == 0:
                    continue
                reader = csv.reader(file)
                value_list = [] # concatenate each day's values
                for index, row in enumerate(reader):
                    if index + 1 in f_d_list:
                        for value in row:
                            value_list.append(float(value))
                        numbers_of_days += 1

            time_indexes = pd.date_range(start=_start_date, freq=frequency,  # dummy start_date since I Don't have one
                                         periods=periods_number * numbers_of_days)
            dict_csv = {0: value_list} # 0 is just temporary
            df = pd.DataFrame(index=time_indexes, data=dict_csv)


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
            for freq in freq_list:
                daily_profile = raw.average_daily_activity(freq=freq, cyclic=False, binarize=False)
                Path(f"daily_avarages/{freq}/csvs").mkdir(parents=True, exist_ok=True)
                daily_profile.to_csv(f"daily_avarages/{freq}/csvs/{name}_{freq}.csv", header=None)

                x_names = []
                for time in daily_profile.index.astype(str):
                    time = time.split(" ")[2:]
                    time = ' '.join(t for t in time)
                    x_names.append(time)

                layout = go.Layout(title=name,
                                   xaxis=dict(title="Date time"),
                                   yaxis=dict(title="Counts/period"),
                                   showlegend=False,
                                   width=2000,
                                   height=600)

                fig = go.Figure(data=[go.Scatter(x=x_names, y=daily_profile)], layout=layout)
                Path(f"daily_avarages/{freq}/pictures").mkdir(parents=True, exist_ok=True)
                fig.write_image(f"daily_avarages/{freq}/pictures/{name}_{freq}.png", scale=3)



make_baseraws_days_consecutively(dir_path, periods_number, frequency, _start_date, _uuid, _format)