import matplotlib.pyplot as plt
import csv
import os
from pathlib import Path
import pandas as pd
import numpy as np
from math import nan, isnan
import ZCM.zcm_helper as zcm_helper

"""10 measurement / sec == 10Hz == 864.000 / day"""
periods_number = 864000
dir_path = "I:\\Munka\\Elso\\project\\RAW_VALUES\\RAWS_FILTERED_PLUS_BANDPASS"
dir_path2 = "..\\..\\jupyter_notebook\\aggregalt\\Aggregált"
dir_path3 = "../RAW_VALUES/RAWS_FILTERED_PLUS_BANDPASS"
dirpath_filtered_raws_days = "../RAW_VALUES/RAW"
frequency = "100ms"
_start_date = "01/01/2020" # dummy data
_uuid = "IdontHaveOne"  #dummy data
_format = "Pandas"


def calc_zcm_raw(mx, my, mz, day, max_day, dominator, _lvl, _file,):
    zcm_list_full = []
    if day * 86400 < len(mx):
        while day < max_day:

            print(f"{day} day is started! ZCM")
            chunks = [range(x, x + dominator) for x in range(day * 864000, (day + 1) * 864000, dominator)]

            # [10,12,15,16,17] stb
            zcm_list = []
            threshold = 0

            for c in chunks:
                crossed = 0
                crossed_x = 0
                crossed_y = 0
                crossed_z = 0
                for idx, val in enumerate(c):

                    pi = c[idx - 1]
                    ni = c[idx]
                    if 0 < idx < len(c) - 1:
                        if (mx[pi] < threshold - _lvl and threshold + _lvl < mx[ni]) or (
                                mx[ni] < threshold - _lvl and threshold + _lvl < mx[pi]):
                            crossed += 1
                            crossed_x += 1

                        if (my[pi] < threshold - _lvl and threshold + _lvl < my[ni]) or (
                                my[ni] < threshold - _lvl and threshold + _lvl < my[pi]):
                            crossed += 1
                            crossed_y += 1

                        if (mz[pi] < threshold - _lvl and threshold + _lvl < mz[ni]) or (
                                mz[ni] < threshold - _lvl and threshold + _lvl < mz[pi]):
                            crossed += 1
                            crossed_z += 1
                zcm_list.append((crossed, crossed_x, crossed_y, crossed_z))

            zcm_dict = {f"threshold": threshold, "values": zcm_list, "day": day + 1}
            zcm_list_full.append(zcm_dict)

            day += 1

    list_to_write = []

    name = _file.split(".")[0]
    for li in zcm_list_full:
        for value in li["values"]:
            dict_to_write = {"id": name, "threshold": li["threshold"], "day": li["day"],
                             "zcm_value": value[0], "zcm_value_x": value[1],
                             "zcm_value_y": value[2], "zcm_value_z": value[3],
                             "freq": "1min"}
            list_to_write.append(dict_to_write)
    headers = ["id", "zcm_value", "zcm_value_x", "zcm_value_y", "zcm_value_z", "day", "threshold", "freq"]
    Path(f"Values_RAW_{dominator}/ZCM/{_lvl}").mkdir(parents=True, exist_ok=True)
    with open(f"Values_RAW_{dominator}/ZCM/{_lvl}/{name}_{str(_lvl)}.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(list_to_write)

def calc_tat_raw(mx, my, mz, day, max_day, dominator, _lvl, _file):

    tat_list_full = []
    if day * 86400 < len(mx):
        while day < max_day:

            print(f"{day} day is started! TAT")
            chunks = [range(x, x + dominator) for x in range(day * 864000, (day+1) * 864000, dominator)]

            #[10,12,15,16,17] stb
            tat_list = []
            threshold = 0

            for c in chunks:
                crossed_x = 0
                crossed_y = 0
                crossed_z = 0

                for idx, val in enumerate(c):

                    ni = c[idx]

                    if mx[ni] < threshold - _lvl or mx[ni] > threshold + _lvl:
                        crossed_x += 1

                    if my[ni] < threshold - _lvl or my[ni] > threshold + _lvl:
                        crossed_y += 1

                    if mz[ni] < threshold - _lvl or mz[ni] > threshold + _lvl:
                        crossed_z += 1
                tat_list.append((crossed_x + crossed_y + crossed_z, crossed_x, crossed_y, crossed_z))

            tat_dict = {f"threshold": threshold, "values": tat_list, "day": day+1}
            tat_list_full.append(tat_dict)

            day += 1

    list_to_write = []

    name = _file.split(".")[0]
    for li in tat_list_full:
        for value in li["values"]:
            dict_to_write = {"id": name, "threshold": li["threshold"], "day": li["day"],
                             "tat_value": value[0], "tat_value_x": value[1],
                             "tat_value_y": value[2], "tat_value_z": value[3],
                             "freq": "1min"}
            list_to_write.append(dict_to_write)
    headers = ["id", "tat_value", "tat_value_x", "tat_value_y", "tat_value_z", "day", "threshold", "freq"]
    Path(f"Values_RAW_{dominator}/TAT/{_lvl}").mkdir(parents=True, exist_ok=True)
    with open(f"Values_RAW_{dominator}/TAT/{_lvl}/{name}_{str(_lvl)}.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(list_to_write)


def calc_mad_raw(mx, my, mz, day, max_day, dominator, _lvl, _file):


    mad_list_full = []
    if day * 86400 < len(mx):
        while day < max_day:

            print(f"{day} day is started! MAD")
            chunks = [range(x, x + dominator) for x in range(day * 864000, (day+1) * 864000, dominator)]

            #[10,12,15,16,17] stb
            mad_list = []
            threshold = 0

            for c in chunks:
                mad_x = 0
                mad_y = 0
                mad_z = 0
                mad_x_list = mx[c[0]:c[-1]]
                mad_y_list = my[c[0]:c[-1]]
                mad_z_list = mz[c[0]:c[-1]]

                mad_mean_x = np.mean(mad_x_list)
                mad_mean_y = np.mean(mad_y_list)
                mad_mean_z = np.mean(mad_z_list)

                for idx, val in enumerate(c):

                    ni = c[idx]

                    mad_x += np.abs(mx[ni] - mad_mean_x)
                    mad_y += np.abs(my[ni] - mad_mean_y)
                    mad_z += np.abs(mz[ni] - mad_mean_z)
                mad_x *= (1/dominator)
                mad_y *= (1/dominator)
                mad_z *= (1/dominator)
                mad_list.append((mad_x + mad_y + mad_z, mad_x, mad_y, mad_z))

            mad_list = {f"threshold": threshold, "values": mad_list, "day": day+1}
            mad_list_full.append(mad_list)

            day += 1

    list_to_write = []

    name = _file.split(".")[0]
    for li in mad_list_full:
        for value in li["values"]:
            dict_to_write = {"id": name, "threshold": li["threshold"], "day": li["day"],
                             "mad_value": value[0], "mad_value_x": value[1],
                             "mad_value_y": value[2], "mad_value_z": value[3],
                             "freq": "1min"}
            list_to_write.append(dict_to_write)
    headers = ["id", "mad_value", "mad_value_x", "mad_value_y", "mad_value_z", "day", "threshold", "freq"]
    Path(f"Values_RAW_{dominator}/MAD/{_lvl}").mkdir(parents=True, exist_ok=True)
    with open(f"Values_RAW_{dominator}/MAD/{_lvl}/{name}_{str(_lvl)}.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(list_to_write)


def calc_raw_features(dir_path: str, periods_number: int, frequency: str, _start_date: str, _uuid: str, _format: str):
    """
    @param _lvl: the zero-crossing between  +- _lvl will be ignored
    Calculates and writes the ZCM values into a .csv per-axis and summed as well."""
    limit = 0
    _date_splitted = _start_date.split("/")
    _date_splitted = [int(d) for d in _date_splitted]

    for _file in os.listdir(dir_path):
        if int(_file[1:3]) > 0 and _file.endswith(".csv"):
            print(f"Making of {_file} has started!")
            _temp_df = pd.read_csv(f"{dir_path}/{_file}")
            for vl in np.arange(0.065, 0.075, 0.005):
                _lvl = round(vl, 3)
                print(f"_lvl: {_lvl}")
                mx = _temp_df["x_filtered"].tolist()
                my = _temp_df["y_filtered"].tolist()
                mz = _temp_df["z_filtered"].tolist()

                dominator = 600 #convert 10ms to 1min)
                day = 0
                max_day = len(mx) // 864000

                calc_zcm_raw(mx, my, mz, day, max_day, dominator, _lvl, _file)
                calc_tat_raw(mx, my, mz, day, max_day, dominator, _lvl, _file)
                calc_mad_raw(mx, my, mz, day, max_day, dominator, _lvl, _file)


calc_raw_features(dir_path3, periods_number, frequency, _start_date, _uuid, _format)

def plot_zcm_days(dirpath_filtered: str ,dirpath_zcm: str,  periods_number: int, frequency: str, _start_date: str, _uuid: str, _format: str,
                 _outdir:str = "FILTERED_DAYS_FULLY_PLOTTED", aggregate=False, window_left=5, window_right=5, filter=False):
    """
    :param dirpath_filtered: directory containing the proper_raw_days.csv
    :param dirpath_zcm: directory containing the zcm values (generated in the function above)
    :param periods_number: number of measurements per day
    :param frequency: used frequency
    :param _start_date: start date
    :param _uuid: uuid
    :param _format: pandas
    :param _outdir: output directory
    :param aggregate: set it to True if you want to apply a sliding window on the data before calculating the wake-sleep periods
    :param window_left: sliding-window left threshold
    :param window_right: sliding-window right threshold
    :param filter: only plots/calculates the longest sleep cycle
    :return: None
    Plots the wake-sleep cycles on the data, and writes the wake-sleep cycles into a .csv file.
    """

    limit = 0
    df_with_filtered_days = pd.read_csv(f"{dirpath_filtered}/proper_raw_days.csv")

    _filtered_string = "\n" if not filter else "\nSleep Filtered\n"
    aggregate_str = f"aggregated_{window_left}_{window_right}" if aggregate else ""
    _outdir_csv = _outdir + "/csvs"

    if aggregate:
        _outdir = _outdir + "/aggregated"
        _outdir_csv = _outdir_csv + "/aggregated"
    else:
        _outdir = _outdir + "/non-aggregated"
        _outdir_csv = _outdir_csv + "/non-aggregated"

    if filter:
        _outdir = _outdir + "/filtered"
        _outdir_csv = _outdir_csv + "/filtered"
        aggregate_str = aggregate_str + "_filtered"
    else:
        _outdir = _outdir + "/non_filtered"
        _outdir_csv = _outdir_csv + "/non_filtered"
        aggregate_str = aggregate_str + "_non_filtered"

    for _file in os.listdir(dirpath_zcm):
        if _file.endswith(".csv") and int(_file[1:3]) > 0:
            print(f"Making of {_file} has started!")
            _name = _file.split("_")
            name = _name[0] + "_" + _name[1]
            limit += 1
            if limit < 2000:
                days_list = df_with_filtered_days[name].tolist()
                days_list = [x for x in days_list if isnan(x) == False]
                print(days_list)
                if len(days_list) > 1:
                    z_df = pd.read_csv(f"{dirpath_zcm}/{_file}")
                    shift = periods_number*0.5
                    aggregated_zcm_list = zcm_helper.aggregate_zcm_values(z_df["zcm_value"].tolist(), window_left, window_right)
                    z_df["zcm_aggregated"] = aggregated_zcm_list
                    zcm_df_list = []

                    for day in days_list:
                        zcm_df_list.append(z_df.iloc[int((day-1) * periods_number + shift):int(day * periods_number + shift)])

                    time_indexes = pd.date_range(start=_start_date, freq=frequency,
                                               periods=periods_number)

                    x_names = []
                    for time in time_indexes:
                        _time = str(time).split(" ")[1:]
                        _time = ' '.join(t for t in _time)
                        _time = _time[:5]
                        x_names.append(_time)

                    x_names_temp = x_names
                    g_idx = 0

                    fig, AXESES = plt.subplots(len(zcm_df_list), 2, figsize=(24, 2 * len(zcm_df_list)), dpi=70,
                                               facecolor="white",
                                             edgecolor="k", )

                    list_to_write = []

                    for AX in AXESES:

                        for i_idx, ax in enumerate(AX):
                            x_names = x_names_temp
                            __day = zcm_df_list[g_idx].iloc[0]["day"]
                            if g_idx == 0:
                                ax.set_title(f'{name}{_filtered_string}Day {__day}\nFreq: {frequency}\n', ha='left', va="center", position=(1,3), rotation=-90, fontsize=15)
                            else:
                                ax.set_title(f"Day {__day}\n", ha='left', va="center", position=(1,3), rotation=-90, fontsize=15)

                            ax.set_ylabel(f'"ZCM Value')
                            if g_idx != len(zcm_df_list) - 1:
                                ax.set_xticks([])
                                ax.set_xlabel("")
                            else:
                                ax.set_xticks(np.arange(0, len(x_names), 60))
                                ax.set_xlabel(f'Time')
                            plt.tight_layout()
                            plt.subplots_adjust(right=0.95)
                            ax.tick_params(axis="x", rotation=45)
                            ax.tick_params(axis='both', which='major', labelsize=14)
                            ax.tick_params(axis='both', which='minor', labelsize=8)
                            characteristic_color = "r-" if i_idx == 0 else "g-"
                            #l1, l2 = ax.plot(x_names, zcm_df_list[g_idx]["zcm_aggregated"], 'b-', x_names, [_c * 70 for _c in ck_df_list[g_idx]["value"].tolist()], 'r-')
                            #fig.legend((l1,l2), ('ZCM Value', 'Cole-Kripke Value'), 'upper left')

                            ZCM_THRESHOLD = 5
                            SMOOTHING_THRESHOLD = 30

                            if aggregate:
                                zcm_to_plot = zcm_df_list[g_idx]["zcm_aggregated"].tolist()
                            else:
                                zcm_to_plot = zcm_df_list[g_idx]["zcm_value"].tolist()

                            if filter:
                                zcm_to_plot, constr_list = zcm_helper._zcm_filter(zcm_to_plot, ZCM_THRESHOLD)

                                x_names_filtered = [x_names[idx] for idx, val in enumerate(constr_list) if val is not None]

                                zcm_to_plot_target = zcm_helper._get_longest_zcm_target(zcm_to_plot, ZCM_THRESHOLD)
                                zcm_to_plot = zcm_to_plot[zcm_to_plot_target[2] - zcm_to_plot_target[1] + 1:zcm_to_plot_target[2] + 1]
                                x_names_filtered = x_names_filtered[zcm_to_plot_target[2] - zcm_to_plot_target[1] + 1:zcm_to_plot_target[2] + 1]
                                x_names = x_names_filtered

                            switch_list = zcm_helper._get_switchlist(zcm_to_plot, threshold=ZCM_THRESHOLD)

                            if i_idx == 1:
                                switch_list = zcm_helper._switchlist_smoother(switch_list, threshold=SMOOTHING_THRESHOLD)

                            Y_LINE = int(max(zcm_to_plot) * 0.9)

                            list_to_plot = zcm_helper._construct_list_from_switchlist(switch_list)
                            list_to_plot = [0 if l == 0 else Y_LINE for l in list_to_plot]

                            l1, l2 = ax.plot(x_names, zcm_to_plot, 'b-', x_names, list_to_plot, characteristic_color)
                            fig.legend((l1,l2), ('ZCM Value', 'Sleep'), 'upper left')

                            if i_idx == 0:
                                for _zcm in zcm_to_plot:
                                    dict_to_write = {}
                                    dict_to_write ["name"] = name
                                    dict_to_write["day"] = __day
                                    dict_to_write["zcm_values"] = _zcm
                                    list_to_write.append(dict_to_write)
                        g_idx += 1

                    Path(f"{_outdir}/zcm_plots").mkdir(parents=True, exist_ok=True)
                    plt.savefig(
                        f"{_outdir}/zcm_plots/{name}_days_{aggregate_str}_shifted.png")
                    plt.close()

                    Path(f"{_outdir_csv}").mkdir(parents=True, exist_ok=True)
                    with open(f"{_outdir_csv}/{name}_zcm_values_{aggregate_str}.csv", 'w',
                              newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=["name", "zcm_values", "day"])
                        writer.writeheader()
                        writer.writerows(list_to_write)

#plot_zcm_days(dirpath_filtered_raws_days, "Values_RAW_3000",  1440//5, "5min", "2020-01-01 12:00:00", _uuid, _format, "FILTERED_DAYS_FULLY_PLOTTED_3000", False)
#plot_zcm_days(dirpath_filtered_raws_days, "Values_RAW_1800",  1440//3, "3min", "2020-01-01 12:00:00", _uuid, _format, "FILTERED_DAYS_FULLY_PLOTTED_1800", False)


#plot_zcm_days(dirpath_filtered_raws_days, "Values_RAW",  1440, "1min", "2020-01-01 12:00:00", _uuid, _format, "FILTERED_DAYS_FULLY_PLOTTED", True, 5, 5)
#plot_zcm_days(dirpath_filtered_raws_days, "Values_RAW",  1440, "1min", "2020-01-01 12:00:00", _uuid, _format, "FILTERED_DAYS_FULLY_PLOTTED", False, 5, 5)

#plot_zcm_days(dirpath_filtered_raws_days, "Values_RAW",  1440, "1min", "2020-01-01 12:00:00", _uuid, _format, "FILTERED_DAYS_FULLY_PLOTTED", True, 5, 5, True)
#plot_zcm_days(dirpath_filtered_raws_days, "Values_RAW",  1440, "1min", "2020-01-01 12:00:00", _uuid, _format, "FILTERED_DAYS_FULLY_PLOTTED", False, 5, 5, True)




