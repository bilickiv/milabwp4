import numpy as np
from pathlib import Path
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import ZCM.zcm_helper as zcm_helper
import ZCM.zcm_analyzer as zcm_analyzer
import copy, csv
from numpy import isnan
import datetime
import sys



def plot_sleep_wake_statuses(zcm_values, timestamps, name, switchlist, f_d_list, additional_str="", zcm_compared=[], COMPARE_STR=""):

    days = len(zcm_values)//1440
    if days < 2:
        return
    print(len(timestamps))
    print(days)
    print()
    _cl = zcm_helper._construct_list_from_switchlist(switchlist)

    fig, AXESES = plt.subplots(days, 1, figsize=(24, 2 * days), dpi=280,
                               facecolor="white",
                               edgecolor="k", )

    time_indexes = pd.date_range(start=timestamps[0], freq="1min",
                                 periods=1440)
    x_names = []
    for time in time_indexes:
        _time = str(time).split(" ")[1:]
        _time = ' '.join(t for t in _time)
        _time = _time[:5]
        x_names.append(_time)

    for d, AX in enumerate(AXESES):
        zcm_color = "-b" if (d+1) in f_d_list else "-g"
        zcm_to_plot = zcm_compared[d*1440: (d+1)*1440]
        cl = _cl[d*1440: (d+1)*1440]
        max_Y = max(zcm_to_plot)*0.9
        cl_to_plot = [max_Y if c == 1 else 0 for idx, c in enumerate(cl)]

        if d == 0:
            AX.set_title(f'{name}{COMPARE_STR}\n{timestamps[d].month}.{timestamps[d].day} ', ha='left', va="center",
                         position=(1, 3), rotation=-90, fontsize=15)
        else:
            AX.set_title(f'{timestamps[d].month}.{timestamps[d].day} ',
                         ha='left',
                         va="center", position=(1, 3), rotation=-90, fontsize=15)

        AX.set_ylabel(f'"ZCM Value')

        if d != len(AXESES) - 1:
            AX.set_xticks([])
            AX.set_xlabel("")
        else:
            AX.set_xticks(np.arange(0, len(x_names), 60))
            AX.set_xlabel(f'Time')

        plt.tight_layout()
        plt.subplots_adjust(right=0.95)
        AX.tick_params(axis="x", rotation=45)
        AX.tick_params(axis='both', which='major', labelsize=14)
        AX.tick_params(axis='both', which='minor', labelsize=8)

        if d == 0:
            AX.plot(x_names, zcm_to_plot, zcm_color, label="activity")
            AX.plot(x_names, cl_to_plot, 'r-', label="wake/sleep")
            fig.legend(loc='upper left')
        else:
            AX.plot(x_names, zcm_to_plot, zcm_color, label="activity")
            AX.plot(x_names, cl_to_plot, 'r-', label="wake/sleep")
    Path(f"ROUND2_WITH_OURS/plots/sleepwake/compared{COMPARE_STR}/").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"ROUND2_WITH_OURS/plots/sleepwake/compared{COMPARE_STR}/{name}{additional_str}{COMPARE_STR}.png")
    plt.close()


def calculate_zcm_characteristics_others_above_limit(dir_path, outdir, additional_name: str="",
                                                     additional_filename: str="",
                                                     split_humps=False, ZCM_THRESHOLD=500, LIMIT=239,  WTS=20, WTS2=90,
                                                     MODE_OR=False, L_WINDOW=20, R_WINDOW=20, PLOT=False,
                                                     COMPARE_STR="",):
    "Calculates zcm characteristics for the other datasets for szte data, filtered with 'good' days!"
    list_to_write = []


    filtered_days_path = "I:\Munka\Elso\project\RAW_VALUES\RAW\proper_raw_days.csv"
    filtered_days_df = pd.read_csv(filtered_days_path)

    for _file in os.listdir(dir_path):
        print(f"{dir_path}/{_file} is in the making! {split_humps}")
        if _file.endswith(".csv"):
            name = additional_name + "_" + _file.split(".")[0]
            df = pd.read_csv(f"{dir_path}/{_file}")
            timestamps = df.timestamp.values.tolist()

            f_name = "".join(name[1:6])
            f_d_list = filtered_days_df[f_name].values.tolist()
            f_d_list = [int(d) for d in f_d_list if isnan(d) == False]
            zcm_values = df["zcm_value"].tolist()
            zcm_values = zcm_helper.aggregate_zcm_values(zcm_values, 5, 5)


            zcm_values_compared = zcm_values

            original_switchlist = zcm_helper._get_switchlist(zcm_values, threshold=ZCM_THRESHOLD)

            ###
            if PLOT:
                timestamps_to_plot = [pd.to_datetime(t) for t in timestamps]
                timestamps_to_plot = [timestamps_to_plot[d * 1440] for d in range(len(timestamps)//1440)]
                plot_sleep_wake_statuses(zcm_values, timestamps_to_plot, f_name, original_switchlist, f_d_list, "_before")
            ###

            original_switchlist = zcm_helper._switchlist_smoother(original_switchlist, WTS, 1, 0, 0, MODE_OR=MODE_OR, OR_TARGET=1)
            original_switchlist = zcm_helper._switchlist_smoother(original_switchlist, WTS, 0, L_WINDOW, R_WINDOW)
            original_switchlist = zcm_helper._switchlist_smoother(original_switchlist, WTS2, 1, 0, 0, MODE_OR=MODE_OR,
                                                                  OR_TARGET=1)
            original_switchlist = zcm_helper._switchlist_smoother(original_switchlist, WTS2, 0, 0, 0, MODE_OR=MODE_OR,
                                                                  OR_TARGET=1)

            ###
            if PLOT:
                plot_sleep_wake_statuses(zcm_values, timestamps_to_plot, f_name, original_switchlist, f_d_list, "_after", zcm_values_compared, COMPARE_STR)
            ###

            original_constr_list = zcm_helper._construct_list_from_switchlist(original_switchlist)

            zcm_values_filtered = [] # list for each day's
            found_overlap = False
            overlap_index = 0
            for idx, fd in enumerate(f_d_list): #list of good days

                if not found_overlap or fd - f_d_list[idx-1] > 1: #if the current day is not the continuation of the previously analyzed day, or that day doesn't end with sleep
                    begin_index = (fd - 1) * 1440
                else:
                    begin_index = overlap_index + 1 #if the current day is the previously analyzed day's continuation and that day ended with sleep
                last_index = fd * 1440
                ix = 0
                while original_switchlist[ix][2] < last_index:
                    ix += 1
                if original_switchlist[ix][0] == 0: #if the day ends with sleep
                    last_index = original_switchlist[ix][2]
                    found_overlap = True
                    overlap_index = last_index
                else:
                    found_overlap = False

                zcm_values_filtered.append((zcm_values[begin_index: last_index],
                                            fd-1,
                                            begin_index, original_constr_list[begin_index: last_index],
                                            zcm_values_compared[begin_index: last_index]))


            timestamps = df.timestamp.values.tolist()
            for zcm_day_values in zcm_values_filtered:


                zcm_to_plot, constr_list = zcm_helper._zcm_filter(zcm_day_values[0], ZCM_THRESHOLD, given_constr_list=zcm_day_values[3])

                constr_list = [c for c in constr_list if c is not None]
                sw = zcm_helper._get_switchlist(constr_list, 1)

                zcm_to_plot_targets = zcm_helper._get_all_zcm_target(zcm_to_plot, ZCM_THRESHOLD, 0, LIMIT, False, given_switchlist=sw)
                if zcm_to_plot_targets:
                    for ind, target in enumerate(zcm_to_plot_targets):
                        try:
                            zcm_to_plot_temp = zcm_day_values[0][target[2] - target[1] + 1:target[2] + 1]
                            timestamp = timestamps[zcm_day_values[2] + (target[2] - target[1])]
                            zcm_to_plot_ten_percent = len(zcm_to_plot_temp) // 10
                            zcm_data_copy = zcm_to_plot_temp[zcm_to_plot_ten_percent: len(zcm_to_plot_temp) - zcm_to_plot_ten_percent]
                            return_dict = zcm_analyzer.get_characteristics(zcm_data_copy, name, timestamp, split_humps)
                            return_dict["zcm_threshold"] = ZCM_THRESHOLD
                            return_dict["sleep_to_wake_1"] = WTS
                            return_dict["wake_to_sleep_1"] = WTS
                            return_dict["left_window_1"] = L_WINDOW
                            return_dict["right_window_1"] = R_WINDOW
                            return_dict["sleep_to_wake_OR_1"] = WTS2
                            return_dict["wake_to_sleep_OR_1"] = WTS2
                            return_dict["timestamp_beginning"] = return_dict.pop("day")
                            list_to_write.append(return_dict)

                        except:
                            continue



    for di in list_to_write:
        print(di)
    headers = []
    for _dict in list_to_write:
        for key, value in _dict.items():
            headers.append(key)
        break
    split_humps_str = "_quartile" if split_humps else ""
    Path(f"{outdir}/adatvizsgalat").mkdir(parents=True, exist_ok=True)
    with open(f"{outdir}/adatvizsgalat/{additional_filename}{split_humps_str}_adatvizsgalat.csv", 'w',
              newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(list_to_write)




def calculate_zcm_characteristics_others_above_limit2(dir_path, outdir, additional_name: str="",
                                                      additional_filename: str="",
                                                      split_humps=False, PIM_THRESHOLD=500, LIMIT=239, WTS=20, WTS2=90,
                                                      MODE_OR=False, L_WINDOW=20, R_WINDOW=20, PLOT=False):
    """Calculates pim characteristics for the other datasets for psykose!"""
    list_to_write = []

    for _file in os.listdir(dir_path):
        print(f"{dir_path}/{_file} is in the making! {split_humps}")
        if _file.endswith(".csv"):
            name = additional_name + "_" + _file.split(".")[0]
            df = pd.read_csv(f"{dir_path}/{_file}")
            timestamps = df.timestamp.values.tolist()
            pim_values = df["activity"].tolist()

            original_switchlist = zcm_helper._get_switchlist(pim_values, threshold=PIM_THRESHOLD)
            #Smoothes wake values
            original_switchlist = zcm_helper._switchlist_smoother(original_switchlist, WTS, 1, 0, 0, MODE_OR=True, OR_TARGET=1)
            #Smoothes sleep values
            original_switchlist = zcm_helper._switchlist_smoother(original_switchlist, WTS, 0, L_WINDOW, R_WINDOW)
            #Smoothes wake values
            original_switchlist = zcm_helper._switchlist_smoother(original_switchlist, WTS2, 1, 0, 0, MODE_OR=MODE_OR,
                                                                  OR_TARGET=1)
            #Smoothes sleep values
            original_switchlist = zcm_helper._switchlist_smoother(original_switchlist, WTS2, 0, 0, 0, MODE_OR=MODE_OR,
                                                                  OR_TARGET=1)

            original_constr_list = zcm_helper._construct_list_from_switchlist(original_switchlist)

            pim_to_plot, constr_list = zcm_helper._zcm_filter(pim_values, PIM_THRESHOLD,
                                                              given_constr_list=original_constr_list)

            constr_list = [c for c in constr_list if c is not None]
            sw = zcm_helper._get_switchlist(constr_list, 1)

            pim_to_plot_targets = zcm_helper._get_all_zcm_target(pim_to_plot, PIM_THRESHOLD, 0, LIMIT, False,
                                                                 given_switchlist=sw)
            if pim_to_plot_targets:
                for ind, target in enumerate(pim_to_plot_targets):
                    try:
                        pim_to_plot_temp = pim_to_plot[target[2] - target[1] + 1:target[2] + 1]
                        timestamp = timestamps[(target[2] - target[1] + 1)]
                        pim_to_plot_ten_percent = len(pim_to_plot_temp) // 10
                        pim_data_copy = pim_to_plot_temp[
                                        pim_to_plot_ten_percent: len(pim_to_plot_temp) - pim_to_plot_ten_percent]
                        return_dict = zcm_analyzer.get_characteristics(pim_data_copy, name, timestamp, split_humps)
                        return_dict["zcm_threshold"] = PIM_THRESHOLD
                        return_dict["sleep_to_wake_1"] = WTS
                        return_dict["wake_to_sleep_1"] = WTS
                        return_dict["left_window_1"] = L_WINDOW
                        return_dict["right_window_1"] = R_WINDOW
                        return_dict["sleep_to_wake_OR_1"] = WTS2
                        return_dict["wake_to_sleep_OR_1"] = WTS2
                        return_dict["timestamp_beginning"] = return_dict.pop("day")
                        list_to_write.append(return_dict)
                    except:
                        print("hiba")
                        pass

    for di in list_to_write:
        print(di)

    headers = []
    for _dict in list_to_write:
        for key, value in _dict.items():
            headers.append(key)
        break
    split_humps_str = "_quartile" if split_humps else ""
    Path(f"{outdir}").mkdir(parents=True, exist_ok=True)
    with open(f"{outdir}/{additional_filename}{split_humps_str}_adatvizsgalat.csv", 'w',
              newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(list_to_write)


if __name__ == "__main__":


    """Analyzing PIM humps for the psykose values"""
    #used for indexing inside the csv
    additional_name = "depresjson"

    additional_file_name = "0"
    _quartile = True
    PIM_THRESHOLD = 200
    LIMIT = 179
    WTS = 20
    WTS2 = 20
    MODE_OR = True
    L_WINDOW = 20
    R_WINDOW = 20

    calculate_zcm_characteristics_others_above_limit2(
    "directory_for_the_psykose_csvs",
    "out_directory_for_the_psykose_csvs", additional_name,
    additional_file_name,
    _quartile, PIM_THRESHOLD, 179, WTS, WTS2, MODE_OR, L_WINDOW, R_WINDOW, PLOT=False)



    """Analyzing ZCM humps for the szte values"""
    # used for indexing inside the csv
    additional_name = "skizo"

    additional_file_name = "0"
    _quartile = True
    ZCM_THRESHOLD = 10
    LIMIT = 179
    WTS = 20
    WTS2 = 20
    MODE_OR = True
    L_WINDOW = 20
    R_WINDOW = 20

    calculate_zcm_characteristics_others_above_limit(
        "directory_for_the_szte_csvs",
        "out_directory_for_the_szte_csvs", additional_name,
        additional_file_name,
        _quartile, ZCM_THRESHOLD, 179, WTS, WTS2, MODE_OR, L_WINDOW, R_WINDOW, PLOT=False)




