import numpy as np
from pathlib import Path
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import ZCM.zcm_helper as zcm_helper
import copy, csv
from numpy import isnan

def count_near_big_humps(switchlist_upper, threshold=10):

    counter = 0

    for i in range(0, len(switchlist_upper) - 1):
        if (switchlist_upper[i + 1][2] - switchlist_upper[i + 1][1] + 1) - switchlist_upper[i][2] <= threshold:
            counter += 1
    return counter



def get_characteristics(zcm_data, name="", day=0, split_humps=False):
    """

    :param zcm_data: calculated zcm data
    :param name: patient's name
    :param day: which day is it
    :param split_humps: splits the humps into bigger/lower. If False, it's splits by the median, if True lower == 1st quartile, upper == 4th quartile
    :return: A dictionary with the datas
    """

    def try_except(func, data):
        try:
            return func(data)
        except:
            return 0.0


    data_length = len(zcm_data)
    _data_bigger = [z for z in zcm_data if z > 0]
    data_length_bigger = len(_data_bigger)
    try:
        percentage_of_bigger = (data_length_bigger / data_length) * 100
    except:
        percentage_of_bigger = 0.0
    avg_bigger_values = try_except(np.mean, _data_bigger)
    max_bigger_values = try_except(np.max, _data_bigger)
    min_bigger_values = try_except(np.min, _data_bigger)

    lengths_between_bigger_values = []

    zero = False
    length = -1

    for val in zcm_data:
        length += 1
        if val != 0 and zero is False:
            lengths_between_bigger_values.append(length)
            length = 0
            zero = True

        elif val == 0 and zero is True:
            zero = False

    median_length_between_bigger = try_except(np.median, lengths_between_bigger_values)
    avg_length_between_bigger = try_except(np.mean, lengths_between_bigger_values)
    max_length_between_bigger = try_except(np.max, lengths_between_bigger_values)
    min_length_between_bigger = try_except(np.min, lengths_between_bigger_values)

    if isinstance(zcm_data, list):
        switchlist = zcm_helper._get_switchlist(zcm_data, 1)
    else:
        switchlist = zcm_helper._get_switchlist(zcm_data.tolist(), 1)

    switchlist = [s for s in switchlist if s[0] != 0]

    _temp_switchlist = switchlist
    switchlist = []
    for val in _temp_switchlist:
        _array = zcm_data[val[2] - val[1] + 1 :val[2]+1]
        val = (*val , np.max(_array))
        val = (*val , np.mean(_array))
        val = (*val , len(_array))
        switchlist.append(val)

    v3_s = [v[3] for v in switchlist]
    median = np.median(v3_s)


    if not split_humps:
        lower_humps = [v for v in switchlist if v[4] < median]
        upper_humps = [v for v in switchlist if v[4] >= median]
    else:
        lower_humps = [v for v in switchlist if v[4] <= np.quantile(v3_s, 0.25)]
        upper_humps = [v for v in switchlist if v[4] >= np.quantile(v3_s, 0.75)]

    if len(lower_humps) == 0:
        lower_humps = [(0,0,0,0,0,0)]

    if len(upper_humps) == 0:
        upper_humps = [(0,0,0,0,0,0)]

    first_big_hump_length = 0
    last_big_hump_length = 0
    if len(upper_humps) > 0:
        first_big_hump_length = upper_humps[0][5]
        last_big_hump_length = upper_humps[-1][5]

    close_big_humps_counter = count_near_big_humps(upper_humps, threshold=10)

    upper_humps_median = np.median([v[4] for v in upper_humps])
    upper_humps_mean = np.mean([v[4] for v in upper_humps])
    upper_humps_max = np.max([v[4] for v in upper_humps])
    upper_humps_min = np.min([v[4] for v in upper_humps])
    lower_humps_median = np.median([v[4] for v in lower_humps])
    lower_humps_mean = np.mean([v[4] for v in lower_humps])
    lower_humps_max = np.max([v[4] for v in lower_humps])
    lower_humps_min = np.min([v[4] for v in lower_humps])

    upper_humps_width_median = np.median([v[5] for v in upper_humps])
    upper_humps_width_mean = np.mean([v[5] for v in upper_humps])
    upper_humps_width_max = np.max([v[5] for v in upper_humps])
    upper_humps_width_min = np.min([v[5] for v in upper_humps])
    lower_humps_width_median = np.median([v[5] for v in lower_humps])
    lower_humps_width_mean = np.mean([v[5] for v in lower_humps])
    lower_humps_width_max = np.max([v[5] for v in lower_humps])
    lower_humps_width_min = np.min([v[5] for v in lower_humps])

    lower_humps_avg_distance = []
    for idx, val in enumerate(lower_humps):
        if 0 < idx < len(lower_humps):
            lower_humps_avg_distance.append(val[2] - lower_humps[idx-1][2])
    lower_humps_median_distance = np.median(lower_humps_avg_distance)
    lower_humps_avg_distance = np.mean(lower_humps_avg_distance)
    lower_humps_max_distance = np.max(lower_humps_avg_distance)
    lower_humps_min_distance = np.min(lower_humps_avg_distance)

    upper_humps_avg_distance = []
    for idx, val in enumerate(upper_humps):
        if 0 < idx < len(upper_humps):
            upper_humps_avg_distance.append(val[2] - upper_humps[idx - 1][2])

    upper_humps_median_distance = np.median(upper_humps_avg_distance)
    upper_humps_avg_distance = np.mean(upper_humps_avg_distance)
    upper_humps_max_distance = np.max(upper_humps_avg_distance)
    upper_humps_min_distance = np.min(upper_humps_avg_distance)

    number_of_lower_humps = len(lower_humps)
    number_of_upper_humps = len(upper_humps)

    dict_to_write = {}

    dict_to_write[f"name"] = name
    dict_to_write[f"day"] = day


    if not split_humps:
        dict_to_write[f"length_of_sleep_in_minutes"] = data_length
        dict_to_write[f"data_length_bigger"] = data_length_bigger
        dict_to_write[f"percentage_of_bigger"] = percentage_of_bigger
        dict_to_write[f"avg_bigger_values"] = avg_bigger_values
        dict_to_write[f"max_bigger_values"] = max_bigger_values
        dict_to_write[f"min_bigger_values"] = min_bigger_values
        dict_to_write[f"median_length_between_bigger"] = median_length_between_bigger
        dict_to_write[f"avg_length_between_bigger"] = avg_length_between_bigger
        dict_to_write[f"max_length_between_bigger"] = max_length_between_bigger
        dict_to_write[f"min_length_between_bigger"] = min_length_between_bigger

    dict_to_write[f"upper_humps_median"] = upper_humps_median
    dict_to_write[f"upper_humps_mean"] = upper_humps_mean
    dict_to_write[f"upper_humps_max"] = upper_humps_max
    dict_to_write[f"upper_humps_min"] = upper_humps_min
    dict_to_write[f"upper_humps_width_median"] = upper_humps_width_median
    dict_to_write[f"upper_humps_width_mean"] = upper_humps_width_mean
    dict_to_write[f"upper_humps_width_max"] = upper_humps_width_max
    dict_to_write[f"upper_humps_width_min"] = upper_humps_width_min
    dict_to_write[f"upper_humps_median_distance"] = upper_humps_median_distance
    dict_to_write[f"upper_humps_avg_distance"] = upper_humps_avg_distance
    dict_to_write[f"upper_humps_max_distance"] = upper_humps_max_distance
    dict_to_write[f"upper_humps_min_distance"] = upper_humps_min_distance
    dict_to_write[f"lower_humps_median"] = lower_humps_median
    dict_to_write[f"lower_humps_mean"] = lower_humps_mean
    dict_to_write[f"lower_humps_max"] = lower_humps_max
    dict_to_write[f"lower_humps_min"] = lower_humps_min
    dict_to_write[f"lower_humps_width_median"] = lower_humps_width_median
    dict_to_write[f"lower_humps_width_mean"] = lower_humps_width_mean
    dict_to_write[f"lower_humps_width_max"] = lower_humps_width_max
    dict_to_write[f"lower_humps_width_min"] = lower_humps_width_min
    dict_to_write[f"lower_humps_max_distance"] = lower_humps_max_distance
    dict_to_write[f"lower_humps_min_distance"] = lower_humps_min_distance
    dict_to_write[f"lower_humps_median_distance"] = lower_humps_median_distance
    dict_to_write[f"lower_humps_avg_distance"] = lower_humps_avg_distance
    dict_to_write[f"number_of_lower_humps"] = number_of_lower_humps
    dict_to_write[f"number_of_upper_humps"] = number_of_upper_humps
    dict_to_write[f"number_of_close_big_humps"] = close_big_humps_counter
    dict_to_write[f"first_big_hump_length"] = first_big_hump_length
    dict_to_write[f"last_big_hump_length"] = last_big_hump_length

    for key, value in dict_to_write.items():
        if pd.isna(value) is True:
            dict_to_write[key] = 0
    return dict_to_write


def calculate_dimensions(number):
    width = 1
    height = 1

    iteration = 0

    while width * height < number:
        if iteration == 0:
            iteration = 1
            width += 1
        else:
            iteration = 0
            height += 1

    return width, height


def plot_characteristics(dir_path, outdir):
    """Plots the characteristics"""
    for _file in os.listdir(dir_path):
        if _file.endswith("characteristics.csv"):
            full_df = pd.read_csv(f"{dir_path}/{_file}")

            names = full_df.name.unique().tolist()

            columns = full_df.columns
            columns = [c for c in columns if c != "name" and c != "day" and not c.startswith("data_length")]

            for name in names:
                print(f"{name}!")
                df = full_df[full_df["name"] == name]
                days = df.day.unique().tolist()
                avg_df = df[df["day"] == 0]
                avg_df_without_0 = df[df["day"] != 0]

                WIDTH, HEIGHT = calculate_dimensions(len(columns))
                subplot_to_hide = WIDTH * HEIGHT - len(columns)

                fig, AXESES = plt.subplots(WIDTH, HEIGHT, figsize=(WIDTH*10, HEIGHT*5), dpi=50, facecolor="white", edgecolor="k")

                global_index = 0
                _days = [str(d) if d != 0 else "Aggregated" for d in days ]

                colors = []
                for d in days:
                    colors.append("blue")
                colors[len(colors) - 1] = "red"

                for AX in AXESES:
                    for i_idx, ax in enumerate(AX):
                        if global_index < len(columns):
                            ax.set_title(f"{name}")
                            ax.set_ylabel(f"{columns[global_index]}")
                            ax.set_xticks(np.arange(0, len(days), 1))
                            ax.set_xlabel(f'Day')
                            y_list = []

                            for d in days:
                                _temp_df = df[df["day"] == d]
                                y_list.append(_temp_df[columns[global_index]].values.tolist())
                            y_list = [y[0] for y in y_list]
                            l1 = ax.bar(_days, y_list, color=colors)


                        else:
                            ax.axis('off')
                        global_index += 1

                fig.tight_layout()
                Path(f"{outdir}/barplots_tied").mkdir(parents=True, exist_ok=True)
                plt.savefig(dpi=150, fname=f"{outdir}/barplots_tied/{name}_barplot.png")
                plt.cla()
                plt.clf()
                plt.close()
                for c in columns:
                    Path(f"{outdir}/barplots/{c}").mkdir(parents=True, exist_ok=True)

                    ax = sns.barplot(data=df, x="day", y=c,)
                    plt.savefig(dpi=300, fname=f"{outdir}/barplots/{c}/{name}_barplot_{c}.png")
                    plt.close()




def calculate_zcm_characteristics_others_above_limit(dir_path,outdir, additional_name: str="", additional_filename: str="", split_humps=False, ZCM_THRESHOLD=500, LIMIT=239):
    "Calculates zcm characteristics for the other datasets not for ours!"
    list_to_write = []
    # dir_path = "I:\\Munka\\Elso\\tovabbi_csvk\\psykose\\control"

    filtered_days_path = "I:\Munka\Elso\project\RAW_VALUES\RAW\proper_raw_days.csv"
    filtered_days_df = pd.read_csv(filtered_days_path)

    for _file in os.listdir(dir_path):
        print(f"{dir_path}/{_file} is in the making! {split_humps}")
        if _file.endswith(".csv"):
            name = additional_name + "_" + _file.split(".")[0]
            df = pd.read_csv(f"{dir_path}/{_file}")
            timestamps = df.timestamp.values.tolist()
            #zcm_values = df["activity"].tolist()


            f_name = "".join(name[1:6])
            f_d_list = filtered_days_df[f_name].values.tolist()
            f_d_list = [int(d) for d in f_d_list if isnan(d) == False]
            zcm_values = []
            for fd in f_d_list:
                asd = df["zcm_value"].tolist()[(fd-1)*1440: fd*1440]
                for dv in asd:
                    zcm_values.append(dv)

            zcm_data_aggregated = zcm_helper.aggregate_zcm_values(zcm_values, 5, 5)

            ZCM_THRESHOLD = 5
            zcm_to_plot, constr_list = zcm_helper._zcm_filter(zcm_data_aggregated, ZCM_THRESHOLD)
            zcm_to_plot_targets = zcm_helper._get_all_zcm_target(zcm_to_plot, ZCM_THRESHOLD, 0, LIMIT, True)
            zcm_values_conjugated = []
            if zcm_to_plot_targets:
                for target in zcm_to_plot_targets:
                    try:
                        if target[1] < 70000:
                            zcm_to_plot_temp = zcm_to_plot[target[2] - target[1] + 1:target[2] + 1]
                            timestamp = timestamps[target[2] - target[1] + 1]
                            zcm_to_plot_ten_percent = len(zcm_to_plot_temp) // 10
                            zcm_data_copy = zcm_to_plot_temp[zcm_to_plot_ten_percent: len(zcm_to_plot_temp) - zcm_to_plot_ten_percent]
                            return_dict = get_characteristics(zcm_data_copy, name, timestamp, split_humps)
                            return_dict["zcm_threshold"] = ZCM_THRESHOLD
                            return_dict["timestamp_beginning"] = return_dict.pop("day")
                            list_to_write.append(return_dict)
                            for val in zcm_data_copy:
                                zcm_values_conjugated.append(val)
                    except:
                        continue
                    #if target[1] > 600 and target[1] < 700:
                        #switch_list = zcm_helper._get_switchlist(zcm_data_copy, threshold=ZCM_THRESHOLD)
                        #plot_shit(zcm_data_copy, switch_list, timestamp, name)




                return_dict_ = get_characteristics(zcm_values_conjugated, name, 0, split_humps)
                return_dict_["timestamp_beginning"] = return_dict_.pop("day")
                return_dict_["zcm_threshold"] = ZCM_THRESHOLD

                def make_avg(c_key):
                    dl = return_dict_[c_key]
                    return_dict_[c_key] = dl / len(zcm_to_plot_targets)

                make_avg("number_of_lower_humps")
                make_avg("number_of_upper_humps")
                make_avg("number_of_close_big_humps")
                list_to_write.append(return_dict_)

    for di in list_to_write:
        print(di)
    headers = []
    for _dict in list_to_write:
        for key, value in _dict.items():
            headers.append(key)
        break
    split_humps_str = "_quartile" if split_humps else ""
    Path(f"{outdir}").mkdir(parents=True, exist_ok=True)
    with open(f"{outdir}/characteristics_2_{additional_filename}_{split_humps_str}_ver2.csv", 'w',
              newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(list_to_write)

    df = pd.read_csv(f"{outdir}/characteristics_2_{additional_filename}_{split_humps_str}_ver2.csv")

    names = df.name.unique()

    full_df = pd.DataFrame()

    for n in names:
        _temp_df = df[df["name"] == n]

        fbhl = _temp_df["first_big_hump_length"].tolist()
        if len(fbhl) > 1:
            del fbhl[-1]
            last = sum(fbhl) / len(fbhl)
            fbhl.append(last)
            _temp_df["first_big_hump_length"] = fbhl

        lbhl = _temp_df["last_big_hump_length"].tolist()
        if len(lbhl) > 1:
            del lbhl[-1]
            last = sum(lbhl) / len(lbhl)
            lbhl.append(last)
            _temp_df["last_big_hump_length"] = lbhl

        if full_df.shape[0] > 0:
            full_df = pd.concat([full_df, _temp_df], axis=0)
        else:
            full_df = _temp_df

    full_df.to_csv(f"{outdir}/characteristics_2_{additional_filename}_{split_humps_str}_ver2.csv")



def calculate_zcm_characteristics_others(dir_path,outdir, additional_name: str="", additional_filename: str="", split_humps=False, ZCM_THRESHOLD=500):
    "Calculates zcm characteristics for the other datasets not for ours!"
    list_to_write = []
    #dir_path = "I:\\Munka\\Elso\\tovabbi_csvk\\psykose\\control"
    for _file in os.listdir(dir_path):
        print(f"{dir_path}/{_file} is in the making! {split_humps}")
        if _file.endswith(".csv"):
            name = additional_name + "_" +  _file.split(".")[0]
            df = pd.read_csv(f"{dir_path}/{_file}")
            days = df.date.unique().tolist()
            zcm_values_conjugated = []

            for day in days:
                filtered_df = df[df["date"] == day]
                zcm_data_copy = copy.deepcopy(filtered_df["activity"])
                zcm_data_copy = zcm_helper.aggregate_zcm_values(zcm_data_copy.tolist(), 5, 5)
                try:
                    zcm_to_plot, constr_list = zcm_helper._zcm_filter(zcm_data_copy, ZCM_THRESHOLD, )
                    zcm_to_plot_target = zcm_helper._get_longest_zcm_target(zcm_to_plot, ZCM_THRESHOLD)
                    zcm_to_plot = zcm_to_plot[zcm_to_plot_target[2] - zcm_to_plot_target[1] + 1:zcm_to_plot_target[2] + 1]

                    zcm_to_plot_ten_percent = len(zcm_to_plot) // 10
                    zcm_data_copy = zcm_to_plot[zcm_to_plot_ten_percent: len(zcm_to_plot) - zcm_to_plot_ten_percent]

                    return_dict = get_characteristics(zcm_data_copy, name, day, split_humps)
                    return_dict["zcm_threshold"] = ZCM_THRESHOLD
                    list_to_write.append(return_dict)
                    #switch_list = zcm_helper._get_switchlist(zcm_data_copy, threshold=ZCM_THRESHOLD)
                    # plot_shit(zcm_data_copy, switch_list, day, name)
                except:
                    continue



                for val in zcm_data_copy:
                    zcm_values_conjugated.append(val)

            return_dict_ = get_characteristics(zcm_values_conjugated, name, 0, split_humps)
            def make_avg(c_key):
                dl = return_dict_[c_key]
                return_dict_[c_key] = dl/len(days)

            make_avg("number_of_lower_humps")
            make_avg("number_of_upper_humps")
            make_avg("number_of_close_big_humps")
            list_to_write.append(return_dict_)

    headers = []
    for _dict in list_to_write:

        for key, value in _dict.items():
            headers.append(key)
        break
    split_humps_str = "_quartile" if split_humps else ""
    Path(f"{outdir}").mkdir(parents=True, exist_ok=True)
    with open(f"{outdir}/characteristics_2_{additional_filename}_{split_humps_str}.csv", 'w',
              newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(list_to_write)


    df = pd.read_csv(f"{outdir}/characteristics_2_{additional_filename}_{split_humps_str}.csv")

    names = df.name.unique()

    full_df = pd.DataFrame()

    for n in names:
        _temp_df = df[df["name"] == n]

        fbhl = _temp_df["first_big_hump_length"].tolist()
        if len(fbhl) > 1:
            del fbhl[-1]
            last = sum(fbhl)/len(fbhl)
            fbhl.append(last)
            _temp_df["first_big_hump_length"] = fbhl

        lbhl = _temp_df["last_big_hump_length"].tolist()
        if len(lbhl) > 1:
            del lbhl[-1]
            last = sum(lbhl) / len(lbhl)
            lbhl.append(last)
            _temp_df["last_big_hump_length"] = lbhl

        if full_df.shape[0] > 0:
            full_df = pd.concat([full_df, _temp_df], axis=0)
        else:
            full_df = _temp_df

    full_df.to_csv(f"{outdir}/characteristics_2_{additional_filename}_{split_humps_str}.csv")


def calculate_rhythym(timestamps, freq='D'):

    if freq == 'D':
        freq_ts = {t.floor(freq) for t in timestamps}
        timestamps2 = [t.floor(freq) for t in timestamps]
    elif freq == 'H':
        freq_ts = {t.floor(freq).hour for t in timestamps}
        timestamps2 = [t.floor(freq).hour for t in timestamps]
    else:
        print("Wrong 'freq' keyword!")
        return {}

    return_dict = {}
    for fr in freq_ts:
        counter = 0
        for t in timestamps2:
            if t == fr:
                counter += 1
        return_dict[str(fr)] = counter
    return return_dict

def calculate_rhythym_full_hour(timestamps, lengths, freq='D'):

    if freq == 'H':
        freq_ts = {t.floor(freq).hour for t in timestamps}
        timestamps2 = [t.floor(freq).hour for t in timestamps]
        return_dict = {}
        for i in range(24):
            return_dict[str(i)] = 0
    else:
        print("Wrong 'freq' keyword!")
        return {}

    for idx, ts in enumerate(timestamps2):
        len_hour = lengths[idx] // 60
        for fr in freq_ts:
            if fr == ts:
                num = return_dict[str(fr)]
                num += 1
                return_dict[str(fr)] = num

                if len_hour > 0:
                    h_counter = 0
                    while h_counter != len_hour:
                        h_counter += 1
                        new_h = fr + h_counter
                        new_h = new_h if new_h < 24 else new_h - 24
                        num = return_dict[str(new_h)]
                        num += 1
                        return_dict[str(new_h)] = num
    return return_dict
def count_sleep_cycles():

    df1 = pd.read_csv(f"OTHER_DATASETS/DEPRESJON_FEATURES/condition/characteristics_2_depresjson_condition__ver2.csv")
    df2 = pd.read_csv(f"OTHER_DATASETS/PSYKOSE_FEATURES/patient/characteristics_2_psykose_patient__ver2.csv")
    df3 = pd.read_csv(f"OTHER_DATASETS/PSYKOSE_FEATURES/control/characteristics_2_psykose_control__ver2.csv")
    df = pd.DataFrame(pd.concat([df1, df2, df3], axis=0))

    df = df[df["timestamp_beginning"] != "0"]
    names = df.name.unique().tolist()
    LIST_TO_WRITE = []
    KEYS_TO_WRITE = []
    for name in names:
        print(name)
        temp_df = df[df["name"] == name]
        timestamps = temp_df["timestamp_beginning"].values.tolist()
        timestamps = [pd.to_datetime(t) for t in timestamps]
        sleep_lengths = temp_df['length_of_sleep_in_minutes'].values.tolist()
        sleep_lengths = [int(k) for k in sleep_lengths]


        ###########################################################
        #calculate how many times patient slept each day - probably not needed
        day_dict = calculate_rhythym(timestamps, "D")

        ###########################################################
        # calculate how many times patient slept each hour
        hour_dict = calculate_rhythym_full_hour(timestamps, sleep_lengths, "H")


        #calculate sleep timedelta features
        tsl = [(timestamps[i], sleep_lengths[i]) for i in range(len(timestamps))]
        tsl.sort(key=lambda e: e[0])
        timestamps_differences = [(tsl[i+1][0] - (tsl[i][0] + pd.Timedelta(seconds=tsl[i][1]))).total_seconds() for i in
                                  range(len(timestamps) - 1)]
        avg_ts_difference = round(sum(timestamps_differences) / len(timestamps_differences), 0)
        std_ts_difference = round(np.std(timestamps_differences), 3)
        median_ts_difference = round(np.median(timestamps_differences), 3)

        dict_to_write = {}
        dict_to_write["name"] = name

        #dict_to_write = dict(dict_to_write, **day_dict)
        #dict_to_write.update(dict_to_write)
        dict_to_write = dict(dict_to_write, **hour_dict)
        dict_to_write.update(dict_to_write)

        dict_to_write['avg_diff_between_sleeps_sec'] = avg_ts_difference
        dict_to_write['std_diff_between_sleeps_sec'] = std_ts_difference
        dict_to_write['median_diff_between_sleeps_sec'] = median_ts_difference

        LIST_TO_WRITE.append(dict_to_write)

        for key, item in dict_to_write.items():
            if key not in KEYS_TO_WRITE:
                KEYS_TO_WRITE.append(key)

    #so each dict will contain the same keys, if it didn't contain previously then it's value will be 0
    for dic in LIST_TO_WRITE:
        for _k in KEYS_TO_WRITE:
            if _k not in dic.keys():
                dic[_k] = 0

    #sort the keys so the hours are the first
    KEYS_TO_WRITE_SORTED = []
    counter = 0
    for _k in copy.deepcopy(KEYS_TO_WRITE):
        counter += 1
        try:
            if _k == "name":
                KEYS_TO_WRITE.remove(_k)
            else:
                KEYS_TO_WRITE_SORTED.append(int(_k))
                KEYS_TO_WRITE.remove(_k)
        except:
            continue
    KEYS_TO_WRITE_SORTED.sort()
    KEYS_TO_WRITE_SORTED = [str(k) for k in KEYS_TO_WRITE_SORTED]
    KEYS_TO_WRITE_SORTED = KEYS_TO_WRITE_SORTED + KEYS_TO_WRITE
    KEYS_TO_WRITE_SORTED.insert(0, "name")

    Path(f"OTHER_DATASETS/sleep_features").mkdir(parents=True, exist_ok=True)
    with open(f"OTHER_DATASETS/sleep_features/full_sleep_cycles.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=KEYS_TO_WRITE_SORTED)
        writer.writeheader()
        writer.writerows(LIST_TO_WRITE)