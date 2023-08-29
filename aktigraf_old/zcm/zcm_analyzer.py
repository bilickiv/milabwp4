import numpy as np
from pathlib import Path
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import ZCM.zcm_helper as zcm_helper


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
    _data_bigger = [z for z in zcm_data if z > 1]
    data_length_bigger = len(_data_bigger)
    percentage_of_bigger = (data_length_bigger / data_length) * 100
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
