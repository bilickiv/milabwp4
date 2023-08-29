import mne
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import mne_microstates



def plot_difference(data):
    xticks = range(0, len(data))

    plt.figure(num=None, figsize=(16, 9), dpi=300, facecolor='lightgrey', edgecolor='k')
    plt.plot(xticks, data, '-', linewidth=0.5, label=f"State positive/negative")

    plt.title(f"Difference")
    plt.xticks(np.arange(0, len(xticks), 50000))
    plt.xlabel("Time Elapse")
    plt.ylabel("State")
    plt.grid(False)
    plt.legend()
    plt.show()
    plt.cla()
    plt.clf()
    plt.close()


def calc_polarities(data, fname, channel_names, scale_up=1000000):

    LIST_TO_WRITE = []
    for idx, d in enumerate(data):
        percentile_0 = np.round(np.min(d) * scale_up,3)
        percentile_100 = np.round(np.max(d) * scale_up,3)
        percentile_5 = np.round(np.percentile(d, 5) * scale_up,3)
        percentile_95 = np.round(np.percentile(d, 95) * scale_up,3)
        percentile_30 = np.round(np.percentile(d, 30) * scale_up,3)
        percentile_70 = np.round(np.percentile(d, 70) * scale_up,3)
        ch_name = channel_names[idx]

        dict_to_write = {}
        dict_to_write["name"] = fname
        dict_to_write["percentile_0"] = percentile_0
        dict_to_write["percentile_100"] = percentile_100
        dict_to_write["percentile_5"] = percentile_5
        dict_to_write["percentile_95"] = percentile_95
        dict_to_write["percentile_30"] = percentile_30
        dict_to_write["percentile_70"] = percentile_70
        dict_to_write["ch_name"] = ch_name

        print(dict_to_write)
        LIST_TO_WRITE.append(dict_to_write)
    fieldnames = LIST_TO_WRITE[0].keys()

    if not os.path.isfile(f"characteristics/polarity_percentiles_scaledup_{scale_up}.csv"):
        with open(f"characteristics/polarity_percentiles_scaledup_{scale_up}.csv", 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(LIST_TO_WRITE)
    else:
        with open(f"characteristics/polarity_percentiles_scaledup_{scale_up}.csv", 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerows(LIST_TO_WRITE)



def difference_in_moments(data, unique_limit=False, lower_percentile=35, upper_percentile=65,
                          unique_filter=False,
                          lower_percentile_filter=5, upper_percentile_filter=95):

    if unique_filter:
        data_temp = []
        for idx, d in enumerate(data):
            lower_limit = np.percentile(d, 5)
            upper_limit = np.percentile(d, 95)
            data_temp.append(d[(d >= lower_limit) & (d <= upper_limit)])
    else:
        data_temp = data

    percentiles = []

    if unique_limit:
        for d in data_temp:
            quartiles_dict = {}
            quartiles_dict["negative_limit"] = np.percentile(d, lower_percentile)
            quartiles_dict["positive_limit"] = np.percentile(d, upper_percentile)
            percentiles.append(quartiles_dict)
    else:
        for _ in data_temp:
            quartiles_dict = {}
            quartiles_dict["negative_limit"] = np.float64("-1e-06")
            quartiles_dict["positive_limit"] = np.float64("1e-06")
            percentiles.append(quartiles_dict)
    for idx, q in enumerate(percentiles):
        print(idx)
        print(f"negative limit = {q['negative_limit']}")
        print(f"positive limit = {q['positive_limit']}")
        print()

    data_states = []
    for idx, d in enumerate(data_temp):
        neg_lim = percentiles[idx]["negative_limit"]
        pos_lim = percentiles[idx]["positive_limit"]
        data_states.append(np.where(d <= neg_lim, -1, np.where(d >= pos_lim, 1, 0)))

    return data_states
