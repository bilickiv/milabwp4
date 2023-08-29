import pyedflib
from pyedflib import highlevel
import numpy as np
import csv
import os
from pathlib import Path
import copy
import sys
import matplotlib.pyplot as plt
from preprocess.resample import get_resampled_data
from preprocess.butter_bandpass import get_butter_bandpass_data
from preprocess.notch_filter import get_notch_filtered_data
from preprocess.sliding_window_average import get_aggregated_values_data
from sklearn.cluster import KMeans

"""
ch names: ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz', 'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'Status']
"""


def write_csv(data, name):
    LIST_TO_WRITE = []
    for d in data:
        dict_to_write = {"data": d}
        LIST_TO_WRITE.append(dict_to_write)

    Path(f"csvs").mkdir(parents=True, exist_ok=True)
    with open(f"csvs/{name}.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["data"])
        writer.writeheader()
        writer.writerows(LIST_TO_WRITE)


def plot_this(x_ticks, y_ticks, name, save_name, savedir="", save=False):
    plt.figure(num=None, figsize=(21, 6), dpi=150, facecolor='lightgrey', edgecolor='k')
    plt.plot(x_ticks, y_ticks, '-', linewidth=0.5, label=f"activity - resampled")
    plt.title(f"{name} Activity")
    plt.xticks(np.arange(0, len(x_ticks), 80000))
    plt.xlabel("Time Elapse")
    plt.ylabel("Activity")
    plt.grid(False)
    plt.legend()
    if save:
        if savedir:
            Path(f"{savedir}").mkdir(parents=True, exist_ok=True)
            plt.savefig(f"{savedir}/{save_name}")
        else:
            plt.savefig(f"{save_name}")
        print(f"{save_name} has been saved!")
    #plt.show()
    plt.cla()
    plt.clf()
    plt.close()


def proba():

    LIST_TO_WRITE = []
    LIST_TO_WRITE2 = []
    calcul = 0
    for file in os.listdir(dirpath):
        try:
            calcul += 1
            name = file.split(".")[0]
            print(f"{file} is in the making!")
            signals, signal_headers, header = highlevel.read_edf(f"{dirpath}/{file}", ch_names=['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz',])
            headers = ["id", "value", "startdate"]
            name = file.split(".")[0]
            list_to_write = []
        except:
            continue

        frequency = int(signal_headers[0]["sample_frequency"])

        #fájlok elején / végén van néha egy-egy nagy ugrás, ez csak levágja azokat - legyen osztható az alap frekvenciával a resample miatt (bár valószínűleg nem probléma, ha nem)
        cut_these = 128

        resampled_array = []

        print("elektródák számolása!")
        for idx, s_header in enumerate(signal_headers): #41 dolgon végigmegy
            print(f"{idx + 1}/{len(signal_headers)}. tömb elkezdve!")
            raw_data = signals[idx]


            resampled_data = get_resampled_data(raw_data, frequency, 128)
            #resampled_data = resampled_data[cut_these : len(resampled_data) - cut_these]
            resampled_array.append(resampled_data)

        print("GFP számolása!")
        n = len(resampled_array)
        gfp_array = []
        gfp_array2 = []
        print(f"len resampeld data: {len(resampled_array[0])}")
        for i in range(len(resampled_array[0])):
            electrode_powers = [r[i] ** 2 for r in resampled_array]  # Vi(t)
            sump = sum(electrode_powers)
            vt = np.sqrt((1/n) * sump)
            differences = [(ep - vt) ** 2 for ep in electrode_powers]
            gfp = np.sqrt((1/n) * sum(differences))
            gfp_array.append(gfp)

        for i in range(len(resampled_array[0])):
            electrode_powers = [r[i] for r in resampled_array]  # Vi(t)
            sump = sum(electrode_powers)
            vt = (1/n) * sump
            differences = [(ep - vt) ** 2 for ep in electrode_powers]
            gfp = np.sqrt((1/n) * sum(differences))
            gfp_array2.append(gfp)


        gfp_peaks = [gfp_array[i] for i in range(1, len(gfp_array) - 1) if gfp_array[i - 1] < gfp_array[i] > gfp_array[i + 1]]
        gfp_peaks2 = [gfp_array2[i] for i in range(1, len(gfp_array2) - 1) if gfp_array2[i - 1] < gfp_array2[i] > gfp_array2[i + 1]]

        kmeans = KMeans(n_clusters=4)
        kmeans.fit(np.array(gfp_peaks).reshape(-1, 1))

        print(f"gfp_peakek nagysága: {len(gfp_peaks)}")

        dict_to_write = {"name": name, "microstates": list(kmeans.labels_)}
        LIST_TO_WRITE.append(dict_to_write)

        kmeans2 = KMeans(n_clusters=4)
        kmeans2.fit(np.array(gfp_peaks2).reshape(-1, 1))

        print(f"gfp_peakek2 nagysága: {len(gfp_peaks2)}")

        dict_to_write2 = {"name": name, "microstates": list(kmeans2.labels_)}
        LIST_TO_WRITE2.append(dict_to_write2)


    Path(f"csvs").mkdir(parents=True, exist_ok=True)
    with open(f"csvs/microstates_sajat_eredo.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["name", "microstates"])
        writer.writeheader()
        writer.writerows(LIST_TO_WRITE)

    Path(f"csvs").mkdir(parents=True, exist_ok=True)
    with open(f"csvs/microstates_sajat_sima.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["name", "microstates"])
        writer.writeheader()
        writer.writerows(LIST_TO_WRITE2)






if __name__ == "__main__":
    dirpath = "../converter/files"
    proba()
