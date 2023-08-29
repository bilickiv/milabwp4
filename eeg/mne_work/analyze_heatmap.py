import mne
import numpy as np
import matplotlib.pyplot as plt
import mne_microstates
import mne_work.analyzer as analyzer
import os
import pandas as pd
import seaborn as sns

HEATMAP_DICTS = [{},{},{}]
counters = [0, 0, 0]

channel_names = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3',
 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6',
  'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz']

category_df = pd.read_csv("categories.csv")

for file in os.listdir("characteristics/heatmaps"):

    name = file.split("_")[0]
    category = category_df[name].values[0] - 1
    HEATMAP_DICT = HEATMAP_DICTS[category]
    counter = counters[category - 1]

    df = pd.read_csv(f"characteristics/heatmaps/{file}")
    df.drop(["Unnamed: 0"], axis=1, inplace=True)
    corr_vals = df.values

    inner_counter = 0
    for i in range(len(channel_names) - 1):
        for j in range(i+1, len(channel_names)):
            key_name = f"{channel_names[i]}_{channel_names[j]}"
            if i != j:
                if counter == 0:
                    HEATMAP_DICT[key_name] = [corr_vals[i][j]]
                    counters[category - 1] = counter + 1
                else:
                    corr_list = HEATMAP_DICT[key_name]
                    corr_list.append(corr_vals[i][j])
                    HEATMAP_DICT[key_name] = corr_list
                inner_counter += 1
    counter += 1

STD_DICT = {}
for i in range(len(channel_names) - 1):
    for j in range(i + 1, len(channel_names)):
        key_name = f"{channel_names[i]}_{channel_names[j]}"

        std_vals = []
        for hd in HEATMAP_DICTS:
            std_vals.append(np.std(hd[key_name]))
        STD_DICT[key_name] = std_vals



df_std = pd.DataFrame(STD_DICT)
df_std["category"] = df_std.index
stds = df_std.std()


stds = stds.sort_values(ascending=False)
first_10 = stds.keys()[1:40]

dict_to_Df = {}
for idx, fir in enumerate(first_10):
    dict_val = df_std[fir].tolist()
    dict_val.append(stds[idx+1])
    dict_to_Df[fir] = dict_val
    print(dict_to_Df[fir])

dict_to_Df["category"] = [0,1,2,"std_of_stds"]

df_stds = pd.DataFrame(dict_to_Df)

df_stds.to_csv("best_stds.csv")