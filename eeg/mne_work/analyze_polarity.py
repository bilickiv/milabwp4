import mne
import numpy as np
import matplotlib.pyplot as plt
import mne_microstates
import mne_work.analyzer as analyzer
import os
import pandas as pd
import seaborn as sns

def idk(asd):
 if asd == "sz_hr":
  return 3
 elif asd == "bp_hr":
  return 2
 elif asd == "normal":
  return 1
 return 0

df2 = pd.read_csv("../Demográfiai adatok_2019_12_10_Anita.xlsx - Demográfiai adatok.csv")
df2["category"] = df2["csoport"].apply(idk)
dict_c_cs = {}
for index, row in df2.iterrows():
 dict_c_cs[row["Kód"].replace("E", "e")] = row["category"]

channel_names = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3',
 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6',
  'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz']

df = pd.read_csv("characteristics/polarity_percentiles_scaledup_1000000.csv")
df["category"] = df["name"].apply(lambda e: dict_c_cs[e.split("_")[0]])



columns = df.columns
percentiles = [c for c in columns if "percentile" in c]

for p in percentiles:
 df[p] = pd.to_numeric(df[p], errors='coerce')

for ch in channel_names:
 fig, AXESES = plt.subplots(1, 3, figsize=(24, 9), dpi=280,
                            facecolor="white",
                            edgecolor="k", )
 counter = 0
 df_to_plot = df[df["ch_name"] == ch]

 for AX in AXESES:

  AX.set_title(f'{ch}\nPolarities\n')
  AX.set_xticks(np.linspace(df[percentiles[0]].min(), df[percentiles[0]].max(), num=30))
  AX.set_yticks(np.linspace(df[percentiles[1]].min(), df[percentiles[1]].max(), num=30))
  AX.tick_params(axis="x", rotation=45)
  sns.scatterplot(data=df_to_plot, x=percentiles[counter], y=percentiles[counter+1], hue="category", ax=AX,)
  #sns.histplot(data=df_to_plot, x=percentiles[counter], hue="category", ax=AX)
  AX.grid()
  counter += 2

 plt.savefig(f"characteristics_plots/polarity_scatter/{ch}_polarity_percentiles.png")
