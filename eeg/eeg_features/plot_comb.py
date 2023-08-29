import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import csv
from pathlib import Path
from itertools import combinations


def plot_df_col_combs(t_df, outdir, cols_not_to_compare=None):
    if cols_not_to_compare is None:
        cols_not_to_compare = ["id","category"]
    t_cols = t_df.columns.tolist()
    print(f"{outdir} / {t_cols}")
    cols_combs = [c for c in t_cols if c not in cols_not_to_compare]

    _comb = set(combinations(cols_combs, 2))
    comb_list = []
    for c in _comb:
        comb_list.append(list(c))

    for ix, comb in enumerate(comb_list):
        df_t = t_df[comb]
        df_tc = pd.concat([df_t, t_df['category']], axis=1)

        fig, ax1 = plt.subplots(1, 1, figsize=(21, 12))
        ax1.set_title(f'{comb[0]} + {comb[1]} alapján')
        ax1.set_xlabel(f'{comb[0]}')
        ax1.set_ylabel(f'{comb[1]}')
        scatter = ax1.scatter(df_tc[comb[0]], df_tc[comb[1]], c=df_tc['category'], cmap='rainbow')
        legend1 = ax1.legend(*scatter.legend_elements())
        ax1.add_artist(legend1)

        Path(f"{outdir}/").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{outdir}/{ix + 1}_{comb[0]}_{comb[1]}.png")
        plt.close()

df1 = pd.read_csv("characteristics/eeg_transition_metrics.csv")
df1.drop(["id"], axis=1, inplace=True)

switches = ["AA", "BB", "CC", "DD",
            "AB", "AC", "AD", "BC", "BD", "CD",
            "BA", "CA", "DA", "CB", "DB", "DC"]

df1_switches = df1[switches]
df1_switches = pd.concat([df1_switches, df1['category']], axis=1)

df2_non_switches = df1[[c for c in df1.columns.tolist() if c not in switches]]


plot_df_col_combs(df1_switches, "plots/plot_switches_statistical")
plot_df_col_combs(df2_non_switches, "plots/plot_statistical")

def read_dfs_with_multiple_endings(filepath, outdir):

    df = pd.read_csv(filepath)

    columns = df.columns.tolist()

    unique_numbers = set([c.split("_")[-1] for c in columns if len(c.split("_")) > 1])

    df_list = []

    for num in unique_numbers:
        temp_cols = [c for c in columns if c.endswith(num)]
        temp_cols += [c for c in columns if len(c.split("_")) == 1]
        temp_df = df[temp_cols]
        df_list.append(temp_df)

    for t_df in df_list:
        plot_df_col_combs(t_df, outdir)

read_dfs_with_multiple_endings("characteristics/eeg_transition_homogenity_metric.csv", "plots/plot_homogenity")
read_dfs_with_multiple_endings("characteristics/eeg_transition_shannon_metric.csv", "plots/plot_shannon")