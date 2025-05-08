from itertools import combinations
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import csv
from pathlib import Path
import sys
from PYACTYGRAPHY_FILTERED.clustering import helper
import copy
import networkx as nx


"""This script searches for all the cliques in a heatmap of features."""



adatok = ['upper_humps_width_min_qrt',
 'min_length_between_bigger',
 'lower_humps_width_min_qrt',
 'daily_activity_mean',
 'structure_pm_stdev',
 'upper_humps_max_qrt',
 'std_upper',
 'upper_humps_width_max_qrt',
 'upper_humps_max_distance',
 'std_avg',
 'daily_activity_stdev',
 'upper_humps_min_distance_qrt',
 'lower_humps_width_min',
 'upper_humps_median_distance',
 'min_bigger_values',
 'number_of_lower_humps_qrt',
 'upper_humps_mean_qrt',
 'upper_humps_width_mean',
 '3_thrd',
 'min_bigger_values_qrt',
 'avg_length_between_bigger_qrt',
 'length_of_sleep_in_minutes',
 '2_thrd',
 'median_length_between_bigger',
 'max_length_between_bigger',
 'max_length_between_bigger_qrt',
 'M10',
 'RA',
 'upper_humps_avg_distance_qrt',
 'avg_length_between_bigger',
 'structure_pm',
 'upper_humps_width_mean_qrt',
 'upper_humps_min_qrt',
 'upper_humps_avg_distance',
 'L5',
 'first_big_hump_length',
 'length_of_sleep_in_minutes_qrt',
 'number_of_upper_humps',
 'upper_humps_min_distance',
 'upper_humps_median_distance_qrt',
 'frg_index',
 'upper_humps_median_qrt',
 'median_length_between_bigger_qrt',
 'first_big_hump_length_qrt',
 'upper_humps_max_distance_qrt',
 'last_big_hump_length_qrt',
 'zero_ratio',
 'upper_humps_width_max',
 'upper_humps_width_median',
 'IS',
 'number_of_upper_humps_qrt',
 'min_length_between_bigger_qrt',
 'std_lower',
 'upper_humps_width_median_qrt']

df = pd.read_csv("I:/Munka/Elso/project/ML/features_20200915.csv", usecols=adatok)

#df.drop(["id", "class"], inplace=True, axis=1)
#df = df.drop("Unnamed: 0", axis=1)



columns = df.columns


corr_matrix = abs(df[columns].corr())
#plt.figure(figsize=[12,10])
#sns.heatmap(corr_matrix, linewidth=0.3)



columns = df.columns.tolist()
corr_values = corr_matrix.values.tolist()
print(columns)
indexes_below_limit = []


limit = 0.3
for i in range(1, len(corr_values)):
    indexes = []
    for j in range(0, i):
        if corr_values[i][j] < limit:
            dict_to_append = {"col1": columns[i], "col2": columns[j], "value": corr_values[i][j], "limit": str(round(limit, 2))}
            indexes_below_limit.append(dict_to_append)


Path(f"slctd").mkdir(parents=True, exist_ok=True)
with open(f"slctd/slctd_correlation_20221017.csv", 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["col1", "col2", "value", "limit"])
    writer.writeheader()
    writer.writerows(indexes_below_limit)



#######################################################

_df = pd.read_csv(f"slctd/slctd_correlation_20221017.csv", usecols=["col1", "col2", "value", "limit"])
#_df = _df[(_df["limit"] >= 0.4) & (_df["limit"] <= 0.6)]

dfs = [_df[_df["limit"] == x] for x in _df.limit.unique().tolist()]

headers = []
list_to_write = []

for df in dfs:
    limit = df.limit.unique().tolist()[0]
    edges = pd.DataFrame({
        "source": df.col1.values,
        "target": df.col2.values,
        "weight": df.value.values,
    }
    )

    G = nx.from_pandas_edgelist(edges, edge_attr=True)

    complete_graphs = [g for g in nx.find_cliques(G) if len(g) > 7]
    print(len(complete_graphs))
    for g in complete_graphs:
        dict_to_write = {}
        dict_to_write["limit"] = limit
        for idx, v in enumerate(g):
            dict_to_write[f"col_{idx+1}"] = v
            if f"col_{idx+1}" not in headers:
                headers.append(f"col_{idx+1}")
        dict_to_write["feature"] = g
        list_to_write.append(dict_to_write)


print("Sorok készen")
for _dict in list_to_write:
    for header in headers:
        if header not in _dict.keys():
            _dict[header] = np.nan


headers.append("feature")
headers.append("limit")
Path(f"slctd/cliques").mkdir(parents=True, exist_ok=True)
with open(f"slctd/cliques/slctd_cliques_length_20221017.csv", 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)
    writer.writeheader()
    writer.writerows(list_to_write)



