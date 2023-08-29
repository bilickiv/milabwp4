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

cols_new = ["daily_activity_mean", "first_big_hump_length_y",
            "upper_humps_median_distance_x",
            "first_big_hump_length_x",
            "zero_ratio", "RA_linear_val",
            "std_avg_linear_val",
            "std_lower_rank_empiric"]

df = pd.read_csv("dep_scaled_features.csv")

df.drop(["Unnamed: 0", "class", 'name'], inplace=True, axis=1)
columns = df.columns



corr_matrix = abs(df[columns].corr())
plt.figure(figsize=[12,10])
sns.heatmap(corr_matrix, linewidth=0.3)
#Path(f"kepek").mkdir(parents=True, exist_ok=True)
#plt.savefig("kepek/heatmap_kiindulo.png",dpi=300)
plt.close()

corr_values = corr_matrix.values.tolist()
corr_indexes = []
limit = 0.6

for i in range(1, len(corr_values)):
    for j in range(0, i):
        dict_to_append = {"col1": columns[i], "col2": columns[j], "value": corr_values[i][j]}
        corr_indexes.append(dict_to_append)



Path(f"dep/kepek/csvk").mkdir(parents=True, exist_ok=True)
with open(f"dep/kepek/csvk/alap_corr_matrix.csv", 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["col1", "col2", "value"])
    writer.writeheader()
    writer.writerows(corr_indexes)

df_corr = pd.read_csv("dep/kepek/csvk/alap_corr_matrix.csv")
#df_corr = df_corr[(df_corr["value"] <= 0.7)]
import networkx as nx
from matplotlib.pyplot import figure



# ALAP GRÁF KIRAJZOLÁSA
edges = pd.DataFrame({
    "source": df_corr.col1.values,
    "target": df_corr.col2.values,
    "weight": [d * 3 for d in df_corr.value.values],
}
)



figure(figsize=(10, 10), dpi=250)
G = nx.from_pandas_edgelist(edges, edge_attr=True)
weights = nx.get_edge_attributes(G,'weight').values()
pos = nx.circular_layout(G)
nx.draw(G, pos, with_labels=True, width=list(weights))
labels = nx.get_edge_attributes(G,'weight')
#nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)

plt.savefig("dep/kepek/teljes_graf_kiindulo.png")
plt.close()



# GRÁF NAGY ÉLEK NÉLKÜL
df_corr = df_corr[(df_corr["value"] <= 0.7)]
edges = pd.DataFrame({
    "source": df_corr.col1.values,
    "target": df_corr.col2.values,
    "weight": [d * 3 for d in df_corr.value.values],
}

)


figure(figsize=(10, 10), dpi=250)
G = nx.from_pandas_edgelist(edges, edge_attr=True)
weights = nx.get_edge_attributes(G,'weight').values()
pos = nx.circular_layout(G)
nx.draw(G, pos, with_labels=True, width=list(weights))

labels = nx.get_edge_attributes(G,'weight')
#nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)

plt.savefig("dep/kepek/teljes_graf_kis_sulyuak.png")
plt.close()




complete_graphs = [g for g in nx.find_cliques(G) if len(g) > 5]
G2 = nx.Graph()

import copy

df_corr_2 = copy.deepcopy(df_corr)

df_corr_2["color"] = ["b" for i in range(0, df_corr_2.shape[0])]

for attr in complete_graphs[0]:
    for attr2 in complete_graphs[0]:
        if attr != attr2:
            df_corr_2.loc[(df_corr_2["col1"] == attr) & (df_corr_2["col2"] == attr2), ["color"]] = "r"


# GRÁF SZíNEZETT ÉLEKKEL
edges = pd.DataFrame({
    "source": df_corr_2.col1.values,
    "target": df_corr_2.col2.values,
    "weight": [d * 3 for d in df_corr_2.value.values],
    "color": df_corr_2.color.values
}
)


figure(figsize=(10, 10), dpi=250)
G = nx.from_pandas_edgelist(edges, edge_attr=True)
weights = nx.get_edge_attributes(G,'weight').values()
colors = nx.get_edge_attributes(G, "color").values()
pos = nx.circular_layout(G)
nx.draw(G, pos, edge_color=colors, with_labels=True, width=list(weights))

labels = nx.get_edge_attributes(G,'weight')
#nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)

plt.savefig("dep/kepek/teljes_graf_kis_sulyuak_szinezve.png")
plt.close()