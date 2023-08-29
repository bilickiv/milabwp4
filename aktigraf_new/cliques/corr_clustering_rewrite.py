from itertools import combinations
import pandas as pd
import numpy as np
import csv
from pathlib import Path
import sys
import networkx as nx


def make_corr(df_path, date_str, min_lim, max_lim, step_lim, cc=2, outdir="dep/slctd"):
    """
    Calculates correlation matrix (lower triangular part) below <limits> for the make_cliques function.
    :param df_path: .csv to calculate the filtered corr_matrix from
    :param date_str: String to differentiate the result file. It is used for file naming purposes only.
    :param min_lim: Minimum limit for the values in the corr matrix.
    :param max_lim: Maximum limit for the values in the corr matrix.
    :param step_lim: Stepping limit. Used for the for loop.
    For example if set_limit is set to 50,
    then it calculates all the correlations at every 0.05 limit between min_lim, max_lim.
    :param cc: Which class' values to keep besides the one labeled 0.
    :param outdir: Directory to write the result files.
    :return: None
    """
    """This script searches for all the cliques in a heatmap of features."""

    classes_to_drop = ["id", "class", "name", "group", "Unnamed: 0", 'timestamp_beginning', 'zcm_threshold', "wake_time"]

    df = pd.read_csv(df_path)
    #df = df[(df["name"].str[5] == "1") | (df["name"].str[5] == cc)]
    df = df[(df["class"] == 1) | (df["class"] == cc)]
    columns = df.columns

    for cla in classes_to_drop:
        if cla in columns:
            df.drop([cla], inplace=True, axis=1)

    #scaler = RobustScaler()
    #df[df.columns] = scaler.fit_transform(df[df.columns])
    print(df)
    columns = df.columns
    corr_matrix = abs(df[columns].corr())
    #plt.figure(figsize=[12,10])
    #sns.heatmap(corr_matrix, linewidth=0.3)

    corr_matrix.to_csv(f"korr_matrix{date_str}.csv")
    columns = df.columns.tolist()
    corr_values = corr_matrix.values.tolist()
    indexes_below_limit = []

    for lim in range(min_lim, max_lim, step_lim):
        limit = round(lim/1000, 2)
        for i in range(1, len(corr_values)):
            for j in range(0, i):
                if corr_values[i][j] < limit:
                    dict_to_append = {"col1": columns[i], "col2": columns[j], "value": corr_values[i][j], "limit": str(round(limit, 2))}
                    indexes_below_limit.append(dict_to_append)

    Path(f"{outdir}").mkdir(parents=True, exist_ok=True)
    with open(f"{outdir}/slctd_correlation_{date_str}.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["col1", "col2", "value", "limit"])
        writer.writeheader()
        writer.writerows(indexes_below_limit)




def make_corr2(df_path, date_str, min_lim, max_lim, step_lim,  outdir=r"I:\Munka\Masodik\eeg_munka\characteristics\sajat_absolute\smoothed\filtered_corr_matrices\corr_matrices"):
    """
    Calculates correlation matrix (lower triangular part) below <limits> for the make_cliques function. Use this, if the .csv at @df_path is a corr matrix itself.
    :param df_path: .csv to calculate the filtered corr_matrix from
    :param date_str: String to differentiate the result file. It is used for file naming purposes only.
    :param min_lim: Minimum limit for the values in the corr matrix.
    :param max_lim: Maximum limit for the values in the corr matrix.
    :param step_lim: Stepping limit. Used for the for loop.
    For example if set_limit is set to 50,
    then it calculates all the correlations at every 0.05 limit between min_lim, max_lim.
    :param outdir: Directory to write the result files.
    :return: None
    """
    classes_to_drop = ["id", "class", "name", "group", "Unnamed: 0", 'timestamp_beginning', 'zcm_threshold', ]

    df = pd.read_csv(df_path, index_col=0)
    columns = df.columns
    for cla in classes_to_drop:
        if cla in columns:
            df.drop([cla], inplace=True, axis=1)
            df.drop([cla], inplace=True, axis=0)


    corr_matrix = df

    corr_matrix.to_csv(f"korr_matrix{date_str}.csv")
    columns = df.columns.tolist()
    corr_values = corr_matrix.values.tolist()
    indexes_below_limit = []

    for lim in range(min_lim, max_lim, step_lim):
        limit = round(lim / 1000, 2)
        for i in range(1, len(corr_values)):
            for j in range(0, i):
                if corr_values[i][j] < limit:
                    dict_to_append = {"col1": columns[i], "col2": columns[j], "value": corr_values[i][j],
                                      "limit": str(round(limit, 2))}
                    indexes_below_limit.append(dict_to_append)
    print(len(indexes_below_limit))
    Path(f"{outdir}").mkdir(parents=True, exist_ok=True)
    with open(f"{outdir}/slctd_correlation_{date_str}.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["col1", "col2", "value", "limit"])
        writer.writeheader()
        writer.writerows(indexes_below_limit)



def get_disjunct_cliques(cliques):
    disjunkt_klikks = []

    for cg in sorted(cliques, key=len, reverse=True):
        if not disjunkt_klikks:
            disjunkt_klikks.append(set(cg))
        else:
            for dkl in disjunkt_klikks:
                if set(cg).issubset(dkl):
                    break
            else:
                disjunkt_klikks.append(set(cg))

    disjunkt_klikks = [list(d) for d in disjunkt_klikks]
    return disjunkt_klikks

def make_cliques(corr_df, date_str, min_clique, max_clique, outdir="dep/slctd/cliques"):
    #_df = _df[(_df["limit"] >= 0.4) & (_df["limit"] <= 0.6)]

    dfs = [corr_df[corr_df["limit"] == x] for x in corr_df.limit.unique().tolist()]

    headers = []
    list_to_write = []
    klikks_to_append = []

    for df in dfs:
        print(df)
        weights_dict = {}

        limit = df.limit.unique().tolist()[0]
        edges = pd.DataFrame({
            "source": df.col1.values,
            "target": df.col2.values,
            "weight": df.value.values,
        }
        )
        #####################

        # megnézni, hogy egy csúcsból ellehet-e jutni minden másik csúcsba
        csucsok = set(edges.source.values)
        csucsok.update(set(edges.target.values))
        csucsok = list(csucsok)
        talalt_csucsok = [csucsok[0]]
        sources = edges.source.values.tolist()
        targets = edges.target.values.tolist()

        index = 0

        while index < len(csucsok):
            try:
                cs = talalt_csucsok[index]
                for z1, z2 in zip(sources,targets):
                    if z1 == cs:
                        if z2 not in talalt_csucsok:
                            talalt_csucsok.append(z2)
                    elif z2 == cs:
                        if z1 not in talalt_csucsok:
                            talalt_csucsok.append(z1)
                index += 1
            except:
                break


        ######################
        if len(csucsok) == len(talalt_csucsok):

            G = nx.from_pandas_edgelist(edges, edge_attr=True)

            complete_graphs = [g for g in nx.enumerate_all_cliques(G) if len(g) > 2]
            print(f"összesen: {len(complete_graphs)}\n eloszlás:")
            klikks = []

            for _num_ in range(min_clique, max_clique):
                print(f"{_num_} -> {len([g for g in complete_graphs if len(g) == _num_])}")
                klikks.append(len([g for g in complete_graphs if len(g) == _num_]))

            klikks_to_append.append({"limit": limit,"data":klikks})
            print()


            disjunct_graphs = get_disjunct_cliques(complete_graphs)

            print("Diszjunkt klikkek: ")
            for _num_ in range(min_clique, max_clique):
                print(f"{_num_} -> {len([g for g in disjunct_graphs if len(g) == _num_])}")

            for g in disjunct_graphs:
                combs = combinations(g, 2)
                weights = []
                for c in combs:
                    if f"{c[0]}_{c[1]}" not in weights_dict.keys() and f"{c[1]}_{c[0]}" not in weights_dict.keys():
                        for index, row in edges.iterrows():
                            if (row["source"] == c[0] and row["target"] == c[1]) or (row["source"] == c[1] and row["target"] == c[0]):
                                if row["source"] == c[0] and row["target"] == c[1]:
                                    weights_dict[f"{c[0]}_{c[1]}"] = row["weight"]
                                elif row["source"] == c[1] and row["target"] == c[0]:
                                    weights_dict[f"{c[1]}_{c[0]}"] = row["weight"]
                                weights.append(row["weight"])
                    else:
                        if f"{c[0]}_{c[1]}" in weights_dict.keys():
                            weights.append(weights_dict[f"{c[0]}_{c[1]}"])
                        elif f"{c[1]}_{c[0]}" in weights_dict.keys():
                            weights.append(weights_dict[f"{c[1]}_{c[0]}"])
                max_weight = max(weights)
                min_weight = min(weights)
                avg_weight = np.mean(weights)
                std_weight = np.std(weights)

                dict_to_write = {}
                dict_to_write["limit"] = limit
                dict_to_write["feature"] = g
                dict_to_write["weights"] = weights
                dict_to_write["max_weight"] = max_weight
                dict_to_write["min_weight"] = min_weight
                dict_to_write["avg_weight"] = avg_weight
                dict_to_write["std_weight"] = std_weight
                dict_to_write["clique_length"] = len(g)
                list_to_write.append(dict_to_write)
        else:
            print(f"{limit} not general graph!!!")

    headers.append("feature")
    headers.append("weights")
    headers.append("max_weight")
    headers.append("min_weight")
    headers.append("avg_weight")
    headers.append("std_weight")
    headers.append("clique_length")
    headers.append("limit")
    Path(f"{outdir}").mkdir(parents=True, exist_ok=True)
    with open(f"{outdir}/slctd_cliques_length_{date_str}.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(list_to_write)


if __name__ == "__main__":

    """
    Calculates the correlation matrix use make_corr2() if the .csv at @df_path is a corr matrix itself.
    Then it calculates the cliques between min_lim and max_lim at every step_lim.
    For exmaple: min_lim == 200, max_lim == 301, step_lim == 50 then it calculates the cliques for limit 0.2, 0.25, 0.3
    """



    filepath = r"path_of_csv/for/data"
    #files = [f for f in os.listdir(dirpath)]

    fn = filepath.split("\\")[-1]

    #might need to change this, depends on the filepath
    strname = "_".join(fn.split("_")[0:3])
    print(strname)

    date_str = strname + "_20230829" #for example

    #which class to compare
    cc = 2

    #divided by 1000  so  min_lim = 150 /1000  -> 0.15
    min_lim = 150
    max_lim = 200
    step_lim = 50

    corr_dirpath = r"Elso\random\replaced"
    make_corr(f"{filepath}", date_str, min_lim, max_lim, step_lim, 3, outdir=r"Elso\random\replaced\slctd")
    corr_df = pd.read_csv(f"{corr_dirpath}\\slctd\\slctd_correlation_nrv_newest.csv_20230810.csv")
    print("make_cliques has started!")
    make_cliques(corr_df, date_str, 3, 10, outdir=r"Elso\random\replaced\cliques")


    sys.exit()



