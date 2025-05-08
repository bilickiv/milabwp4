import os
import sys



#csv  for features
features = r"I:\Munka\Elso\random\replaced\nrv_newest.csv"

#csv for cliques
cliques = [r"I:\Munka\Elso\random\replaced\cliques\slctd_cliques_length_nrv_newest.csv_20230810.csv"]

#list of  which class to compare in which run
ccs = [3]

outfile_name = "3kfold_norveg_dep_javitott"
date = "20230815"

steps = 100
for idx, cli in enumerate(cliques):
    text_arg = f"{outfile_name}_1{ccs[idx]}_{date}"
    for i in range(0, 2000, steps):
        error_code = command = f"python shap_rewrite_uj.py {i} {i + steps} {text_arg}_{i} {ccs[idx]} {features} {cli} "
        if error_code == 42:
            sys.exit()
        os.system(command)









