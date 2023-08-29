import pandas as pd
import numpy as np
import ast
from pathlib import Path
pd.set_option('display.width', 300)
np.set_printoptions(linewidth=300)
pd.set_option('display.max_columns',300)
import os

replace_dict = {
    "length_of_sleep_in_minutes": "slp_tm",
    "data_length_bigger" : "p_l",
    "avg_bigger_values" : "avg(p_a)",
    "max_bigger_values": "max(p_a)",
    "min_bigger_values": "min(p_a)",
    "median_length_between_bigger": "med(s_l)",
    "avg_length_between_bigger": "avg(s_l)",
    "max_length_between_bigger": "max(s_l)",
    "min_length_between_bigger": "min(s_l)",

    "upper_humps_median": "med(lp_a)",
    "upper_humps_mean": "avg(lp_a)",
    "upper_humps_max": "max(lp_a)",
    "upper_humps_min": "min(lp_a)",

    "upper_humps_width_median": "med(lp_l)",
    "upper_humps_width_mean": "avg(lp_l)",
    "upper_humps_width_max": "max(lp_l)",
    "upper_humps_width_min": "min(lp_l)",

    "upper_humps_median_distance": "med(lp_d)",
    "upper_humps_avg_distance": "avg(lp_d)",
    "upper_humps_min_distance": "min(lp_d)",
    "upper_humps_max_distance": "max(lp_d)",

    "lower_humps_median": "med(sp_a)",
    "lower_humps_mean": "avg(sp_a)",
    "lower_humps_max": "max(sp_a)",
    "lower_humps_min": "min(sp_a)",

    "lower_humps_width_median": "med(sp_l)",
    "lower_humps_width_mean": "avg(sp_l)",
    "lower_humps_width_max": "max(sp_l)",
    "lower_humps_width_min": "min(sp_l)",

    "lower_humps_median_distance": "med(sp_d)",
    "lower_humps_avg_distance": "avg(sp_d)",
    "lower_humps_min_distance": "min(sp_d)",
    "lower_humps_max_distance": "max(sp_d)",

    "number_of_lower_humps": "nbr(sp)",
    "number_of_upper_humps": "nbr(lp)",

    "number_of_close_big_humps": "nbr(lp_cls)",
    "first_big_hump_length": "lp_first_l",
    "last_big_hump_length": "lp_last_l",

    "upper_humps_median_qrt": "med(lp_q_a)",
    "upper_humps_mean_qrt": "avg(lp_q_a)",
    "upper_humps_max_qrt": "max(lp_q_a)",
    "upper_humps_min_qrt": "min(lp_q_a)",

    "upper_humps_width_median_qrt": "med(lp_q_l)",
    "upper_humps_width_mean_qrt": "avg(lp_q_l)",
    "upper_humps_width_max_qrt": "max(lp_q_l)",
    "upper_humps_width_min_qrt": "min(lp_q_l)",

    "upper_humps_median_distance_qrt": "med(lp_q_d)",
    "upper_humps_avg_distance_qrt": "avg(lp_q_d)",
    "upper_humps_min_distance_qrt": "min(lp_q_d)",
    "upper_humps_max_distance_qrt": "max(lp_q_d)",

    "lower_humps_median_qrt": "med(sp_q_a)",
    "lower_humps_mean_qrt": "avg(sp_q_a)",
    "lower_humps_max_qrt": "max(sp_q_a)",
    "lower_humps_min_qrt": "min(sp_q_a)",

    "lower_humps_width_median_qrt": "med(sp_q_l)",
    "lower_humps_width_mean_qrt": "avg(sp_q_l)",
    "lower_humps_width_max_qrt": "max(sp_q_l)",
    "lower_humps_width_min_qrt": "min(sp_q_l)",

    "lower_humps_median_distance_qrt": "med(sp_d)",
    "lower_humps_avg_distance_qrt": "avg(sp_q_d)",
    "lower_humps_min_distance_qrt": "min(sp_q_d)",
    "lower_humps_max_distance_qrt": "max(sp_q_d)",

    "number_of_lower_humps_qrt": "nbr(sp_q)",
    "number_of_upper_humps_qrt": "nbr(lp_q)",

    "number_of_close_big_humps_qrt": "nbr(lp_q_cls))",
    "first_big_hump_length_qrt": "lp_q_first_l",
    "last_big_hump_length_qrt": "lp_q_last_l",


}

def replace_features(feature_list):
    try:
        features = ast.literal_eval(feature_list)
        replaced_features = [replace_dict.get(item, item) if item in replace_dict else item for item in features]
        return str(replaced_features)
    except (SyntaxError, ValueError):
        return feature_list


if __name__ == "__main__":
    outdir = r"I:\Munka\Elso\project\ML\SHAPML\NEW\20230808\új2\merged\replaced"

    dfs = [r"I:\Munka\Elso\project\ML\SHAPML\NEW\20230808\új2\merged\ML_model_merged.csv",
           r"I:\Munka\Elso\project\ML\SHAPML\NEW\20230808\új2\merged\shap_merged.csv",
           ]

    

    for df_path in dfs:
        fname = df_path.split('\\')[-1]

        df = pd.read_csv(df_path)
        target = "features" if "features" in df.columns.values.tolist() else "columns" if "columns" in df.columns.values.tolist() else None
        df[target] = df[target].apply(replace_features)
        #df.drop(columns=["Unnamed: 0.1", "Unnamed: 0"], inplace=True)
        #old_cols = df.columns.values.tolist()
        #new_cols = [replace_dict.get(d, d) for d in old_cols]
        #print(old_cols)
        #print(new_cols)
        #df.rename(columns=dict(zip(old_cols, new_cols)), inplace=True)

        Path(f"{outdir}").mkdir(parents=True, exist_ok=True)
        df.to_csv(f"{outdir}/{fname}")


    df1 = pd.read_csv(r"I:\Munka\Elso\csvk_col_csere\ML_szte_20230621\shap_valuesszte_23230620_merged.csv")
    df2 = pd.read_csv(r"I:\Munka\Elso\csvk_col_csere\shap_valuesszte_23230620_merged.csv")

    comparison = np.array_equal(df1['SHAP_values'].values, df2['SHAP_values'].values)
    print(f"The 'SHAP_value' columns are equal: {comparison}")

    """
    dirname = r"I:\Munka\Elso\csvk_col_csere\alap"

    for dfp in os.listdir(dirname):
        if dfp.endswith(".csv"):
            df = pd.read_csv(f"{dirname}/{dfp}")
            df = df.rename(columns=replace_dict)
            df.to_csv(f"{dirname}\\replaced\\{dfp}")
    """


